"""RAG service: retrieval → context assembly → LLM generation → cited answer."""

import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.paperlens.api.prompt_loader import PromptLoader
from src.paperlens.api.schemas import Citation, QueryRequest, QueryResponse
from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import RetrievalResult, SemanticRetriever
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.llm.base import LLMProvider
from src.paperlens.observability.langfuse_client import get_langfuse_client
from src.paperlens.observability.metrics import (
    track_citation,
    track_confidence,
    track_latency,
    track_tokens,
)
from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.api")


@dataclass
class GenerationResult:
    """Raw result from Ollama generation."""

    text: str
    model_used: str
    latency_ms: float


# Module-level singleton for EmbeddingModel (loaded once per process)
_embedding_model_cache: EmbeddingModel | None = None


def _get_embedding_model(settings: Settings) -> EmbeddingModel:
    """Return a cached EmbeddingModel instance."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        from src.paperlens.embedding.embedder import EmbeddingModel

        _embedding_model_cache = EmbeddingModel(settings)
    return _embedding_model_cache


def get_cached_embedding_model(settings: Settings) -> EmbeddingModel:
    """Return the process-wide EmbeddingModel. Used by Ragas runner."""
    return _get_embedding_model(settings=settings)


# Module-level singleton for Langfuse Observablity
_observer = None


def _get_observer():
    global _observer
    if _observer is None:
        _observer = get_langfuse_client()
    return _observer


class LLMOutput(BaseModel):
    """Structured validation for LLM output."""

    answer: str

    @classmethod
    def validate_response(
        cls, text: str, refusal_message: str, has_context: bool = True
    ) -> tuple[bool, str]:
        if not text or not str(text).strip():
            return False, "invalid"

        text = str(text).strip()

        # Only an exact refusal is a refusal. A missing [chunk_id] is malformed
        # and must be retried, not rewritten into the refusal sentence.
        if text == refusal_message:
            return False, refusal_message

        has_citations = bool(re.search(r"\[[^\]]+\]", text))
        if has_context and not has_citations:
            return False, "invalid"

        return True, text


class RAGService:
    """Orchestrates the end-to-end RAG query pipeline."""

    def __init__(
        self,
        settings: Settings,
        llm_provider: LLMProvider,
        retriever: SemanticRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        observer: Any | None = None,
    ) -> None:
        self.settings = settings
        self._hybrid_retriever = hybrid_retriever
        cached_embedder = _get_embedding_model(settings)
        if hybrid_retriever is not None:
            self.retriever = hybrid_retriever
        elif retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(settings=settings, embedder=cached_embedder)

        # VectorStore for health checks
        self._vector_store = VectorStore(self.settings)

        self._prompt_loader = PromptLoader(self.settings)
        self._prompt_template = self._prompt_loader.load_prompt()

        self.llm_provider = llm_provider
        self.primary_model = settings.llm_model or settings.ollama_model
        self.fallback_model = settings.llm_fallback_model or self.primary_model
        # Health check cache (TTL 10 seconds)
        self._health_cache: tuple[float, dict] | None = None
        self._health_ttl = 10.0
        self._observer = observer or _get_observer()

    def _user_prompt(self, context: str, query: str) -> str:
        """Build the user prompt from context and query."""
        return self._prompt_template.user_template.format(context=context, query=query)

    def _llm_options(self) -> dict:
        """Return standard LLM options shared by query() and query_stream()."""
        return {
            "temperature": 0.1,
            "num_predict": 1024,
            "system": self._prompt_template.system_prompt,
        }

    def _confidence_from_results(self, results: list[RetrievalResult]) -> float:
        """Compute confidence heuristic from top-3 retrieval scores."""
        if not results:
            return 0.0
        top_scores = [r.score for r in results[:3]]
        return max(0.0, min(1.0, sum(top_scores) / len(top_scores)))

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute the full RAG pipeline for a single query."""
        response, _contexts = await self.query_with_contexts(request)
        return response

    async def query_with_contexts(self, request: QueryRequest) -> tuple[QueryResponse, list[str]]:
        """query() plus the chunk texts the prompt saw, in rank order.

        Not used by routes.py. citations is empty on refusal; the text list
        is not. Empty retrieval returns contexts=[].
        """
        total_start = time.perf_counter()
        observer = self._observer

        def _finish(
            trace: Any,
            resp: QueryResponse,
            *,
            cited: bool,
            confidence: float,
        ) -> QueryResponse:
            trace.update(
                output={
                    "answer": resp.answer,
                    "mode": resp.mode,
                    "confidence": resp.confidence,
                    "citation_count": len(resp.citations or []),
                }
            )
            if observer.enabled:
                observer.latency_score(resp.total_time_ms, "p50")
                observer.score(name="citation_coverage", value=1.0 if cited else 0.0)
                observer.score(name="confidence", value=confidence)
            return resp

        try:
            with observer.trace(
                "rag.query",
                input=request.query,
                user_id=None,
                metadata={"query_length": len(request.query)},
            ) as trace:
                retrieval_start = time.perf_counter()
                results: list[RetrievalResult] = self.retriever.search(
                    query=request.query, top_k=request.top_k
                )
                retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
                track_latency("retrieval", retrieval_time_ms)
                contexts = [r.chunk.text for r in results]

                if not results:
                    total_time_ms = (time.perf_counter() - total_start) * 1000
                    track_citation(False)
                    track_confidence(0.0)
                    resp = QueryResponse(
                        mode="conversational",
                        answer="I cannot answer from the provided sources.",
                        citations=[],
                        confidence=0.0,
                        retrieval_time_ms=retrieval_time_ms,
                        generation_time_ms=0.0,
                        total_time_ms=total_time_ms,
                    )
                    logger.info(
                        "Retrieved %d chunks: %s",
                        len(results),
                        [f"{r.chunk.chunk_id} | {r.chunk.title[:80]}" for r in results],
                    )
                    return _finish(trace, resp, cited=False, confidence=0.0), contexts

                context = self._assemble_context(results)
                generation_start = time.perf_counter()
                gen_result = await self._generate_with_fallback(
                    request.query, context, request.model
                )
                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                track_latency("generation", generation_time_ms)
                track_tokens(
                    self.primary_model, len(context.split()) + len(gen_result.text.split())
                )
                total_time_ms = (time.perf_counter() - total_start) * 1000
                track_latency("rag.total", total_time_ms)

                if gen_result.text.strip() == self._prompt_template.refusal_message:
                    track_citation(False)
                    track_confidence(0.0)
                    resp = QueryResponse(
                        mode="conversational",
                        answer=self._prompt_template.refusal_message,
                        citations=[],
                        confidence=0.0,
                        retrieval_time_ms=retrieval_time_ms,
                        generation_time_ms=generation_time_ms,
                        total_time_ms=total_time_ms,
                    )
                    return _finish(trace, resp, cited=False, confidence=0.0), contexts

                citations = self._build_citations(results)
                is_cited = len(citations) > 0
                track_citation(is_cited)
                top_scores = [r.score for r in results[:3]]
                confidence = max(0.0, min(1.0, sum(top_scores) / len(top_scores)))
                track_confidence(confidence)
                resp = QueryResponse(
                    mode="conversational",
                    answer=gen_result.text.strip(),
                    citations=citations,
                    confidence=confidence,
                    retrieval_time_ms=retrieval_time_ms,
                    generation_time_ms=generation_time_ms,
                    total_time_ms=total_time_ms,
                )
                return _finish(trace, resp, cited=is_cited, confidence=confidence), contexts
        except Exception:
            track_citation(False)
            track_confidence(0.0)
            raise
        finally:
            if observer.enabled:
                observer.flush()

    async def query_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Execute the full RAG pipeline with SSE streaming."""
        total_start = time.perf_counter()
        observer = self._observer

        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            with observer.trace(
                "rag.query_stream",
                input=request.query,
                metadata={"query_length": len(request.query)},
            ) as trace:
                retrieval_start = time.perf_counter()
                results: list[RetrievalResult] = self.retriever.search(
                    query=request.query, top_k=request.top_k
                )
                retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
                track_latency("retrieval", retrieval_time_ms)

                if not results:
                    total_time_ms = (time.perf_counter() - total_start) * 1000
                    track_citation(False)
                    track_confidence(0.0)
                    trace.update(
                        output={
                            "answer": self._prompt_template.refusal_message,
                            "mode": "conversational",
                            "confidence": 0.0,
                            "citation_count": 0,
                        }
                    )
                    if observer.enabled:
                        observer.latency_score(total_time_ms, "p50")
                        observer.score(name="citation_coverage", value=0.0)
                        observer.score(name="confidence", value=0.0)
                    yield _sse(
                        {
                            "answer": self._prompt_template.refusal_message,
                            "citations": [],
                            "confidence": 0.0,
                            "retrieval_time_ms": round(retrieval_time_ms, 2),
                            "generation_time_ms": 0,
                            "total_time_ms": round(total_time_ms, 2),
                            "done": True,
                        }
                    )
                    return

                context = self._assemble_context(results)
                citations = self._build_citations(results)
                yield _sse(
                    {
                        "citations": [c.model_dump() for c in citations],
                        "retrieval_time_ms": round(retrieval_time_ms, 2),
                    }
                )

                user_prompt = self._user_prompt(context, request.query)
                options = self._llm_options()
                model = request.model or self.primary_model

                text = ""
                used_model = model
                ttft_ms: float | None = None
                chunk_count = 0
                generation_time_ms = 0.0

                for attempt, m in enumerate((model, self.fallback_model)):
                    attempt_text = ""
                    attempt_ttft: float | None = None
                    attempt_chunks = 0
                    stream_start = time.perf_counter()
                    try:
                        logger.info("Streaming with model=%s (attempt %d)", m, attempt + 1)
                        async for token in self.llm_provider.stream(
                            prompt=user_prompt, model=m, options=options
                        ):
                            if not token:
                                continue
                            if attempt_ttft is None:
                                attempt_ttft = (time.perf_counter() - stream_start) * 1000
                            attempt_chunks += 1
                            attempt_text += token
                            yield _sse({"token": token})
                        text = attempt_text
                        used_model = m
                        ttft_ms = attempt_ttft
                        chunk_count = attempt_chunks
                        generation_time_ms = (time.perf_counter() - stream_start) * 1000
                        break
                    except Exception as exc:
                        logger.warning("LLM stream failed with %s: %s", m, exc)
                        if attempt_text:
                            # Already yielded chunks — do not fall back
                            raise RuntimeError(
                                f"Stream failed after {attempt_chunks} chunks; not falling back"
                            ) from exc
                        if attempt == 0 and m != self.fallback_model:
                            continue
                        raise RuntimeError(
                            f"Both primary ({self.primary_model}) and "
                            f"fallback ({self.fallback_model}) models failed"
                        ) from exc

                total_time_ms = (time.perf_counter() - total_start) * 1000
                refused = text.strip() == self._prompt_template.refusal_message
                confidence = 0.0 if refused else self._confidence_from_results(results)
                track_latency("generation", generation_time_ms)
                track_latency("rag.total", total_time_ms)
                if ttft_ms is not None:
                    track_latency("stream.ttft", ttft_ms)
                track_citation(not refused and len(citations) > 0)
                track_confidence(confidence)

                elapsed_after_first = generation_time_ms - (ttft_ms or 0.0)
                chunks_per_sec = (
                    chunk_count / (elapsed_after_first / 1000.0) if elapsed_after_first > 0 else 0.0
                )
                final = {
                    "generation_time_ms": round(generation_time_ms, 2),
                    "total_time_ms": round(total_time_ms, 2),
                    "confidence": round(confidence, 4),
                    "model": used_model,
                    "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                    "chunk_count": chunk_count,
                    "chunks_per_sec": round(chunks_per_sec, 2),
                    "done": True,
                }
                if refused:
                    final["refused"] = True
                    final["citations"] = []
                trace.update(
                    output={
                        "answer": text.strip()
                        if not refused
                        else self._prompt_template.refusal_message,
                        "mode": "conversational",
                        "confidence": confidence,
                        "citation_count": 0 if refused else len(citations),
                    }
                )
                if observer.enabled:
                    observer.latency_score(total_time_ms, "p50")
                    observer.score(name="citation_coverage", value=0.0 if refused else 1.0)
                    observer.score(name="confidence", value=confidence)
                    if ttft_ms is not None:
                        observer.score(name="ttft_ms", value=round(ttft_ms, 2))
                yield _sse(final)
        except Exception:
            track_citation(False)
            track_confidence(0.0)
            raise
        finally:
            if observer.enabled:
                observer.flush()

    def _assemble_context(self, results: list[RetrievalResult]) -> str:
        """Concatenate chunk texts with [chunk_id] markers."""
        parts = []
        for r in results:
            cid = r.chunk.chunk_id
            txt = r.chunk.text.strip()
            parts.append(
                f"[{cid}] title: {r.chunk.title} | section: {r.chunk.section_label}\n{txt}"
            )
        return "\n---\n".join(parts)

    async def _generate_with_fallback(
        self, query: str, context: str, model_override: str | None, attempt: int = 0
    ) -> GenerationResult:
        """Call LLM provider generate; on failure, retry once for malformed output, then fall back.

        Args:
            query: Original query string
            context: Assembled context chunks
            model_override: Optional model override
            attempt: Current retry attempt (0 = first try, 1 = first retry, 2 = fallback)

        Retry logic:
        - First call (attempt 0): with primary model
        - If malformed output: retry with same model (attempt 1)
        - If second attempt also malformed: fall back to secondary model (attempt 2)
        - Maximum 2 retries to avoid compounding latency on slow CPU
        """
        user_prompt = self._user_prompt(context, query)
        options = self._llm_options()
        model = model_override or self.primary_model
        refusal_msg = self._prompt_template.refusal_message
        has_context = bool(context and context.strip())

        max_retries = 2
        last_text = ""
        last_latency_ms = 0.0
        for retry in range(max_retries + 1):
            try:
                logger.info(
                    "Generating with model=%s (attempt %d, retry %d)",
                    model,
                    attempt + 1,
                    retry + 1,
                )
                start = time.perf_counter()
                text = await self.llm_provider.generate(
                    prompt=user_prompt,
                    model=model,
                    options=options,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                last_text = text or ""
                last_latency_ms = latency_ms
                logger.info(
                    "LLM raw response (%d chars): %s",
                    len(last_text),
                    last_text[:400].replace("\n", " "),
                )

                is_valid, result = LLMOutput.validate_response(text, refusal_msg, has_context)

                if is_valid:
                    return GenerationResult(text=text, model_used=model, latency_ms=latency_ms)

                if result == refusal_msg:
                    return GenerationResult(
                        text=refusal_msg, model_used=model, latency_ms=latency_ms
                    )

                logger.warning(
                    "LLM output missing [chunk_id] citations (attempt %d)",
                    retry + 1,
                )
                if retry < max_retries:
                    continue

            except Exception as exc:
                logger.warning("LLM generation failed with %s: %s", model, exc)
                if attempt == 0:
                    model = self.fallback_model
                    attempt = 1
                    continue
                raise RuntimeError(
                    f"Both primary ({self.primary_model}) and fallback "
                    f"({self.fallback_model}) models failed"
                ) from exc

        # Retries exhausted but the model did write an answer. Show it.
        # Retrieval citations are still attached by query().
        if last_text.strip() and last_text.strip() != refusal_msg:
            logger.warning("Returning model output without citation markers after retries")
            return GenerationResult(text=last_text, model_used=model, latency_ms=last_latency_ms)

        raise RuntimeError("Generation retry logic exhausted unexpectedly")

    def _build_citations(self, results: list[RetrievalResult]) -> list[Citation]:
        """Convert RetrievalResult objects to Citation schema objects."""
        citations = []
        for r in results:
            c = r.chunk
            citations.append(
                Citation(
                    chunk_id=c.chunk_id,
                    arxiv_id=c.arxiv_id,
                    title=c.title,
                    authors=c.authors,
                    section_label=c.section_label,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    score=r.score,
                    rank=r.rank,
                )
            )
        return citations

    async def health_check(self) -> dict:
        """Check ChromaDB collection count and LLM provider reachability (cached 10s)."""
        now = time.perf_counter()
        if self._health_cache and (now - self._health_cache[0]) < self._health_ttl:
            return self._health_cache[1]

        chroma_count = self._vector_store.count()

        llm_ok = False
        try:
            llm_ok = await self.llm_provider.is_reachable()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM provider health check failed: %s", exc)

        result = {
            "chroma_vector_count": chroma_count,
            "llm_reachable": llm_ok,
        }
        self._health_cache = (now, result)
        return result

    def __repr__(self) -> str:
        return (
            f"RAGService(primary_model={self.primary_model!r}, "
            f"fallback_model={self.fallback_model!r}, "
            f"retriever_type={type(self.retriever).__name__})"
        )
