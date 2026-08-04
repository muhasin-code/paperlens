"""RAG service: retrieval → context assembly → Ollama generation → cited answer."""

import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import ollama
from pydantic import BaseModel

from src.paperlens.api.prompt_loader import PromptLoader
from src.paperlens.api.schemas import Citation, QueryRequest, QueryResponse
from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import RetrievalResult, SemanticRetriever
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.retrieval.reranker import CrossEncoderReranker
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.api")

# Fallback model if primary fails (CPU-friendly, already small)
FALLBACK_MODEL = "llama3.2:1b"


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


class LLMOutput(BaseModel):
    """Structured validation for LLM output."""

    answer: str

    @classmethod
    def validate_response(
        cls, text: str, refusal_message: str, has_context: bool = True
    ) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, "invalid"

        text = text.strip()

        # Exact match
        if text == refusal_message:
            return False, refusal_message

        # Check for citation markers
        has_citations = bool(re.search(r"\[[^\]]+\]", text))

        if not has_citations:
            # If we have context but no citations, likely a refusal
            if has_context:
                # Check for refusal-like language
                refusal_phrases = [
                    "cannot answer",
                    "can't answer",
                    "unable to answer",
                    "don't know",
                    "do not know",
                    "insufficient information",
                    "not enough information",
                    "not in the context",
                    "not provided in the context",
                    "context does not contain",
                    "no information",
                    "cannot find",
                    "i don't know",
                    "i cannot",
                    "i am unable",
                    "not mentioned",
                    "not found",
                ]
                text_lower = text.lower()
                if any(phrase in text_lower for phrase in refusal_phrases):
                    return False, refusal_message
                # Even without explicit refusal phrases, if context exists but no citations,
                # treat as implicit refusal to avoid retries
                return False, refusal_message
            return False, "invalid"

        return True, text


class RAGService:
    """Orchestrates the end-to-end RAG query pipeline."""

    def __init__(
        self,
        settings: Settings,
        retriever: SemanticRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self.settings = settings
        self._hybrid_retriever = hybrid_retriever
        # # Use cached embedding model via singleton retriever
        # self.retriever = retriever or SemanticRetriever(
        #     settings, embedder=_get_embedding_model(settings)
        # )
        cached_embedder = _get_embedding_model(settings)
        if hybrid_retriever is not None:
            self.retriever = hybrid_retriever
        elif retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(settings=settings, embedder=cached_embedder)

        # Store reranker instance (model is a class-level singleton, loaded once)
        self._reranker = CrossEncoderReranker(self.settings)

        # Store VectorStore for health checks
        self._vector_store = VectorStore(self.settings)

        self._prompt_loader = PromptLoader(self.settings)
        self._prompt_template = self._prompt_loader.load_prompt()

        self._ollama_client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.primary_model = settings.ollama_model
        self.fallback_model = FALLBACK_MODEL
        # Health check cache (TTL 10 seconds)
        self._health_cache: tuple[float, dict] | None = None
        self._health_ttl = 10.0

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute the full RAG pipeline for a single query."""
        total_start = time.perf_counter()

        # 1) Retrieval
        retrieval_start = time.perf_counter()
        results: list[RetrievalResult] = self.retriever.search(
            query=request.query, top_k=request.top_k
        )
        # 1b) Reranking (after hybrid retrieval)
        rerank_start = time.perf_counter()
        results = self._reranker.rerank(
            query=request.query, results=results, top_k=self.settings.rerank_top_k
        )
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        logger.info("Reranking completed: %d results, %.1f ms", len(results), rerank_time_ms)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if not results:
            return QueryResponse(
                answer="I cannot answer from the provided sources.",
                citations=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0.0,
                total_time_ms=(time.perf_counter() - total_start) * 1000,
            )

        # 2) Context assembly
        context = self._assemble_context(results)

        # 3) LLM generation (with fallback)
        generation_start = time.perf_counter()
        gen_result = await self._generate_with_fallback(request.query, context, request.model)
        generation_time_ms = (time.perf_counter() - generation_start) * 1000
        total_time_ms = (time.perf_counter() - total_start) * 1000
        # Check for refusal response
        if self._prompt_template.refusal_message in gen_result.text:
            return QueryResponse(
                answer=self._prompt_template.refusal_message,
                citations=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=total_time_ms,
            )

        # 4) Build citations from retrieval results
        citations = self._build_citations(results)

        # Heuristic confidence from retrieval scores
        if results:
            top_scores = [r.score for r in results[:3]]
            confidence = max(0.0, min(1.0, sum(top_scores) / len(top_scores)))
        else:
            confidence = 0.0

        return QueryResponse(
            answer=gen_result.text.strip(),
            citations=citations,
            confidence=confidence,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
        )

    async def query_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Execute the full RAG pipeline with SSE streaming."""
        total_start = time.perf_counter()

        # 1) Retrieval
        retrieval_start = time.perf_counter()
        results: list[RetrievalResult] = self.retriever.search(
            query=request.query, top_k=request.top_k
        )
        # 1b) Reranking
        rerank_start = time.perf_counter()
        results = self._reranker.rerank(
            query=request.query, results=results, top_k=self.settings.rerank_top_k
        )
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        logger.info("Reranking completed: %d results, %.1f ms", len(results), rerank_time_ms)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if not results:
            yield f"data: {json.dumps({'answer': 'I cannot answer from the provided sources.', 'citations': [], 'confidence': 0.0, 'retrieval_time_ms': retrieval_time_ms, 'generation_time_ms': 0, 'total_time_ms': (time.perf_counter() - total_start) * 1000, 'done': True})}\n\n"
            return

        # 2) Context assembly
        context = self._assemble_context(results)

        # 3) Build citations
        citations = self._build_citations(results)

        # Send initial metadata
        yield f"data: {json.dumps({'citations': [c.model_dump() for c in citations], 'retrieval_time_ms': retrieval_time_ms})}\n\n"

        # 4) LLM generation (with fallback) - stream tokens
        generation_start = time.perf_counter()
        prompt = self._prompt_template.format(context=context, query=request.query)
        model = request.model or self.primary_model
        response_text = ""

        for attempt, m in enumerate((model, self.fallback_model)):
            try:
                logger.info("Generating with model=%s (attempt %d)", m, attempt + 1)
                stream = await self._ollama_client.generate(
                    model=m,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 256},
                    stream=True,
                )
                async for chunk in stream:
                    token = chunk.get("response", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                        response_text += chunk
                # Generation complete
                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                total_time_ms = (time.perf_counter() - total_start) * 1000
                # Heuristic confidence from retrieval scores
                if results:
                    top_scores = [r.score for r in results[:3]]
                    confidence = max(0.0, min(1.0, sum(top_scores) / len(top_scores)))
                else:
                    confidence = 0.0
                yield f"data: {json.dumps({'generation_time_ms': generation_time_ms, 'total_time_ms': total_time_ms, 'confidence': confidence, 'model': m, 'done': True})}\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama generation failed with %s: %s", m, exc)
                if attempt == 0:
                    continue  # try fallback
                raise RuntimeError(
                    f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed"
                ) from exc

        # Check for refusal
        if self._prompt_template.refusal_message in response_text:
            yield f"data: {json.dumps({'answer': self._prompt_template.refusal_message, 'citations': [], 'confidence': 0.0, 'generation_time_ms': generation_time_ms, 'total_time_ms': total_time_ms, 'done': True})}\n\n"
            return

        raise RuntimeError("Generation loop exited unexpectedly")

    def _assemble_context(self, results: list[RetrievalResult]) -> str:
        """Concatenate chunk texts with [chunk_id] markers."""
        parts = []
        for r in results:
            cid = r.chunk.chunk_id
            txt = r.chunk.text.strip()
            parts.append(f"[{cid}] {txt}")
        return "\n---\n".join(parts)

    async def _generate_with_fallback(
        self, query: str, context: str, model_override: str | None, attempt: int = 0
    ) -> GenerationResult:
        """Call Ollama generate; on failure, retry once for malformed output, then fall back.

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
        prompt = self._prompt_template.build_prompt(context=context, query=query)
        model = model_override or self.primary_model
        refusal_msg = self._prompt_template.refusal_message
        has_context = bool(context and context.strip())

        max_retries = 2
        for retry in range(max_retries + 1):
            try:
                logger.info(
                    "Generating with model=%s (attempt %d, retry %d)",
                    model,
                    attempt + 1,
                    retry + 1,
                )
                start = time.perf_counter()
                resp = await self._ollama_client.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 256},
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = resp.get("response", "")
                logger.debug("LLM raw response: %s", text[:200])

                is_valid, result = LLMOutput.validate_response(text, refusal_msg, has_context)

                if is_valid:
                    return GenerationResult(text=text, model_used=model, latency_ms=latency_ms)

                if result == refusal_msg:
                    return GenerationResult(text=text, model_used=model, latency_ms=latency_ms)

                logger.warning(
                    "LLM output validation failed (attempt %d): response missing citation markers",
                    retry + 1,
                )

                if retry < max_retries:
                    continue

            except Exception as exc:
                logger.warning("Ollama generation failed with %s: %s", model, exc)
                if attempt == 0:
                    # Switch to fallback model for the retry
                    model = self.fallback_model
                    attempt = 1
                    continue
                raise RuntimeError(
                    f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed"
                ) from exc

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
        """Check ChromaDB collection count and Ollama reachability (cached 10s)."""
        now = time.perf_counter()
        if self._health_cache and (now - self._health_cache[0]) < self._health_ttl:
            return self._health_cache[1]

        chroma_count = self._vector_store.count()

        ollama_ok = False
        try:
            await self._ollama_client.list()
            ollama_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama health check failed: %s", exc)

        result = {
            "chroma_vector_count": chroma_count,
            "ollama_reachable": ollama_ok,
        }
        self._health_cache = (now, result)
        return result
