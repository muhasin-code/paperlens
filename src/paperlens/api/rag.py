"""RAG service: retrieval → context assembly → Ollama generation → cited answer."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import ollama

from src.paperlens.api.schemas import Citation, QueryRequest, QueryResponse
from src.paperlens.embedding.embedder import EmbeddingModel
from src.paperlens.embedding.retriever import RetrievalResult, SemanticRetriever
from src.paperlens.retrieval.hybrid import HybridRetriever
from src.paperlens.settings import Settings

logger = logging.getLogger("paperlens.api")

# Fallback model if primary fails (CPU-friendly, already small)
FALLBACK_MODEL = "llama3.2:1b"

# Prompt template (moved to configs/prompts/ in Phase 2.4)
RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions about ML/AI papers from arXiv.
Use ONLY the provided context chunks to answer. Each chunk is marked with its [chunk_id].
Cite every claim by including the [chunk_id] in square brackets at the end of the sentence.
If the context does not contain enough information to answer, respond exactly:
"I cannot answer from the provided sources."

Context chunks:
{context}

Question: {query}

Answer:"""


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

        # 4) Build citations from retrieval results
        citations = self._build_citations(results)

        # 5) Confidence heuristic (placeholder for Phase 2.4 citation validator)
        confidence = self._compute_confidence(results)

        total_time_ms = (time.perf_counter() - total_start) * 1000

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
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=request.query)
        model = request.model or self.primary_model

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
                # Generation complete
                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                total_time_ms = (time.perf_counter() - total_start) * 1000
                confidence = self._compute_confidence(results)
                yield f"data: {json.dumps({'generation_time_ms': generation_time_ms, 'total_time_ms': total_time_ms, 'confidence': confidence, 'model': m, 'done': True})}\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama generation failed with %s: %s", m, exc)
                if attempt == 0:
                    continue  # try fallback
                raise RuntimeError(
                    f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed"
                ) from exc

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
        self, query: str, context: str, model_override: str | None
    ) -> GenerationResult:
        """Call Ollama generate; on failure, retry once with fallback model."""
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
        model = model_override or self.primary_model

        for attempt, m in enumerate((model, self.fallback_model)):
            try:
                logger.info("Generating with model=%s (attempt %d)", m, attempt + 1)
                start = time.perf_counter()
                resp = await self._ollama_client.generate(
                    model=m,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 256},
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = resp.get("response", "")
                if not text.strip():
                    raise ValueError("Empty response from Ollama")
                return GenerationResult(text=text, model_used=m, latency_ms=latency_ms)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama generation failed with %s: %s", m, exc)
                if attempt == 0:
                    continue  # try fallback
                raise RuntimeError(
                    f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed"
                ) from exc

        raise RuntimeError("Generation loop exited unexpectedly")

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

    def _compute_confidence(self, results: list[RetrievalResult]) -> float:
        """Heuristic: mean of top-3 scores, clamped to [0, 1]."""
        if not results:
            return 0.0
        top_scores = [r.score for r in results[:3]]
        mean_score = sum(top_scores) / len(top_scores)
        return max(0.0, min(1.0, mean_score))

    async def health_check(self) -> dict:
        """Check ChromaDB collection count and Ollama reachability (cached 10s)."""
        now = time.perf_counter()
        if self._health_cache and (now - self._health_cache[0]) < self._health_ttl:
            return self._health_cache[1]

        from src.paperlens.embedding.vector_store import VectorStore

        store = VectorStore(self.settings)
        chroma_count = store.count()

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
