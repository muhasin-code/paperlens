"""Pydantic request/response schemas for the PaperLens API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for the /query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question over the arXiv corpus.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override default retrieval top-k (settings.retrieval_top_k).",
    )
    model: str | None = Field(
        default=None, description="Override default Ollama model (settings.ollama_model)."
    )


class Citation(BaseModel):
    """A single source citation returned with the answer."""

    chunk_id: str = Field(
        ..., description="Globally unique chunk identifier, e.g. '2301.07597v2_chunk_0001'."
    )
    arxiv_id: str = Field(..., description="Source paper arXiv ID with version.")
    title: str = Field(..., description="Source paper title.")
    authors: list[str] = Field(..., description="Source paper author list.")
    section_label: str = Field(
        ..., description="Section this chunk belongs to (e.g. 'method', 'abstract')."
    )
    page_start: int = Field(..., description="1-indexed starting page of the chunk.")
    page_end: int = Field(..., description="1-indexed ending page of the chunk.")
    score: float = Field(
        ..., description="Retrieval similarity score (cosine, higher = more relevant)."
    )
    rank: int = Field(..., description="1-based rank in the retrieved list.")


class QueryResponse(BaseModel):
    """Response payload for the /query endpoint."""

    answer: str = Field(..., description="Generated answer text.")
    citations: list[Citation] = Field(..., description="Source chunks used for the answer.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the answer (heuristic: mean of top-3 retrieval scores, capped).",
    )
    retrieval_time_ms: float = Field(
        ..., description="Time spent in semantic retrieval (embed + search)."
    )
    generation_time_ms: float = Field(..., description="Time spent in Ollama LLM generation.")
    total_time_ms: float = Field(..., description="End-to-end latency.")


class HealthResponse(BaseModel):
    """Response payload for the /health endpoint."""

    status: str = Field(..., description="'ok' if server is up and dependencies reachable.")
    chroma_collection: str = Field(..., description="ChromaDB collection name.")
    chroma_vector_count: int = Field(..., description="Number of vectors in the collection.")
    ollama_reachable: bool = Field(..., description="Whether Ollama responded to /api/tags.")
    ollama_model: str = Field(..., description="Configured Ollama model (settings.ollama_model).")
    ollama_fallback_model: str = Field(..., description="Fallback model name (llama3.2:1b).")


class Bm25SearchRequest(BaseModel):
    """Request payload for the /retrieval/bm25 debug endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language query for BM25 keyword search.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of BM25 results to return.")
