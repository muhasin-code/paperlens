"""
Tests for the EmbeddingModel wrapper.

The real sentence-transformers model is mocked so tests run offline in CI.
We verify: batch embedding returns one EmbeddedChunk per Chunk, vectors are
normalized (unit length), and the query prefix is applied to queries only.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.paperlens.embedding.embedder import BGE_QUERY_PREFIX, EmbeddingModel
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


def _make_chunk(text: str, cid: str = "2401.00001v1_chunk_0001") -> Chunk:
    return Chunk(
        chunk_id=cid,
        arxiv_id="2401.00001v1",
        title="Test Paper",
        authors=["Alice"],
        section_label="method",
        chunk_index=0,
        page_start=3,
        page_end=3,
        text=text,
        token_count=10,
        total_chunks_in_section=1,
        total_chunks_in_paper=1,
    )


@pytest.fixture
def fake_model() -> MagicMock:
    """A mocked SentenceTransformer returning fixed unit vectors."""
    m = MagicMock()
    m.get_embedding_dimension.return_value = 4  # This matches embedder.py line 36

    def _encode(texts, **kwargs):
        out = []
        for i, _ in enumerate(texts):
            vec = np.zeros(4, dtype=np.float32)
            vec[i % 4] = 1.0
            out.append(vec)
        return np.array(out, dtype=np.float32)

    m.encode.side_effect = _encode
    return m


@pytest.fixture
def embedder(fake_model: MagicMock, request) -> EmbeddingModel:
    patcher = patch("src.paperlens.embedding.embedder.SentenceTransformer", return_value=fake_model)
    e = EmbeddingModel(Settings(), batch_size=2)
    request.addfinalizer(patcher.stop)
    return e


class TestEmbedChunks:
    def test_one_embedded_chunk_per_input(self, embedder: EmbeddingModel) -> None:
        chunks = [_make_chunk("alpha"), _make_chunk("beta", cid="2401.00001v1_chunk_0002")]
        embedded = embedder.embed_chunks(chunks)
        assert len(embedded) == 2
        assert embedded[0].chunk.chunk_id == "2401.00001v1_chunk_0001"

    def test_vectors_are_unit_normalized(self, embedder: EmbeddingModel) -> None:
        embedded = embedder.embed_chunks([_make_chunk("gamma")])
        vec = np.array(embedded[0].embedding)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_dimension_exposed(self, embedder: EmbeddingModel) -> None:
        embedded = embedder.embed_chunks([_make_chunk("delta")])
        assert embedded[0].dimension == 4


class TestEmbedQuery:
    def test_query_is_prefixed(self, embedder: EmbeddingModel, fake_model: MagicMock) -> None:
        embedder.embed_query("how to train?")
        # The first positional arg to encode should carry the BGE prefix.
        called_texts = fake_model.encode.call_args[0][0]
        assert called_texts[0].startswith(BGE_QUERY_PREFIX)

    def test_query_vector_is_unit_normalized(self, embedder: EmbeddingModel) -> None:
        vec = np.array(embedder.embed_query("normalize me"))
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)
