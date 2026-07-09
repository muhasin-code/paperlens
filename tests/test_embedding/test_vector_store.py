"""
Tests for the ChromaDB VectorStore.

Uses a temp persist dir so it never touches the real data/chroma index.
"""

from pathlib import Path

import pytest

from src.paperlens.embedding.models import EmbeddedChunk
from src.paperlens.embedding.vector_store import VectorStore
from src.paperlens.parsing.models import Chunk
from src.paperlens.settings import Settings


def _embedded(text: str, cid: str, dim: int = 4) -> EmbeddedChunk:
    chunk = Chunk(
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
    vec = [0.0] * dim
    vec[0] = 1.0
    return EmbeddedChunk(chunk=chunk, embedding=vec, dimension=dim)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    settings = Settings(chroma_persist_dir=tmp_path / "chroma")
    return VectorStore(settings)


class TestUpsertAndCount:
    def test_empty_count(self, store: VectorStore) -> None:
        assert store.count() == 0

    def test_upsert_increments_count(self, store: VectorStore) -> None:
        store.upsert([_embedded("a", "c1"), _embedded("b", "c2")])
        assert store.count() == 2

    def test_upsert_is_idempotent(self, store: VectorStore) -> None:
        store.upsert([_embedded("a", "c1")])
        store.upsert([_embedded("a-updated", "c1")])  # same id
        assert store.count() == 1


class TestQuery:
    def test_query_returns_nearest(self, store: VectorStore) -> None:
        m1 = _embedded("method text", "m1")
        m1.embedding = [1.0, 0.0, 0.0, 0.0]
        i1 = _embedded("intro text", "i1")
        i1.embedding = [0.0, 1.0, 0.0, 0.0]
        store.upsert([m1, i1])
        # Query vector identical to m1's vector -> m1 should rank first.
        resp = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert resp["ids"][0][0] == "m1"


class TestReset:
    def test_reset_clears_collection(self, store: VectorStore) -> None:
        store.upsert([_embedded("x", "x1")])
        store.reset()
        assert store.count() == 0
