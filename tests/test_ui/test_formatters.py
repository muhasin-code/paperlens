"""Tests for Gradio UI formatting helpers (pure functions, no network)."""

from src.paperlens.ui.app import format_citations, format_health, format_latency


class TestFormatCitations:
    def test_empty_list_returns_message(self) -> None:
        assert format_citations([]) == "_No sources retrieved._"

    def test_single_citation_renders_details(self) -> None:
        citations = [
            {
                "chunk_id": "2401.00001v1_chunk_0001",
                "arxiv_id": "2401.00001v1",
                "title": "Test Paper: Learning Rate Schedules",
                "authors": ["Alice Smith", "Bob Jones"],
                "section_label": "method",
                "page_start": 3,
                "page_end": 4,
                "score": 0.8512,
                "rank": 1,
            }
        ]
        md = format_citations(citations)
        assert "**[1] Test Paper: Learning Rate Schedules**" in md
        assert "2401.00001v1_chunk_0001" in md
        assert "arXiv ID:** 2401.00001v1" in md
        assert "Authors:** Alice Smith, Bob Jones" in md
        assert "Section:** method" in md
        assert "Pages:** 3–4" in md
        assert "Score:** 0.8512" in md
        assert "<details>" in md and "</details>" in md

    def test_multiple_citations_separated(self) -> None:
        citations = [
            {
                "chunk_id": "c1",
                "arxiv_id": "a1",
                "title": "Paper A",
                "authors": ["A"],
                "section_label": "intro",
                "page_start": 1,
                "page_end": 1,
                "score": 0.9,
                "rank": 1,
            },
            {
                "chunk_id": "c2",
                "arxiv_id": "a2",
                "title": "Paper B",
                "authors": ["B"],
                "section_label": "method",
                "page_start": 2,
                "page_end": 3,
                "score": 0.8,
                "rank": 2,
            },
        ]
        md = format_citations(citations)
        assert md.count("<details>") == 2
        assert "Paper A" in md and "Paper B" in md


class TestFormatLatency:
    def test_formats_all_fields(self) -> None:
        md = format_latency(150.5, 2500.3, 2650.8, 0.847)
        assert "Retrieval:** 150.5 ms" in md
        assert "Generation:** 2500.3 ms" in md
        assert "Total:** 2650.8 ms" in md
        assert "Confidence:** 0.847" in md

    def test_handles_zero_values(self) -> None:
        md = format_latency(0, 0, 0, 0.0)
        assert "Retrieval:** 0.0 ms" in md
        assert "Confidence:** 0.000" in md


class TestFormatHealth:
    def test_ok_status_green_badge(self) -> None:
        health = {
            "status": "ok",
            "chroma_collection": "paperlens_chunks",
            "chroma_vector_count": 17330,
            "ollama_reachable": True,
            "ollama_model": "phi4-mini",
            "ollama_fallback_model": "llama3.2:1b",
        }
        md = format_health(health)
        assert "status-ok-green" in md or "status-ok-" in md  # shields.io badge
        assert "paperlens_chunks (17330 vectors)" in md
        assert "✓ reachable" in md
        assert "phi4-mini" in md
        assert "llama3.2:1b" in md

    def test_degraded_status_orange_badge(self) -> None:
        health = {
            "status": "degraded",
            "chroma_collection": "paperlens_chunks",
            "chroma_vector_count": 17330,
            "ollama_reachable": False,
            "ollama_model": "phi4-mini",
            "ollama_fallback_model": "llama3.2:1b",
        }
        md = format_health(health)
        assert "status-degraded-orange" in md or "status-degraded-" in md
        assert "✗ unreachable" in md
