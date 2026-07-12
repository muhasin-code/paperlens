"""Tests for Gradio UI formatting helpers (pure functions, no network)."""

from src.paperlens.ui.formatters import format_citations, format_health, format_latency


class TestFormatCitations:
    def test_empty_list_returns_message(self) -> None:
        assert format_citations([]) == "*No sources cited.*"
        assert format_citations(None) == "*No sources cited.*"

    def test_single_citation_renders_details(self) -> None:
        citations = [
            {
                "chunk_id": "chunk_1",
                "arxiv_id": "2305.14314",
                "title": "QLoRA",
                "authors": ["Dettmers", "Lewis"],
                "section_label": "Abstract",
                "page_start": 1,
                "page_end": 2,
                "score": 0.85,
                "rank": 1,
            }
        ]
        md = format_citations(citations)
        assert "[chunk_1]" in md
        assert "QLoRA" in md
        assert "2305.14314" in md
        assert "Dettmers, Lewis" in md
        assert "Abstract" in md
        assert "0.850" in md
        assert "p. 1" in md

    def test_multiple_citations_sorted_by_rank(self) -> None:
        citations = [
            {
                "chunk_id": "c2",
                "arxiv_id": "a2",
                "title": "B",
                "authors": ["B"],
                "section_label": "M",
                "page_start": 2,
                "page_end": 2,
                "score": 0.7,
                "rank": 2,
            },
            {
                "chunk_id": "c1",
                "arxiv_id": "a1",
                "title": "A",
                "authors": ["A"],
                "section_label": "I",
                "page_start": 1,
                "page_end": 1,
                "score": 0.9,
                "rank": 1,
            },
        ]
        md = format_citations(citations)
        # c1 should appear before c2
        assert md.index("c1") < md.index("c2")


class TestFormatLatency:
    def test_formats_all_fields(self) -> None:
        md = format_latency(100.0, 400.0, 500.0)
        assert "500.0ms" in md
        assert "Retrieval" in md
        assert "Generation" in md
        assert "100.0ms" in md
        assert "400.0ms" in md

    def test_handles_none_values(self) -> None:
        md = format_latency(None, None, None)
        assert "0.0ms" in md


class TestFormatHealth:
    def test_healthy_status(self) -> None:
        health = {
            "status": "ok",
            "chroma_collection": "arxiv_papers",
            "chroma_vector_count": 524,
            "ollama_reachable": True,
            "ollama_model": "phi4-mini",
            "ollama_fallback_model": "llama3.2:1b",
        }
        status_text, color = format_health(health)
        assert "Healthy" in status_text
        assert "524" in status_text
        assert "phi4-mini" in status_text
        assert color == "green"

    def test_degraded_status(self) -> None:
        health = {
            "status": "ok",
            "chroma_collection": "arxiv_papers",
            "chroma_vector_count": 524,
            "ollama_reachable": False,
            "ollama_model": "phi4-mini",
            "ollama_fallback_model": "llama3.2:1b",
        }
        status_text, color = format_health(health)
        assert "Degraded" in status_text
        assert "llama3.2:1b" in status_text
        assert color == "amber"

    def test_unhealthy_status(self) -> None:
        status_text, color = format_health(None)
        assert "Offline" in status_text
        assert color == "red"
