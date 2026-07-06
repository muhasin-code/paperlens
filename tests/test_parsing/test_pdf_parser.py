"""
Tests for the PDF parser.

Covers: section detection from synthetic page texts, heading matching, filtering.
"""

from src.paperlens.parsing.pdf_parser import PdfParser


class TestHeadingDetection:
    def test_detects_numbered_introduction(self) -> None:
        parser = PdfParser()
        pages = [
            "Title\nAuthors\nAbstract text here.",
            "1. Introduction\nWe study the problem of...",
            "2. Related Work\nPrior work has shown...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "introduction" in labels

    def test_detects_method_section(self) -> None:
        parser = PdfParser()
        pages = [
            "Abstract\nThis paper presents...",
            "3. Method\nOur approach consists of...",
            "3.1 Architecture\nThe model has...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "method" in labels

    def test_filters_references(self) -> None:
        parser = PdfParser()
        pages = [
            "1. Introduction\nWe present...",
            "5. Conclusion\nIn summary...",
            "References\n[1] Smith et al...",
        ]
        sections = parser.detect_sections(pages)
        labels = [s.label for s in sections]
        assert "references" not in labels

    def test_empty_pages_returns_empty_list(self) -> None:
        parser = PdfParser()
        assert parser.detect_sections([]) == []

    def test_pages_before_first_heading_assigned_to_abstract(self) -> None:
        parser = PdfParser()
        pages = [
            "Title and abstract text on page 1.",
            "Some more abstract content.",
            "1. Introduction\nNow we begin...",
        ]
        sections = parser.detect_sections(pages)
        assert sections[0].label == "abstract"
        assert sections[0].start_page == 1
