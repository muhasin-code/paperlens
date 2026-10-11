"""Unit tests for QueryRouter."""

import pytest

from src.paperlens.extraction.router import QueryRouter, RouterDecision
from src.paperlens.settings import Settings


class TestQueryRouter:
    """Tests for QueryRouter classification."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        """Create a router instance with default settings."""
        return QueryRouter()

    def test_structured_query_detection(self, router: QueryRouter):
        """Queries with extraction keywords should be classified as structured."""
        query = "What datasets were used in this method?"
        result = router.classify(query)

        assert result.mode == "structured"
        assert result.confidence > 0
        assert "what datasets" in result.matched_keywords or result.confidence >= 0.3

    def test_conversational_query_detection(self, router: QueryRouter):
        """Queries asking general questions should be classified as conversational."""
        query = "How does this model work?"
        result = router.classify(query)

        assert result.mode == "conversational"
        assert result.confidence > 0
        assert "how does" in result.matched_keywords

    def test_extract_keyword_triggers_structured(self, router: QueryRouter):
        """Explicit 'extract' queries should route to structured."""
        query = "Extract the method and key findings from this paper."
        result = router.classify(query)

        assert result.mode == "structured"

    def test_research_question_triggers_structured(self, router: QueryRouter):
        """Queries about research questions should be structured."""
        query = "What research question does this paper address?"
        result = router.classify(query)

        assert result.mode == "structured"

    def test_key_finding_keyword(self, router: QueryRouter):
        """Queries about key findings should be structured."""
        query = "What is the key finding of this work?"
        result = router.classify(query)

        assert result.mode == "structured"

    def test_summary_request_is_structured(self, router: QueryRouter):
        """Requests to summarize method should be structured extraction."""
        query = "Summarize the methodology used in this paper."
        result = router.classify(query)

        assert result.mode == "structured"

    def test_compare_query_is_conversational(self, router: QueryRouter):
        """Comparison queries are conversational, not extraction."""
        query = "Compare this approach with prior work."
        result = router.classify(query)

        assert result.mode == "conversational"

    def test_explain_query_is_conversational(self, router: QueryRouter):
        """Explain queries are conversational."""
        query = "Explain how the attention mechanism works."
        result = router.classify(query)

        assert result.mode == "conversational"

    def test_empty_query_defaults_conversational(self, router: QueryRouter):
        """Empty queries should default to conversational (safer fallback)."""
        result = router.classify("")

        assert result.mode == "conversational"
        assert result.confidence > 0

    def test_gibberish_defaults_conversational(self, router: QueryRouter):
        """Gibberish queries with no keywords should default to conversational."""
        result = router.classify("asdf qwer zxcv")

        assert result.mode == "conversational"

    def test_router_decision_is_valid_pydantic(self, router: QueryRouter):
        """RouterDecision should be valid Pydantic model."""
        result = router.classify("What method was used?")

        assert isinstance(result, RouterDecision)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.matched_keywords, list)
        assert isinstance(result.total_keywords, int)

    def test_custom_threshold(self):
        """Router should respect custom threshold from settings."""
        settings = Settings(router_threshold=0.1)
        router = QueryRouter(settings)

        assert router.threshold == 0.1

    def test_custom_keywords(self):
        """Router should support custom keyword sets."""
        settings = Settings(
            router_structured_keywords="custom,keywords,only",
            router_conversational_keywords="other,kinds",
        )
        router = QueryRouter(settings)

        assert "custom" in router.structured_keywords
        assert "other" in router.conversational_keywords

    def test_mixed_query_with_structured_keywords(self, router: QueryRouter):
        """Queries with both structured and conversational keywords: structured wins."""
        query = "What datasets were used and how does the model work?"
        result = router.classify(query)

        # Structured keyword present, should be structured
        assert result.mode == "structured"

    def test_router_repr(self, router: QueryRouter):
        """Router __repr__ should show key configuration."""
        repr_str = repr(router)

        assert "QueryRouter" in repr_str
        assert "threshold=" in repr_str
        assert "structured_keywords=" in repr_str
