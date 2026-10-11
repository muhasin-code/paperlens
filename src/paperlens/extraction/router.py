"""Query router/classifier for routing between conversational RAG and structured extraction."""

from typing import Literal

from pydantic import BaseModel, Field

from src.paperlens.settings import Settings


class RouterDecision(BaseModel):
    """Result of query classification with confidence and reasoning."""

    mode: Literal["conversational", "structured"] = Field(..., description="Query routing decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Reason for the decision")
    matched_keywords: list[str] = Field(default_factory=list, description="Keywords that matched")
    total_keywords: int = Field(..., description="Total keywords checked")


class QueryRouter:
    """Keyword-based query classifier for routing decisions.

    Uses simple string matching on lowercase query text to classify queries
    as either "structured" (extraction) or "conversational" (RAG).

    Designed for CPU-only operation with <50ms latency.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize router with configurable keywords and threshold.

        Args:
            settings: Application settings (provides router_threshold, keywords).
                      If None, uses defaults.
        """
        self.settings = settings or Settings()
        self._parse_keywords()

    def _parse_keywords(self) -> None:
        """Parse keyword strings from settings into sets."""
        self.structured_keywords: set[str] = {
            kw.strip().lower()
            for kw in self.settings.router_structured_keywords.split(",")
            if kw.strip()
        }
        self.conversational_keywords: set[str] = {
            kw.strip().lower()
            for kw in self.settings.router_conversational_keywords.split(",")
            if kw.strip()
        }
        self.threshold: float = self.settings.router_threshold

    def classify(self, query: str) -> RouterDecision:
        """Classify a query as conversational or structured extraction.

        Args:
            query: The user's query string

        Returns:
            RouterDecision with mode, confidence, reasoning, and matched keywords
        """
        query_lower = query.lower().strip()

        structured_matches = self._find_keyword_matches(query_lower, self.structured_keywords)
        conversational_matches = self._find_keyword_matches(
            query_lower, self.conversational_keywords
        )

        structured_count = len(structured_matches)
        conversational_count = len(conversational_matches)

        structured_total = len(self.structured_keywords)
        total_keywords = structured_total + len(self.conversational_keywords)
        confidence = structured_count / structured_total if structured_total > 0 else 0.0

        if confidence >= self.threshold and structured_count >= conversational_count:
            return RouterDecision(
                mode="structured",
                confidence=min(1.0, confidence),
                reasoning=f"Matched {structured_count} structured extraction keywords out of {total_keywords} total",
                matched_keywords=list(structured_matches),
                total_keywords=total_keywords,
            )
        else:
            # For empty query (no matches), use 1.0 confidence since it's a clear fallback
            if structured_count == 0 and conversational_count == 0:
                effective_confidence = 1.0
            else:
                effective_confidence = (
                    conversational_count / total_keywords if total_keywords > 0 else 1.0
                )
            return RouterDecision(
                mode="conversational",
                confidence=min(1.0, effective_confidence),
                reasoning=f"Defaulting to conversational (fallback) - {conversational_count} conversational matches vs {structured_count} structured, threshold={self.threshold:.2f}",
                matched_keywords=list(conversational_matches),
                total_keywords=total_keywords,
            )

    def _find_keyword_matches(self, query: str, keywords: set[str]) -> list[str]:
        """Find which keywords match the query.

        Args:
            query: Lowercase query string
            keywords: Set of lowercase keywords to match

        Returns:
            List of keywords that were found in the query
        """
        matches = []
        for kw in keywords:
            if kw in query:
                matches.append(kw)
        return matches

    def __repr__(self) -> str:
        return (
            f"QueryRouter(threshold={self.threshold}, "
            f"structured_keywords={len(self.structured_keywords)}, "
            f"conversational_keywords={len(self.conversational_keywords)})"
        )


def get_router(settings: Settings | None = None) -> QueryRouter:
    """Factory function for QueryRouter singleton.

    Args:
        settings: Application settings

    Returns:
        QueryRouter instance
    """
    return QueryRouter(settings)
