"""
FastAPI application entry point.

Routers and dependencies are added in Phase 01 (Milestone 1.4)
"""

from fastapi import FastAPI

app = FastAPI(
    title="PaperLens", description="Research intelligence over AI/ML arXiv papers.", version="0.0.1"
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check - returns 200 when the server is up."""
    return {"status": "ok"}
