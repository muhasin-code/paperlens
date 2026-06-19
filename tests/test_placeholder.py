"""
Placeholder test - replaced by real tests in Phase 01.

Verifies that the test suite runs and the PaperLens package is importable.
"""


def test_paperlense_package_importable() -> None:
    """Smoke test: src/paperlens/__init__.py is importable and has a version."""
    from src.paperlens import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_health_endpoint_schema() -> None:
    """Smoke test: FastAPI app instance is created and /health route exists."""
    from fastapi.testclient import TestClient

    from src.paperlens.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
