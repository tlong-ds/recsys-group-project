"""Tests proving /health and /ready are isolated from catalog DB state.

These tests exercise the full FastAPI route layer, ensuring that:
- /health never depends on the catalog database
- /ready depends only on model loadability, not catalog DB
- /health returns 200 even when the model cannot load (degraded)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from recsys.serving.api import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STUB_META = {
    "source": "filesystem",
    "artifact_path": "/fake",
    "model_name": "test",
    "model_version": "1",
    "run_id": "abc123",
}


class _StubPredictor:
    """Minimal predictor that satisfies ModelProvider."""

    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10,
    ) -> list[int]:
        return list(range(1, top_k + 1))

    def input_quality(self, item_sequence: list[int]) -> dict:
        return {
            "sequence_length": len(item_sequence),
            "known_items": len(item_sequence),
            "unknown_items": 0,
            "oov_ratio": 0.0,
            "known_catalog_items": 50,
        }


def _make_client(
    *, model_loads: bool = True, catalog_available: bool = True,
) -> TestClient:
    """Create a TestClient with controlled model and catalog availability.

    Patches at the ``recsys.serving.api`` module level so that ``create_app``
    picks up our mocks instead of the real classes.
    """
    # Build a mock ModelProvider
    mock_provider = MagicMock()
    if model_loads:
        mock_provider.preload.return_value = None
        mock_provider.health_payload.return_value = {
            "status": "ok",
            "model_source": "filesystem",
            "model_name": "test",
            "model_version": "1",
            "run_id": "abc123",
        }
        mock_provider.readiness_payload.return_value = {
            "status": "ready",
            "model_source": "filesystem",
            "model_name": "test",
            "model_version": "1",
            "run_id": "abc123",
        }
    else:
        mock_provider.preload.return_value = None
        mock_provider.health_payload.return_value = {
            "status": "degraded",
            "model_source": "unavailable",
        }
        mock_provider.readiness_payload.side_effect = FileNotFoundError("model missing")

    # Build a mock CatalogRepository
    mock_catalog = MagicMock()
    mock_catalog.available = catalog_available
    mock_catalog.connect = AsyncMock()
    mock_catalog.close = AsyncMock()
    if not catalog_available:
        mock_catalog.pool = None

    with (
        patch("recsys.serving.api.ModelProvider", return_value=mock_provider),
        patch("recsys.serving.api.CatalogRepository", return_value=mock_catalog),
    ):
        app = create_app(
            model_path="/fake/model",
            serving_config={
                "preload_model_on_startup": model_loads,
                "security": {"enabled": False},
            },
            mlflow_config={},
        )
        return TestClient(app)


# ---------------------------------------------------------------------------
# /health isolation
# ---------------------------------------------------------------------------


class TestHealthIsolation:
    """Prove /health never depends on catalog DB."""

    def test_health_ok_with_model_and_no_catalog(self) -> None:
        client = _make_client(model_loads=True, catalog_available=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_ok_with_model_and_catalog(self) -> None:
        client = _make_client(model_loads=True, catalog_available=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_degraded_when_model_fails(self) -> None:
        client = _make_client(model_loads=False, catalog_available=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_source"] == "unavailable"

    def test_health_degraded_when_both_fail(self) -> None:
        client = _make_client(model_loads=False, catalog_available=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# /ready isolation
# ---------------------------------------------------------------------------


class TestReadyIsolation:
    """Prove /ready depends ONLY on model load, not catalog DB."""

    def test_ready_ok_with_model_and_no_catalog(self) -> None:
        client = _make_client(model_loads=True, catalog_available=False)
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_ok_with_model_and_catalog(self) -> None:
        client = _make_client(model_loads=True, catalog_available=True)
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_503_when_model_fails(self) -> None:
        client = _make_client(model_loads=False, catalog_available=True)
        resp = client.get("/ready")
        assert resp.status_code == 503

    def test_ready_503_when_model_fails_and_no_catalog(self) -> None:
        client = _make_client(model_loads=False, catalog_available=False)
        resp = client.get("/ready")
        assert resp.status_code == 503
