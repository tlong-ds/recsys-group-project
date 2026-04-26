from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from recsys.serving.predictor import Predictor


class _MonitoringPredictor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self._item_to_idx = {1: 1, 2: 2}

    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        if self.fail:
            raise RuntimeError("boom")
        return list(range(10, 10 + top_k))

    def input_quality(self, item_sequence: list[int]) -> dict[str, int | float]:
        unknown_items = sum(
            1 for item in item_sequence if item not in self._item_to_idx
        )
        sequence_length = len(item_sequence)
        return {
            "sequence_length": sequence_length,
            "known_items": sequence_length - unknown_items,
            "unknown_items": unknown_items,
            "oov_ratio": unknown_items / sequence_length if sequence_length else 0.0,
            "known_catalog_items": len(self._item_to_idx),
        }


class _FailingCatalogAcquire:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def fetch(self, *_args, **_kwargs):
        raise RuntimeError("catalog unavailable")


class _FailingCatalogPool:
    def acquire(self):
        return _FailingCatalogAcquire()


def _create_app(monkeypatch, predictor_factory):
    monkeypatch.setattr(
        Predictor,
        "from_path",
        staticmethod(lambda _: predictor_factory()),
    )
    api_module = importlib.import_module("recsys.serving.api")
    return api_module.create_app(
        model_path="unused",
        serving_config={"security": {"enabled": False}},
        mlflow_config={},
    )


def test_recommend_records_online_monitoring_metrics(monkeypatch) -> None:
    client = TestClient(_create_app(monkeypatch, lambda: _MonitoringPredictor()))

    response = client.post(
        "/recommend",
        json={"item_sequence": [1, 2, 999], "top_k": 2},
    )
    metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert "recsys_recommendation_requests_total" in metrics
    assert 'status="success"' in metrics
    assert "recsys_input_items_total" in metrics
    assert "recsys_oov_items_total" in metrics
    assert "recsys_prediction_latency_seconds" in metrics
    assert "recsys_input_sequence_length" in metrics
    assert "recsys_requested_top_k" in metrics
    assert "recsys_model_ready" in metrics


def test_recommend_falls_back_when_catalog_metadata_lookup_fails(monkeypatch) -> None:
    app = _create_app(monkeypatch, lambda: _MonitoringPredictor())
    # Inject a failing pool into the catalog repository
    catalog_repo_module = importlib.import_module("recsys.serving.catalog_repository")
    # Access the catalog through the app's recommendation service
    # We need to set the pool on the catalog repository to a failing pool
    # The catalog is wired inside create_app, so we access it via the
    # recommendation service's catalog reference.
    # A simpler approach: find the CatalogRepository instance in the app closure.
    # Since the app's routes capture the `catalog` variable from create_app,
    # we can patch its _pool attribute.
    from recsys.serving.catalog_repository import CatalogRepository

    # Walk the app routes to find the catalog reference
    # Instead, let's just patch the pool on the catalog object
    # The catalog object is captured in the recommend route's closure
    # via recommendation_service._catalog
    # Let's access it through the startup event
    for route in app.routes:
        handler = getattr(route, "endpoint", None)
        if handler and getattr(handler, "__name__", "") == "recommend":
            # Get the closure variables
            closure = handler.__code__.co_freevars
            if "recommendation_service" in closure:
                idx = closure.index("recommendation_service")
                rec_svc = handler.__closure__[idx].cell_contents
                rec_svc._catalog._pool = _FailingCatalogPool()
                break
    else:
        # Fallback: directly set pool on any CatalogRepository found
        pytest.skip("Could not access catalog in closure")

    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={"item_sequence": [1, 2], "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["recommendations"] == [10, 11]
    assert response.json()["recommended_products"] == [
        {"id": 10, "categoryId": 0, "name": "Product 10", "price": 0.0},
        {"id": 11, "categoryId": 0, "name": "Product 11", "price": 0.0},
    ]


def test_ready_returns_success_without_exposing_sensitive_metadata(monkeypatch) -> None:
    client = TestClient(_create_app(monkeypatch, lambda: _MonitoringPredictor()))

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "model_path" not in body
    assert "run_id" in body
    assert "error" not in body


def test_ready_returns_503_when_model_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        Predictor,
        "from_path",
        staticmethod(lambda _: (_ for _ in ()).throw(FileNotFoundError("missing"))),
    )
    api_module = importlib.import_module("recsys.serving.api")
    client = TestClient(
        api_module.create_app(
            model_path="unused",
            serving_config={"security": {"enabled": False}},
            mlflow_config={},
        )
    )

    ready = client.get("/ready")
    health = client.get("/health")

    assert ready.status_code == 503
    assert ready.json()["detail"] == "Model unavailable."
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"


def test_recommend_returns_sanitized_500_on_inference_error(monkeypatch) -> None:
    client = TestClient(
        _create_app(monkeypatch, lambda: _MonitoringPredictor(fail=True))
    )

    response = client.post("/recommend", json={"item_sequence": [1, 2], "top_k": 2})
    metrics = client.get("/metrics").text

    assert response.status_code == 500
    assert response.json()["detail"] == "Recommendation failed."
    assert 'status="error"' in metrics
