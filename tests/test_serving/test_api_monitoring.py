from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


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


def _create_app(monkeypatch, predictor_factory):
    api_module = importlib.import_module("recsys.serving.api")
    monkeypatch.setattr(
        api_module.Predictor,
        "from_path",
        staticmethod(lambda _: predictor_factory()),
    )
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
    api_module = importlib.import_module("recsys.serving.api")
    monkeypatch.setattr(
        api_module.Predictor,
        "from_path",
        staticmethod(lambda _: (_ for _ in ()).throw(FileNotFoundError("missing"))),
    )
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
