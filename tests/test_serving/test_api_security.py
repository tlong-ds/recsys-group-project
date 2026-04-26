from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


class FakePredictor:
    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        return list(range(100, 100 + top_k))

    def input_quality(self, item_sequence: list[int]) -> dict[str, int | float]:
        return {
            "sequence_length": len(item_sequence),
            "known_items": len(item_sequence),
            "unknown_items": 0,
            "oov_ratio": 0.0,
            "known_catalog_items": 100,
        }


def _app(monkeypatch, *, rate_limit_per_minute: int = 120, max_body_bytes: int = 65536):
    monkeypatch.setenv("RECSYS_API_KEYS", "test-key,rotated-key")
    api_module = importlib.import_module("recsys.serving.api")
    monkeypatch.setattr(
        api_module.Predictor,
        "from_path",
        staticmethod(lambda _: FakePredictor()),
    )
    return api_module.create_app(
        model_path="unused",
        serving_config={
            "security": {
                "enabled": True,
                "api_keys_env_var": "RECSYS_API_KEYS",
                "public_paths": ["/health"],
                "rate_limit_per_minute": rate_limit_per_minute,
                "max_body_bytes": max_body_bytes,
                "docs_enabled": False,
            }
        },
        mlflow_config={},
    )


def _auth_headers(key: str = "test-key") -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_recommend_requires_api_key(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))
    payload = {"item_sequence": [1, 2, 3], "top_k": 3}

    assert client.post("/recommend", json=payload).status_code == 401
    bad_key_response = client.post(
        "/recommend", json=payload, headers=_auth_headers("bad")
    )
    assert bad_key_response.status_code == 401

    response = client.post("/recommend", json=payload, headers=_auth_headers())
    assert response.status_code == 200
    assert response.json()["recommendations"] == [100, 101, 102]


def test_metrics_requires_api_key_and_docs_are_disabled(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers=_auth_headers()).status_code == 200
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_health_is_public_and_sanitized(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_path" not in body
    assert "run_id" in body
    assert "error" not in body


def test_validation_rejects_unsafe_payloads(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch))

    cases = [
        {"item_sequence": [0], "top_k": 1},
        {"item_sequence": [1], "top_k": 101},
        {"item_sequence": [1], "session_id": "x" * 129},
        {"item_sequence": [1], "top_k": 1, "unexpected": True},
        {"item_sequence": list(range(1, 102)), "top_k": 1},
    ]

    for payload in cases:
        response = client.post("/recommend", json=payload, headers=_auth_headers())
        assert response.status_code == 422


def test_body_size_limit(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch, max_body_bytes=20))

    response = client.post(
        "/recommend",
        json={"item_sequence": [1, 2, 3], "top_k": 1},
        headers=_auth_headers(),
    )

    assert response.status_code == 413


def test_rate_limit(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch, rate_limit_per_minute=1))
    payload = {"item_sequence": [1], "top_k": 1}

    first_response = client.post("/recommend", json=payload, headers=_auth_headers())
    second_response = client.post("/recommend", json=payload, headers=_auth_headers())
    assert first_response.status_code == 200
    assert second_response.status_code == 429
