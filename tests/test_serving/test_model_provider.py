"""Unit tests for ModelProvider — no FastAPI, no MLflow, no live DB."""

from __future__ import annotations

from typing import Any

import pytest

from recsys.serving.model_provider import ModelProvider


class _StubPredictor:
    def __init__(self, tag: str = "stub") -> None:
        self.tag = tag

    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        return list(range(1, top_k + 1))

    def input_quality(self, item_sequence: list[int]) -> dict[str, int | float]:
        return {
            "sequence_length": len(item_sequence),
            "known_items": len(item_sequence),
            "unknown_items": 0,
            "oov_ratio": 0.0,
            "known_catalog_items": 50,
        }


def _make_provider(
    monkeypatch, *, serving_config: dict[str, Any] | None = None
) -> ModelProvider:
    from recsys.serving import predictor as pred_mod

    monkeypatch.setattr(
        pred_mod.Predictor, "from_path", staticmethod(lambda _: _StubPredictor())
    )
    return ModelProvider(
        serving_config=serving_config or {},
        mlflow_config={},
        model_path="unused",
    )


def test_get_bundle_loads_from_filesystem(monkeypatch) -> None:
    provider = _make_provider(monkeypatch)

    predictor, meta = provider.get_bundle()

    assert isinstance(predictor, _StubPredictor)
    assert meta["source"] == "filesystem"


def test_get_bundle_caches_result(monkeypatch) -> None:
    provider = _make_provider(monkeypatch)

    first = provider.get_bundle()
    second = provider.get_bundle()

    assert first is second  # exact same tuple object


def test_preload_sets_ready_on_success(monkeypatch) -> None:
    from recsys.serving.model_provider import RECSYS_MODEL_READY

    provider = _make_provider(monkeypatch)
    RECSYS_MODEL_READY.set(0)

    provider.preload()

    assert RECSYS_MODEL_READY._value.get() == 1.0


def test_preload_sets_not_ready_on_failure(monkeypatch) -> None:
    from recsys.serving import predictor as pred_mod
    from recsys.serving.model_provider import RECSYS_MODEL_READY

    monkeypatch.setattr(
        pred_mod.Predictor,
        "from_path",
        staticmethod(lambda _: (_ for _ in ()).throw(FileNotFoundError("nope"))),
    )
    provider = ModelProvider(
        serving_config={}, mlflow_config={}, model_path="unused"
    )
    RECSYS_MODEL_READY.set(1)

    provider.preload()

    assert RECSYS_MODEL_READY._value.get() == 0.0


def test_health_payload_returns_ok_on_success(monkeypatch) -> None:
    provider = _make_provider(monkeypatch)

    payload = provider.health_payload()

    assert payload["status"] == "ok"
    assert payload["model_source"] == "filesystem"


def test_health_payload_returns_degraded_on_failure(monkeypatch) -> None:
    from recsys.serving import predictor as pred_mod

    monkeypatch.setattr(
        pred_mod.Predictor,
        "from_path",
        staticmethod(lambda _: (_ for _ in ()).throw(FileNotFoundError("missing"))),
    )
    provider = ModelProvider(
        serving_config={}, mlflow_config={}, model_path="unused"
    )

    payload = provider.health_payload()

    assert payload["status"] == "degraded"
    assert payload["model_source"] == "unavailable"


def test_readiness_payload_raises_on_failure(monkeypatch) -> None:
    from recsys.serving import predictor as pred_mod

    monkeypatch.setattr(
        pred_mod.Predictor,
        "from_path",
        staticmethod(lambda _: (_ for _ in ()).throw(FileNotFoundError("missing"))),
    )
    provider = ModelProvider(
        serving_config={}, mlflow_config={}, model_path="unused"
    )

    with pytest.raises(FileNotFoundError):
        provider.readiness_payload()


def test_registry_enabled_falls_back_to_filesystem(monkeypatch) -> None:
    from recsys.serving import predictor as pred_mod

    monkeypatch.setattr(
        pred_mod.Predictor,
        "from_model_registry",
        staticmethod(lambda **_kw: (_ for _ in ()).throw(RuntimeError("down"))),
    )
    monkeypatch.setattr(
        pred_mod.Predictor,
        "from_path",
        staticmethod(lambda _: _StubPredictor("fs-fallback")),
    )

    provider = ModelProvider(
        serving_config={
            "model_registry": {
                "enabled": True,
                "model_name": "test",
                "model_alias": "Production",
                "fallback_to_filesystem": True,
            },
        },
        mlflow_config={},
        model_path="unused",
    )

    predictor, meta = provider.get_bundle()

    assert predictor.tag == "fs-fallback"
    assert meta["source"] == "filesystem"
