from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from recsys.serving import predictor as predictor_module


class _FakePredictor:
    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        return list(range(1, top_k + 1))


def _serving_config() -> dict[str, object]:
    return {
        "security": {"enabled": False},
    }


def test_preload_model_on_startup_loads_once(monkeypatch) -> None:
    api_module = importlib.import_module("recsys.serving.api")
    calls = {"from_path": 0}

    def _from_path(_model_path: str):
        calls["from_path"] += 1
        return _FakePredictor()

    monkeypatch.setattr(api_module.Predictor, "from_path", staticmethod(_from_path))

    app = api_module.create_app(
        model_path="unused",
        serving_config={**_serving_config(), "preload_model_on_startup": True},
        mlflow_config={},
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        rec = client.post("/recommend", json={"item_sequence": [1, 2], "top_k": 3})
        assert rec.status_code == 200
        assert rec.json()["recommendations"] == [1, 2, 3]

    assert calls["from_path"] == 1


def test_registry_falls_back_to_filesystem_when_enabled(monkeypatch) -> None:
    api_module = importlib.import_module("recsys.serving.api")
    calls = {"from_registry": 0, "from_path": 0}

    def _from_registry(**_kwargs):
        calls["from_registry"] += 1
        raise RuntimeError("registry unavailable")

    def _from_path(_model_path: str):
        calls["from_path"] += 1
        return _FakePredictor()

    monkeypatch.setattr(
        api_module.Predictor, "from_model_registry", staticmethod(_from_registry)
    )
    monkeypatch.setattr(api_module.Predictor, "from_path", staticmethod(_from_path))

    app = api_module.create_app(
        model_path="unused",
        serving_config={
            **_serving_config(),
            "preload_model_on_startup": True,
            "model_registry": {
                "enabled": True,
                "model_name": "recsys-srgnn",
                "model_alias": "Production",
                "artifact_path": "registered_model",
                "local_cache_dir": ".cache/recsys_registry",
                "fallback_to_filesystem": True,
            },
        },
        mlflow_config={},
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_source"] == "filesystem"

    assert calls["from_registry"] >= 1
    assert calls["from_path"] == 1


def test_registry_artifact_cache_reuses_download(tmp_path: Path) -> None:
    class _FakeClient:
        def __init__(self, source: Path) -> None:
            self.source = source
            self.download_calls = 0

        def download_artifacts(self, _run_id: str, _artifact_path: str) -> str:
            self.download_calls += 1
            return str(self.source)

    source = tmp_path / "downloaded"
    source.mkdir(parents=True)
    (source / "model.json").write_text("{}", encoding="utf-8")

    client = _FakeClient(source)
    cache_dir = tmp_path / "cache"

    first_path, first_hit = predictor_module._resolve_registry_artifact(
        client=client,
        run_id="run-1",
        artifact_path="registered_model",
        cache_dir=str(cache_dir),
    )
    second_path, second_hit = predictor_module._resolve_registry_artifact(
        client=client,
        run_id="run-1",
        artifact_path="registered_model",
        cache_dir=str(cache_dir),
    )

    assert first_path == second_path
    assert first_hit is False
    assert second_hit is True
    assert client.download_calls == 1


def test_from_path_dispatches_tagnn_loader(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact_tagnn"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.json").write_text(
        json.dumps({"model_type": "tagnn"}),
        encoding="utf-8",
    )

    sentinel = object()
    calls = {"tagnn": 0}

    def _fake_tagnn_load(_path: Path):
        calls["tagnn"] += 1
        return sentinel

    monkeypatch.setattr("recsys.models.tagnn.TAGNNRecommender.load", _fake_tagnn_load)

    predictor = predictor_module.Predictor.from_path(str(artifact_dir))
    assert predictor.model is sentinel
    assert calls["tagnn"] == 1


def test_from_path_dispatches_ggnn_loader(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact_ggnn"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.json").write_text(
        json.dumps({"model_type": "ggnn"}),
        encoding="utf-8",
    )

    sentinel = object()
    calls = {"ggnn": 0}

    def _fake_ggnn_load(_path: Path):
        calls["ggnn"] += 1
        return sentinel

    monkeypatch.setattr("recsys.models.ggnn.GGNNRecommender.load", _fake_ggnn_load)

    predictor = predictor_module.Predictor.from_path(str(artifact_dir))
    assert predictor.model is sentinel
    assert calls["ggnn"] == 1
