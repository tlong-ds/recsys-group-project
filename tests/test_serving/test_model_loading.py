from __future__ import annotations

import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from recsys.serving import predictor as predictor_module


class _FakePredictor:
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
            "known_catalog_items": 100,
        }


class _DummyAcquire:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def executemany(self, *_args, **_kwargs):
        return None

    async def fetch(self, *_args, **_kwargs):
        return []

    async def fetchval(self, *_args, **_kwargs):
        return 0


class _DummyPool:
    def acquire(self):
        return _DummyAcquire()

    async def close(self) -> None:
        return None


def _serving_config() -> dict[str, object]:
    return {
        "security": {"enabled": False},
    }


def test_preload_model_on_startup_loads_once(monkeypatch) -> None:
    api_module = importlib.import_module("recsys.serving.api")
    calls = {"from_path": 0}

    async def _fake_create_pool(*_args, **_kwargs):
        return _DummyPool()

    def _from_path(_model_path: str):
        calls["from_path"] += 1
        return _FakePredictor()

    monkeypatch.setattr(api_module.asyncpg, "create_pool", _fake_create_pool)
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

    async def _fake_create_pool(*_args, **_kwargs):
        return _DummyPool()

    def _from_registry(**_kwargs):
        calls["from_registry"] += 1
        raise RuntimeError("registry unavailable")

    def _from_path(_model_path: str):
        calls["from_path"] += 1
        return _FakePredictor()

    monkeypatch.setattr(api_module.asyncpg, "create_pool", _fake_create_pool)
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
        model_name="recsys-serving",
        model_version="7",
        run_id="run-1",
        artifact_path="registered_model",
        cache_dir=str(cache_dir),
    )
    second_path, second_hit = predictor_module._resolve_registry_artifact(
        client=client,
        model_name="recsys-serving",
        model_version="7",
        run_id="run-1",
        artifact_path="registered_model",
        cache_dir=str(cache_dir),
    )

    assert first_path == second_path
    assert first_hit is False
    assert second_hit is True
    assert client.download_calls == 1


def test_registry_cache_hit_with_ready_marker_skips_download(tmp_path: Path) -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.download_calls = 0

        def download_artifacts(self, _run_id: str, _artifact_path: str) -> str:
            self.download_calls += 1
            raise AssertionError("download should not be called")

    cache_target = predictor_module.resolve_registry_cache_path(
        cache_root=tmp_path,
        model_name="recsys-serving",
        model_version="9",
        run_id="run-abc",
        artifact_path="registered_model",
    )
    cache_target.mkdir(parents=True)
    (cache_target / "model.json").write_text("{}", encoding="utf-8")
    (cache_target / ".ready").write_text("ready\n", encoding="utf-8")

    client = _FakeClient()
    path, hit = predictor_module._resolve_registry_artifact(
        client=client,
        model_name="recsys-serving",
        model_version="9",
        run_id="run-abc",
        artifact_path="registered_model",
        cache_dir=str(tmp_path),
    )

    assert path == cache_target
    assert hit is True
    assert client.download_calls == 0


def test_registry_cache_lock_allows_single_download_on_concurrent_start(
    tmp_path: Path,
) -> None:
    class _FakeClient:
        def __init__(self, source: Path) -> None:
            self.source = source
            self.download_calls = 0
            self._lock = threading.Lock()

        def download_artifacts(self, _run_id: str, _artifact_path: str) -> str:
            with self._lock:
                self.download_calls += 1
            time.sleep(0.25)
            return str(self.source)

    source = tmp_path / "downloaded"
    source.mkdir(parents=True)
    (source / "model.json").write_text("{}", encoding="utf-8")

    client = _FakeClient(source)
    cache_dir = tmp_path / "cache"

    def _resolve_once() -> tuple[Path, bool]:
        return predictor_module._resolve_registry_artifact(
            client=client,
            model_name="recsys-serving",
            model_version="11",
            run_id="run-xyz",
            artifact_path="registered_model",
            cache_dir=str(cache_dir),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: _resolve_once(), [1, 2]))

    assert first[0] == second[0]
    assert sorted([first[1], second[1]]) == [False, True]
    assert client.download_calls == 1


def test_deploy_pin_env_overrides_alias_selection(monkeypatch) -> None:
    api_module = importlib.import_module("recsys.serving.api")
    captured: dict[str, str | None] = {}

    async def _fake_create_pool(*_args, **_kwargs):
        return _DummyPool()

    def _from_registry(**kwargs):
        captured.update(kwargs)
        return _FakePredictor(), {
            "source": "mlflow_registry",
            "model_name": "recsys-serving",
            "model_version": "99",
            "model_alias": "",
            "run_id": "run-99",
            "artifact_path": "/tmp/fake",
            "cache_hit": "true",
        }

    monkeypatch.setattr(api_module.asyncpg, "create_pool", _fake_create_pool)
    monkeypatch.setattr(
        api_module.Predictor,
        "from_model_registry",
        staticmethod(_from_registry),
    )

    monkeypatch.setenv("RECSYS_DEPLOY_MODEL_NAME", "recsys-serving")
    monkeypatch.setenv("RECSYS_DEPLOY_MODEL_VERSION", "99")
    monkeypatch.setenv("RECSYS_DEPLOY_RUN_ID", "run-99")
    monkeypatch.setenv("RECSYS_MODEL_CACHE_ROOT", "/app/models/cache")

    app = api_module.create_app(
        model_path="unused",
        serving_config={
            **_serving_config(),
            "preload_model_on_startup": True,
            "model_registry": {
                "enabled": True,
                "model_name": "recsys-serving",
                "model_alias": "Production",
                "artifact_path": "registered_model",
                "local_cache_dir": "/tmp/ignored",
                "fallback_to_filesystem": False,
            },
        },
        mlflow_config={},
    )

    with TestClient(app) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert captured["model_name"] == "recsys-serving"
    assert captured["model_version"] == "99"
    assert captured["run_id"] == "run-99"
    assert captured["model_alias"] is None
    assert captured["cache_dir"] == "/app/models/cache"


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
