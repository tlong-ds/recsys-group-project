from __future__ import annotations

import json
from pathlib import Path

from recsys.training.select_model import (
    promote_best_model,
    promote_model_from_training_metrics,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_promote_best_model_creates_canonical_version_and_alias(
    monkeypatch, tmp_path: Path
) -> None:
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    _write_json(
        tmp_path / "metrics/best_model.json",
        {
            "best_model": {
                "data_version": "v2_sliding_window",
                "model_profile": "srgnn_fc",
            }
        },
    )
    _write_json(
        tmp_path
        / "metrics/experiments/v2_sliding_window/srgnn_fc/training_metrics.json",
        {
            "model_registry": {
                "model_name": "recsys-srgnn_fc-data-v2",
                "model_version": "7",
                "run_id": "run-007",
                "source": "runs:/run-007/model_core",
            }
        },
    )

    captured: dict[str, list[dict[str, str]]] = {
        "create_registered_model": [],
        "create_model_version": [],
        "set_registered_model_alias": [],
    }

    class _FakeVersion:
        version = "42"

    class _FakeClient:
        def create_registered_model(self, name: str) -> None:
            captured["create_registered_model"].append({"name": name})

        def create_model_version(
            self, *, name: str, source: str, run_id: str
        ) -> _FakeVersion:
            captured["create_model_version"].append(
                {"name": name, "source": source, "run_id": run_id}
            )
            return _FakeVersion()

        def set_registered_model_alias(
            self, *, name: str, alias: str, version: str
        ) -> None:
            captured["set_registered_model_alias"].append(
                {"name": name, "alias": alias, "version": version}
            )

    monkeypatch.setattr(
        "recsys.training.select_model._mlflow_client",
        lambda: _FakeClient(),
    )

    output_path = tmp_path / "metrics/promotion_result.json"
    result = promote_best_model(
        training_config_path=str(training_config),
        best_model_path=str(tmp_path / "metrics/best_model.json"),
        experiments_root=str(tmp_path / "metrics/experiments"),
        output_path=str(output_path),
        canonical_model_name="recsys-serving",
        target_alias="Production",
    )

    assert result == {
        "model_name": "recsys-serving",
        "model_version": "42",
        "run_id": "run-007",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert captured["create_registered_model"] == [{"name": "recsys-serving"}]
    assert captured["create_model_version"] == [
        {
            "name": "recsys-serving",
            "source": "runs:/run-007/model_core",
            "run_id": "run-007",
        }
    ]
    assert captured["set_registered_model_alias"] == [
        {"name": "recsys-serving", "alias": "Production", "version": "42"}
    ]


def test_promote_model_from_training_metrics_uses_direct_payload(
    monkeypatch, tmp_path: Path
) -> None:
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")
    training_metrics = tmp_path / "metrics/retrained_selected/training_metrics.json"
    _write_json(
        training_metrics,
        {
            "model_registry": {
                "model_name": "recsys-srgnn-data-v2",
                "model_version": "11",
                "run_id": "run-011",
                "source": "runs:/run-011/model_core",
            }
        },
    )

    captured: dict[str, list[dict[str, str]]] = {
        "create_registered_model": [],
        "create_model_version": [],
        "set_registered_model_alias": [],
    }

    class _FakeVersion:
        version = "77"

    class _FakeClient:
        def create_registered_model(self, name: str) -> None:
            captured["create_registered_model"].append({"name": name})

        def create_model_version(
            self, *, name: str, source: str, run_id: str
        ) -> _FakeVersion:
            captured["create_model_version"].append(
                {"name": name, "source": source, "run_id": run_id}
            )
            return _FakeVersion()

        def set_registered_model_alias(
            self, *, name: str, alias: str, version: str
        ) -> None:
            captured["set_registered_model_alias"].append(
                {"name": name, "alias": alias, "version": version}
            )

    monkeypatch.setattr(
        "recsys.training.select_model._mlflow_client",
        lambda: _FakeClient(),
    )

    output_path = tmp_path / "metrics/promotion_result.json"
    result = promote_model_from_training_metrics(
        training_config_path=str(training_config),
        training_metrics_path=str(training_metrics),
        output_path=str(output_path),
        canonical_model_name="recsys-serving",
        target_alias="Production",
    )

    assert result == {
        "model_name": "recsys-serving",
        "model_version": "77",
        "run_id": "run-011",
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert captured["create_registered_model"] == [{"name": "recsys-serving"}]
    assert captured["create_model_version"] == [
        {
            "name": "recsys-serving",
            "source": "runs:/run-011/model_core",
            "run_id": "run-011",
        }
    ]
    assert captured["set_registered_model_alias"] == [
        {"name": "recsys-serving", "alias": "Production", "version": "77"}
    ]
