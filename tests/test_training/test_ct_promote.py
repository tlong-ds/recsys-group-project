from __future__ import annotations

import json
from pathlib import Path

import pytest

from recsys.training.ct_promote import promote_from_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_promote_from_metrics_promotes_when_gates_pass(
    monkeypatch, tmp_path: Path
) -> None:
    train_metrics = tmp_path / "training_metrics.json"
    eval_metrics = tmp_path / "evaluation_metrics.json"
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    _write_json(
        train_metrics,
        {
            "model_registry": {
                "model_name": "recsys-srgnn-data-v2",
                "model_version": "7",
            }
        },
    )
    _write_json(eval_metrics, {"test_metrics": {"hr@k": 0.42, "mrr@k": 0.21}})

    captured: dict[str, str] = {}

    def _capture_alias(*, model_name: str, alias: str, version: str) -> None:
        captured["model_name"] = model_name
        captured["alias"] = alias
        captured["version"] = version

    monkeypatch.setattr(
        "recsys.training.ct_promote.set_registered_model_alias",
        _capture_alias,
    )

    result = promote_from_metrics(
        training_config_path=str(training_config),
        train_metrics_path=str(train_metrics),
        evaluation_metrics_path=str(eval_metrics),
        metric_key="hr@k",
        min_threshold=0.4,
        target_alias="Production",
    )

    assert result["candidate_metric"] == 0.42
    assert captured == {
        "model_name": "recsys-srgnn-data-v2",
        "alias": "Production",
        "version": "7",
    }


def test_promote_from_metrics_fails_threshold_gate(monkeypatch, tmp_path: Path) -> None:
    train_metrics = tmp_path / "training_metrics.json"
    eval_metrics = tmp_path / "evaluation_metrics.json"
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    _write_json(
        train_metrics,
        {
            "model_registry": {
                "model_name": "recsys-srgnn-data-v2",
                "model_version": "7",
            }
        },
    )
    _write_json(eval_metrics, {"test_metrics": {"hr@k": 0.12}})

    monkeypatch.setattr(
        "recsys.training.ct_promote.set_registered_model_alias",
        lambda **_: pytest.fail("alias update should not run when gate fails"),
    )

    with pytest.raises(RuntimeError, match="Metric gate failed"):
        promote_from_metrics(
            training_config_path=str(training_config),
            train_metrics_path=str(train_metrics),
            evaluation_metrics_path=str(eval_metrics),
            metric_key="hr@k",
            min_threshold=0.2,
        )


def test_promote_from_metrics_fails_improvement_gate(tmp_path: Path) -> None:
    train_metrics = tmp_path / "training_metrics.json"
    eval_metrics = tmp_path / "evaluation_metrics.json"
    baseline_metrics = tmp_path / "baseline.json"
    training_config = tmp_path / "training_config.yaml"
    training_config.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    _write_json(
        train_metrics,
        {
            "model_registry": {
                "model_name": "recsys-srgnn-data-v2",
                "model_version": "7",
            }
        },
    )
    _write_json(eval_metrics, {"test_metrics": {"hr@k": 0.3}})
    _write_json(baseline_metrics, {"test_metrics": {"hr@k": 0.31}})

    with pytest.raises(RuntimeError, match="Improvement gate failed"):
        promote_from_metrics(
            training_config_path=str(training_config),
            train_metrics_path=str(train_metrics),
            evaluation_metrics_path=str(eval_metrics),
            metric_key="hr@k",
            min_threshold=0.2,
            baseline_evaluation_metrics_path=str(baseline_metrics),
            require_improvement=True,
        )
