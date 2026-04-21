"""CT gate + MLflow registry promotion helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recsys.training.mlflow_registry import set_registered_model_alias
from recsys.training.tracking import configure_tracking
from recsys.utils.config import load_config


def promote_from_metrics(
    *,
    training_config_path: str,
    train_metrics_path: str,
    evaluation_metrics_path: str,
    metric_key: str = "hr@k",
    min_threshold: float = 0.0,
    baseline_evaluation_metrics_path: str | None = None,
    require_improvement: bool = False,
    target_alias: str = "Production",
) -> dict[str, Any]:
    """Validate candidate metrics and promote registry alias when gates pass."""
    training_cfg = load_config(training_config_path)
    configure_tracking(training_cfg)

    training_payload = _load_json(train_metrics_path)
    evaluation_payload = _load_json(evaluation_metrics_path)

    model_name, model_version = _extract_model_registry_info(training_payload)
    candidate_metric = _extract_metric(evaluation_payload, metric_key)

    if candidate_metric < float(min_threshold):
        raise RuntimeError(
            f"Metric gate failed for {metric_key}: "
            f"{candidate_metric:.6f} < {float(min_threshold):.6f}"
        )

    baseline_metric: float | None = None
    if baseline_evaluation_metrics_path:
        baseline_payload = _load_json(baseline_evaluation_metrics_path)
        baseline_metric = _extract_metric(baseline_payload, metric_key)
        if require_improvement and candidate_metric <= baseline_metric:
            raise RuntimeError(
                f"Improvement gate failed for {metric_key}: "
                f"{candidate_metric:.6f} <= {baseline_metric:.6f}"
            )

    set_registered_model_alias(
        model_name=model_name,
        alias=target_alias,
        version=model_version,
    )

    return {
        "model_name": model_name,
        "model_version": model_version,
        "target_alias": target_alias,
        "metric_key": metric_key,
        "candidate_metric": candidate_metric,
        "baseline_metric": baseline_metric,
        "min_threshold": float(min_threshold),
        "require_improvement": bool(require_improvement),
    }


def _load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_model_registry_info(training_payload: dict[str, Any]) -> tuple[str, str]:
    registry_info = training_payload.get("model_registry")
    if not isinstance(registry_info, dict):
        raise ValueError(
            "Training metrics payload does not contain model_registry metadata. "
            "Ensure MLflow registry is enabled during training."
        )

    model_name = registry_info.get("model_name")
    model_version = registry_info.get("model_version")
    if not model_name or not model_version:
        raise ValueError(
            "model_registry metadata must include model_name and model_version."
        )
    return str(model_name), str(model_version)


def _extract_metric(payload: dict[str, Any], metric_key: str) -> float:
    metrics = payload.get("test_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Expected test_metrics in evaluation metrics payload.")
    if metric_key not in metrics:
        available = ", ".join(sorted(str(key) for key in metrics.keys()))
        raise KeyError(
            f"Metric {metric_key!r} not found in evaluation payload. "
            f"Available metrics: [{available}]"
        )
    return float(metrics[metric_key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate CT evaluation metrics and promote an MLflow alias."
    )
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--train-metrics-path", required=True)
    parser.add_argument("--evaluation-metrics-path", required=True)
    parser.add_argument("--metric-key", default="hr@k")
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--baseline-evaluation-metrics-path", default=None)
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--target-alias", default="Production")
    args = parser.parse_args()

    result = promote_from_metrics(
        training_config_path=args.training_config,
        train_metrics_path=args.train_metrics_path,
        evaluation_metrics_path=args.evaluation_metrics_path,
        metric_key=args.metric_key,
        min_threshold=args.min_threshold,
        baseline_evaluation_metrics_path=args.baseline_evaluation_metrics_path,
        require_improvement=bool(args.require_improvement),
        target_alias=args.target_alias,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

