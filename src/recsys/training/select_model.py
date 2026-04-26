from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from recsys.training.tracking import configure_tracking
from recsys.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS_METRICS_ROOT = Path("metrics/experiments")
BEST_MODEL_METRICS_PATH = Path("metrics/best_model.json")
PROMOTION_RESULT_PATH = Path("metrics/promotion_result.json")
CANONICAL_MODEL_NAME = "recsys-serving"
CANONICAL_ALIAS = "Production"


def _extract_metric_value(metrics_payload: dict[str, Any], *keys: str) -> float:
    """Return first matching metric value from root/test/validation metric sections."""
    metric_sections: list[dict[str, Any]] = [
        metrics_payload,
        metrics_payload.get("test_metrics", {}),
        metrics_payload.get("validation_metrics", {}),
    ]

    normalized_keys = {_normalize_metric_key(key) for key in keys}
    for section in metric_sections:
        if not isinstance(section, dict):
            continue
        for raw_key, value in section.items():
            if _normalize_metric_key(str(raw_key)) in normalized_keys:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return 0.0


def _normalize_metric_key(metric_key: str) -> str:
    return metric_key.lower().replace("-", "").replace("_", "").replace(" ", "")


def _score_metrics(metrics_payload: dict[str, Any]) -> tuple[float, float]:
    """Score payload as (primary, secondary) for deterministic best-model selection."""
    primary = _extract_metric_value(metrics_payload, "hr@k")
    secondary = _extract_metric_value(metrics_payload, "mrr@k")
    return primary, secondary


def _discover_evaluation_metrics(metrics_root: Path) -> list[Path]:
    return sorted(metrics_root.glob("*/*/evaluation_metrics.json"))


def _parse_experiment_identifiers(metrics_root: Path, metrics_path: Path) -> tuple[str, str]:
    # Expected pattern:
    # metrics/experiments/<data_version>/<model_profile>/evaluation_metrics.json
    rel = metrics_path.relative_to(metrics_root)
    return rel.parts[0], rel.parts[1]


def select_best_model(
    *,
    metrics_root: str = str(EXPERIMENTS_METRICS_ROOT),
    output_path: str = str(BEST_MODEL_METRICS_PATH),
) -> dict[str, Any]:
    metrics_root_path = Path(metrics_root)
    metrics_files = _discover_evaluation_metrics(metrics_root_path)
    if not metrics_files:
        raise ValueError(f"No evaluation metrics found at {metrics_root_path}")

    best_record: dict[str, Any] | None = None
    best_sort_key: tuple[float, float, str, str] | None = None

    for metrics_file in metrics_files:
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        data_version, model_profile = _parse_experiment_identifiers(
            metrics_root_path, metrics_file
        )
        primary, secondary = _score_metrics(payload)
        sort_key = (primary, secondary, data_version, model_profile)

        if best_sort_key is None or sort_key > best_sort_key:
            model_source_dir = (
                Path("models/experiments")
                / data_version
                / model_profile
                / "latest"
            )
            best_sort_key = sort_key
            best_record = {
                "data_version": data_version,
                "model_profile": model_profile,
                "metrics": payload,
                "source": str(model_source_dir),
                "selection_metrics": {
                    "primary": primary,
                    "secondary": secondary,
                },
            }

    if best_record is None:
        raise ValueError("Could not determine the best model from evaluation metrics.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"best_model": best_record}, indent=2), encoding="utf-8"
    )

    logger.info(
        "Best model: %s/%s (primary=%.6f, secondary=%.6f)",
        best_record["data_version"],
        best_record["model_profile"],
        best_record["selection_metrics"]["primary"],
        best_record["selection_metrics"]["secondary"],
    )
    return best_record


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _extract_registry_metadata(
    training_payload: dict[str, Any], *, source_path: Path
) -> dict[str, str]:
    registry_info = training_payload.get("model_registry")
    if not isinstance(registry_info, dict):
        raise ValueError(
            f"Missing model_registry metadata in {source_path}. "
            "Ensure training registered model versions in MLflow."
        )

    source_model_name = str(registry_info.get("model_name", "")).strip()
    source_model_version = str(registry_info.get("model_version", "")).strip()
    source_run_id = str(registry_info.get("run_id", "")).strip()
    source_uri = str(registry_info.get("source", "")).strip()
    if not source_model_name or not source_model_version or not source_run_id:
        raise ValueError(
            "model_registry metadata must include model_name, model_version, and run_id."
        )
    if not source_uri:
        raise ValueError("model_registry metadata must include source artifact URI.")
    return {
        "model_name": source_model_name,
        "model_version": source_model_version,
        "run_id": source_run_id,
        "source": source_uri,
    }


def _promote_registry_source(
    *,
    source_run_id: str,
    source_uri: str,
    output_path: str,
    canonical_model_name: str,
    target_alias: str,
) -> dict[str, str]:
    client = _mlflow_client()
    try:
        client.create_registered_model(canonical_model_name)
    except Exception as exc:
        err = str(exc).lower()
        if "resource_already_exists" not in err and "already exists" not in err:
            raise

    promoted_version = client.create_model_version(
        name=canonical_model_name,
        source=source_uri,
        run_id=source_run_id,
    )
    promoted_version_str = str(promoted_version.version)
    client.set_registered_model_alias(
        name=canonical_model_name,
        alias=target_alias,
        version=promoted_version_str,
    )

    result = {
        "model_name": canonical_model_name,
        "model_version": promoted_version_str,
        "run_id": source_run_id,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def promote_best_model(
    *,
    training_config_path: str = "configs/training_config.yaml",
    best_model_path: str = str(BEST_MODEL_METRICS_PATH),
    experiments_root: str = str(EXPERIMENTS_METRICS_ROOT),
    output_path: str = str(PROMOTION_RESULT_PATH),
    canonical_model_name: str = CANONICAL_MODEL_NAME,
    target_alias: str = CANONICAL_ALIAS,
) -> dict[str, str]:
    """Promote selected winner metrics into canonical serving model + alias."""
    training_cfg = load_config(training_config_path)
    configure_tracking(training_cfg)

    best_payload = _load_json(Path(best_model_path))
    winner = best_payload.get("best_model")
    if not isinstance(winner, dict):
        raise ValueError("best_model.json must contain a 'best_model' object.")

    data_version = str(winner.get("data_version", "")).strip()
    model_profile = str(winner.get("model_profile", "")).strip()
    if not data_version or not model_profile:
        raise ValueError("best_model must include non-empty data_version/model_profile.")

    training_metrics_path = (
        Path(experiments_root) / data_version / model_profile / "training_metrics.json"
    )
    training_payload = _load_json(training_metrics_path)
    metadata = _extract_registry_metadata(
        training_payload,
        source_path=training_metrics_path,
    )
    return _promote_registry_source(
        source_run_id=metadata["run_id"],
        source_uri=metadata["source"],
        output_path=output_path,
        canonical_model_name=canonical_model_name,
        target_alias=target_alias,
    )


def promote_model_from_training_metrics(
    *,
    training_config_path: str = "configs/training_config.yaml",
    training_metrics_path: str,
    output_path: str = str(PROMOTION_RESULT_PATH),
    canonical_model_name: str = CANONICAL_MODEL_NAME,
    target_alias: str = CANONICAL_ALIAS,
) -> dict[str, str]:
    """Promote a model directly from one training_metrics.json payload."""
    training_cfg = load_config(training_config_path)
    configure_tracking(training_cfg)

    metrics_path = Path(training_metrics_path)
    training_payload = _load_json(metrics_path)
    metadata = _extract_registry_metadata(training_payload, source_path=metrics_path)
    return _promote_registry_source(
        source_run_id=metadata["run_id"],
        source_uri=metadata["source"],
        output_path=output_path,
        canonical_model_name=canonical_model_name,
        target_alias=target_alias,
    )


def select_and_promote_best_model(
    *,
    training_config_path: str = "configs/training_config.yaml",
    experiments_root: str = str(EXPERIMENTS_METRICS_ROOT),
    best_model_path: str = str(BEST_MODEL_METRICS_PATH),
    promotion_output_path: str = str(PROMOTION_RESULT_PATH),
    canonical_model_name: str = CANONICAL_MODEL_NAME,
    target_alias: str = CANONICAL_ALIAS,
) -> dict[str, str]:
    select_best_model(metrics_root=experiments_root, output_path=best_model_path)
    return promote_best_model(
        training_config_path=training_config_path,
        best_model_path=best_model_path,
        experiments_root=experiments_root,
        output_path=promotion_output_path,
        canonical_model_name=canonical_model_name,
        target_alias=target_alias,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select best model by hr@k,mrr@k from metrics/experiments and optionally "
            "promote it to canonical MLflow model recsys-serving."
        )
    )
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--experiments-root", default=str(EXPERIMENTS_METRICS_ROOT))
    parser.add_argument("--best-model-path", default=str(BEST_MODEL_METRICS_PATH))
    parser.add_argument(
        "--training-metrics-path",
        default=None,
        help=(
            "Promote directly from this training_metrics.json payload "
            "(skips best-model selection)."
        ),
    )
    parser.add_argument("--output-path", default=str(PROMOTION_RESULT_PATH))
    parser.add_argument("--canonical-model-name", default=CANONICAL_MODEL_NAME)
    parser.add_argument("--target-alias", default=CANONICAL_ALIAS)
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only write metrics/best_model.json and skip MLflow promotion.",
    )
    args = parser.parse_args()

    if args.select_only:
        result = select_best_model(
            metrics_root=args.experiments_root,
            output_path=args.best_model_path,
        )
        print(json.dumps({"best_model": result}, ensure_ascii=True))
        return

    if args.training_metrics_path:
        result = promote_model_from_training_metrics(
            training_config_path=args.training_config,
            training_metrics_path=args.training_metrics_path,
            output_path=args.output_path,
            canonical_model_name=args.canonical_model_name,
            target_alias=args.target_alias,
        )
        print(json.dumps(result, ensure_ascii=True))
        return

    result = select_and_promote_best_model(
        training_config_path=args.training_config,
        experiments_root=args.experiments_root,
        best_model_path=args.best_model_path,
        promotion_output_path=args.output_path,
        canonical_model_name=args.canonical_model_name,
        target_alias=args.target_alias,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
