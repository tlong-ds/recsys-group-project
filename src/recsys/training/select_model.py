from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENTS_METRICS_ROOT = Path("metrics/experiments")
BEST_MODEL_METRICS_PATH = Path("metrics/best_model.json")
LATEST_MODEL_TARGET = Path("models/trained/latest")


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


def _parse_experiment_identifiers(metrics_path: Path) -> tuple[str, str]:
    # Expected pattern:
    # metrics/experiments/<data_version>/<model_profile>/evaluation_metrics.json
    rel = metrics_path.relative_to(EXPERIMENTS_METRICS_ROOT)
    return rel.parts[0], rel.parts[1]


def select_best_model() -> None:
    metrics_files = _discover_evaluation_metrics(EXPERIMENTS_METRICS_ROOT)
    if not metrics_files:
        logger.error(
            "No experiment evaluation metrics found at %s", EXPERIMENTS_METRICS_ROOT
        )
        return

    best_record: dict[str, Any] | None = None
    best_sort_key: tuple[float, float, str, str] | None = None

    for metrics_file in metrics_files:
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read %s: %s", metrics_file, exc)
            continue

        data_version, model_profile = _parse_experiment_identifiers(metrics_file)
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
        logger.error("Could not determine the best model from evaluation metrics.")
        return

    BEST_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_METRICS_PATH.write_text(
        json.dumps({"best_model": best_record}, indent=2), encoding="utf-8"
    )

    logger.info(
        "Best model: %s/%s (primary=%.6f, secondary=%.6f)",
        best_record["data_version"],
        best_record["model_profile"],
        best_record["selection_metrics"]["primary"],
        best_record["selection_metrics"]["secondary"],
    )

    source = Path(best_record["source"])
    if LATEST_MODEL_TARGET.exists():
        if LATEST_MODEL_TARGET.is_symlink():
            LATEST_MODEL_TARGET.unlink()
        else:
            shutil.rmtree(LATEST_MODEL_TARGET)

    if source.exists():
        shutil.copytree(source, LATEST_MODEL_TARGET)
        logger.info("Copied %s to %s", source, LATEST_MODEL_TARGET)
    else:
        logger.error("Source model directory %s does not exist.", source)


if __name__ == "__main__":
    select_best_model()
