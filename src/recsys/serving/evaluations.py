"""Load offline evaluation metrics from experiment directories."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from recsys.serving.schemas import ModelMetrics


def load_evaluation_metrics(
    experiments_dir: Path,
) -> list[ModelMetrics]:
    """Walk ``experiments_dir`` and collect ``metrics.json`` files.

    Returns an empty list when the directory does not exist or contains
    no valid metric files.
    """
    results: list[ModelMetrics] = []
    if not experiments_dir.exists():
        return results

    for data_version_dir in experiments_dir.iterdir():
        if not data_version_dir.is_dir():
            continue
        for model_dir in data_version_dir.iterdir():
            if not model_dir.is_dir():
                continue
            metrics_file = model_dir / "latest" / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    profile = _format_profile_name(model_dir.name)

                    results.append(
                        ModelMetrics(
                            profile=profile,
                            dataVersion=data_version_dir.name,
                            hrAtK=metrics.get("hr@k", 0.0),
                            mrrAtK=metrics.get("mrr@k", 0.0),
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load metrics for {}: {}",
                        model_dir.name,
                        exc,
                    )

    return results


def _format_profile_name(raw_name: str) -> str:
    """Convert a directory name like ``srgnn_avg`` into a display label."""
    profile = raw_name.upper()
    if profile.startswith("SRGNN_"):
        return f"SR-GNN ({profile.split('_')[1].upper()})"
    if profile == "SRGNN":
        return "SR-GNN"
    if profile == "TAGNN":
        return "TAGNN"
    if profile == "GGNN":
        return "GGNN"
    return profile
