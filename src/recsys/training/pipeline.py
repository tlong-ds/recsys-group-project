"""End-to-end training pipeline: data → features → train → evaluate."""

from __future__ import annotations

from typing import Any


def run_training_pipeline(config: dict[str, Any]) -> None:
    """Execute the full training pipeline from config."""
    # TODO: wire DataLoader → Preprocessor → FeatureEngineer → Trainer → Evaluator
    raise NotImplementedError
