"""Trainer: orchestrates model fitting and MLflow tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from recsys.models.base_model import BaseRecsysModel


class Trainer:
    """Fit a model, log metrics, and save artefacts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def train(
        self,
        model: BaseRecsysModel,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> BaseRecsysModel:
        """Run the training loop and return the fitted model."""
        # TODO: implement training loop with MLflow logging
        raise NotImplementedError

    def save_checkpoint(self, model: BaseRecsysModel, path: str | Path) -> None:
        """Persist a model checkpoint."""
        # TODO: implement
        raise NotImplementedError
