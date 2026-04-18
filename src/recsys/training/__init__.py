"""Training pipeline, orchestration, and local registry."""

from recsys.training.registry import ModelRegistry
from recsys.training.trainer import Trainer, TrainingResult


def run_training_pipeline(config):
    from recsys.training.pipeline import run_training_pipeline as _run_training_pipeline

    return _run_training_pipeline(config)


__all__ = ["Trainer", "TrainingResult", "ModelRegistry", "run_training_pipeline"]
