"""Training pipeline, orchestration, and local registry."""

from recsys.training.pipeline import run_training_pipeline
from recsys.training.registry import ModelRegistry
from recsys.training.trainer import Trainer, TrainingResult

__all__ = ["Trainer", "TrainingResult", "ModelRegistry", "run_training_pipeline"]
