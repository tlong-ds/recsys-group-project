"""Training pipeline, orchestration, and local registry."""

def run_training_pipeline(config):
    from recsys.training.pipeline import run_training_pipeline as _run_training_pipeline

    return _run_training_pipeline(config)


def __getattr__(name):
    if name == "ModelRegistry":
        from recsys.training.registry import ModelRegistry

        return ModelRegistry
    if name == "Trainer":
        from recsys.training.trainer import Trainer

        return Trainer
    if name == "TrainingResult":
        from recsys.training.trainer import TrainingResult

        return TrainingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Trainer", "TrainingResult", "ModelRegistry", "run_training_pipeline"]
