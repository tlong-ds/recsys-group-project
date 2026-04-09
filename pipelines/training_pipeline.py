"""End-to-end training pipeline entry point."""

from __future__ import annotations

import argparse

from recsys.training.pipeline import run_training_pipeline
from recsys.utils.config import load_config, merge_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    args = parser.parse_args()

    config = merge_configs(
        load_config(args.model_config),
        load_config(args.training_config),
    )
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
