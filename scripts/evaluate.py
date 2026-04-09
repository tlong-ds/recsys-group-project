#!/usr/bin/env python
"""CLI entry point: evaluate a trained recommendation model."""

from __future__ import annotations

import argparse

from recsys.evaluation.evaluator import Evaluator
from recsys.serving.predictor import Predictor
from recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # TODO: load test data, run evaluator, print/log metrics
    raise NotImplementedError


if __name__ == "__main__":
    main()
