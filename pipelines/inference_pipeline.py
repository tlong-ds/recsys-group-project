"""Batch inference pipeline entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference pipeline")
    parser.add_argument("--model-path", required=True, help="Path to saved model")
    parser.add_argument("--output-path", required=True, help="Where to write predictions")
    args = parser.parse_args()

    # TODO: load model, run inference, save predictions
    raise NotImplementedError


if __name__ == "__main__":
    main()
