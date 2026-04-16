"""Dispatcher CLI and backward-compatible orchestrator for data pipeline stages."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger

from recsys.data.stages import STAGE_ALL, STAGE_NAMES, run_stage


class DataProcessingPipeline:
    """Compatibility wrapper around stage-based execution."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path or "configs/data_config.yaml")

    def run(self, stage: str = STAGE_ALL) -> dict[str, Any]:
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting data pipeline stage: {}", stage)
        logger.info("=" * 80)

        outputs = run_stage(stage=stage, config_path=self.config_path)

        elapsed = time.time() - start_time
        logger.info("Stage '{}' completed in {:.1f}s", stage, elapsed)
        for name, artifact_path in outputs.items():
            logger.info("{} -> {}", name, artifact_path)

        return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one stage or all stages of the data pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to data pipeline configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=STAGE_ALL,
        choices=STAGE_NAMES,
        help="Pipeline stage to run.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if not args.config.exists():
        logger.error("Config file not found: {}", args.config)
        raise SystemExit(2)

    pipeline = DataProcessingPipeline(config_path=args.config)
    try:
        pipeline.run(stage=args.stage)
    except Exception as exc:
        logger.error("Pipeline execution failed: {}", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
