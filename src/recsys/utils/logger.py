"""Centralised logger using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def get_logger(log_file: str | Path | None = None):
    """Configure and return the application logger."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if log_file is not None:
        logger.add(log_file, rotation="10 MB", retention="7 days", level="DEBUG")
    return logger
