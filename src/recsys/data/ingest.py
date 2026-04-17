"""Data ingestion script and utilities for raw item-view datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from recsys.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load session interactions from disk."""

    def __init__(
        self,
        raw_path: str | Path = "data/raw",
        interim_path: str | Path = "data/interim",
        **kwargs,
    ) -> None:
        self.raw_path = Path(raw_path)
        self.interim_path = Path(interim_path)

    def _load_csv(self, filename: str, required: bool = True) -> pd.DataFrame:
        path = self.raw_path / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Expected dataset file at {path}")
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
        logger.info(f"Loading {path}")
        return pd.read_csv(path, sep=";")

    def ingest(self, params: dict) -> None:
        """Process item views and save to interim Parquet files."""
        self.interim_path.mkdir(parents=True, exist_ok=True)

        # 1. Process Interactions (ONLY item views)
        views_file = params["ingest"]["item_views"]
        interactions = self._load_csv(views_file, required=True)
        interactions["event_type"] = "view"

        # Sort by sessionId and timeframe if columns exist
        sort_cols = []
        if "sessionId" in interactions.columns:
            sort_cols.append("sessionId")
        if "timeframe" in interactions.columns:
            sort_cols.append("timeframe")
            
        if sort_cols:
            interactions = interactions.sort_values(sort_cols)

        interactions_path = self.interim_path / "interactions.parquet"
        interactions.to_parquet(interactions_path, index=False)
        logger.info(f"Saved {len(interactions):,} item-view interactions to {interactions_path}")

        # 2. Generate ingest report
        report = {
            "n_interactions": len(interactions),
            "n_sessions": interactions["sessionId"].nunique() if "sessionId" in interactions.columns else 0,
            "n_items": interactions["itemId"].nunique() if "itemId" in interactions.columns else 0,
            "event_type_counts": {"view": len(interactions)},
        }

        report_path = Path("metrics/ingest_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved ingest report to {report_path}")

    def load_interactions(self) -> pd.DataFrame:
        """Load processed interactions from interim storage."""
        path = self.interim_path / "interactions.parquet"
        if not path.exists():
            logger.error(f"Interim interactions not found at {path}. Run ingest first.")
            raise FileNotFoundError(path)
        return pd.read_parquet(path)


def main() -> None:
    """Main ingestion entry point."""
    # Lazy import avoids circular dependency:
    # stages -> ingest.DataLoader and ingest.main -> stages
    from recsys.data.stages import run_ingest_stage

    # Keep this module as a valid stage entrypoint for legacy callers.
    run_ingest_stage(config_path="configs/data_config.yaml", params_path="params.yaml")


if __name__ == "__main__":
    main()
