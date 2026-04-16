"""Data ingestion script and utilities for raw recommender datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from recsys.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load interactions and optional metadata tables from disk."""

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
        """Process raw CSVs and save to interim Parquet files."""
        self.interim_path.mkdir(parents=True, exist_ok=True)

        # 1. Process Items
        products_df = self._load_csv(params["ingest"]["products"])
        categories_df = self._load_csv(params["ingest"]["product_categories"])

        items_df = pd.merge(products_df, categories_df, on="itemId", how="left")
        items_path = self.interim_path / "items.parquet"
        items_df.to_parquet(items_path, index=False)
        logger.info(f"Saved items to {items_path}")

        # 2. Process Queries
        queries_df = self._load_csv(params["ingest"]["queries"])
        queries_path = self.interim_path / "queries.parquet"
        queries_df.to_parquet(queries_path, index=False)
        logger.info(f"Saved queries to {queries_path}")

        # 3. Process Interactions
        # Combine views, clicks (if mapped), and purchases
        views_df = self._load_csv(params["ingest"]["item_views"])
        purchases_df = self._load_csv(params["ingest"]["purchases"])
        clicks_df = self._load_csv(params["ingest"]["clicks"])

        # Views and purchases have sessionId
        views_df["event_type"] = "view"
        purchases_df["event_type"] = "purchase"

        interactions = pd.concat(
            [
                views_df[["sessionId", "itemId", "timeframe", "eventdate", "event_type"]],
                purchases_df[["sessionId", "itemId", "timeframe", "eventdate", "event_type"]],
            ],
            ignore_index=True,
        )

        # Clicks need to be mapped to sessionId via queries
        if not clicks_df.empty and not queries_df.empty:
            clicks_with_session = pd.merge(
                clicks_df,
                queries_df[["queryId", "sessionId", "eventdate"]],
                on="queryId",
                how="inner",
            )
            clicks_with_session["event_type"] = "click"
            # clicks don't have eventdate in the original file, but we get it from queries
            # we might need to adjust column names if they differ
            # In clicks_df, we have queryId, timeframe, itemId
            # In queries_df, we have queryId, sessionId, userId, timeframe, duration, eventdate, ...
            # The timeframe in clicks_df might be relative to query timeframe? 
            # Let's check the head output again.
            # train-clicks.csv: queryId;timeframe;itemId. 1;16338861;24857
            # train-queries.csv: queryId;sessionId;userId;timeframe;... 1;1;NA;16327074;...
            # Yes, they both have timeframe.
            interactions = pd.concat(
                [
                    interactions,
                    clicks_with_session[
                        ["sessionId", "itemId", "timeframe", "eventdate", "event_type"]
                    ],
                ],
                ignore_index=True,
            )

        # Sort by sessionId and timeframe
        interactions = interactions.sort_values(["sessionId", "timeframe"])

        interactions_path = self.interim_path / "interactions.parquet"
        interactions.to_parquet(interactions_path, index=False)
        logger.info(f"Saved interactions to {interactions_path}")

        # 4. Generate ingest report
        report = {
            "n_items": len(items_df),
            "n_queries": len(queries_df),
            "n_interactions": len(interactions),
            "n_sessions": interactions["sessionId"].nunique(),
            "event_type_counts": interactions["event_type"].value_counts().to_dict(),
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

    def load_items(self) -> pd.DataFrame:
        """Load processed items from interim storage."""
        path = self.interim_path / "items.parquet"
        if not path.exists():
            logger.error(f"Interim items not found at {path}. Run ingest first.")
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
