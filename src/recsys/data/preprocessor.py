"""Data preprocessing for session-based interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


class SessionPreprocessor:
    """Clean, type-cast, and sort interaction data."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "eventdate",
        event_date_col: str | None = None,
        timeframe_col: str | None = None,
    ) -> None:
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.event_date_col = event_date_col
        self.timeframe_col = timeframe_col

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Return cleaned interactions ordered within each session.
        
        Args:
            interactions: Raw interaction DataFrame.
        
        Returns:
            Cleaned and sorted DataFrame.
        
        Raises:
            ValueError: If required columns are missing.
        """
        timestamp_source_col = self.timestamp_col
        if timestamp_source_col not in interactions.columns and self.event_date_col:
            timestamp_source_col = self.event_date_col

        columns = [self.session_col, self.item_col, timestamp_source_col]
        missing = [column for column in columns if column not in interactions.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        cleaned = interactions.copy()
        logger.info(f"Starting with {len(cleaned):,} rows")
        
        # Drop rows with missing values in key columns
        initial_len = len(cleaned)
        cleaned = cleaned.dropna(subset=columns)
        dropped_nulls = initial_len - len(cleaned)
        if dropped_nulls > 0:
            logger.warning(f"Dropped {dropped_nulls:,} rows with null values")
        
        # Build canonical timestamp from timestamp or event_date
        try:
            cleaned[self.timestamp_col] = pd.to_datetime(
                cleaned[timestamp_source_col], utc=True, errors="coerce"
            )
        except Exception as e:
            logger.error(f"Failed to parse timestamp: {e}")
            raise

        # Diginetica has eventdate (date-only) and timeframe (ordering within session).
        # Keep date as canonical timestamp for split; timeframe is used for in-session order.
        if self.event_date_col and self.event_date_col in cleaned.columns:
            cleaned[self.timestamp_col] = pd.to_datetime(
                cleaned[self.event_date_col], utc=True, errors="coerce"
            )
        
        # Drop rows where timestamp parsing failed
        initial_len = len(cleaned)
        cleaned = cleaned.dropna(subset=[self.timestamp_col])
        dropped_ts_errors = initial_len - len(cleaned)
        if dropped_ts_errors > 0:
                logger.warning(
                    f"Dropped {dropped_ts_errors:,} rows with unparseable timestamps"
                )
        
        # Convert item_id to integer
        try:
            cleaned[self.item_col] = pd.to_numeric(cleaned[self.item_col], errors="coerce")
        except Exception as e:
            logger.error(f"Failed to convert item_id to numeric: {e}")
            raise
        
        # Drop rows where item_id conversion failed
        initial_len = len(cleaned)
        cleaned = cleaned.dropna(subset=[self.item_col])
        dropped_item_errors = initial_len - len(cleaned)
        if dropped_item_errors > 0:
            logger.warning(f"Dropped {dropped_item_errors:,} rows with invalid item_id")
        
        cleaned[self.item_col] = cleaned[self.item_col].astype(int)
        
        sort_cols = [self.session_col, self.timestamp_col]
        if self.timeframe_col and self.timeframe_col in cleaned.columns:
            cleaned[self.timeframe_col] = pd.to_numeric(
                cleaned[self.timeframe_col], errors="coerce"
            )
            cleaned = cleaned.dropna(subset=[self.timeframe_col])
            sort_cols.append(self.timeframe_col)

        # Sort by session/date and optionally by timeframe to preserve click order.
        cleaned = cleaned.sort_values(sort_cols).reset_index(drop=True)
        
        logger.info(f"✓ After preprocessing: {len(cleaned):,} rows")
        return cleaned

    def filter_sessions(
        self,
        interactions: pd.DataFrame,
        min_length: int = 2,
        max_length: int | None = None,
    ) -> pd.DataFrame:
        """Filter sessions by length constraints.
        
        Args:
            interactions: Cleaned interaction DataFrame.
            min_length: Minimum number of items per session.
            max_length: Keep at most this many latest items per session
                (None = no limit).
        
        Returns:
            Filtered DataFrame.
        """
        initial_sessions = interactions[self.session_col].nunique()
        initial_rows = len(interactions)

        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be >= 1 when provided")
        
        # Filter minimum length
        session_lengths = interactions.groupby(self.session_col).size()
        valid_sessions = session_lengths[session_lengths >= min_length].index
        filtered = interactions[interactions[self.session_col].isin(valid_sessions)].copy()
        
        dropped_min = initial_rows - len(filtered)
        if dropped_min > 0:
            logger.info(
                f"Dropped {dropped_min:,} rows from sessions with < {min_length} items"
            )
        
        # Cap maximum length per session by keeping most recent rows.
        if max_length is not None:
            before_cap_rows = len(filtered)
            long_session_count = int(
                filtered.groupby(self.session_col).size().gt(max_length).sum()
            )
            filtered = (
                filtered.groupby(self.session_col, group_keys=False)
                .tail(max_length)
                .copy()
            )

            dropped_max = before_cap_rows - len(filtered)
            if dropped_max > 0:
                logger.info(
                    f"Trimmed {dropped_max:,} rows from {long_session_count:,} "
                    f"sessions longer than {max_length} items"
                )
        
        final_sessions = filtered[self.session_col].nunique()
        logger.info(
            f"Sessions: {initial_sessions:,} → {final_sessions:,} "
            f"(removed {initial_sessions - final_sessions:,})"
        )
        
        return filtered

    def filter_items(
        self,
        interactions: pd.DataFrame,
        min_frequency: int = 5,
        max_frequency: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Filter items by frequency constraints.
        
        Args:
            interactions: Cleaned interaction DataFrame.
            min_frequency: Minimum number of interactions per item.
            max_frequency: Maximum number of interactions per item (None = no limit).
        
        Returns:
            Tuple of (filtered DataFrame, statistics).
        """
        initial_items = interactions[self.item_col].nunique()
        initial_rows = len(interactions)
        
        # Count item frequencies
        item_counts = interactions[self.item_col].value_counts()
        
        # Filter by minimum frequency
        frequent_items = item_counts[item_counts >= min_frequency].index
        filtered = interactions[interactions[self.item_col].isin(frequent_items)].copy()
        
        dropped_min = initial_rows - len(filtered)
        if dropped_min > 0:
            logger.info(
                f"Dropped {dropped_min:,} rows from items with < {min_frequency} interactions"
            )
        
        # # Filter by maximum frequency (optional)
        # if max_frequency is not None:
        #     item_counts = filtered[self.item_col].value_counts()
        #     frequent_items = item_counts[item_counts <= max_frequency].index
        #     filtered = filtered[filtered[self.item_col].isin(frequent_items)].copy()
            
        #     dropped_max = len(filtered) - dropped_min
        #     if dropped_max > 0:
        #         logger.info(
        #             f"Dropped {dropped_max:,} rows from items with > {max_frequency} interactions"
        #         )
        
        final_items = filtered[self.item_col].nunique()
        logger.info(
            f"Items: {initial_items:,} → {final_items:,} "
            f"(removed {initial_items - final_items:,})"
        )
        
        stats = {
            "initial_items": initial_items,
            "final_items": final_items,
            "removed_items": initial_items - final_items,
            "initial_rows": initial_rows,
            "final_rows": len(filtered),
            "removed_rows": initial_rows - len(filtered),
            "removed_item_list": list(set(interactions[self.item_col]) - set(filtered[self.item_col])),
        }
        
        return filtered, stats

    def remove_duplicate_items_in_session(
        self,
        interactions: pd.DataFrame,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate items within the same session (keeping first or last).
        
        Args:
            interactions: Cleaned interaction DataFrame.
            keep: Which duplicate to keep ('first', 'last').
        
        Returns:
            DataFrame with duplicates removed.
        """
        initial_rows = len(interactions)
        
        # Find and remove duplicates within each session
        deduplicated = interactions.drop_duplicates(
            subset=[self.session_col, self.item_col],
            keep=keep,
        ).reset_index(drop=True)
        
        dropped = initial_rows - len(deduplicated)
        if dropped > 0:
            logger.info(f"Removed {dropped:,} duplicate items within sessions")
        
        return deduplicated

