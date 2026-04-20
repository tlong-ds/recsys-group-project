"""Schema validation for interaction data using Pandera."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


class DataValidator:
    """Validate required columns and basic data quality constraints."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "eventdate",
    ) -> None:
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def validate_interactions(self, interactions: pd.DataFrame) -> None:
        required = {self.session_col, self.item_col, self.timestamp_col}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        if interactions.empty:
            raise ValueError("Interaction dataset is empty")

    def validate_items(self, items: pd.DataFrame, item_col: str = "item_id") -> None:
        if items.empty:
            return
        if item_col not in items.columns:
            raise ValueError(f"Missing item column: {item_col}")


class InteractionValidator:
    """Validate interaction data against expected schema using Pandera."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "eventdate",
    ) -> None:
        """Initialize validator.

        Args:
            session_col: Name of session ID column.
            item_col: Name of item ID column.
            timestamp_col: Name of timestamp column.
        """
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def validate_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate DataFrame schema.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with validation results.

        Raises:
            ValueError: If schema validation fails.
        """
        required_cols = [self.session_col, self.item_col, self.timestamp_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"✗ Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.empty:
            logger.error("✗ DataFrame is empty")
            raise ValueError("DataFrame is empty")

        logger.info("✓ All required columns present")

        return {
            "valid": True,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }

    def validate_semantics(
        self,
        df: pd.DataFrame,
        min_session_length: int = 1,
        max_session_length: int | None = None,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        """Validate semantic constraints on data.

        Args:
            df: Input DataFrame.
            min_session_length: Minimum items per session.
            max_session_length: Maximum items per session.
            allow_duplicates: Whether to allow duplicate items in same session.

        Returns:
            Dictionary with validation results.
        """
        issues = []
        stats = {
            "n_rows": len(df),
            "n_sessions": df[self.session_col].nunique(),
            "n_items": df[self.item_col].nunique(),
            "avg_session_length": 0.0,
            "min_session_length": 0,
            "max_session_length": 0,
            "sessions_by_length": {},
            "duplicate_items_per_session": 0,
        }

        # Check session lengths
        session_lengths = df.groupby(self.session_col).size()
        stats["min_session_length"] = int(session_lengths.min())
        stats["max_session_length"] = int(session_lengths.max())
        stats["avg_session_length"] = float(session_lengths.mean())
        stats["sessions_by_length"] = session_lengths.value_counts().to_dict()

        short_sessions = (session_lengths < min_session_length).sum()
        if short_sessions > 0:
            issues.append(
                f"Found {short_sessions} sessions with < {min_session_length} items"
            )

        if max_session_length is not None:
            long_sessions = (session_lengths > max_session_length).sum()
            if long_sessions > 0:
                issues.append(
                    f"Found {long_sessions} sessions with > {max_session_length} items"
                )

        # Check for duplicates
        if not allow_duplicates:
            dup_count = (
                df.groupby(self.session_col)[self.item_col]
                .apply(lambda x: x.duplicated().sum())
                .sum()
            )
            stats["duplicate_items_per_session"] = int(dup_count)
            if dup_count > 0:
                issues.append(f"Found {dup_count} duplicate items within sessions")

        # Check for null values
        null_counts = (
            df[[self.session_col, self.item_col, self.timestamp_col]].isnull().sum()
        )
        if null_counts.sum() > 0:
            for col, count in null_counts[null_counts > 0].items():
                issues.append(f"Found {count} null values in column '{col}'")

        valid = len(issues) == 0

        if valid:
            logger.info("✓ Semantic validation passed")
        else:
            logger.warning("✗ Semantic validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return {
            "valid": valid,
            "issues": issues,
            "stats": stats,
        }

    def generate_report(
        self,
        df: pd.DataFrame,
        min_session_length: int = 1,
        max_session_length: int | None = None,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        """Generate comprehensive validation report.

        Args:
            df: Input DataFrame.
            min_session_length: Minimum items per session.
            max_session_length: Maximum items per session.
            allow_duplicates: Whether to allow duplicate items in same session.

        Returns:
            Comprehensive validation report.
        """
        schema_report = self.validate_schema(df)
        semantic_report = self.validate_semantics(
            df,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            allow_duplicates=allow_duplicates,
        )

        return {
            "stage": "validation",
            "schema": schema_report,
            "semantics": semantic_report,
            "valid": schema_report["valid"] and semantic_report["valid"],
        }
