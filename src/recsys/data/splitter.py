"""Temporal splitting for session-based training data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from typing import Literal


def split_by_time(
    df: pd.DataFrame,
    timestamp_col: str = "eventdate",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split a DataFrame into train, validation, and test sets by ratio.
    
    Args:
        df: Input DataFrame sorted by timestamp.
        timestamp_col: Name of timestamp column.
        val_ratio: Fraction of data for validation [0, 1).
        test_ratio: Fraction of data for test [0, 1).
    
    Returns:
        Tuple of (train_df, val_df, test_df).
    
    Raises:
        ValueError: If ratios are invalid.
    """
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Split ratios must be in the [0, 1) range")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1")
    
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n_rows = len(ordered)
    test_start = int(n_rows * (1 - test_ratio))
    val_start = int(n_rows * (1 - test_ratio - val_ratio))

    train_df = ordered.iloc[:val_start].reset_index(drop=True)
    val_df = ordered.iloc[val_start:test_start].reset_index(drop=True)
    test_df = ordered.iloc[test_start:].reset_index(drop=True)
    
    logger.info(
        f"Temporal split (ratio-based): "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    
    return train_df, val_df, test_df


def split_by_days(
    df: pd.DataFrame,
    timestamp_col: str = "eventdate",
    test_days: int = 7,
    val_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by date boundaries (last N days for test, etc).
    
    Args:
        df: Input DataFrame with timestamp column.
        timestamp_col: Name of timestamp column.
        test_days: Number of days in test set from the end.
        val_days: Number of days in validation set (before test set).
    
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()
    
    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(ordered[timestamp_col]):
        ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col])
    
    max_date = ordered[timestamp_col].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)
    
    train_df = ordered[ordered[timestamp_col] < val_cutoff].reset_index(drop=True)
    val_df = ordered[
        (ordered[timestamp_col] >= val_cutoff) & (ordered[timestamp_col] < test_cutoff)
    ].reset_index(drop=True)
    test_df = ordered[ordered[timestamp_col] >= test_cutoff].reset_index(drop=True)
    
    logger.info(
        f"Temporal split (time-based): "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    logger.info(f"Date boundaries: {val_cutoff} | {test_cutoff} | {max_date}")
    
    return train_df, val_df, test_df


def split_by_session_time(
    df: pd.DataFrame,
    session_col: str = "session_id",
    timestamp_col: str = "eventdate",
    test_days: int = 7,
    val_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally on complete sessions (no session split across sets).
    
    Args:
        df: Input DataFrame with session and timestamp columns.
        session_col: Name of session ID column.
        timestamp_col: Name of timestamp column.
        test_days: Number of days in test set from the end.
        val_days: Number of days in validation set (before test set).
    
    Returns:
        Tuple of (train_df, val_df, test_df) with complete sessions.
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()
    
    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(ordered[timestamp_col]):
        ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col])
    
    # Get session-level timestamps (last interaction in each session)
    session_timestamps = ordered.groupby(session_col)[timestamp_col].max()
    
    max_date = session_timestamps.max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)
    
    # Classify sessions
    train_sessions = session_timestamps[session_timestamps < val_cutoff].index
    val_sessions = session_timestamps[
        (session_timestamps >= val_cutoff) & (session_timestamps < test_cutoff)
    ].index
    test_sessions = session_timestamps[session_timestamps >= test_cutoff].index
    
    train_df = ordered[ordered[session_col].isin(train_sessions)].reset_index(drop=True)
    val_df = ordered[ordered[session_col].isin(val_sessions)].reset_index(drop=True)
    test_df = ordered[ordered[session_col].isin(test_sessions)].reset_index(drop=True)
    
    logger.info(
        f"Temporal split (session-based): "
        f"train={len(train_df):,} (sessions={len(train_sessions):,}) "
        f"val={len(val_df):,} (sessions={len(val_sessions):,}) "
        f"test={len(test_df):,} (sessions={len(test_sessions):,})"
    )
    logger.info(f"Date boundaries: {val_cutoff} | {test_cutoff} | {max_date}")
    
    return train_df, val_df, test_df


def split_diginetica_legacy(
    df: pd.DataFrame,
    session_col: str = "session_id",
    timestamp_col: str = "eventdate",
    test_days: int = 7,
    val_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sessions following the original SR-GNN Diginetica idea.

    Original behavior:
    - Session date is used for split boundaries.
    - Train sessions: session_date < split_date
    - Test sessions: session_date > split_date
    - Sessions exactly on split_date are excluded.

    For MLOps compatibility, optional validation split is carved from train when
    ``val_days > 0`` while keeping the same temporal idea.
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    ordered = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(ordered[timestamp_col]):
        ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], errors="coerce")

    ordered = ordered.dropna(subset=[timestamp_col]).reset_index(drop=True)
    if ordered.empty:
        return ordered.copy(), ordered.copy(), ordered.copy()

    session_dates = ordered.groupby(session_col)[timestamp_col].max()
    max_date = session_dates.max()
    split_date = max_date - pd.Timedelta(days=test_days)

    test_sessions = session_dates[session_dates > split_date].index
    train_candidate_sessions = session_dates[session_dates < split_date].index

    if val_days > 0:
        train_candidate_dates = session_dates.loc[train_candidate_sessions]
        val_cutoff = split_date - pd.Timedelta(days=val_days)
        val_sessions = train_candidate_dates[train_candidate_dates >= val_cutoff].index
        train_sessions = train_candidate_dates[train_candidate_dates < val_cutoff].index
    else:
        train_sessions = train_candidate_sessions
        val_sessions = pd.Index([])

    train_df = ordered[ordered[session_col].isin(train_sessions)].reset_index(drop=True)
    val_df = ordered[ordered[session_col].isin(val_sessions)].reset_index(drop=True)
    test_df = ordered[ordered[session_col].isin(test_sessions)].reset_index(drop=True)

    logger.info(
        "Temporal split (diginetica-legacy): "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    logger.info(f"Date boundaries: split={split_date} max={max_date}")

    return train_df, val_df, test_df

