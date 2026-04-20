"""Feature extraction helpers for offline drift monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_COLUMN_CANDIDATES = ("sessionId", "session_id")
ITEM_COLUMN_CANDIDATES = ("itemId", "item_id")
TIMESTAMP_COLUMN_CANDIDATES = ("eventdate", "event_date", "timestamp")


@dataclass(frozen=True)
class InteractionColumns:
    """Resolved column names for an interaction table."""

    session: str
    item: str
    timestamp: str


@dataclass(frozen=True)
class MonitoringView:
    """Derived monitoring features and aggregate counts for one data window."""

    session_features: pd.DataFrame
    item_counts: dict[int, int]
    total_interactions: int
    unique_items: int
    oov_interactions: int


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load an interaction table from parquet or csv."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported interaction format: {source}")


def load_item_vocab(path: str | Path) -> set[int]:
    """Load original item IDs from an item vocabulary JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("item2id"), dict):
        keys = payload["item2id"].keys()
    elif isinstance(payload, dict):
        keys = payload.keys()
    else:
        raise ValueError(f"Unsupported item vocab payload: {path}")

    vocab: set[int] = set()
    for key in keys:
        try:
            vocab.add(int(key))
        except (TypeError, ValueError):
            continue
    return vocab


def resolve_interaction_columns(
    interactions: pd.DataFrame,
    *,
    session_col: str | None = None,
    item_col: str | None = None,
    timestamp_col: str | None = None,
) -> InteractionColumns:
    """Resolve interaction columns using explicit names or repo defaults."""
    return InteractionColumns(
        session=_resolve_column(
            interactions,
            explicit=session_col,
            candidates=SESSION_COLUMN_CANDIDATES,
            logical_name="session",
        ),
        item=_resolve_column(
            interactions,
            explicit=item_col,
            candidates=ITEM_COLUMN_CANDIDATES,
            logical_name="item",
        ),
        timestamp=_resolve_column(
            interactions,
            explicit=timestamp_col,
            candidates=TIMESTAMP_COLUMN_CANDIDATES,
            logical_name="timestamp",
        ),
    )


def build_monitoring_view(
    interactions: pd.DataFrame,
    *,
    vocab_items: set[int] | None = None,
    columns: InteractionColumns | None = None,
) -> MonitoringView:
    """Build session-level features and item-level aggregates."""
    cols = columns or resolve_interaction_columns(interactions)
    working = interactions[[cols.session, cols.item, cols.timestamp]].copy()
    working[cols.timestamp] = pd.to_datetime(
        working[cols.timestamp],
        errors="coerce",
    )
    working["_item_int"] = pd.to_numeric(working[cols.item], errors="coerce")

    if vocab_items is None:
        working["_is_oov"] = False
    else:
        working["_is_oov"] = ~working["_item_int"].apply(
            _item_in_vocab,
            args=(vocab_items,),
        )

    grouped = working.groupby(cols.session, dropna=False)
    session_features = grouped.agg(
        session_length=("_item_int", "size"),
        unique_items=("_item_int", "nunique"),
        first_ts=(cols.timestamp, "min"),
        last_ts=(cols.timestamp, "max"),
        oov_count=("_is_oov", "sum"),
    )
    session_features = session_features.reset_index().rename(
        columns={cols.session: "session_id"}
    )
    duration = session_features["last_ts"] - session_features["first_ts"]
    session_features["duration_seconds"] = (
        duration.dt.total_seconds().fillna(0.0).clip(lower=0.0)
    )
    session_features["first_event_hour"] = (
        session_features["first_ts"].dt.hour.fillna(-1).astype(int)
    )
    session_features["repeat_ratio"] = 1.0 - (
        session_features["unique_items"]
        / session_features["session_length"].replace(0, np.nan)
    )
    session_features["repeat_ratio"] = session_features["repeat_ratio"].fillna(0.0)
    session_features["oov_ratio"] = (
        session_features["oov_count"]
        / session_features["session_length"].replace(0, np.nan)
    ).fillna(0.0)
    session_features = session_features[
        [
            "session_id",
            "session_length",
            "unique_items",
            "repeat_ratio",
            "duration_seconds",
            "first_event_hour",
            "oov_count",
            "oov_ratio",
        ]
    ]

    item_series = working["_item_int"].dropna().astype(int)
    item_counts = {
        int(item): int(count)
        for item, count in item_series.value_counts().to_dict().items()
    }

    return MonitoringView(
        session_features=session_features,
        item_counts=item_counts,
        total_interactions=int(len(working)),
        unique_items=int(item_series.nunique()),
        oov_interactions=int(working["_is_oov"].sum()),
    )


def sample_sessions(
    interactions: pd.DataFrame,
    *,
    columns: InteractionColumns,
    sample_size: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Sample up to ``sample_size`` sessions for faster demo runs."""
    if sample_size is None or sample_size <= 0:
        return interactions

    sessions = pd.Series(interactions[columns.session].dropna().unique())
    if len(sessions) <= sample_size:
        return interactions

    sampled_sessions = sessions.sample(
        n=sample_size,
        random_state=random_seed,
    )
    return interactions[interactions[columns.session].isin(sampled_sessions)].copy()


def inject_oov_items(
    interactions: pd.DataFrame,
    *,
    columns: InteractionColumns,
    vocab_items: set[int],
    rate: float,
    random_seed: int,
) -> pd.DataFrame:
    """Replace a controlled share of item IDs with unseen synthetic IDs."""
    if rate <= 0:
        return interactions

    injected = interactions.copy()
    rng = np.random.default_rng(random_seed)
    n_rows = len(injected)
    if n_rows == 0:
        return injected

    n_inject = min(n_rows, max(1, int(round(n_rows * rate))))
    row_positions = rng.choice(n_rows, size=n_inject, replace=False)
    existing_max = pd.to_numeric(injected[columns.item], errors="coerce").max()
    start = int(max(vocab_items or {0}) + 1)
    if pd.notna(existing_max):
        start = max(start, int(existing_max) + 1)

    injected_items = np.arange(start, start + n_inject, dtype=np.int64)
    item_col_position = injected.columns.get_loc(columns.item)
    injected.iloc[row_positions, item_col_position] = injected_items
    return injected


def shift_session_lengths(
    interactions: pd.DataFrame,
    *,
    columns: InteractionColumns,
    factor: float,
    random_seed: int,
) -> pd.DataFrame:
    """Create a synthetic session-length shift for demo scenarios."""
    if factor == 1.0:
        return interactions
    if factor <= 0:
        raise ValueError("--inject-session-length-shift must be positive")

    if factor > 1.0:
        whole = int(np.floor(factor))
        frac = factor - whole
        parts = [interactions.copy()]
        for _ in range(max(0, whole - 1)):
            parts.append(interactions.copy())
        if frac > 0:
            parts.append(
                interactions.sample(frac=frac, random_state=random_seed).copy()
            )
        return pd.concat(parts, ignore_index=True)

    reduced = (
        interactions.groupby(columns.session, group_keys=False)
        .apply(lambda group: group.head(max(1, int(np.ceil(len(group) * factor)))))
        .reset_index(drop=True)
    )
    return reduced


def _resolve_column(
    interactions: pd.DataFrame,
    *,
    explicit: str | None,
    candidates: tuple[str, ...],
    logical_name: str,
) -> str:
    if explicit:
        if explicit not in interactions.columns:
            raise ValueError(f"Missing configured {logical_name} column: {explicit}")
        return explicit

    for candidate in candidates:
        if candidate in interactions.columns:
            return candidate
    raise ValueError(
        f"Could not resolve {logical_name} column. "
        f"Tried: {', '.join(candidates)}"
    )


def _item_in_vocab(value: object, vocab_items: set[int]) -> bool:
    if pd.isna(value):
        return False
    try:
        return int(value) in vocab_items
    except (TypeError, ValueError):
        return False
