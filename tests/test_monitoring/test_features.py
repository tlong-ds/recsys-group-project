from __future__ import annotations

import pandas as pd

from recsys.monitoring.features import (
    build_monitoring_view,
    inject_oov_items,
    resolve_interaction_columns,
    sample_sessions,
    shift_session_lengths,
)


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sessionId": [1, 1, 2, 2],
            "itemId": [10, 10, 20, 999],
            "eventdate": [
                "2021-01-01 00:00:00",
                "2021-01-01 00:00:10",
                "2021-01-01 01:00:00",
                "2021-01-01 01:01:00",
            ],
        }
    )


def test_build_monitoring_view_computes_session_features() -> None:
    view = build_monitoring_view(_interactions(), vocab_items={10, 20})

    first = view.session_features.sort_values("session_id").iloc[0]
    second = view.session_features.sort_values("session_id").iloc[1]

    assert first["session_length"] == 2
    assert first["unique_items"] == 1
    assert first["repeat_ratio"] == 0.5
    assert first["duration_seconds"] == 10.0
    assert first["first_event_hour"] == 0
    assert first["oov_count"] == 0

    assert second["session_length"] == 2
    assert second["unique_items"] == 2
    assert second["repeat_ratio"] == 0.0
    assert second["duration_seconds"] == 60.0
    assert second["first_event_hour"] == 1
    assert second["oov_count"] == 1
    assert second["oov_ratio"] == 0.5
    assert view.oov_interactions == 1


def test_sample_sessions_limits_number_of_sessions() -> None:
    interactions = _interactions()
    columns = resolve_interaction_columns(interactions)

    sampled = sample_sessions(
        interactions,
        columns=columns,
        sample_size=1,
        random_seed=7,
    )

    assert sampled["sessionId"].nunique() == 1


def test_inject_oov_items_replaces_requested_share() -> None:
    interactions = _interactions()
    columns = resolve_interaction_columns(interactions)

    injected = inject_oov_items(
        interactions,
        columns=columns,
        vocab_items={10, 20},
        rate=0.5,
        random_seed=3,
    )
    view = build_monitoring_view(injected, vocab_items={10, 20}, columns=columns)

    assert view.oov_interactions >= 2


def test_shift_session_lengths_can_increase_window_size() -> None:
    interactions = _interactions()
    columns = resolve_interaction_columns(interactions)

    shifted = shift_session_lengths(
        interactions,
        columns=columns,
        factor=2.0,
        random_seed=42,
    )

    assert len(shifted) == len(interactions) * 2

