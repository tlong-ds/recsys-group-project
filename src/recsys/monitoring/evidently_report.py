"""Optional Evidently HTML report generation for drift monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERICAL_COLUMNS = [
    "session_length",
    "unique_items",
    "repeat_ratio",
    "duration_seconds",
    "oov_count",
    "oov_ratio",
]
CATEGORICAL_COLUMNS = ["first_event_hour"]


def write_evidently_report(
    *,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write an Evidently data summary + drift report as HTML."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is not installed. Install it with: "
            "pip install -e .[monitoring]"
        ) from exc

    available_numeric = [
        column for column in NUMERICAL_COLUMNS if column in reference_features.columns
    ]
    available_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in reference_features.columns
    ]
    schema = DataDefinition(
        numerical_columns=available_numeric,
        categorical_columns=available_categorical,
    )

    reference_dataset = Dataset.from_pandas(
        reference_features[available_numeric + available_categorical],
        data_definition=schema,
    )
    current_dataset = Dataset.from_pandas(
        current_features[available_numeric + available_categorical],
        data_definition=schema,
    )
    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests=True,
    )
    snapshot = report.run(current_dataset, reference_dataset)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(destination))
    return destination

