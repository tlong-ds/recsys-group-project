from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recsys.monitoring.drift import (
    STATUS_CRITICAL,
    STATUS_OK,
    build_drift_report,
    jensen_shannon_divergence,
    main,
    population_stability_index,
    run_drift_monitoring,
    top_n_overlap,
)
from recsys.monitoring.features import build_monitoring_view


def _interactions(items: list[int]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(items):
        rows.append(
            {
                "sessionId": idx // 2,
                "itemId": item,
                "eventdate": f"2021-01-01 00:00:{idx:02d}",
            }
        )
    return pd.DataFrame(rows)


def test_psi_is_zero_for_same_distribution() -> None:
    score = population_stability_index([1, 2, 3, 4], [1, 2, 3, 4])

    assert score == pytest.approx(0.0)


def test_psi_detects_shifted_distribution() -> None:
    score = population_stability_index([1] * 100, [10] * 100)

    assert score >= 0.25


def test_item_distribution_metrics_detect_overlap_and_divergence() -> None:
    reference = {1: 10, 2: 9, 3: 8}
    current_same = {1: 8, 2: 7, 3: 6}
    current_shifted = {4: 10, 5: 9, 6: 8}

    assert top_n_overlap(reference, current_same, top_n=3) == 1.0
    assert top_n_overlap(reference, current_shifted, top_n=3) == 0.0
    assert jensen_shannon_divergence(reference, current_same) < 0.05
    assert jensen_shannon_divergence(reference, current_shifted) >= 0.15


def test_drift_report_status_is_ok_for_same_window() -> None:
    interactions = _interactions([1, 2, 1, 2, 3, 3])
    view = build_monitoring_view(interactions, vocab_items={1, 2, 3})

    report = build_drift_report(
        reference_view=view,
        current_view=view,
        reference_path="reference.parquet",
        current_path="current.parquet",
        top_n=3,
    )

    assert report["status"] == STATUS_OK
    assert report["summary"]["critical_checks"] == 0


def test_drift_report_status_escalates_for_oov_window() -> None:
    reference = build_monitoring_view(
        _interactions([1, 2, 1, 2, 3, 3]),
        vocab_items={1, 2, 3},
    )
    current = build_monitoring_view(
        _interactions([999, 1000, 1001, 1002]),
        vocab_items={1, 2, 3},
    )

    report = build_drift_report(
        reference_view=reference,
        current_view=current,
        reference_path="reference.parquet",
        current_path="current.parquet",
        top_n=3,
    )

    assert report["status"] == STATUS_CRITICAL
    assert report["checks"]["oov_ratio"]["status"] == STATUS_CRITICAL


def test_run_drift_monitoring_writes_json(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    vocab_path = tmp_path / "item_vocab.json"
    output_path = tmp_path / "metrics" / "drift.json"

    _interactions([1, 2, 1, 2]).to_parquet(reference_path, index=False)
    _interactions([1, 2, 999, 1000]).to_parquet(current_path, index=False)
    vocab_path.write_text(
        json.dumps({"item2id": {"1": 1, "2": 2}}),
        encoding="utf-8",
    )

    report = run_drift_monitoring(
        reference_path=reference_path,
        current_path=current_path,
        vocab_path=vocab_path,
        output_path=output_path,
        top_n=2,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == report["status"]
    assert "summary" in written
    assert "checks" in written


def test_cli_writes_json(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    vocab_path = tmp_path / "item_vocab.json"
    output_path = tmp_path / "drift.json"

    _interactions([1, 2, 1, 2]).to_parquet(reference_path, index=False)
    _interactions([1, 2, 1, 2]).to_parquet(current_path, index=False)
    vocab_path.write_text(
        json.dumps({"item2id": {"1": 1, "2": 2}}),
        encoding="utf-8",
    )

    main(
        [
            "--reference",
            str(reference_path),
            "--current",
            str(current_path),
            "--vocab",
            str(vocab_path),
            "--output",
            str(output_path),
        ]
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == STATUS_OK


def test_evidently_html_generation_is_optional(tmp_path: Path) -> None:
    pytest.importorskip("evidently")
    from recsys.monitoring.evidently_report import write_evidently_report

    features = build_monitoring_view(
        _interactions([1, 2, 1, 2]),
        vocab_items={1, 2},
    ).session_features
    output_path = tmp_path / "report.html"

    written = write_evidently_report(
        reference_features=features,
        current_features=features,
        output_path=output_path,
    )

    assert written == output_path
    assert output_path.exists()

