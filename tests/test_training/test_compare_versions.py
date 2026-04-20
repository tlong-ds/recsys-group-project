from __future__ import annotations

import json
from pathlib import Path

import pytest

from recsys.training.compare_versions import (
    build_comparison_report,
    write_comparison_report,
)


def test_build_comparison_report_merges_train_and_eval_payloads(tmp_path: Path) -> None:
    train_metrics = tmp_path / "train.json"
    eval_metrics = tmp_path / "eval.json"

    train_metrics.write_text(
        json.dumps(
            {
                "artifact_path": "models/trained/v1/latest/model.pt",
                "validation_metrics": {"hr@k": 0.5, "mrr@k": 0.25},
                "data_version": "v1_strict_filter",
                "data_params_path": "configs/data_versions/v1_strict_filter.yaml",
            }
        ),
        encoding="utf-8",
    )
    eval_metrics.write_text(
        json.dumps(
            {
                "model_path": "models/trained/v1/latest",
                "test_metrics": {"hr@k": 0.4, "mrr@k": 0.2},
                "data_version": "v1_strict_filter",
                "data_params_path": "configs/data_versions/v1_strict_filter.yaml",
            }
        ),
        encoding="utf-8",
    )

    payload = build_comparison_report(
        [("v1_strict_filter", train_metrics, eval_metrics)]
    )

    assert len(payload["versions"]) == 1
    version = payload["versions"][0]
    assert version["name"] == "v1_strict_filter"
    assert version["validation_metrics"] == {"hr@k": 0.5, "mrr@k": 0.25}
    assert version["test_metrics"] == {"hr@k": 0.4, "mrr@k": 0.2}


def test_build_comparison_report_rejects_mismatched_versions(tmp_path: Path) -> None:
    train_metrics = tmp_path / "train.json"
    eval_metrics = tmp_path / "eval.json"
    train_metrics.write_text(json.dumps({"data_version": "v1"}), encoding="utf-8")
    eval_metrics.write_text(json.dumps({"data_version": "v2"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Mismatched data_version"):
        build_comparison_report([("requested", train_metrics, eval_metrics)])


def test_write_comparison_report_persists_json(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics" / "comparison.json"
    payload = {"versions": [{"name": "v1"}]}

    written_path = write_comparison_report(payload, output_path)

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
