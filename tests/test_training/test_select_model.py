from __future__ import annotations

import json
from pathlib import Path

from recsys.training.select_model import select_best_model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_best_model_compares_model_and_data_versions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    # Baseline/srgnn -> better HR than baseline/ggnn.
    _write_json(
        tmp_path / "metrics/experiments/baseline/srgnn/evaluation_metrics.json",
        {"test_metrics": {"hr@k": 0.42, "mrr@k": 0.10}},
    )
    _write_json(
        tmp_path / "metrics/experiments/baseline/ggnn/evaluation_metrics.json",
        {"test_metrics": {"hr@k": 0.35, "mrr@k": 0.12}},
    )
    # v2/srgnn_fc should win overall.
    _write_json(
        tmp_path
        / "metrics/experiments/v2_sliding_window/srgnn_fc/evaluation_metrics.json",
        {"test_metrics": {"hr@k": 0.49, "mrr@k": 0.16}},
    )

    select_best_model()

    best_payload = json.loads(
        (tmp_path / "metrics/best_model.json").read_text(encoding="utf-8")
    )
    best_model = best_payload["best_model"]
    assert best_model["data_version"] == "v2_sliding_window"
    assert best_model["model_profile"] == "srgnn_fc"
    assert best_model["selection_metrics"]["primary"] == 0.49
    assert not (tmp_path / "models/trained/latest").exists()
