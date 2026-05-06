from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from recsys.training.retrain_selected import run_retrain_selected_model


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_retrain_selected_model_uses_selected_profile_and_train_plus_val(
    tmp_path: Path, monkeypatch
) -> None:
    _write_json(
        tmp_path / "metrics/best_model.json",
        {
            "best_model": {
                "data_version": "v2_sliding_window",
                "model_profile": "srgnn_fc",
            }
        },
    )

    data_root = tmp_path / "data/versions/v2_sliding_window/processed"
    data_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1, 2], "pos_items": [11, 12]}).to_parquet(
        data_root / "train_examples.parquet", index=False
    )
    pd.DataFrame({"x": [3], "pos_items": [13]}).to_parquet(
        data_root / "val_examples.parquet", index=False
    )
    pd.DataFrame({"x": [4], "pos_items": [14]}).to_parquet(
        data_root / "test_examples.parquet", index=False
    )
    _write_json(data_root / "item_vocab.json", {"item2id": {"11": 1, "12": 2}})

    _write_yaml(
        tmp_path / "configs/data_config.yaml", {"data": {"processed_path": "x"}}
    )
    _write_yaml(
        tmp_path / "configs/training_config.yaml",
        {"training": {"num_epochs": 2}, "mlflow": {"enabled": False}},
    )
    _write_yaml(
        tmp_path / "configs/model_profiles/srgnn_fc.yaml",
        {"model": {"type": "srgnn", "variant": "srgnn_fc", "name": "srgnn_fc"}},
    )
    _write_yaml(
        tmp_path / "configs/data_versions/v2_sliding_window.yaml",
        {
            "data": {
                "train_examples_path": str(data_root / "train_examples.parquet"),
                "val_examples_path": str(data_root / "val_examples.parquet"),
                "test_examples_path": str(data_root / "test_examples.parquet"),
                "item_vocab_path": str(data_root / "item_vocab.json"),
            }
        },
    )

    captured: dict[str, dict] = {}

    def _fake_run_training_pipeline(config: dict) -> dict:
        captured["config"] = config
        train_metrics_path = Path(config["training"]["train_metrics_path"])
        eval_metrics_path = Path(config["training"]["evaluation_metrics_path"])
        _write_json(
            train_metrics_path,
            {
                "model_registry": {
                    "model_name": "recsys-srgnn_fc-data-v2",
                    "model_version": "4",
                    "run_id": "run-004",
                    "source": "runs:/run-004/model_core",
                }
            },
        )
        _write_json(eval_metrics_path, {"test_metrics": {"hr@k": 0.5}})
        return {
            "artifact_path": "models/retrained_selected/latest/model.json",
            "training_metrics": str(train_metrics_path),
            "evaluation_metrics": str(eval_metrics_path),
        }

    monkeypatch.setattr(
        "recsys.training.retrain_selected.run_training_pipeline",
        _fake_run_training_pipeline,
    )

    result = run_retrain_selected_model(
        best_model_path=tmp_path / "metrics/best_model.json",
        data_config_path=tmp_path / "configs/data_config.yaml",
        training_config_path=tmp_path / "configs/training_config.yaml",
        data_version_config_root=tmp_path / "configs/data_versions",
        model_profile_config_root=tmp_path / "configs/model_profiles",
        retrain_data_root=tmp_path / "data/retrained_selected",
        registry_root=tmp_path / "models/retrained_selected",
        train_metrics_path=tmp_path
        / "metrics/retrained_selected/training_metrics.json",
        evaluation_metrics_path=tmp_path
        / "metrics/retrained_selected/evaluation_metrics.json",
        output_path=tmp_path / "metrics/retrained_selected/retrain_result.json",
    )

    cfg = captured["config"]
    assert cfg["model"]["variant"] == "srgnn_fc"
    assert cfg["lineage"]["data_version"] == "v2_sliding_window"
    assert cfg["lineage"]["selected_model_profile"] == "srgnn_fc"

    train_plus_val_path = Path(cfg["data"]["train_examples_path"])
    empty_val_path = Path(cfg["data"]["val_examples_path"])
    assert train_plus_val_path.exists()
    assert empty_val_path.exists()
    assert len(pd.read_parquet(train_plus_val_path)) == 3
    assert len(pd.read_parquet(empty_val_path)) == 0

    training_payload = json.loads(
        (tmp_path / "metrics/retrained_selected/training_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert training_payload["selected_model_profile"] == "srgnn_fc"
    assert training_payload["selected_data_version"] == "v2_sliding_window"

    assert result["data_version"] == "v2_sliding_window"
    assert result["model_profile"] == "srgnn_fc"
