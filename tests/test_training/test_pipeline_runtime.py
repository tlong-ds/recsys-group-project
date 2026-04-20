from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recsys.training.pipeline import main, run_training_pipeline


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1, 3], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
            ],
            "edge_index": [
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
            ],
            "alias_inputs": [
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ],
            "item_seq_len": [2, 2, 2],
            "pos_items": [3, 4, 4],
            "session_id": [11, 12, 13],
        }
    )


def _write_examples_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.table(
        {
            "x": [list(map(int, v.tolist())) for v in df["x"]],
            "edge_index": [
                [list(map(int, v[0].tolist())), list(map(int, v[1].tolist()))]
                for v in df["edge_index"]
            ],
            "alias_inputs": [list(map(int, v.tolist())) for v in df["alias_inputs"]],
            "item_seq_len": df["item_seq_len"].astype(int).tolist(),
            "pos_items": df["pos_items"].astype(int).tolist(),
            "session_id": df["session_id"].astype(int).tolist(),
        },
        schema=pa.schema(
            [
                pa.field("x", pa.list_(pa.int64())),
                pa.field("edge_index", pa.list_(pa.list_(pa.int64()))),
                pa.field("alias_inputs", pa.list_(pa.int64())),
                pa.field("item_seq_len", pa.int64()),
                pa.field("pos_items", pa.int64()),
                pa.field("session_id", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, path)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    data_path = tmp_path / "data" / "processed"
    data_path.mkdir(parents=True)

    _write_examples_parquet(_examples(), data_path / "train_examples.parquet")
    _write_examples_parquet(
        _examples().iloc[:1].copy(), data_path / "val_examples.parquet"
    )
    _write_examples_parquet(
        _examples().iloc[1:2].copy(), data_path / "test_examples.parquet"
    )
    (data_path / "item_vocab.json").write_text(
        json.dumps({"item2id": {"101": 1, "102": 2, "103": 3, "104": 4}}),
        encoding="utf-8",
    )
    return data_path


def _config(data_dir: Path, registry_root: Path) -> dict[str, object]:
    return {
        "data": {
            "train_examples_path": str(data_dir / "train_examples.parquet"),
            "val_examples_path": str(data_dir / "val_examples.parquet"),
            "test_examples_path": str(data_dir / "test_examples.parquet"),
            "item_vocab_path": str(data_dir / "item_vocab.json"),
            "processed_path": str(data_dir),
        },
        "model": {
            "type": "srgnn",
            "variant": "srgnn",
            "name": "srgnn",
            "embedding_dim": 8,
            "hidden_size": 8,
            "step": 1,
            "max_session_length": 5,
            "fallback_weight": 0.0,
            "version": "test",
        },
        "training": {
            "seed": 7,
            "batch_size": 2,
            "num_epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "early_stopping_patience": 1,
            "top_k": 3,
            "num_workers": 0,
        },
        "registry": {"root_path": str(registry_root)},
        "mlflow": {"enabled": False},
    }


def test_run_training_pipeline_uses_custom_output_paths(
    data_dir: Path, tmp_path: Path
) -> None:
    metrics_dir = tmp_path / "metrics" / "v1_strict_filter"
    registry_root = tmp_path / "models" / "trained" / "v1_strict_filter"
    versioned_dir = tmp_path / "data" / "versions" / "v1_strict_filter" / "processed"
    versioned_dir.mkdir(parents=True)

    for filename in (
        "train_examples.parquet",
        "val_examples.parquet",
        "test_examples.parquet",
        "item_vocab.json",
    ):
        source = data_dir / filename
        target = versioned_dir / filename
        if source.suffix == ".json":
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            target.write_bytes(source.read_bytes())

    config = _config(versioned_dir, registry_root)
    config["training"]["train_metrics_path"] = str(
        metrics_dir / "training_metrics.json"
    )
    config["training"]["evaluation_metrics_path"] = str(
        metrics_dir / "evaluation_metrics.json"
    )
    config["lineage"] = {
        "data_version": "v1_strict_filter",
        "data_params_path": "configs/data_versions/v1_strict_filter.yaml",
    }

    result = run_training_pipeline(config)

    train_metrics = Path(result["training_metrics"])
    eval_metrics = Path(result["evaluation_metrics"])
    assert train_metrics == metrics_dir / "training_metrics.json"
    assert eval_metrics == metrics_dir / "evaluation_metrics.json"
    assert (registry_root / "latest" / "model.json").exists()

    train_payload = json.loads(train_metrics.read_text(encoding="utf-8"))
    eval_payload = json.loads(eval_metrics.read_text(encoding="utf-8"))
    assert train_payload["data_version"] == "v1_strict_filter"
    assert (
        eval_payload["data_params_path"]
        == "configs/data_versions/v1_strict_filter.yaml"
    )


def test_main_applies_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_load_training_runtime_config(**kwargs):
        captured["load_kwargs"] = kwargs
        return {
            "data": {"processed_path": "data/processed"},
            "training": {},
            "registry": {"root_path": "models/trained"},
        }

    def fake_run_train_stage(config):
        captured["config"] = config
        return {"ok": True}

    monkeypatch.setattr(
        "recsys.training.pipeline.load_training_runtime_config",
        fake_load_training_runtime_config,
    )
    monkeypatch.setattr(
        "recsys.training.pipeline.run_train_stage", fake_run_train_stage
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recsys-train",
            "--stage",
            "train",
            "--params",
            "params.yaml",
            "--data-params",
            "configs/data_versions/v1_strict_filter.yaml",
            "--registry-root",
            str(tmp_path / "models" / "trained" / "v1_strict_filter"),
            "--train-metrics-path",
            str(tmp_path / "metrics" / "v1" / "training_metrics.json"),
            "--device",
            "cpu",
        ],
    )

    main()

    assert (
        captured["load_kwargs"]["data_params_path"]
        == "configs/data_versions/v1_strict_filter.yaml"
    )
    config = captured["config"]
    assert config["registry"]["root_path"].endswith("models/trained/v1_strict_filter")
    assert config["training"]["train_metrics_path"].endswith(
        "metrics/v1/training_metrics.json"
    )
    assert config["training"]["device"] == "cpu"
    assert config["lineage"]["data_version"] == "v1_strict_filter"
