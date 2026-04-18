from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from recsys.training.pipeline import run_training_pipeline


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
            "x": [list(map(int, value.tolist())) for value in df["x"]],
            "edge_index": [
                [list(map(int, value[0].tolist())), list(map(int, value[1].tolist()))]
                for value in df["edge_index"]
            ],
            "alias_inputs": [list(map(int, value.tolist())) for value in df["alias_inputs"]],
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


def test_pipeline_runs_end_to_end_on_processed_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    train_df = _examples()
    val_df = _examples().iloc[:1].copy()
    test_df = _examples().iloc[1:2].copy()

    _write_examples_parquet(train_df, data_dir / "train_examples.parquet")
    _write_examples_parquet(val_df, data_dir / "val_examples.parquet")
    _write_examples_parquet(test_df, data_dir / "test_examples.parquet")
    (data_dir / "item_vocab.json").write_text(json.dumps({"item2id": {"101": 1, "102": 2, "103": 3, "104": 4}}), encoding="utf-8")

    config = {
        "data": {
            "train_examples_path": str(data_dir / "train_examples.parquet"),
            "val_examples_path": str(data_dir / "val_examples.parquet"),
            "test_examples_path": str(data_dir / "test_examples.parquet"),
            "item_vocab_path": str(data_dir / "item_vocab.json"),
        },
        "model": {
            "name": "srgnn",
            "version": "test",
            "embedding_dim": 8,
            "hidden_size": 8,
            "step": 1,
            "max_session_length": 5,
            "fallback_weight": 0.0,
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
        "registry": {"root_path": str(tmp_path / "models")},
        "mlflow": {"enabled": False},
    }
    result = run_training_pipeline(config)

    assert Path(result["artifact_path"]).exists()
    assert set(result["validation_metrics"]) == {"hr@k", "mrr@k"}
    assert set(result["test_metrics"]) == {"hr@k", "mrr@k"}
