from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.models.srgnn import SRGNNRecommender
from recsys.training.registry import ModelRegistry


def _graph_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [np.asarray([1, 2], dtype=np.int64), np.asarray([2, 3], dtype=np.int64)],
            "edge_index": [np.asarray([[0], [1]], dtype=np.int64), np.asarray([[0], [1]], dtype=np.int64)],
            "alias_inputs": [np.asarray([0, 1], dtype=np.int64), np.asarray([0, 1], dtype=np.int64)],
            "item_seq_len": [2, 2],
            "pos_items": [3, 4],
            "session_id": [1, 2],
        }
    )


def test_register_writes_latest_artifact(tmp_path: Path) -> None:
    model = SRGNNRecommender(embedding_dim=8, hidden_size=8, seed=3).fit(
        _graph_examples(),
        num_epochs=1,
        batch_size=2,
    )

    registry = ModelRegistry(tmp_path)
    artifact_path = registry.register(model, config={"model": {"name": "srgnn"}}, metrics={})

    assert artifact_path.exists()
    assert registry.latest_model_path().exists()


def test_register_can_skip_versioned_dirs(tmp_path: Path) -> None:
    model = SRGNNRecommender(embedding_dim=8, hidden_size=8, seed=3).fit(
        _graph_examples(),
        num_epochs=1,
        batch_size=2,
    )

    registry = ModelRegistry(tmp_path, create_versioned=False)
    artifact_path = registry.register(model, config={"model": {"name": "srgnn"}}, metrics={})

    latest_model_path = registry.latest_model_path()
    assert artifact_path.parent == latest_model_path
    assert (latest_model_path / "model.json").exists()
    assert not (tmp_path / "srgnn").exists()
