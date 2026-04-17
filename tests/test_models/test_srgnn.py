from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.models.srgnn import SRGNNRecommender


def _graph_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [
                np.asarray([1, 2, 3], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([2, 3, 4], dtype=np.int64),
                np.asarray([2, 4], dtype=np.int64),
            ],
            "edge_index": [
                np.asarray([[0, 1], [1, 2]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0, 1], [1, 2]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
            ],
            "alias_inputs": [
                np.asarray([0, 1, 2], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1, 2], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ],
            "item_seq_len": [3, 2, 3, 2],
            "pos_items": [4, 3, 5, 5],
            "session_id": [1, 2, 3, 4],
        }
    )


def test_srgnn_smoke_train_recommend_save_load(tmp_path: Path) -> None:
    model = SRGNNRecommender(embedding_dim=16, hidden_size=16, seed=7).fit(
        _graph_examples(),
        num_epochs=1,
        batch_size=2,
        lr=1e-2,
    )

    recommendations = model.recommend([1, 2], top_k=3)
    assert len(recommendations) == 3

    artifact_path = model.save(tmp_path)
    restored = SRGNNRecommender.load(artifact_path.parent)
    assert len(restored.recommend([1, 2], top_k=3)) == 3
