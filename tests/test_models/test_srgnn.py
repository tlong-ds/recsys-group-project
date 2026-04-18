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
def test_recommend_from_graph_consistent_with_recommend() -> None:
    model = SRGNNRecommender(embedding_dim=16, hidden_size=16, seed=7).fit(
        _graph_examples(), num_epochs=1, batch_size=2,
    )
    # recommend_from_graph và recommend phải trả cùng kết quả
    # cho cùng một session [1, 2]
    recs_seq   = model.recommend([1, 2], top_k=3)
    x          = np.asarray([1, 2], dtype=np.int64)
    edge_index = np.asarray([[0], [1]], dtype=np.int64)
    alias      = np.asarray([0, 1], dtype=np.int64)
    recs_graph = model.recommend_from_graph(x, edge_index, alias, top_k=3)

    assert recs_seq == recs_graph
