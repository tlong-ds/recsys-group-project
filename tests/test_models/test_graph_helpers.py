from __future__ import annotations

from types import MethodType

import numpy as np
import pandas as pd
import torch

from recsys.models.graph_helpers import (
    _build_ngc_row_totals,
    _prefetch_batches,
    build_adjacency,
    build_adjacency_fc,
    build_adjacency_ngc,
)
from recsys.models.srgnn import SRGNNRecommender


def test_prefetch_batches_cpu_path_moves_all_batches() -> None:
    loader = [
        {"items": torch.tensor([1, 2], dtype=torch.long)},
        {"items": torch.tensor([3], dtype=torch.long)},
    ]

    batches = list(_prefetch_batches(loader, torch.device("cpu"), non_blocking=True))

    assert len(batches) == 2
    assert batches[0]["items"].device.type == "cpu"
    assert batches[1]["items"].device.type == "cpu"
    assert batches[0]["items"].tolist() == [1, 2]
    assert batches[1]["items"].tolist() == [3]


def test_build_adjacency_ngc_row_totals_cache_matches_default() -> None:
    alias_inputs = np.asarray([0, 1, 2], dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    node_items = [10, 20, 30]
    global_freq = {
        (10, 20): 4.0,
        (20, 30): 2.0,
        (10, 40): 6.0,
    }

    default_adj = build_adjacency_ngc(
        alias_inputs=alias_inputs,
        edge_index=edge_index,
        n_nodes=3,
        global_freq=global_freq,
        node_items=node_items,
    )
    cached_adj = build_adjacency_ngc(
        alias_inputs=alias_inputs,
        edge_index=edge_index,
        n_nodes=3,
        global_freq=global_freq,
        global_row_totals=_build_ngc_row_totals(global_freq),
        node_items=node_items,
    )

    assert np.allclose(default_adj, cached_adj)


def test_build_adjacency_handles_invalid_edges() -> None:
    alias_inputs = np.asarray([0, 1], dtype=np.int64)
    edge_index = np.asarray([[0, 2, 1], [1, 0, -1]], dtype=np.int64)
    adjacency = build_adjacency(alias_inputs, edge_index, n_nodes=2)

    expected_a_in = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    expected_a_out = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    expected = np.concatenate([expected_a_in, expected_a_out], axis=1)
    assert np.allclose(adjacency, expected)


def test_build_adjacency_fc_creates_off_diagonal_connections() -> None:
    alias_inputs = np.asarray([0, 1, 0], dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    adjacency = build_adjacency_fc(alias_inputs, edge_index, n_nodes=2)

    # fc_in (cols 4:6) and fc_out (cols 6:8) should link node0 <-> node1 only.
    expected_fc = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    fc_in = adjacency[:, 4:6]
    fc_out = adjacency[:, 6:8]
    assert np.allclose(fc_in, expected_fc)
    assert np.allclose(fc_out, expected_fc)


def _tiny_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1, 3], dtype=np.int64),
            ],
            "edge_index": [
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
            ],
            "alias_inputs": [
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ],
            "item_seq_len": [2, 2],
            "pos_items": [3, 2],
        }
    )


def test_fit_early_stopping_uses_min_delta_with_validation() -> None:
    train_df = _tiny_examples()
    val_df = _tiny_examples().iloc[:1].copy()
    model = SRGNNRecommender(
        embedding_dim=8,
        hidden_size=8,
        step=1,
        max_session_length=5,
        fallback_weight=0.0,
        seed=7,
        device="cpu",
    )

    counters = {"train": 0, "val": 0}

    def fake_run_epoch(self, loader, optimizer, criterion):  # noqa: ARG001
        counters["train"] += 1
        return 1.0

    def fake_eval_epoch(self, loader, criterion):  # noqa: ARG001
        counters["val"] += 1
        return 1.0 - (0.0001 * (counters["val"] - 1))

    model._run_epoch = MethodType(fake_run_epoch, model)
    model._eval_epoch = MethodType(fake_eval_epoch, model)

    model.fit(
        train_df=train_df,
        val_df=val_df,
        num_epochs=1000,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-3,
        num_workers=0,
    )

    assert counters["train"] == 3
    assert counters["val"] == 3


def test_fit_early_stops_on_train_loss_when_validation_missing() -> None:
    train_df = _tiny_examples()
    model = SRGNNRecommender(
        embedding_dim=8,
        hidden_size=8,
        step=1,
        max_session_length=5,
        fallback_weight=0.0,
        seed=7,
        device="cpu",
    )

    counters = {"train": 0}
    train_losses = [1.0, 0.9, 0.9, 0.91, 0.92]

    def fake_run_epoch(self, loader, optimizer, criterion):  # noqa: ARG001
        idx = counters["train"]
        counters["train"] += 1
        return train_losses[min(idx, len(train_losses) - 1)]

    model._run_epoch = MethodType(fake_run_epoch, model)

    model.fit(
        train_df=train_df,
        val_df=None,
        num_epochs=1000,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        num_workers=0,
    )

    assert counters["train"] == 4
