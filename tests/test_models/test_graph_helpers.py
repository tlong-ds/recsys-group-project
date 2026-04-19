from __future__ import annotations

import numpy as np
import torch

from recsys.models.graph_helpers import (
    _build_ngc_row_totals,
    _prefetch_batches,
    build_adjacency,
    build_adjacency_fc,
    build_adjacency_ngc,
)


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
