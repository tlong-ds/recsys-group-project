from __future__ import annotations

import torch

from recsys.models.graph_helpers import _prefetch_batches


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
