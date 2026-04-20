"""GGNN – Gated Graph Neural Network for session-based recommendation.

Reference
---------
Yujia Li, Daniel Tarlow, et al.  "Gated Graph Sequence Neural Networks."
ICLR 2016.  https://github.com/yujiali/ggnn

Session-based adaptation
-------------------------
Li et al. (SR-GNN, 2019) and subsequent works apply GGNN to session graphs.
This implementation follows that adaptation:

1. Build the session directed graph (same schema as SR-GNN).
2. Run T steps of **GGNN propagation**:
       a(v, t) = A_v · [h_1^(t-1) || ... || h_n^(t-1)]  +  b
       h_v^(t) = GRU(a(v,t), h_v^(t-1))
   where A_v is the v-th row of the concatenated [A_in | A_out] matrix.
   Unlike SR-GNN's custom GRU, here we use a proper nn.GRUCell so the
   recurrence is strictly GRU(input = neighbourhood aggregation, hidden = h_v).
3. Readout strategy: soft-attention hybrid pooling, identical to SR-GNN's
   original readout (local last-item + global attention context).  This is the
   standard choice from the session-graph literature.


Input schema (same parquet columns as SR-GNN)
---------------------------------------------
x, edge_index, alias_inputs, item_seq_len, pos_items
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from recsys.models.graph_helpers import (
    VARIANT_SRGNN,
    GraphRecommenderBase,
    SessionEncoderBase,
    SessionGraphDataset,
    build_adjacency,
)

__all__ = ["GGNNRecommender"]


# ---------------------------------------------------------------------------
# GGNN propagation cell
# ---------------------------------------------------------------------------


class _GGNNPropagation(nn.Module):
    """One step of GGNN propagation for a batched session graph.

    Input  : hidden  (B, n_nodes, D)
             adjacency (B, n_nodes, 2*n_nodes)   [a_in | a_out]
    Output : hidden  (B, n_nodes, D)   after one GRU step

    The neighbourhood aggregation is:
        a_in_msg  = A_in  @ (W_in  · hidden)
        a_out_msg = A_out @ (W_out · hidden)
        a(v)      = W_agg( [a_in_msg | a_out_msg] )  + b_agg
    then the GRUCell is applied node-wise:
        h(v) = GRU( a(v), h_prev(v) )
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Neighbourhood projection for incoming / outgoing edges
        self.W_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, hidden_size, bias=False)
        # Aggregation projection: combines in + out → GRU input dimension
        self.W_agg = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        # Standard GRU cell (input_size = hidden_size, hidden_size = hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        B, n_nodes, D = hidden.shape
        a_in = adjacency[:, :, :n_nodes]  # (B, n, n)
        a_out = adjacency[:, :, n_nodes:]  # (B, n, n)

        msg_in = torch.bmm(a_in, self.W_in(hidden))  # (B, n, D)
        msg_out = torch.bmm(a_out, self.W_out(hidden))  # (B, n, D)
        a_v = self.W_agg(torch.cat([msg_in, msg_out], dim=-1))  # (B, n, D)

        # Apply GRUCell node-wise: reshape (B, n, D) → (B*n, D)
        a_flat = a_v.reshape(B * n_nodes, D)
        h_flat = hidden.reshape(B * n_nodes, D)
        h_new = self.gru(a_flat, h_flat)  # (B*n, D)
        return h_new.reshape(B, n_nodes, D)


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------


class _GGNNCore(SessionEncoderBase):
    """GGNN encoder: strict GRU propagation + attention-hybrid readout.

    Parameters
    ----------
    n_items:
        Total number of items (embedding table has n_items+1 rows; 0 = pad).
    embedding_dim:
        Item / hidden state dimension.
    step:
        Number of GGNN propagation steps (T in the paper).
    """

    def __init__(self, n_items: int, embedding_dim: int, step: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.step = step

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.propagation = _GGNNPropagation(embedding_dim)

        # Readout: same as SR-GNN's attention-hybrid pooling
        self.linear_one = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_two = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_three = nn.Linear(embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(embedding_dim * 2, embedding_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.embedding_dim)
        for p in self.parameters():
            p.data.uniform_(-bound, bound)

    def forward(
        self,
        items: torch.Tensor,
        alias_inputs: torch.Tensor,
        adjacency: torch.Tensor,
        seq_mask: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        # ── GGNN propagation ─────────────────────────────────────────────────
        hidden = self.item_embedding(items)  # (B, n_nodes, D)
        for _ in range(self.step):
            hidden = self.propagation(hidden, adjacency)

        # Gather in sequence order
        D = hidden.size(-1)
        gather_idx = alias_inputs.unsqueeze(-1).expand(-1, -1, D)
        seq_hidden = torch.gather(hidden, 1, gather_idx)  # (B, L, D)

        # Last-item representation
        lengths = seq_mask.sum(dim=1) - 1
        batch_i = torch.arange(hidden.size(0), device=hidden.device)
        ht = seq_hidden[batch_i, lengths]  # (B, D)

        # ── Attention-hybrid readout (same as SR-GNN) ────────────────────────
        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)
        alpha = alpha.masked_fill(~seq_mask, float("-inf"))
        alpha = torch.softmax(alpha, dim=1).unsqueeze(-1)
        l_ctx = (alpha * seq_hidden).sum(dim=1)

        return self.linear_transform(torch.cat([l_ctx, ht], dim=1))

    def compute_scores(self, session_rep: torch.Tensor) -> torch.Tensor:
        return session_rep @ self.item_embedding.weight.t()


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class GGNNRecommender(GraphRecommenderBase):
    """Gated Graph Neural Network recommender for session-based recommendation.

    API is identical to SRGNNRecommender / TAGNNRecommender so it can be used
    as a drop-in replacement inside Trainer / Evaluator / Pipeline.
    """

    MODEL_TYPE = "ggnn"

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        step: int = 1,
        max_session_length: int = 20,
        fallback_weight: float = 0.0,
        model_name: str | None = None,
        model_version: str = "0.1.0",
        seed: int = 42,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            step=step,
            max_session_length=max_session_length,
            fallback_weight=fallback_weight,
            model_name=model_name if model_name is not None else "ggnn",
            model_version=model_version,
            seed=seed,
            device=device,
        )

    # ------------------------------------------------------------------
    # GraphRecommenderBase hooks
    # ------------------------------------------------------------------

    def _build_core(self) -> _GGNNCore:
        return _GGNNCore(self.n_items, self.embedding_dim, self.step)

    def _make_dataset(self, df: pd.DataFrame) -> SessionGraphDataset:
        # GGNN uses the same 2n adjacency as base SR-GNN
        return SessionGraphDataset(df, variant=VARIANT_SRGNN)

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self._core is not None
        session_rep = self._core(
            items=batch["items"],
            alias_inputs=batch["alias_inputs"],
            adjacency=batch["adjacency"],
            seq_mask=batch["seq_mask"],
        )
        scores = self._core.compute_scores(session_rep)
        scores[:, 0] = float("-inf")
        return scores

    def _graph_from_sequence(
        self, encoded_items: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_nodes: list[int] = []
        node_to_idx: dict[int, int] = {}
        alias_list: list[int] = []
        edges_src: list[int] = []
        edges_dst: list[int] = []

        for item in encoded_items:
            if item not in node_to_idx:
                node_to_idx[item] = len(unique_nodes)
                unique_nodes.append(item)
            alias_list.append(node_to_idx[item])

        for i in range(len(alias_list) - 1):
            edges_src.append(alias_list[i])
            edges_dst.append(alias_list[i + 1])

        alias_arr = np.asarray(alias_list, dtype=np.int64)
        edge_index = (
            np.vstack(
                [
                    np.asarray(edges_src, dtype=np.int64),
                    np.asarray(edges_dst, dtype=np.int64),
                ]
            )
            if edges_src
            else np.empty((2, 0), dtype=np.int64)
        )
        adj_np = build_adjacency(alias_arr, edge_index, len(unique_nodes))

        return (
            torch.tensor(unique_nodes, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(alias_list, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(adj_np, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.ones((1, len(alias_list)), dtype=torch.bool, device=self.device),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _extra_metadata(self) -> dict[str, Any]:
        return {}

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device | None = None
    ) -> GGNNRecommender:
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        model, _ = cls._load_common(
            directory,
            extra_init_kwargs={"device": device},
        )
        return model  # type: ignore[return-value]
