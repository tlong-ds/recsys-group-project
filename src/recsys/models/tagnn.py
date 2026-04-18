"""TAGNN – Target Attentive Graph Neural Network for session-based recommendation.

Reference
---------
Yu Zheng, Chang Liu, et al.  "TAGNN: Target Attentive Graph Neural Networks
for Session-based Recommendation."  SIGIR 2020.
https://github.com/CRIPAC-DIG/TAGNN

Architecture overview
---------------------
1. Build the session directed graph (same schema as SR-GNN):
   nodes = unique items, edges = transitions, adjacency = normalised in/out.
2. Run T steps of SR-GNN-style gated GNN propagation.
3. **Target-aware attention readout** – the key TAGNN contribution:
   For each candidate target t compute a target-specific session embedding

       alpha_i(t) = sigmoid( (W_t v_i)^T e_t )  for each sequence position i
       h_s(t)     = sum_i alpha_i(t) * v_i        (normalised)
       s(t)       = W_out( [h_s(t) || h_T] )      (h_T = last-item hidden state)

   The final score for item t is s(t) · e_t.

Memory-efficient scoring
------------------------
A naive fully-batched approach materialises a (B, L, n_items) tensor.
For n_items ≈ 50 k, batch 256, L 20 this is > 4 GB — infeasible.

We instead use **chunked scoring**: iterate over the catalogue in slices of
``score_chunk_size`` items so the peak tensor is (B, L, chunk_size).
With chunk_size=512 the footprint is ~100x smaller while producing
mathematically identical scores.

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
    GraphRecommenderBase,
    GNNCell,
    SessionEncoderBase,
    SessionGraphDataset,
    VARIANT_SRGNN,
    build_adjacency,
)

__all__ = ["TAGNNRecommender"]

# Default chunk size for catalogue scoring – tune down to save VRAM
DEFAULT_SCORE_CHUNK = 512


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------


class _TAGNNCore(SessionEncoderBase):
    """TAGNN encoder: SR-GNN propagation + chunked target-attentive readout.

    Parameters
    ----------
    n_items : int
        Catalogue size (embedding table has n_items+1 rows; index 0 = padding).
    embedding_dim : int
        Item / hidden-state dimension.
    step : int
        Number of GNN propagation steps (T in the paper).
    score_chunk_size : int
        Items scored per chunk in ``compute_scores``.  No effect on quality.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        step: int,
        score_chunk_size: int = DEFAULT_SCORE_CHUNK,
    ) -> None:
        super().__init__()
        self.embedding_dim    = embedding_dim
        self.step             = step
        self.score_chunk_size = score_chunk_size

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.gnn            = GNNCell(embedding_dim, variant=VARIANT_SRGNN)

        # W_t: projects each node hidden state for the target-attention dot product
        self.W_t              = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Merge target-pooled context with last-item embedding
        self.linear_transform = nn.Linear(embedding_dim * 2, embedding_dim, bias=True)

        self._reset_parameters()

        # Runtime cache populated by forward(); consumed by compute_scores()
        self._seq_hidden: torch.Tensor | None = None
        self._seq_mask:   torch.Tensor | None = None
        self._ht:         torch.Tensor | None = None

    def _reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.embedding_dim)
        for p in self.parameters():
            p.data.uniform_(-bound, bound)

    # ------------------------------------------------------------------
    # GNN propagation
    # ------------------------------------------------------------------

    def forward(
        self,
        items:        torch.Tensor,   # (B, n_nodes)
        alias_inputs: torch.Tensor,   # (B, L)
        adjacency:    torch.Tensor,   # (B, n_nodes, 2*n_nodes)
        seq_mask:     torch.Tensor,   # (B, L)
        **_kwargs: Any,
    ) -> torch.Tensor:
        """GNN propagation + cache intermediate states for compute_scores.

        Returns ``ht`` (last-item hidden state, shape (B, D)) as a placeholder
        session representation.  The real per-target scores are computed lazily
        in :meth:`compute_scores`.
        """
        hidden = self.item_embedding(items)                             # (B, n, D)
        for _ in range(self.step):
            hidden = self.gnn(hidden, adjacency)

        D          = hidden.size(-1)
        gather_idx = alias_inputs.unsqueeze(-1).expand(-1, -1, D)
        seq_hidden = torch.gather(hidden, 1, gather_idx)               # (B, L, D)

        lengths = seq_mask.sum(dim=1) - 1
        batch_i = torch.arange(hidden.size(0), device=hidden.device)
        ht      = seq_hidden[batch_i, lengths]                         # (B, D)

        # Cache for compute_scores – avoids re-running the expensive GNN
        self._seq_hidden = seq_hidden
        self._seq_mask   = seq_mask
        self._ht         = ht

        return ht   # placeholder; actual logits come from compute_scores()

    # ------------------------------------------------------------------
    # Target-attentive session rep for a chunk of K candidates
    # ------------------------------------------------------------------

    def _session_rep_chunk(
        self,
        seq_hidden: torch.Tensor,   # (B, L, D)
        seq_mask:   torch.Tensor,   # (B, L)
        ht:         torch.Tensor,   # (B, D)
        E_chunk:    torch.Tensor,   # (K, D)  – embeddings of K candidate items
    ) -> torch.Tensor:
        """Compute target-specific session reps for K candidate items.

        alpha_i(t) = sigmoid( (W_t v_i)^T e_t )
        h_s(t)     = sum_i alpha_i(t) * v_i    (normalised)
        output     = W_out( [h_s(t) || h_T] )   shape (B, K, D)
        """
        B, L, D = seq_hidden.shape
        K       = E_chunk.size(0)

        # Project node hidden states: (B, L, D)
        Wt_seq = self.W_t(seq_hidden)

        # Attention logits for all K candidates: (B, L, K)
        # (B, L, D) x (D, K)  -> (B, L, K)
        raw = torch.matmul(Wt_seq, E_chunk.t())

        # Mask sequence padding, then sigmoid-normalise
        raw   = raw.masked_fill(~seq_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.sigmoid(raw)                                     # (B, L, K)
        alpha = alpha * seq_mask.unsqueeze(-1).float()
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-9)

        # Target-pooled context: (B, K, D)
        g_ctx = torch.einsum("blk,bld->bkd", alpha, seq_hidden)

        # Expand last-item rep: (B, K, D)
        ht_exp = ht.unsqueeze(1).expand(B, K, D)

        # Fuse and project: (B*K, 2D) -> linear -> (B, K, D)
        h_s = self.linear_transform(
            torch.cat([g_ctx, ht_exp], dim=-1).reshape(B * K, 2 * D)
        ).reshape(B, K, D)

        return h_s                                                      # (B, K, D)

    # ------------------------------------------------------------------
    # Chunked full-catalogue scoring
    # ------------------------------------------------------------------

    def compute_scores(self, _session_rep: torch.Tensor) -> torch.Tensor:
        """Score every item in the catalogue using chunked target-attentive reps.

        Peak memory is O(B × L × chunk_size) regardless of catalogue size.
        Results are identical to the fully-batched (B × L × n_items) approach.
        """
        assert self._seq_hidden is not None, "Call forward() before compute_scores()"

        seq_hidden = self._seq_hidden                   # (B, L, D)
        seq_mask   = self._seq_mask                     # (B, L)
        ht         = self._ht                           # (B, D)
        E          = self.item_embedding.weight         # (n+1, D)
        n          = E.size(0)
        B          = seq_hidden.size(0)

        logits = torch.empty(B, n, device=seq_hidden.device, dtype=seq_hidden.dtype)

        for start in range(0, n, self.score_chunk_size):
            end     = min(start + self.score_chunk_size, n)
            E_chunk = E[start:end]                                      # (K, D)

            # (B, K, D) * (1, K, D) -> sum over D -> (B, K)
            h_s = self._session_rep_chunk(seq_hidden, seq_mask, ht, E_chunk)
            logits[:, start:end] = (h_s * E_chunk.unsqueeze(0)).sum(dim=-1)

        return logits                                                    # (B, n+1)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class TAGNNRecommender(GraphRecommenderBase):
    """Target Attentive Graph Neural Network recommender.

    Implements the same public API as SRGNNRecommender / GGNNRecommender so
    it can be swapped into Trainer / Evaluator / Pipeline with zero changes.

    Parameters
    ----------
    score_chunk_size : int
        Number of candidate items processed per chunk during scoring.
        Reduce if you hit GPU / CPU OOM (default 512).  Does not affect
        model quality; only memory usage and throughput.
    """

    MODEL_TYPE = "tagnn"

    def __init__(
        self,
        embedding_dim:    int   = 128,
        hidden_size:      int   = 128,
        step:             int   = 1,
        max_session_length: int = 20,
        fallback_weight:  float = 0.0,
        model_name:       str | None = None,
        model_version:    str   = "0.1.0",
        seed:             int   = 42,
        score_chunk_size: int   = DEFAULT_SCORE_CHUNK,
    ) -> None:
        super().__init__(
            embedding_dim      = embedding_dim,
            hidden_size        = hidden_size,
            step               = step,
            max_session_length = max_session_length,
            fallback_weight    = fallback_weight,
            model_name         = model_name if model_name is not None else "tagnn",
            model_version      = model_version,
            seed               = seed,
        )
        self.score_chunk_size = score_chunk_size

    # ------------------------------------------------------------------
    # GraphRecommenderBase hooks
    # ------------------------------------------------------------------

    def _build_core(self) -> _TAGNNCore:
        return _TAGNNCore(
            n_items          = self.n_items,
            embedding_dim    = self.embedding_dim,
            step             = self.step,
            score_chunk_size = self.score_chunk_size,
        )

    def _make_dataset(self, df: pd.DataFrame) -> SessionGraphDataset:
        # TAGNN uses the same standard 2n adjacency as base SR-GNN
        return SessionGraphDataset(df, variant=VARIANT_SRGNN)

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self._core is not None
        # forward() caches (seq_hidden, seq_mask, ht) as a side-effect
        self._core(
            items        = batch["items"],
            alias_inputs = batch["alias_inputs"],
            adjacency    = batch["adjacency"],
            seq_mask     = batch["seq_mask"],
        )
        scores       = self._core.compute_scores(None)  # type: ignore[arg-type]
        scores[:, 0] = float("-inf")
        return scores

    def _graph_from_sequence(
        self, encoded_items: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_nodes: list[int]     = []
        node_to_idx: dict[int, int] = {}
        alias_list:  list[int]      = []
        edges_src:   list[int]      = []
        edges_dst:   list[int]      = []

        for item in encoded_items:
            if item not in node_to_idx:
                node_to_idx[item] = len(unique_nodes)
                unique_nodes.append(item)
            alias_list.append(node_to_idx[item])

        for i in range(len(alias_list) - 1):
            edges_src.append(alias_list[i])
            edges_dst.append(alias_list[i + 1])

        alias_arr  = np.asarray(alias_list, dtype=np.int64)
        edge_index = (
            np.vstack([
                np.asarray(edges_src, dtype=np.int64),
                np.asarray(edges_dst, dtype=np.int64),
            ])
            if edges_src
            else np.empty((2, 0), dtype=np.int64)
        )
        adj_np = build_adjacency(alias_arr, edge_index, len(unique_nodes))

        return (
            torch.tensor(unique_nodes, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(alias_list,   dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(adj_np,       dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.ones((1, len(alias_list)), dtype=torch.bool, device=self.device),
        )

    # recommend() and recommend_from_graph() are fully inherited from
    # GraphRecommenderBase.  Both call _core.forward() (caches state) then
    # _core.compute_scores() (runs the chunked loop) automatically.

    # ------------------------------------------------------------------
    # Serialisation hooks
    # ------------------------------------------------------------------

    def _extra_metadata(self) -> dict[str, Any]:
        return {"score_chunk_size": self.score_chunk_size}

    def _extra_load_state(self, directory: Path, meta: dict[str, Any]) -> None:
        self.score_chunk_size = int(meta.get("score_chunk_size", DEFAULT_SCORE_CHUNK))
        # Propagate into the already-loaded core
        if self._core is not None:
            self._core.score_chunk_size = self.score_chunk_size

    @classmethod
    def load(cls, path: str | Path) -> "TAGNNRecommender":
        path      = Path(path)
        directory = path if path.is_dir() else path.parent
        model, _  = cls._load_common(directory)
        return model   # type: ignore[return-value]