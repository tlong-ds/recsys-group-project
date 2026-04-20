"""SR-GNN recommender – refactored to use graph_helpers shared scaffold.

Variant taxonomy
----------------
srgnn      : original SR-GNN (local last-item + attention readout)
srgnn-ngc  : global edge weights from corpus co-occurrence (NGC)
srgnn-fc   : local graph augmented with boolean full-connection matrix (FC)
srgnn-l    : local embedding only (last item, no attention readout)
srgnn-avg  : global embedding via average pooling over sequence positions
srgnn-att  : global embedding via attention only (no last-item branch)

Public API matches the original exactly so Trainer / Evaluator / Pipeline
require zero changes.
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
    KNOWN_SRGNN_VARIANTS,
    VARIANT_AVG,
    VARIANT_LOCAL,
    VARIANT_NGC,
    VARIANT_SRGNN,
    GNNCell,
    GraphRecommenderBase,
    SessionEncoderBase,
    SessionGraphDataset,
    _build_global_freq,
    _to_edge_index,
    _to_int_array,
    build_adjacency_for_variant,
)

__all__ = ["SRGNNRecommender"]


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------


class _SRGNNCore(SessionEncoderBase):
    """Batched SR-GNN encoder with pluggable readout strategy.

    All six variants share this single module; the ``variant`` flag selects
    which readout path is taken in ``forward()``.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        step: int,
        variant: str = VARIANT_SRGNN,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.step = step
        self.variant = variant

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.gnn = GNNCell(embedding_dim, variant=variant)

        # Readout projections
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
        # ── GNN propagation ──────────────────────────────────────────────────
        hidden = self.item_embedding(items)
        for _ in range(self.step):
            hidden = self.gnn(hidden, adjacency)

        # Gather in sequence order
        dim = hidden.size(-1)
        gather_idx = alias_inputs.unsqueeze(-1).expand(-1, -1, dim)
        seq_hidden = torch.gather(hidden, 1, gather_idx)

        # Last-item embedding
        lengths = seq_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        ht = seq_hidden[batch_idx, lengths]  # (B, D)

        # ── Readout branch ───────────────────────────────────────────────────
        if self.variant == VARIANT_LOCAL:
            return ht

        if self.variant == VARIANT_AVG:
            mask_f = seq_mask.unsqueeze(-1).float()
            len_f = seq_mask.sum(dim=1, keepdim=True).float().unsqueeze(-1)
            g_ctx = (seq_hidden * mask_f).sum(dim=1) / len_f.squeeze(-1)
            return self.linear_transform(torch.cat([g_ctx, ht], dim=1))

        # srgnn / srgnn-ngc / srgnn-fc / srgnn-att all use attention readout
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


class SRGNNRecommender(GraphRecommenderBase):
    """Session-based SR-GNN recommender supporting six readout variants.

    Parameters
    ----------
    variant:
        One of ``"srgnn"`` (default), ``"srgnn-ngc"``, ``"srgnn-fc"``,
        ``"srgnn-l"``, ``"srgnn-avg"``, ``"srgnn-att"``.
    model_name:
        Artifact name used for registration.  Defaults to ``variant`` so each
        experiment is registered under a distinct name automatically.
    """

    MODEL_TYPE = "srgnn"

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
        variant: str = VARIANT_SRGNN,
        device: str | torch.device | None = None,
    ) -> None:
        if variant not in KNOWN_SRGNN_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose one of: {sorted(KNOWN_SRGNN_VARIANTS)}"
            )
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            step=step,
            max_session_length=max_session_length,
            fallback_weight=fallback_weight,
            model_name=model_name if model_name is not None else variant,
            model_version=model_version,
            seed=seed,
            device=device,
        )
        self.variant = variant
        self._global_freq: dict[tuple[int, int], float] | None = None

    # ------------------------------------------------------------------
    # GraphRecommenderBase hooks
    # ------------------------------------------------------------------

    def _build_core(self) -> _SRGNNCore:
        return _SRGNNCore(self.n_items, self.embedding_dim, self.step, self.variant)

    def _make_dataset(self, df: pd.DataFrame) -> SessionGraphDataset:
        return SessionGraphDataset(
            df, variant=self.variant, global_freq=self._global_freq
        )

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._core is None:
            raise RuntimeError("Model core is not initialized")
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
        n = len(unique_nodes)

        local_freq = (
            self._localise_global_freq(unique_nodes)
            if self.variant == VARIANT_NGC
            else None
        )
        adj_np = build_adjacency_for_variant(
            alias_arr,
            edge_index,
            n,
            self.variant,
            local_freq,
            node_items=unique_nodes,
        )

        return (
            torch.tensor(unique_nodes, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(alias_list, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(adj_np, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.ones((1, len(alias_list)), dtype=torch.bool, device=self.device),
        )

    def _tensors_from_graph(
        self,
        x: Sequence[int],
        edge_index: Any,
        alias_inputs: Sequence[int],
    ) -> tuple[torch.Tensor, ...]:
        x_arr = _to_int_array(x)
        alias_arr = _to_int_array(alias_inputs)
        ei = _to_edge_index(edge_index)
        n = len(x_arr)

        if self.variant == VARIANT_NGC:
            local_freq = self._localise_global_freq(x_arr.tolist())
        else:
            local_freq = None

        adj_np = build_adjacency_for_variant(
            alias_arr,
            ei,
            n,
            self.variant,
            local_freq,
            node_items=x_arr.tolist(),
        )
        items_t = torch.tensor(x_arr, dtype=torch.long, device=self.device).unsqueeze(0)
        alias_t = torch.tensor(
            alias_arr, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        adj_t = torch.tensor(adj_np, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        seq_mask_t = torch.ones(
            (1, alias_t.size(1)), dtype=torch.bool, device=self.device
        )
        return items_t, alias_t, adj_t, seq_mask_t

    # ------------------------------------------------------------------
    # NGC helpers
    # ------------------------------------------------------------------

    def _localise_global_freq(
        self, global_item_ids: list[int]
    ) -> dict[tuple[int, int], float] | None:
        """Subset the global sparse freq dict to only the local node item ids.

        Returns a new sparse dict with the same (u, v) pairs but remapped so
        the caller can pass it directly to build_adjacency_ngc together with
        node_items=global_item_ids.
        """
        if self._global_freq is None:
            return None
        # Pass through as-is; build_adjacency_ngc uses node_items for id lookup
        return self._global_freq

    # ------------------------------------------------------------------
    # Fit override – build global_freq before core is instantiated
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> SRGNNRecommender:
        # GraphRecommenderBase.fit handles vocab init; we need global_freq *after*
        # vocab is ready so we patch it in via a pre-hook inside the base class flow.
        # Solution: temporarily override _build_core to build freq first.
        _original_build_core = self._build_core

        def _patched_build_core() -> _SRGNNCore:
            if self.variant == VARIANT_NGC:
                self._global_freq = _build_global_freq(self, train_df)
            return _original_build_core()

        self._build_core = _patched_build_core  # type: ignore[method-assign]
        result = super().fit(train_df, **kwargs)
        self._build_core = _original_build_core  # type: ignore[method-assign]
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serialisation hooks
    # ------------------------------------------------------------------

    def _extra_metadata(self) -> dict[str, Any]:
        return {"variant": self.variant}

    def _extra_save_state(self, directory: Path) -> None:
        if self.variant == VARIANT_NGC and self._global_freq is not None:
            # Serialise sparse dict as JSON: keys are "u,v" strings
            import json as _json

            serialised = {f"{u},{v}": cnt for (u, v), cnt in self._global_freq.items()}
            (directory / "global_freq.json").write_text(
                _json.dumps(serialised), encoding="utf-8"
            )

    def _extra_load_state(self, directory: Path, meta: dict[str, Any]) -> None:
        freq_path = directory / "global_freq.json"
        if freq_path.exists():
            import json as _json

            raw = _json.loads(freq_path.read_text(encoding="utf-8"))
            self._global_freq = {
                (int(k.split(",")[0]), int(k.split(",")[1])): float(v)
                for k, v in raw.items()
            }

    # ------------------------------------------------------------------
    # Load classmethod
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device | None = None
    ) -> SRGNNRecommender:
        import json as _json

        path = Path(path)
        directory = path if path.is_dir() else path.parent
        meta_text = (directory / "model.json").read_text(encoding="utf-8")
        meta = _json.loads(meta_text)
        variant = str(meta.get("variant", VARIANT_SRGNN))
        model, _ = cls._load_common(
            directory,
            extra_init_kwargs={"variant": variant, "device": device},
        )
        return model  # type: ignore[return-value]
