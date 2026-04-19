"""Shared graph-building utilities for session-based recommender models.

All models in this package (SR-GNN variants, TAGNN, GGNN) share the same
parquet feature schema:

    x            – int64 array, unique node item-ids for the session graph
    edge_index   – object array of shape (2,) containing two int64 arrays
                   [src_indices, dst_indices]  (indices into x, not item ids)
    alias_inputs – int64 array, maps each sequence position → node index in x
    item_seq_len – int,  length of the original interaction sequence
    pos_items    – int,  target item id

Public surface
--------------
Array helpers:
    _to_int_array(value) -> np.ndarray
    _to_edge_index(value) -> np.ndarray   shape (2, n_edges)

Adjacency builders (return float32 np.ndarray):
    build_adjacency(alias, edge_index, n_nodes)           -> (n, 2n)
    build_adjacency_ngc(alias, edge_index, n, global_freq)-> (n, 2n)
    build_adjacency_fc(alias, edge_index, n_nodes)        -> (n, 4n)

Dataset / collation:
    SessionGraphDataset   – torch Dataset wrapping a parquet DataFrame
    collate_graph_batch   – collate_fn for variable-size graph batches

GNN building block:
    GNNCell               – SR-GNN gated propagation cell (variant-aware)

Base classes:
    GraphRecommenderBase  – shared fit / predict / save / load scaffold
    SessionEncoderBase    – nn.Module base for any session encoder
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from recsys.utils.device import resolve_torch_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant constants  (imported by all model modules)
# ---------------------------------------------------------------------------

VARIANT_SRGNN = "srgnn"
VARIANT_NGC   = "srgnn-ngc"
VARIANT_FC    = "srgnn-fc"
VARIANT_LOCAL = "srgnn-l"
VARIANT_AVG   = "srgnn-avg"
VARIANT_ATT   = "srgnn-att"

KNOWN_SRGNN_VARIANTS: frozenset[str] = frozenset(
    [VARIANT_SRGNN, VARIANT_NGC, VARIANT_FC, VARIANT_LOCAL, VARIANT_AVG, VARIANT_ATT]
)

# ---------------------------------------------------------------------------
# Low-level array helpers
# ---------------------------------------------------------------------------


def _to_int_array(value: Any) -> np.ndarray:
    """Coerce parquet object columns to a 1-D int64 numpy array."""
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return np.asarray(list(value), dtype=np.int64)
    if value is None:
        return np.asarray([], dtype=np.int64)
    raise TypeError(f"Unsupported array value: {type(value)!r}")


def _to_edge_index(value: Any) -> np.ndarray:
    """Coerce parquet edge storage to shape (2, n_edges) int64 array."""
    if isinstance(value, np.ndarray) and value.dtype != object:
        edge_index = value.astype(np.int64, copy=False)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        edge_index = np.vstack([_to_int_array(part) for part in value])
    elif isinstance(value, Sequence):
        edge_index = np.vstack([_to_int_array(part) for part in value])
    else:
        raise TypeError(f"Unsupported edge_index value: {type(value)!r}")

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, n_edges), got {edge_index.shape}"
        )
    return edge_index


# ---------------------------------------------------------------------------
# Adjacency builders
# ---------------------------------------------------------------------------


def build_adjacency(
    alias_inputs: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Standard SR-GNN adjacency: row-normalised in/out matrices → (n, 2n).

    Parameters
    ----------
    alias_inputs:
        Sequence of node indices (positions → graph node id).
    edge_index:
        Shape (2, n_edges); src/dst are node indices (into the local graph).
    n_nodes:
        Number of unique nodes in the session graph.

    Returns
    -------
    np.ndarray of shape (n_nodes, 2 * n_nodes), dtype float32.
    Columns 0..n-1 are the normalised *incoming* adjacency (a_in),
    columns n..2n-1 are the normalised *outgoing* adjacency (a_out).
    """
    a_in  = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    a_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        for u, v in zip(src.tolist(), dst.tolist(), strict=False):
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                a_out[u, v] += 1.0
                a_in[v, u]  += 1.0

    out_norm = a_out.sum(axis=1, keepdims=True)
    in_norm  = a_in.sum(axis=1, keepdims=True)
    a_out   /= np.where(out_norm == 0.0, 1.0, out_norm)
    a_in    /= np.where(in_norm  == 0.0, 1.0, in_norm)

    if alias_inputs.size:
        max_alias = int(alias_inputs.max())
        if max_alias >= n_nodes:
            raise ValueError(
                f"alias_inputs contain node index {max_alias} "
                f"with only {n_nodes} nodes"
            )

    return np.concatenate([a_in, a_out], axis=1)


def build_adjacency_ngc(
    alias_inputs: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
    global_freq: dict[tuple[int, int], float] | None,
    node_items: list[int] | None = None,
) -> np.ndarray:
    """SR-GNN-NGC: edge weights rescaled by global co-occurrence frequency.

    Parameters
    ----------
    alias_inputs, edge_index, n_nodes:
        Same as :func:`build_adjacency`.
    global_freq:
        Sparse dict mapping ``(src_item_id, dst_item_id) → count`` across the
        training corpus.  When ``None``, falls back to standard normalisation.
    node_items:
        List of global item IDs for each local node index (i.e. ``x`` array).
        Required when ``global_freq`` is provided so we can look up real item
        IDs rather than local node indices.

    Returns shape (n_nodes, 2 * n_nodes).
    """
    a_in  = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    a_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    if edge_index.size:
        # Pre-compute per-item outgoing totals once (O(|freq|) not O(edges × |freq|))
        row_totals: dict[int, float] = {}
        if global_freq is not None:
            for (s, _), cnt in global_freq.items():
                row_totals[s] = row_totals.get(s, 0.0) + cnt

        src, dst = edge_index[0], edge_index[1]
        for u, v in zip(src.tolist(), dst.tolist(), strict=False):
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                weight = 1.0
                if global_freq is not None and node_items is not None:
                    gi = node_items[u] if u < len(node_items) else 0
                    gj = node_items[v] if v < len(node_items) else 0
                    uv_count  = global_freq.get((gi, gj), 0.0)
                    total_out = row_totals.get(gi, 1.0) or 1.0
                    weight    = uv_count / total_out
                a_out[u, v] += weight
                a_in[v, u]  += weight

    out_norm = a_out.sum(axis=1, keepdims=True)
    in_norm  = a_in.sum(axis=1, keepdims=True)
    a_out   /= np.where(out_norm == 0.0, 1.0, out_norm)
    a_in    /= np.where(in_norm  == 0.0, 1.0, in_norm)

    return np.concatenate([a_in, a_out], axis=1)


def build_adjacency_fc(
    alias_inputs: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """SR-GNN-FC: local adjacency augmented with boolean full-connection matrix.

    Returns shape (n_nodes, 4 * n_nodes):
        [a_in | a_out | fc_in | fc_out]
    where fc_in/fc_out are row-normalised all-pairs connection matrices for
    items that co-appear in the session.
    """
    base = build_adjacency(alias_inputs, edge_index, n_nodes)   # (n, 2n)

    fc = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if alias_inputs.size >= 2:
        unique_nodes = np.unique(alias_inputs)
        for u in unique_nodes.tolist():
            for v in unique_nodes.tolist():
                if u != v and 0 <= u < n_nodes and 0 <= v < n_nodes:
                    fc[u, v] = 1.0

    fc_norm     = fc.sum(axis=1, keepdims=True)
    fc_normed   = fc / np.where(fc_norm == 0.0, 1.0, fc_norm)

    fc_in_norm  = fc.sum(axis=0, keepdims=True)
    fc_in       = (fc / np.where(fc_in_norm == 0.0, 1.0, fc_in_norm)).T

    return np.concatenate([base, fc_in, fc_normed], axis=1)


def build_adjacency_for_variant(
    alias_inputs: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
    variant: str,
    global_freq: dict[tuple[int, int], float] | None = None,
    node_items: list[int] | None = None,
) -> np.ndarray:
    """Dispatch to the correct adjacency builder based on SR-GNN variant."""
    if variant == VARIANT_NGC:
        return build_adjacency_ngc(
            alias_inputs, edge_index, n_nodes, global_freq, node_items
        )
    if variant == VARIANT_FC:
        return build_adjacency_fc(alias_inputs, edge_index, n_nodes)
    return build_adjacency(alias_inputs, edge_index, n_nodes)


# ---------------------------------------------------------------------------
# Dataset  (shared by SR-GNN variants; TAGNN / GGNN subclass or compose)
# ---------------------------------------------------------------------------


class SessionGraphDataset(Dataset):
    """Wraps a parquet examples DataFrame into a PyTorch Dataset.

    Each sample returns a dict of tensors suitable for
    :func:`collate_graph_batch`.  The ``variant`` parameter controls which
    adjacency builder is used at item-load time.

    Extra keyword arguments are forwarded to subclass ``_extra_fields`` hooks
    so that TAGNN / GGNN can inject additional tensors without replacing this
    class.
    """

    def __init__(
        self,
        examples: pd.DataFrame,
        variant: str = VARIANT_SRGNN,
        global_freq: dict[tuple[int, int], float] | None = None,
    ) -> None:
        if examples.empty:
            raise ValueError("Examples dataframe is empty")
        normalized = examples.reset_index(drop=True)
        self.variant = variant
        self.global_freq = global_freq
        # Store only required columns as array-like buffers to lower worker
        # memory overhead versus keeping a full DataFrame in each process.
        self._x_col = normalized["x"].to_numpy(copy=False)
        self._alias_inputs_col = normalized["alias_inputs"].to_numpy(copy=False)
        self._edge_index_col = normalized["edge_index"].to_numpy(copy=False)
        self._pos_items_col = normalized["pos_items"].to_numpy(copy=False)
        self._item_seq_len_col = normalized["item_seq_len"].to_numpy(copy=False)

    def __len__(self) -> int:
        return len(self._x_col)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = _to_int_array(self._x_col[index])
        alias_inputs = _to_int_array(self._alias_inputs_col[index])
        edge_index = _to_edge_index(self._edge_index_col[index])
        adjacency    = self._make_adjacency(x, alias_inputs, edge_index, len(x))

        sample: dict[str, torch.Tensor] = {
            "items":        torch.tensor(x,            dtype=torch.long),
            "alias_inputs": torch.tensor(alias_inputs, dtype=torch.long),
            "adjacency":    torch.tensor(adjacency,    dtype=torch.float32),
            "target":       torch.tensor(int(self._pos_items_col[index]), dtype=torch.long),
            "seq_len":      torch.tensor(int(self._item_seq_len_col[index]), dtype=torch.long),
        }
        sample.update(self._extra_fields(index, x, alias_inputs, edge_index))
        return sample

    def _make_adjacency(
        self,
        x: np.ndarray,
        alias_inputs: np.ndarray,
        edge_index: np.ndarray,
        n_nodes: int,
    ) -> np.ndarray:
        return build_adjacency_for_variant(
            alias_inputs, edge_index, n_nodes,
            self.variant, self.global_freq,
            node_items=x.tolist(),
        )

    def _extra_fields(
        self,
        index: int,
        x: np.ndarray,
        alias_inputs: np.ndarray,
        edge_index: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Hook for subclasses to inject additional tensor fields per sample."""
        return {}


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_graph_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad variable-size session graphs into a dense batch.

    Handles adjacency matrices of any width (2n for base/NGC, 4n for FC).
    Unknown extra keys (injected by subclass datasets) are stacked as-is when
    they are 0-D tensors, otherwise padded along dim-1 to max length.
    """
    batch_size  = len(batch)
    max_nodes   = max(int(e["items"].numel())        for e in batch)
    max_seq_len = max(int(e["alias_inputs"].numel()) for e in batch)

    # adj_width is expressed as a *multiple* of n_nodes — resolve against actual max
    first_adj        = batch[0]["adjacency"]
    adj_width_factor = first_adj.shape[-1] // batch[0]["items"].numel()
    adj_width        = max_nodes * adj_width_factor

    items        = torch.zeros((batch_size, max_nodes),           dtype=torch.long)
    alias_inputs = torch.zeros((batch_size, max_seq_len),         dtype=torch.long)
    adjacency    = torch.zeros((batch_size, max_nodes, adj_width),dtype=torch.float32)
    node_mask    = torch.zeros((batch_size, max_nodes),           dtype=torch.bool)
    seq_mask     = torch.zeros((batch_size, max_seq_len),         dtype=torch.bool)
    targets      = torch.zeros(batch_size,                        dtype=torch.long)
    seq_lens     = torch.zeros(batch_size,                        dtype=torch.long)

    for idx, example in enumerate(batch):
        n_nodes = int(example["items"].numel())
        seq_len = int(example["alias_inputs"].numel())
        adj_w   = example["adjacency"].shape[-1]

        items[idx, :n_nodes]              = example["items"]
        alias_inputs[idx, :seq_len]       = example["alias_inputs"]
        adjacency[idx, :n_nodes, :adj_w]  = example["adjacency"]
        node_mask[idx, :n_nodes]          = True
        seq_mask[idx, :seq_len]           = True
        targets[idx]                      = example["target"]
        seq_lens[idx]                     = example["seq_len"]

    result: dict[str, torch.Tensor] = {
        "items":        items,
        "alias_inputs": alias_inputs,
        "adjacency":    adjacency,
        "node_mask":    node_mask,
        "seq_mask":     seq_mask,
        "targets":      targets,
        "seq_lens":     seq_lens,
    }

    # Pass through any extra keys from subclass datasets
    extra_keys = [k for k in batch[0] if k not in result]
    for key in extra_keys:
        sample_val = batch[0][key]
        if sample_val.dim() == 0:
            result[key] = torch.stack([e[key] for e in batch])
        else:
            max_len = max(int(e[key].numel()) for e in batch)
            padded  = torch.zeros((batch_size, max_len), dtype=sample_val.dtype)
            for idx, example in enumerate(batch):
                length = int(example[key].numel())
                padded[idx, :length] = example[key]
            result[key] = padded

    return result


# ---------------------------------------------------------------------------
# GNN cell  (shared between SR-GNN variants; not used by TAGNN / GGNN)
# ---------------------------------------------------------------------------


class GNNCell(nn.Module):
    """SR-GNN gated-recurrent propagation cell.

    For SR-GNN-FC the adjacency has 4n columns instead of 2n, so the GRU input
    size doubles and extra linear projections are added for the full-connection
    sub-matrix.

    Parameters
    ----------
    hidden_size:
        Dimension of node hidden states.
    variant:
        SR-GNN variant string; only ``"srgnn-fc"`` changes the architecture.
    """

    def __init__(self, hidden_size: int, variant: str = VARIANT_SRGNN) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_fc      = (variant == VARIANT_FC)

        self.linear_edge_in  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)

        if self.use_fc:
            self.linear_fc_in  = nn.Linear(hidden_size, hidden_size, bias=True)
            self.linear_fc_out = nn.Linear(hidden_size, hidden_size, bias=True)
            gru_in = hidden_size * 4
        else:
            gru_in = hidden_size * 2

        self.w_ih = nn.Linear(gru_in,      hidden_size * 3, bias=True)
        self.w_hh = nn.Linear(hidden_size, hidden_size * 3, bias=True)

    def forward(
        self, hidden: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        n_nodes = hidden.size(1)

        a_in  = adjacency[:, :, :n_nodes]
        a_out = adjacency[:, :, n_nodes : 2 * n_nodes]

        input_in  = torch.matmul(a_in,  self.linear_edge_in(hidden))
        input_out = torch.matmul(a_out, self.linear_edge_out(hidden))

        if self.use_fc:
            fc_in  = adjacency[:, :, 2 * n_nodes : 3 * n_nodes]
            fc_out = adjacency[:, :, 3 * n_nodes : 4 * n_nodes]
            input_fc_in  = torch.matmul(fc_in,  self.linear_fc_in(hidden))
            input_fc_out = torch.matmul(fc_out, self.linear_fc_out(hidden))
            inputs = torch.cat([input_in, input_out, input_fc_in, input_fc_out], dim=-1)
        else:
            inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.w_ih(inputs)
        gh = self.w_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate   = torch.tanh(i_n + reset_gate * h_n)
        return new_gate + input_gate * (hidden - new_gate)


# ---------------------------------------------------------------------------
# Base nn.Module for session encoders
# ---------------------------------------------------------------------------


class SessionEncoderBase(nn.Module):
    """Common interface every session-graph encoder must implement.

    Subclasses override :meth:`forward` to produce a session representation
    of shape (batch, embedding_dim) and may optionally override
    :meth:`compute_scores` (defaults to inner product with item embeddings).
    """

    def forward(
        self,
        items: torch.Tensor,
        alias_inputs: torch.Tensor,
        adjacency: torch.Tensor,
        seq_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_scores(self, session_rep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GraphRecommenderBase  – shared scaffold for all recommender classes
# ---------------------------------------------------------------------------


class GraphRecommenderBase:
    """Shared fit / recommend / score / save / load logic for graph recommenders.

    Concrete model classes (SRGNNRecommender, TAGNNRecommender, GGNNRecommender)
    inherit from this and implement:

        _build_core()  → SessionEncoderBase   (builds the nn.Module)
        _make_dataset(df) → SessionGraphDataset  (wraps data + variant)
        _graph_from_sequence(encoded_items) → tuple of tensors  (for live inference)
        _forward_batch(batch) → scores tensor  (model-specific forward pass)
        _extra_save_state(directory)  (optional – save extra artefacts, e.g. global_freq)
        _extra_load_state(directory, meta)  (optional – restore them)
        _extra_metadata() → dict  (optional – extra keys for model.json)
    """

    # subclasses set these
    MODEL_TYPE: str = "base"

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
        if hidden_size != embedding_dim:
            logger.warning(
                "hidden_size=%d != embedding_dim=%d; using embedding_dim",
                hidden_size, embedding_dim,
            )
        self.embedding_dim      = embedding_dim
        self.step               = step
        self.max_session_length = max_session_length
        self.fallback_weight    = fallback_weight
        self.model_name         = model_name if model_name is not None else self.MODEL_TYPE
        self.model_version      = model_version
        self.seed               = seed

        self.device = resolve_torch_device(device)
        self._non_blocking_transfer = False
        self.n_items: int                 = 0
        self._core: SessionEncoderBase | None = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._popularity: dict[int, int]  = {}

    # ------------------------------------------------------------------
    # Abstract hooks – subclasses implement these
    # ------------------------------------------------------------------

    def _build_core(self) -> SessionEncoderBase:
        raise NotImplementedError

    def _make_dataset(
        self, df: pd.DataFrame
    ) -> SessionGraphDataset:
        """Return a SessionGraphDataset (or subclass) for the given split."""
        raise NotImplementedError

    def _graph_from_sequence(
        self, encoded_items: Sequence[int]
    ) -> tuple[torch.Tensor, ...]:
        """Build graph tensors from a live item sequence for single-session inference."""
        raise NotImplementedError

    def _forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run one forward pass and return logit scores (B, n_items+1)."""
        raise NotImplementedError

    def _extra_save_state(self, directory: Path) -> None:
        """Persist any extra artefacts needed beyond model.pt / model.json."""

    def _extra_load_state(self, directory: Path, meta: dict[str, Any]) -> None:
        """Restore extra artefacts after weights are loaded."""

    def _extra_metadata(self) -> dict[str, Any]:
        """Extra key/value pairs to include in model.json."""
        return {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        num_epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_df: pd.DataFrame | None = None,
        early_stopping_patience: int = 3,
        item_vocab: Mapping[str, Any] | None = None,
        num_workers: int = 0,
        pin_memory: bool | str | None = None,
        persistent_workers: bool | str | None = None,
        prefetch_factor: int | str | None = None,
    ) -> "GraphRecommenderBase":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_df = _normalize_examples(train_df)
        val_df   = (
            _normalize_examples(val_df)
            if val_df is not None and not val_df.empty
            else None
        )

        _initialize_vocab(self, train_df, item_vocab)

        self._core = self._build_core().to(self.device)
        num_workers = max(int(num_workers), 0)
        resolved_pin_memory = _resolve_optional_bool(
            pin_memory,
            key="pin_memory",
        )
        resolved_persistent_workers = _resolve_optional_bool(
            persistent_workers,
            key="persistent_workers",
        )
        resolved_prefetch_factor = _resolve_optional_int(
            prefetch_factor,
            key="prefetch_factor",
        )
        pin_memory_enabled = (
            resolved_pin_memory if resolved_pin_memory is not None else self.device.type == "cuda"
        )
        persistent_workers_enabled = (
            resolved_persistent_workers
            if resolved_persistent_workers is not None
            else num_workers > 0
        )
        self._non_blocking_transfer = bool(
            pin_memory_enabled and self.device.type == "cuda"
        )

        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_graph_batch,
            "pin_memory": pin_memory_enabled,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers_enabled
            if resolved_prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
        elif resolved_prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor requires num_workers > 0."
            )

        train_loader = DataLoader(
            self._make_dataset(train_df),
            shuffle=True,
            **loader_kwargs,
        )

        val_loader: DataLoader | None = None
        if val_df is not None and not val_df.empty:
            val_loader = DataLoader(
                self._make_dataset(val_df),
                shuffle=False,
                **loader_kwargs,
            )

        optimizer = torch.optim.Adam(
            self._core.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer, criterion)
            log_msg    = (
                f"[{self.model_name}] epoch={epoch}/{num_epochs} "
                f"train_loss={train_loss:.4f}"
            )

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                log_msg += f" val_loss={val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self._core.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("%s early_stop=true", log_msg)
                        break

            logger.info(log_msg)

        if best_state is not None and self._core is not None:
            self._core.load_state_dict(best_state)

        return self

    # ------------------------------------------------------------------
    # Epoch helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        assert self._core is not None
        self._core.train()
        total_loss, total_n = 0.0, 0

        for batch in loader:
            batch  = _move_batch(
                batch,
                self.device,
                non_blocking=self._non_blocking_transfer,
            )
            scores = self._forward_batch(batch)
            loss   = criterion(scores, batch["targets"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=5.0)
            optimizer.step()
            n = int(batch["targets"].size(0))
            total_loss += loss.item() * n
            total_n    += n

        return total_loss / max(total_n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self._core is not None
        self._core.eval()
        total_loss, total_n = 0.0, 0

        for batch in loader:
            batch  = _move_batch(
                batch,
                self.device,
                non_blocking=self._non_blocking_transfer,
            )
            scores = self._forward_batch(batch)
            loss   = criterion(scores, batch["targets"])
            n = int(batch["targets"].size(0))
            total_loss += loss.item() * n
            total_n    += n

        return total_loss / max(total_n, 1)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self, item_sequence: Sequence[int], top_k: int = 10
    ) -> list[int]:
        """Recommend ``top_k`` items given a live session sequence."""
        encoded = _encode_external_items(self, item_sequence)
        if not encoded or self._core is None:
            return _decode_internal_items(self, _popularity_top_indices(self, top_k))

        self._core.eval()
        tensors     = self._graph_from_sequence(encoded)
        session_rep = self._core(*tensors)
        scores      = self._core.compute_scores(session_rep).squeeze(0)
        scores[0]   = float("-inf")

        if self.fallback_weight > 0:
            pop    = torch.tensor(
                _popularity_distribution(self), dtype=torch.float32, device=self.device
            )
            scores = (1.0 - self.fallback_weight) * scores + self.fallback_weight * pop

        top_idx = torch.topk(scores, k=min(top_k, self.n_items)).indices.cpu().tolist()
        return _decode_internal_items(self, top_idx)

    @torch.no_grad()
    def recommend_from_graph(
        self,
        x: Sequence[int],
        edge_index: Any,
        alias_inputs: Sequence[int],
        top_k: int = 10,
    ) -> list[int]:
        """Recommend using pre-built graph tensors (used by Evaluator)."""
        if self._core is None:
            return []
        tensors     = self._tensors_from_graph(x, edge_index, alias_inputs)
        self._core.eval()
        session_rep = self._core(*tensors)
        scores      = self._core.compute_scores(session_rep).squeeze(0)
        scores[0]   = float("-inf")
        top_idx = torch.topk(scores, k=min(top_k, self.n_items)).indices.cpu().tolist()
        return top_idx

    @torch.no_grad()
    def score(
        self, item_sequence: Sequence[int], item_ids: Sequence[int]
    ) -> list[float]:
        encoded = _encode_external_items(self, item_sequence)
        if not encoded or self._core is None:
            return [0.0] * len(item_ids)

        self._core.eval()
        tensors     = self._graph_from_sequence(encoded)
        session_rep = self._core(*tensors)
        all_scores  = self._core.compute_scores(session_rep).squeeze(0)

        result: list[float] = []
        for item in item_ids:
            item_int = int(item)
            enc = self._item_to_idx.get(
                item_int, item_int if 1 <= item_int <= self.n_items else 0
            )
            result.append(float(all_scores[enc].item()) if enc else 0.0)
        return result

    # ------------------------------------------------------------------
    # Graph tensor construction for recommend_from_graph
    # ------------------------------------------------------------------

    def _tensors_from_graph(
        self,
        x: Sequence[int],
        edge_index: Any,
        alias_inputs: Sequence[int],
    ) -> tuple[torch.Tensor, ...]:
        """Build inference tensors from raw graph arrays (default implementation).

        Subclasses override this when their forward() signature differs.
        """
        x_arr     = _to_int_array(x)
        alias_arr = _to_int_array(alias_inputs)
        ei        = _to_edge_index(edge_index)
        adj_np    = build_adjacency(alias_arr, ei, len(x_arr))

        items_t    = torch.tensor(x_arr,     dtype=torch.long,    device=self.device).unsqueeze(0)
        alias_t    = torch.tensor(alias_arr, dtype=torch.long,    device=self.device).unsqueeze(0)
        adj_t      = torch.tensor(adj_np,    dtype=torch.float32, device=self.device).unsqueeze(0)
        seq_mask_t = torch.ones((1, alias_t.size(1)), dtype=torch.bool, device=self.device)
        return items_t, alias_t, adj_t, seq_mask_t

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._core is None:
            raise ValueError("Cannot save an unfitted model")

        weights_path  = directory / "model.pt"
        metadata_path = directory / "model.json"

        torch.save(self._core.state_dict(), weights_path)

        metadata: dict[str, Any] = {
            "model_type":         self.MODEL_TYPE,
            "model_name":         self.model_name,
            "model_version":      self.model_version,
            "embedding_dim":      self.embedding_dim,
            "step":               self.step,
            "max_session_length": self.max_session_length,
            "fallback_weight":    self.fallback_weight,
            "seed":               self.seed,
            "n_items":            self.n_items,
            "item_to_idx":        {str(k): v for k, v in self._item_to_idx.items()},
            "idx_to_item":        {str(k): v for k, v in self._idx_to_item.items()},
            "popularity":         {str(k): v for k, v in self._popularity.items()},
            **self._extra_metadata(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._extra_save_state(directory)
        return weights_path

    @classmethod
    def _load_common(
        cls,
        directory: Path,
        extra_init_kwargs: dict[str, Any] | None = None,
    ) -> tuple["GraphRecommenderBase", dict[str, Any]]:
        """Load weights + metadata; return (initialised model, meta dict).

        Parameters
        ----------
        extra_init_kwargs:
            Additional keyword arguments forwarded to ``cls.__init__`` beyond
            the standard fields stored in model.json.  Use this to pass
            subclass-specific constructor params (e.g. ``variant`` for SR-GNN)
            that must be set *before* ``_build_core()`` is called.
        """
        weights_path  = directory / "model.pt"
        metadata_path = directory / "model.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        init_kwargs: dict[str, Any] = dict(
            embedding_dim      = int(meta["embedding_dim"]),
            hidden_size        = int(meta["embedding_dim"]),
            step               = int(meta["step"]),
            max_session_length = int(meta["max_session_length"]),
            fallback_weight    = float(meta["fallback_weight"]),
            model_name         = str(meta["model_name"]),
            model_version      = str(meta["model_version"]),
            seed               = int(meta.get("seed", 42)),
        )
        if extra_init_kwargs:
            init_kwargs.update(extra_init_kwargs)
        model = cls(**init_kwargs)   # type: ignore[call-arg]
        model.n_items      = int(meta["n_items"])
        model._item_to_idx = {int(k): int(v) for k, v in meta["item_to_idx"].items()}
        model._idx_to_item = {int(k): int(v) for k, v in meta["idx_to_item"].items()}
        model._popularity  = {int(k): int(v) for k, v in meta["popularity"].items()}

        model._core = model._build_core().to(model.device)
        model._core.load_state_dict(
            torch.load(weights_path, map_location=model.device, weights_only=True)
        )
        model._core.eval()
        model._extra_load_state(directory, meta)
        return model, meta


# ---------------------------------------------------------------------------
# Module-level helpers (shared stateless utilities used by GraphRecommenderBase)
# ---------------------------------------------------------------------------


def _normalize_examples(examples: pd.DataFrame | None) -> pd.DataFrame:
    if examples is None:
        return pd.DataFrame()
    required = {"x", "edge_index", "alias_inputs", "item_seq_len", "pos_items"}
    missing  = required.difference(examples.columns)
    if missing:
        raise ValueError(f"Missing required example columns: {sorted(missing)}")
    out                  = examples.copy()
    out["item_seq_len"]  = out["item_seq_len"].astype(int)
    out["pos_items"]     = out["pos_items"].astype(int)
    return out


def _initialize_vocab(
    model: GraphRecommenderBase,
    train_df: pd.DataFrame,
    item_vocab: Mapping[str, Any] | None,
) -> None:
    if item_vocab and "item2id" in item_vocab:
        raw                 = {int(k): int(v) for k, v in item_vocab["item2id"].items()}
        model._item_to_idx  = raw
        model._idx_to_item  = {idx: item for item, idx in raw.items()}
        model.n_items       = max(model._idx_to_item) if model._idx_to_item else 0
    else:
        max_item = 0
        for row in train_df.itertuples(index=False):
            row_items = _to_int_array(row.x)
            if row_items.size:
                max_item = max(max_item, int(row_items.max()))
            max_item = max(max_item, int(row.pos_items))
        model.n_items       = max_item
        model._idx_to_item  = {i: i for i in range(1, model.n_items + 1)}
        model._item_to_idx  = dict(model._idx_to_item)

    pop: Counter[int] = Counter()
    for row in train_df.itertuples(index=False):
        pop.update(_to_int_array(row.x).tolist())
        pop.update([int(row.pos_items)])
    model._popularity = dict(pop)


def _encode_external_items(
    model: GraphRecommenderBase, item_sequence: Sequence[int]
) -> list[int]:
    encoded: list[int] = []
    for item in item_sequence[-model.max_session_length :]:
        item_int = int(item)
        if item_int in model._item_to_idx:
            encoded.append(model._item_to_idx[item_int])
        elif 1 <= item_int <= model.n_items:
            encoded.append(item_int)
    return encoded


def _decode_internal_items(
    model: GraphRecommenderBase, item_ids: Sequence[int]
) -> list[int]:
    return [model._idx_to_item.get(int(i), int(i)) for i in item_ids]


def _popularity_top_indices(model: GraphRecommenderBase, top_k: int) -> list[int]:
    return [
        item
        for item, _ in sorted(
            model._popularity.items(), key=lambda p: p[1], reverse=True
        )[:top_k]
    ]


def _popularity_distribution(model: GraphRecommenderBase) -> list[float]:
    values = [0.0] * (model.n_items + 1)
    total  = float(sum(model._popularity.values()) or 1)
    for item, count in model._popularity.items():
        if 0 <= item <= model.n_items:
            values[item] = count / total
    return values


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def _resolve_optional_bool(value: bool | str | None, *, key: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"Invalid {key} value '{value}'. Use true, false, or auto."
    )


def _resolve_optional_int(value: int | str | None, *, key: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return resolved


def _build_global_freq(
    model: GraphRecommenderBase, train_df: pd.DataFrame
) -> dict[tuple[int, int], float]:
    """Build a sparse co-occurrence frequency dict for the NGC variant.

    Returns
    -------
    dict mapping (src_item_id, dst_item_id) -> raw count.

    Rationale for sparse representation
    ------------------------------------
    A dense (n_items+1)^2 float32 matrix is O(n^2) memory — for a 50 k item
    catalogue this is ~10 GB, both too large to keep in RAM and unserializable
    in a reasonable time.  In practice the transition graph is extremely sparse
    (each session has at most max_session_length - 1 edges), so the dict stays
    small even for large catalogues.
    """
    size = model.n_items + 1
    freq: dict[tuple[int, int], float] = {}

    for row in train_df.itertuples(index=False):
        alias = _to_int_array(row.alias_inputs)
        x     = _to_int_array(row.x)
        seq   = [int(x[a]) if a < len(x) else 0 for a in alias]
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if 0 < u < size and 0 < v < size:
                key      = (u, v)
                freq[key] = freq.get(key, 0.0) + 1.0

    return freq
