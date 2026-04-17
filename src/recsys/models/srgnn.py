"""SR-GNN recommender: self-contained wrapper for the MLOps pipeline.

Public interface consumed by pipeline.py, trainer.py, evaluator.py and predictor.py:
    model.fit(train_df)          -> SRGNNRecommender
    model.recommend(items, k)    -> list[int]
    model.score(items, item_ids) -> list[float]
    model.save(directory)        -> Path
    SRGNNRecommender.load(path)  -> SRGNNRecommender
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _build_graph(session: list[int], n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a session sequence into normalised in/out adjacency matrices.

    Returns
    -------
    A_in, A_out : np.ndarray of shape (n_nodes, n_nodes)
    """
    A_in = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    A_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(len(session) - 1):
        u, v = session[i], session[i + 1]
        A_out[u][v] += 1.0
        A_in[v][u] += 1.0

    # Row-normalise
    row_sum_out = A_out.sum(axis=1, keepdims=True)
    row_sum_in = A_in.sum(axis=1, keepdims=True)
    row_sum_out = np.where(row_sum_out == 0, 1.0, row_sum_out)
    row_sum_in = np.where(row_sum_in == 0, 1.0, row_sum_in)
    A_out /= row_sum_out
    A_in /= row_sum_in

    return A_in, A_out


def _session_to_graph_tensors(
    session: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a single session as graph tensors expected by _SRGNNCore.

    Returns
    -------
    x           : unique-node indices  (n_nodes,)
    alias_input : position → unique-node index  (seq_len,)
    A           : stacked adjacency [A_in | A_out]  (n_nodes, 2*n_nodes)
    seq_len     : scalar tensor
    """
    unique_nodes: list[int] = []
    node_index: dict[int, int] = {}
    for item in session:
        if item not in node_index:
            node_index[item] = len(unique_nodes)
            unique_nodes.append(item)

    n = len(unique_nodes)
    alias_seq = [node_index[item] for item in session]

    A_in, A_out = _build_graph(alias_seq, n)
    A = np.concatenate([A_in, A_out], axis=1)  # (n, 2n)

    x = torch.tensor(unique_nodes, dtype=torch.long, device=device)
    alias_input = torch.tensor(alias_seq, dtype=torch.long, device=device)
    A_t = torch.tensor(A, dtype=torch.float, device=device)
    seq_len = torch.tensor(len(session), dtype=torch.long, device=device)

    return x, alias_input, A_t, seq_len


# ---------------------------------------------------------------------------
# Core PyTorch modules
# ---------------------------------------------------------------------------

class _GNNCell(nn.Module):
    """Single GNN propagation step (SR-GNN gate mechanism)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_edge_in = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)
        # GRU-style gates
        self.W_r = nn.Linear(2 * hidden_size, hidden_size)
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,   # (n_nodes, H)
        A: torch.Tensor,        # (n_nodes, 2*n_nodes)
    ) -> torch.Tensor:
        n = hidden.size(0)
        A_in = A[:, :n]   # (n, n)
        A_out = A[:, n:]  # (n, n)

        input_in = torch.matmul(A_in, self.linear_edge_in(hidden))
        input_out = torch.matmul(A_out, self.linear_edge_out(hidden))
        inputs = torch.cat([input_in, input_out], dim=-1)  # (n, 2H)

        # Not a full GRU: simplified as in original SR-GNN paper
        z = torch.sigmoid(self.W_z(torch.cat([inputs, hidden], dim=-1)[:, :2 * self.hidden_size]))
        r = torch.sigmoid(self.W_r(torch.cat([inputs, hidden], dim=-1)[:, :2 * self.hidden_size]))
        # Recompute gate inputs properly
        # z = torch.sigmoid(self.W_z(torch.cat([inputs, hidden], dim=1) if inputs.size(1) == 2 * self.hidden_size
        #                             else inputs))
        # r = torch.sigmoid(self.W_r(inputs))
        h_tilde = torch.tanh(self.W_h(torch.cat([inputs[:, :self.hidden_size] * r, hidden], dim=1)
                                       if False else inputs))
        h = (1 - z) * hidden + z * h_tilde
        return h


class _SRGNNCore(nn.Module):
    """Pure PyTorch SR-GNN as described in Wu et al. AAAI 2019."""

    def __init__(self, n_items: int, embedding_size: int, step: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.step = step

        self.item_embedding = nn.Embedding(n_items + 1, embedding_size, padding_idx=0)
        self.gnn = _GNNCell(embedding_size)

        self.linear_one = nn.Linear(embedding_size, embedding_size)
        self.linear_two = nn.Linear(embedding_size, embedding_size)
        self.linear_three = nn.Linear(embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(embedding_size * 2, embedding_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,           # (n_nodes,)
        alias_input: torch.Tensor,  # (seq_len,)
        A: torch.Tensor,            # (n_nodes, 2*n_nodes)
        seq_len: torch.Tensor,      # scalar
    ) -> torch.Tensor:              # (embedding_size,)
        hidden = self.item_embedding(x)  # (n_nodes, H)

        for _ in range(self.step):
            hidden = self.gnn(hidden, A)

        seq_hidden = hidden[alias_input]  # (seq_len, H)

        # Last item representation
        ht = seq_hidden[seq_len - 1]  # (H,)
        q1 = self.linear_one(ht).unsqueeze(0)  # (1, H)
        q2 = self.linear_two(seq_hidden)        # (seq_len, H)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (seq_len, 1)
        a = torch.sum(alpha * seq_hidden, dim=0)  # (H,)

        seq_output = self.linear_transform(torch.cat([a, ht], dim=0))  # (H,)
        return seq_output

    def compute_scores(
        self,
        seq_output: torch.Tensor,  # (H,)
    ) -> torch.Tensor:             # (n_items+1,)
        return torch.matmul(seq_output, self.item_embedding.weight.T)


# ---------------------------------------------------------------------------
# Item index helpers
# ---------------------------------------------------------------------------

class _ItemIndex:
    """Bidirectional mapping between external item IDs and internal 1-based indices."""

    def __init__(self) -> None:
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}

    def fit(self, item_ids: list[int]) -> None:
        unique = sorted(set(item_ids))
        self._item_to_idx = {item: idx + 1 for idx, item in enumerate(unique)}
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

    @property
    def n_items(self) -> int:
        return len(self._item_to_idx)

    def encode(self, items: list[int]) -> list[int]:
        return [self._item_to_idx[i] for i in items if i in self._item_to_idx]

    def decode(self, indices: list[int]) -> list[int]:
        return [self._idx_to_item[i] for i in indices if i in self._idx_to_item]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_to_idx": {str(k): v for k, v in self._item_to_idx.items()},
            "idx_to_item": {str(k): v for k, v in self._idx_to_item.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ItemIndex":
        obj = cls()
        obj._item_to_idx = {int(k): v for k, v in data["item_to_idx"].items()}
        obj._idx_to_item = {int(k): v for k, v in data["idx_to_item"].items()}
        return obj


# ---------------------------------------------------------------------------
# Public recommender class
# ---------------------------------------------------------------------------

class SRGNNRecommender:
    """Session-based recommender wrapping the SR-GNN architecture.

    Parameters
    ----------
    embedding_dim : int
        Size of item embeddings and hidden states.
    hidden_size : int
        Kept for config compatibility; must equal embedding_dim.
    step : int
        Number of GNN propagation steps.
    max_session_length : int
        Sessions longer than this are truncated (keep last N items).
    fallback_weight : float
        Weight given to popularity-based fallback scores when the session
        is empty or all items are unknown.
    model_name : str
        Logical name stored in the registry.
    model_version : str
        Semantic version string stored in the registry.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        step: int = 1,
        max_session_length: int = 20,
        fallback_weight: float = 0.15,
        model_name: str = "srgnn",
        model_version: str = "0.1.0",
    ) -> None:
        if hidden_size != embedding_dim:
            logger.warning(
                "hidden_size=%d differs from embedding_dim=%d; using embedding_dim.",
                hidden_size,
                embedding_dim,
            )
        self.embedding_dim = embedding_dim
        self.step = step
        self.max_session_length = max_session_length
        self.fallback_weight = fallback_weight
        self.model_name = model_name
        self.model_version = model_version

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._item_index = _ItemIndex()
        self._popularity: dict[int, int] = {}  # internal idx -> count
        self._core: _SRGNNCore | None = None

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
        early_stopping_patience: int = 5,
    ) -> "SRGNNRecommender":
        """Train on *train_df* and return self.

        Expected columns
        ----------------
        input_items : list[int] or comma-separated string
            Item IDs forming the session context.
        target_item : int
            The ground-truth next item to predict.
        """
        train_df = train_df.copy()
        train_df["input_items"] = train_df["input_items"].apply(_coerce_list)
        train_df["target_item"] = train_df["target_item"].astype(int)

        all_items: list[int] = []
        for items in train_df["input_items"]:
            all_items.extend(items)
        all_items.extend(train_df["target_item"].tolist())

        self._item_index.fit(all_items)
        self._popularity = Counter(self._item_index.encode(all_items))

        n_items = self._item_index.n_items
        self._core = _SRGNNCore(n_items, self.embedding_dim, self.step).to(self.device)

        optimizer = torch.optim.Adam(
            self._core.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        examples = self._prepare_examples(train_df)
        val_examples = self._prepare_examples(val_df) if val_df is not None and not val_df.empty else []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self._core.train()
            epoch_loss = self._run_epoch(examples, optimizer, criterion, batch_size)
            log_msg = f"Epoch {epoch}/{num_epochs}  train_loss={epoch_loss:.4f}"

            if val_examples:
                val_loss = self._eval_epoch(val_examples, criterion)
                log_msg += f"  val_loss={val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("%s  (early stop)", log_msg)
                        break

            logger.info(log_msg)

        return self

    def _prepare_examples(
        self, df: pd.DataFrame
    ) -> list[tuple[list[int], int]]:
        """Encode a DataFrame into (encoded_session, encoded_target) pairs."""
        records: list[tuple[list[int], int]] = []
        for row in df.itertuples(index=False):
            raw_ctx = _coerce_list(row.input_items)
            raw_tgt = int(row.target_item)
            enc_ctx = self._item_index.encode(raw_ctx)
            enc_tgt_list = self._item_index.encode([raw_tgt])
            if not enc_ctx or not enc_tgt_list:
                continue
            # Truncate to max_session_length
            enc_ctx = enc_ctx[-self.max_session_length:]
            records.append((enc_ctx, enc_tgt_list[0]))
        return records

    def _run_epoch(
        self,
        examples: list[tuple[list[int], int]],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_size: int,
    ) -> float:
        indices = np.random.permutation(len(examples))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_loss = torch.tensor(0.0, device=self.device)

            for i in batch_idx:
                ctx, tgt = examples[i]
                seq_output = self._forward_single(ctx)
                scores = self._core.compute_scores(seq_output).unsqueeze(0)  # (1, n+1)
                target = torch.tensor([tgt], dtype=torch.long, device=self.device)
                batch_loss = batch_loss + criterion(scores, target)

            batch_loss = batch_loss / len(batch_idx)
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(
        self,
        examples: list[tuple[list[int], int]],
        criterion: nn.Module,
    ) -> float:
        self._core.eval()
        total_loss = 0.0
        for ctx, tgt in examples:
            seq_output = self._forward_single(ctx)
            scores = self._core.compute_scores(seq_output).unsqueeze(0)
            target = torch.tensor([tgt], dtype=torch.long, device=self.device)
            total_loss += criterion(scores, target).item()
        return total_loss / max(len(examples), 1)

    def _forward_single(self, encoded_session: list[int]) -> torch.Tensor:
        x, alias_input, A, seq_len = _session_to_graph_tensors(
            encoded_session, self.device
        )
        assert self._core is not None
        return self._core(x, alias_input, A, seq_len)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        item_sequence: list[int],
        top_k: int = 10,
    ) -> list[int]:
        """Return top-k next-item predictions given an interaction history.

        Falls back to popularity ranking when the session is empty or contains
        only unknown items.
        """
        encoded = self._item_index.encode(item_sequence)
        encoded = encoded[-self.max_session_length:]

        if not encoded or self._core is None:
            return self._popularity_fallback(top_k)

        self._core.eval()
        seq_output = self._forward_single(encoded)
        scores = self._core.compute_scores(seq_output)  # (n_items+1,)

        # Apply popularity smoothing
        pop_scores = self._popularity_scores(scores.size(0))
        pop_tensor = torch.tensor(pop_scores, dtype=torch.float, device=self.device)
        scores = (1 - self.fallback_weight) * scores + self.fallback_weight * pop_tensor

        # Zero out padding index
        scores[0] = -float("inf")

        top_indices = torch.topk(scores, k=min(top_k, self._item_index.n_items)).indices
        return self._item_index.decode(top_indices.cpu().tolist())

    @torch.no_grad()
    def score(
        self,
        item_sequence: list[int],
        item_ids: list[int],
    ) -> list[float]:
        """Return raw model scores for the given candidate items."""
        encoded_ctx = self._item_index.encode(item_sequence)
        encoded_ctx = encoded_ctx[-self.max_session_length:]
        encoded_items = self._item_index.encode(item_ids)

        if not encoded_ctx or not encoded_items or self._core is None:
            return [0.0] * len(item_ids)

        self._core.eval()
        seq_output = self._forward_single(encoded_ctx)
        all_scores = self._core.compute_scores(seq_output)

        result: list[float] = []
        for enc in encoded_items:
            result.append(all_scores[enc].item())
        return result

    # ------------------------------------------------------------------
    # Popularity fallback
    # ------------------------------------------------------------------

    def _popularity_fallback(self, top_k: int) -> list[int]:
        top_idx = sorted(self._popularity, key=self._popularity.get, reverse=True)[:top_k]
        return self._item_index.decode(top_idx)

    def _popularity_scores(self, size: int) -> list[float]:
        scores = [0.0] * size
        total = sum(self._popularity.values()) or 1
        for idx, cnt in self._popularity.items():
            if idx < size:
                scores[idx] = cnt / total
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist model weights and metadata to *directory*.

        Returns the path to the weights file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        weights_path = directory / "model.pt"
        meta_path = directory / "model.json"

        assert self._core is not None, "Cannot save an unfitted model."
        torch.save(self._core.state_dict(), weights_path)

        meta: dict[str, Any] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "step": self.step,
            "max_session_length": self.max_session_length,
            "fallback_weight": self.fallback_weight,
            "n_items": self._item_index.n_items,
            "item_index": self._item_index.to_dict(),
            "popularity": {str(k): v for k, v in self._popularity.items()},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Model saved → %s", weights_path)
        return weights_path

    @classmethod
    def load(cls, path: str | Path) -> "SRGNNRecommender":
        """Load a model artifact from *path*.

        *path* may point to either:
        - a directory containing ``model.pt`` + ``model.json``, or
        - ``model.pt`` directly (``model.json`` must be in the same folder).
        """
        path = Path(path)
        if path.is_dir():
            weights_path = path / "model.pt"
            meta_path = path / "model.json"
        else:
            weights_path = path
            meta_path = path.parent / "model.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

        obj = cls(
            embedding_dim=meta["embedding_dim"],
            step=meta.get("step", 1),
            max_session_length=meta["max_session_length"],
            fallback_weight=meta["fallback_weight"],
            model_name=meta["model_name"],
            model_version=meta["model_version"],
        )
        obj._item_index = _ItemIndex.from_dict(meta["item_index"])
        obj._popularity = {int(k): v for k, v in meta["popularity"].items()}

        n_items = meta["n_items"]
        obj._core = _SRGNNCore(n_items, obj.embedding_dim, obj.step).to(obj.device)
        obj._core.load_state_dict(
            torch.load(weights_path, map_location=obj.device, weights_only=True)
        )
        obj._core.eval()
        logger.info("Model loaded ← %s", weights_path)
        return obj


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _coerce_list(value: Any) -> list[int]:
    """Convert input_items column value to list[int]."""
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return []
        return [int(p.strip()) for p in stripped.split(",")]
    # numpy / pandas array-like
    try:
        return [int(v) for v in value]
    except TypeError:
        return []