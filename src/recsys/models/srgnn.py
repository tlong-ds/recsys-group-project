"""PyTorch SR-GNN recommender trained from preprocessed graph examples."""

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

logger = logging.getLogger(__name__)


def _to_int_array(value: Any) -> np.ndarray:
    """Convert parquet object columns into a 1D int64 array."""
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return np.asarray(list(value), dtype=np.int64)
    if value is None:
        return np.asarray([], dtype=np.int64)
    raise TypeError(f"Unsupported array value: {type(value)!r}")


def _to_edge_index(value: Any) -> np.ndarray:
    """Convert parquet edge storage into shape (2, n_edges)."""
    if isinstance(value, np.ndarray) and value.dtype != object:
        edge_index = value.astype(np.int64, copy=False)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        edge_index = np.vstack([_to_int_array(part) for part in value])
    elif isinstance(value, Sequence):
        edge_index = np.vstack([_to_int_array(part) for part in value])
    else:
        raise TypeError(f"Unsupported edge_index value: {type(value)!r}")

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, n_edges), got {edge_index.shape}")
    return edge_index


def _build_adjacency(alias_inputs: np.ndarray, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Build normalized incoming/outgoing adjacency for one session graph."""
    a_in = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    a_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    if edge_index.size:
        src = edge_index[0]
        dst = edge_index[1]
        for u, v in zip(src.tolist(), dst.tolist(), strict=False):
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                a_out[u, v] += 1.0
                a_in[v, u] += 1.0

    out_norm = a_out.sum(axis=1, keepdims=True)
    in_norm = a_in.sum(axis=1, keepdims=True)
    a_out /= np.where(out_norm == 0.0, 1.0, out_norm)
    a_in /= np.where(in_norm == 0.0, 1.0, in_norm)

    if alias_inputs.size:
        max_alias = int(alias_inputs.max())
        if max_alias >= n_nodes:
            raise ValueError(f"alias_inputs contain node index {max_alias} with only {n_nodes} nodes")

    return np.concatenate([a_in, a_out], axis=1)


class SessionGraphDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper around parquet feature examples."""

    def __init__(self, examples: pd.DataFrame) -> None:
        if examples.empty:
            raise ValueError("Examples dataframe is empty")
        self.examples = examples.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.examples.iloc[index]
        x = _to_int_array(row.x)
        alias_inputs = _to_int_array(row.alias_inputs)
        edge_index = _to_edge_index(row.edge_index)
        adjacency = _build_adjacency(alias_inputs, edge_index, len(x))
        return {
            "items": torch.tensor(x, dtype=torch.long),
            "alias_inputs": torch.tensor(alias_inputs, dtype=torch.long),
            "adjacency": torch.tensor(adjacency, dtype=torch.float32),
            "target": torch.tensor(int(row.pos_items), dtype=torch.long),
            "seq_len": torch.tensor(int(row.item_seq_len), dtype=torch.long),
        }


def _collate_graph_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-size graph examples into a dense batch."""
    batch_size = len(batch)
    max_nodes = max(int(example["items"].numel()) for example in batch)
    max_seq_len = max(int(example["alias_inputs"].numel()) for example in batch)

    items = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    alias_inputs = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes * 2), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    seq_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)

    for idx, example in enumerate(batch):
        n_nodes = int(example["items"].numel())
        seq_len = int(example["alias_inputs"].numel())
        items[idx, :n_nodes] = example["items"]
        alias_inputs[idx, :seq_len] = example["alias_inputs"]
        adjacency[idx, :n_nodes, : 2 * n_nodes] = example["adjacency"]
        node_mask[idx, :n_nodes] = True
        seq_mask[idx, :seq_len] = True
        targets[idx] = example["target"]
        seq_lens[idx] = example["seq_len"]

    return {
        "items": items,
        "alias_inputs": alias_inputs,
        "adjacency": adjacency,
        "node_mask": node_mask,
        "seq_mask": seq_mask,
        "targets": targets,
        "seq_lens": seq_lens,
    }


class _GNNCell(nn.Module):
    """SR-GNN gated propagation cell."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_edge_in = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)

        self.w_ih = nn.Linear(hidden_size * 2, hidden_size * 3, bias=True)
        self.w_hh = nn.Linear(hidden_size, hidden_size * 3, bias=True)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n_nodes = hidden.size(1)
        a_in = adjacency[:, :, :n_nodes]
        a_out = adjacency[:, :, n_nodes:]

        input_in = torch.matmul(a_in, self.linear_edge_in(hidden))
        input_out = torch.matmul(a_out, self.linear_edge_out(hidden))
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.w_ih(inputs)
        gh = self.w_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return new_gate + input_gate * (hidden - new_gate)


class _SRGNNCore(nn.Module):
    """Batched SR-GNN encoder."""

    def __init__(self, n_items: int, embedding_dim: int, step: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.step = step

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.gnn = _GNNCell(embedding_dim)
        self.linear_one = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_two = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_three = nn.Linear(embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(embedding_dim * 2, embedding_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.embedding_dim)
        for parameter in self.parameters():
            parameter.data.uniform_(-bound, bound)

    def forward(
        self,
        items: torch.Tensor,
        alias_inputs: torch.Tensor,
        adjacency: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.item_embedding(items)
        for _ in range(self.step):
            hidden = self.gnn(hidden, adjacency)

        hidden_size = hidden.size(-1)
        gather_index = alias_inputs.unsqueeze(-1).expand(-1, -1, hidden_size)
        seq_hidden = torch.gather(hidden, 1, gather_index)

        lengths = seq_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        ht = seq_hidden[batch_index, lengths]

        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)
        alpha = alpha.masked_fill(~seq_mask, float("-inf"))
        alpha = torch.softmax(alpha, dim=1).unsqueeze(-1)

        local_context = torch.sum(alpha * seq_hidden, dim=1)
        session_rep = self.linear_transform(torch.cat([local_context, ht], dim=1))
        return session_rep

    def compute_scores(self, session_rep: torch.Tensor) -> torch.Tensor:
        return session_rep @ self.item_embedding.weight.t()


class SRGNNRecommender:
    """Session-based SR-GNN recommender over preprocessed graph examples."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        step: int = 1,
        max_session_length: int = 20,
        fallback_weight: float = 0.0,
        model_name: str = "srgnn",
        model_version: str = "0.1.0",
        seed: int = 42,
    ) -> None:
        if hidden_size != embedding_dim:
            logger.warning("hidden_size=%d differs from embedding_dim=%d; using embedding_dim", hidden_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.step = step
        self.max_session_length = max_session_length
        self.fallback_weight = fallback_weight
        self.model_name = model_name
        self.model_version = model_version
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_items = 0
        self._core: _SRGNNCore | None = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._popularity: dict[int, int] = {}

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
    ) -> "SRGNNRecommender":
        """Train on preprocessed graph examples."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_df = self._normalize_examples(train_df)
        val_df = self._normalize_examples(val_df) if val_df is not None and not val_df.empty else None

        self._initialize_vocab(train_df, item_vocab)
        self._core = _SRGNNCore(self.n_items, self.embedding_dim, self.step).to(self.device)

        train_loader = DataLoader(
            SessionGraphDataset(train_df),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_graph_batch,
        )

        val_loader = None
        if val_df is not None and not val_df.empty:
            val_loader = DataLoader(
                SessionGraphDataset(val_df),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_graph_batch,
            )

        optimizer = torch.optim.Adam(self._core.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer, criterion)
            log_message = f"epoch={epoch}/{num_epochs} train_loss={train_loss:.4f}"

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                log_message += f" val_loss={val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in self._core.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("%s early_stop=true", log_message)
                        break

            logger.info(log_message)

        if best_state is not None:
            self._core.load_state_dict(best_state)

        return self

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        assert self._core is not None
        self._core.train()
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = self._move_batch(batch)
            scores = self._compute_scores(batch)
            loss = criterion(scores, batch["targets"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = int(batch["targets"].size(0))
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
        criterion: nn.Module,
    ) -> float:
        assert self._core is not None
        self._core.eval()
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = self._move_batch(batch)
            scores = self._compute_scores(batch)
            loss = criterion(scores, batch["targets"])
            batch_size = int(batch["targets"].size(0))
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def _compute_scores(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _normalize_examples(self, examples: pd.DataFrame | None) -> pd.DataFrame:
        if examples is None:
            return pd.DataFrame()
        required = {"x", "edge_index", "alias_inputs", "item_seq_len", "pos_items"}
        missing = required.difference(examples.columns)
        if missing:
            raise ValueError(f"Missing required example columns: {sorted(missing)}")
        normalized = examples.copy()
        normalized["item_seq_len"] = normalized["item_seq_len"].astype(int)
        normalized["pos_items"] = normalized["pos_items"].astype(int)
        return normalized

    def _initialize_vocab(self, train_df: pd.DataFrame, item_vocab: Mapping[str, Any] | None) -> None:
        if item_vocab and "item2id" in item_vocab:
            raw_item_to_idx = {int(key): int(value) for key, value in item_vocab["item2id"].items()}
            self._item_to_idx = raw_item_to_idx
            self._idx_to_item = {idx: item for item, idx in raw_item_to_idx.items()}
            self.n_items = max(self._idx_to_item) if self._idx_to_item else 0
        else:
            max_item = 0
            for row in train_df.itertuples(index=False):
                row_items = _to_int_array(row.x)
                if row_items.size:
                    max_item = max(max_item, int(row_items.max()))
                max_item = max(max_item, int(row.pos_items))
            self.n_items = max_item
            self._idx_to_item = {idx: idx for idx in range(1, self.n_items + 1)}
            self._item_to_idx = dict(self._idx_to_item)

        popularity_counter: Counter[int] = Counter()
        for row in train_df.itertuples(index=False):
            popularity_counter.update(_to_int_array(row.x).tolist())
            popularity_counter.update([int(row.pos_items)])
        self._popularity = dict(popularity_counter)

    def _encode_external_items(self, item_sequence: Sequence[int]) -> list[int]:
        encoded: list[int] = []
        for item in item_sequence[-self.max_session_length :]:
            item_int = int(item)
            if item_int in self._item_to_idx:
                encoded.append(self._item_to_idx[item_int])
            elif 1 <= item_int <= self.n_items:
                encoded.append(item_int)
        return encoded

    def _decode_internal_items(self, item_ids: Sequence[int]) -> list[int]:
        return [self._idx_to_item.get(int(item_id), int(item_id)) for item_id in item_ids]

    def _graph_from_sequence(self, encoded_items: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_nodes: list[int] = []
        node_to_index: dict[int, int] = {}
        alias_inputs: list[int] = []
        edges_src: list[int] = []
        edges_dst: list[int] = []

        for item in encoded_items:
            if item not in node_to_index:
                node_to_index[item] = len(unique_nodes)
                unique_nodes.append(item)
            alias_inputs.append(node_to_index[item])

        for index in range(len(alias_inputs) - 1):
            edges_src.append(alias_inputs[index])
            edges_dst.append(alias_inputs[index + 1])

        items = torch.tensor(unique_nodes, dtype=torch.long)
        alias = torch.tensor(alias_inputs, dtype=torch.long)
        edge_index = np.vstack([np.asarray(edges_src, dtype=np.int64), np.asarray(edges_dst, dtype=np.int64)]) if edges_src else np.empty((2, 0), dtype=np.int64)
        adjacency = torch.tensor(_build_adjacency(np.asarray(alias_inputs, dtype=np.int64), edge_index, len(unique_nodes)), dtype=torch.float32)
        seq_mask = torch.ones((1, len(alias_inputs)), dtype=torch.bool)
        return (
            items.unsqueeze(0).to(self.device),
            alias.unsqueeze(0).to(self.device),
            adjacency.unsqueeze(0).to(self.device),
            seq_mask.to(self.device),
        )

    @torch.no_grad()
    def recommend(self, item_sequence: Sequence[int], top_k: int = 10) -> list[int]:
        encoded_items = self._encode_external_items(item_sequence)
        if not encoded_items or self._core is None:
            return self._decode_internal_items(self._popularity_top_indices(top_k))

        self._core.eval()
        items, alias_inputs, adjacency, seq_mask = self._graph_from_sequence(encoded_items)
        session_rep = self._core(items, alias_inputs, adjacency, seq_mask)
        scores = self._core.compute_scores(session_rep).squeeze(0)
        scores[0] = float("-inf")

        if self.fallback_weight > 0:
            popularity = torch.tensor(self._popularity_distribution(), dtype=torch.float32, device=self.device)
            scores = (1.0 - self.fallback_weight) * scores + self.fallback_weight * popularity

        top_indices = torch.topk(scores, k=min(top_k, self.n_items)).indices.cpu().tolist()
        return self._decode_internal_items(top_indices)

    @torch.no_grad()
    def recommend_from_graph(
        self,
        x: Sequence[int],
        alias_inputs: Sequence[int],
        top_k: int = 10,
    ) -> list[int]:
        if self._core is None:
            return []

        items = torch.tensor(_to_int_array(x), dtype=torch.long, device=self.device).unsqueeze(0)
        alias = torch.tensor(_to_int_array(alias_inputs), dtype=torch.long, device=self.device).unsqueeze(0)

        src = alias[:, :-1]
        dst = alias[:, 1:]
        edge_index = np.vstack([src.squeeze(0).cpu().numpy(), dst.squeeze(0).cpu().numpy()]) if alias.size(1) > 1 else np.empty((2, 0), dtype=np.int64)
        adjacency = torch.tensor(
            _build_adjacency(_to_int_array(alias_inputs), edge_index, int(items.size(1))),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        seq_mask = torch.ones((1, alias.size(1)), dtype=torch.bool, device=self.device)

        self._core.eval()
        session_rep = self._core(items, alias, adjacency, seq_mask)
        scores = self._core.compute_scores(session_rep).squeeze(0)
        scores[0] = float("-inf")
        top_indices = torch.topk(scores, k=min(top_k, self.n_items)).indices.cpu().tolist()
        return top_indices

    @torch.no_grad()
    def score(self, item_sequence: Sequence[int], item_ids: Sequence[int]) -> list[float]:
        encoded_context = self._encode_external_items(item_sequence)
        if not encoded_context or self._core is None:
            return [0.0 for _ in item_ids]

        items, alias_inputs, adjacency, seq_mask = self._graph_from_sequence(encoded_context)
        self._core.eval()
        session_rep = self._core(items, alias_inputs, adjacency, seq_mask)
        all_scores = self._core.compute_scores(session_rep).squeeze(0)

        scores: list[float] = []
        for item in item_ids:
            item_int = int(item)
            encoded_item = self._item_to_idx.get(item_int, item_int if 1 <= item_int <= self.n_items else 0)
            scores.append(float(all_scores[encoded_item].item()) if encoded_item else 0.0)
        return scores

    def _popularity_top_indices(self, top_k: int) -> list[int]:
        return [item for item, _count in sorted(self._popularity.items(), key=lambda pair: pair[1], reverse=True)[:top_k]]

    def _popularity_distribution(self) -> list[float]:
        values = [0.0] * (self.n_items + 1)
        total = float(sum(self._popularity.values()) or 1)
        for item, count in self._popularity.items():
            if 0 <= item <= self.n_items:
                values[item] = count / total
        return values

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights_path = directory / "model.pt"
        metadata_path = directory / "model.json"

        if self._core is None:
            raise ValueError("Cannot save an unfitted model")

        torch.save(self._core.state_dict(), weights_path)
        metadata = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "step": self.step,
            "max_session_length": self.max_session_length,
            "fallback_weight": self.fallback_weight,
            "seed": self.seed,
            "n_items": self.n_items,
            "item_to_idx": {str(key): value for key, value in self._item_to_idx.items()},
            "idx_to_item": {str(key): value for key, value in self._idx_to_item.items()},
            "popularity": {str(key): value for key, value in self._popularity.items()},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return weights_path

    @classmethod
    def load(cls, path: str | Path) -> "SRGNNRecommender":
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        weights_path = directory / "model.pt"
        metadata_path = directory / "model.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(
            embedding_dim=int(metadata["embedding_dim"]),
            hidden_size=int(metadata["embedding_dim"]),
            step=int(metadata["step"]),
            max_session_length=int(metadata["max_session_length"]),
            fallback_weight=float(metadata["fallback_weight"]),
            model_name=str(metadata["model_name"]),
            model_version=str(metadata["model_version"]),
            seed=int(metadata.get("seed", 42)),
        )
        model.n_items = int(metadata["n_items"])
        model._item_to_idx = {int(key): int(value) for key, value in metadata["item_to_idx"].items()}
        model._idx_to_item = {int(key): int(value) for key, value in metadata["idx_to_item"].items()}
        model._popularity = {int(key): int(value) for key, value in metadata["popularity"].items()}

        model._core = _SRGNNCore(model.n_items, model.embedding_dim, model.step).to(model.device)
        model._core.load_state_dict(torch.load(weights_path, map_location=model.device, weights_only=True))
        model._core.eval()
        return model
