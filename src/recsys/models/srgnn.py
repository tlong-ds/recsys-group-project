"""SR-GNN model wrapper backed by a transition-graph baseline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# Develop this lightweight to fullframe srgnn

class SRGNNRecommender:
    """Production-facing SR-GNN wrapper with a lightweight graph baseline."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        max_session_length: int = 20,
        fallback_weight: float = 0.15,
        model_name: str = "srgnn",
        model_version: str = "0.1.0",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_session_length = max_session_length
        self.fallback_weight = fallback_weight
        self.model_name = model_name
        self.model_version = model_version
        self.item_popularity: dict[int, float] = {}
        self.transition_scores: dict[int, dict[int, float]] = {}
        self.item_catalog: list[int] = []

    def fit(self, training_examples: pd.DataFrame) -> "SRGNNRecommender":
        """Fit transition and popularity statistics from training examples."""
        required = {"context_items", "target_item", "last_item_id"}
        missing = required.difference(training_examples.columns)
        if missing:
            raise ValueError(f"Training examples missing columns: {sorted(missing)}")
        if training_examples.empty:
            raise ValueError("Training examples are empty")

        popularity_counter: Counter[int] = Counter()
        transition_counter: dict[int, Counter[int]] = defaultdict(Counter)
        item_catalog: set[int] = set()

        for row in training_examples.itertuples(index=False):
            target_item = int(row.target_item)
            last_item = int(row.last_item_id)
            context_items = [int(item) for item in row.context_items]

            popularity_counter[target_item] += 1
            transition_counter[last_item][target_item] += 1
            item_catalog.add(target_item)
            item_catalog.update(context_items)

        total_targets = sum(popularity_counter.values())
        self.item_popularity = {
            item_id: count / total_targets for item_id, count in popularity_counter.items()
        }
        self.transition_scores = {}
        for source_item, counts in transition_counter.items():
            total = sum(counts.values())
            self.transition_scores[source_item] = {
                target_item: count / total for target_item, count in counts.items()
            }
        self.item_catalog = sorted(item_catalog)
        return self

    def recommend(self, session_items: list[int], top_k: int = 10) -> list[int]:
        """Return the highest-scoring next-item candidates."""
        if not self.item_catalog:
            return []
        candidates = [item for item in self.item_catalog if item not in set(session_items)]
        scored = list(zip(candidates, self.score(session_items, candidates), strict=False))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [item for item, _ in scored[:top_k]]

    def score(self, session_items: list[int], candidate_items: list[int]) -> list[float]:
        """Return scores for candidate next items."""
        last_item = int(session_items[-1]) if session_items else None
        transition_scores = self.transition_scores.get(last_item, {}) if last_item is not None else {}

        results: list[float] = []
        for item_id in candidate_items:
            transition_score = transition_scores.get(int(item_id), 0.0)
            popularity_score = self.item_popularity.get(int(item_id), 0.0)
            score = ((1 - self.fallback_weight) * transition_score) + (
                self.fallback_weight * popularity_score
            )
            results.append(score)
        return results

    def save(self, path: str | Path) -> Path:
        """Persist the model artifact as JSON."""
        target = Path(path)
        if target.suffix != ".json":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "model.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_size": self.hidden_size,
                "max_session_length": self.max_session_length,
                "fallback_weight": self.fallback_weight,
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
            "artifact": {
                "item_popularity": self.item_popularity,
                "transition_scores": self.transition_scores,
                "item_catalog": self.item_catalog,
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SRGNNRecommender":
        """Load the model artifact from JSON."""
        target = Path(path)
        if target.is_dir():
            target = target / "model.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        config = payload["config"]
        artifact = payload["artifact"]

        model = cls(**config)
        model.item_popularity = {
            int(item_id): float(score)
            for item_id, score in artifact.get("item_popularity", {}).items()
        }
        model.transition_scores = {
            int(source_item): {
                int(target_item): float(score)
                for target_item, score in targets.items()
            }
            for source_item, targets in artifact.get("transition_scores", {}).items()
        }
        model.item_catalog = [int(item_id) for item_id in artifact.get("item_catalog", [])]
        return model

    def summary(self) -> dict[str, Any]:
        """Return human-readable model metadata."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "num_items": len(self.item_catalog),
        }
