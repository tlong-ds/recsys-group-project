"""Build training examples from clean interaction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from typing import Any


class ItemVocabBuilder:
    """Create and manage item vocabulary for consistent ID mapping."""

    def __init__(
        self,
        item_col: str = "item_id",
        start_id: int = 1,
    ) -> None:
        """Initialize vocabulary builder.
        
        Args:
            item_col: Name of item ID column.
            start_id: Starting ID for new vocabulary (typically 1 for RNNs).
        """
        self.item_col = item_col
        self.start_id = start_id
        self.item2id = {}
        self.id2item = {}

    def build_from_interactions(
        self,
        interactions: pd.DataFrame,
        sort_by: str = "frequency",
        timestamp_col: str | None = None,
        session_col: str | None = None,
    ) -> dict[int, int]:
        """Build vocabulary from interaction data.
        
        Args:
            interactions: DataFrame with item column.
            sort_by: How to sort items ("frequency", "value").
        
        Returns:
            Mapping of original item IDs to sequential IDs.
        """
        unique_items = interactions[self.item_col].unique()
        
        if sort_by == "frequency":
            # Sort by frequency (most frequent first)
            item_counts = interactions[self.item_col].value_counts()
            sorted_items = item_counts.index.tolist()
        elif sort_by == "first_seen":
            # Preserve stream order (session/time sorted stream for legacy SR-GNN behavior)
            ordered = interactions
            if session_col and timestamp_col and session_col in ordered and timestamp_col in ordered:
                ordered = ordered.sort_values([session_col, timestamp_col])
            sorted_items = ordered[self.item_col].drop_duplicates().tolist()
        else:
            # Sort by item value
            sorted_items = sorted(unique_items)
        
        self.item2id = {item: idx for idx, item in enumerate(sorted_items, start=self.start_id)}
        self.id2item = {idx: item for item, idx in self.item2id.items()}
        
        logger.info(f"✓ Built vocabulary with {len(self.item2id)} items")
        logger.debug(f"Sample mapping: {dict(list(self.item2id.items())[:5])}")
        
        return self.item2id

    def encode_items(self, items: list[int] | pd.Series) -> list[int]:
        """Convert original item IDs to sequential IDs.
        
        Args:
            items: List or Series of original item IDs.
        
        Returns:
            List of encoded item IDs.
        """
        if isinstance(items, pd.Series):
            items = items.tolist()
        
        encoded = []
        for item in items:
            if item not in self.item2id:
                logger.warning(f"Unknown item {item} in vocabulary")
                continue
            encoded.append(self.item2id[item])
        
        return encoded

    def decode_items(self, encoded_ids: list[int]) -> list[int]:
        """Convert sequential IDs back to original item IDs.
        
        Args:
            encoded_ids: List of encoded item IDs.
        
        Returns:
            List of original item IDs.
        """
        decoded = [self.id2item[eid] for eid in encoded_ids if eid in self.id2item]
        return decoded

    def save(self, filepath: Path | str) -> None:
        """Save vocabulary to JSON file.
        
        Args:
            filepath: Destination file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            "item2id": {str(k): v for k, v in self.item2id.items()},
            "id2item": {str(k): v for k, v in self.id2item.items()},
            "size": len(self.item2id),
            "start_id": self.start_id,
        }
        
        with open(filepath, "w") as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"✓ Vocabulary saved to {filepath}")

    def load(self, filepath: Path | str) -> None:
        """Load vocabulary from JSON file.
        
        Args:
            filepath: Source file path.
        """
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            vocab_data = json.load(f)
        
        self.item2id = {int(k): v for k, v in vocab_data["item2id"].items()}
        self.id2item = {int(k): v for k, v in vocab_data["id2item"].items()}
        self.start_id = vocab_data.get("start_id", 1)
        
        logger.info(f"✓ Vocabulary loaded from {filepath} ({len(self.item2id)} items)")


class TrainingExampleBuilder:
    """Build training examples (input sequences + target) from clean interactions."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "eventdate",
    ) -> None:
        """Initialize training example builder.
        
        Args:
            session_col: Name of session ID column.
            item_col: Name of item ID column.
            timestamp_col: Name of timestamp column.
        """
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def build_examples(
        self,
        interactions: pd.DataFrame,
        vocab: dict[int, int] | None = None,
        max_session_length: int | None = None,
        sequence_order: str = "forward",
        drop_unknown_items: bool = False,
    ) -> pd.DataFrame:
        """Create training examples from interaction sequences.
        
        For each session, creates multiple training examples by:
        - Taking all prefixes of length >= 1
        - Each example has: input_items (sequence), target_item, seq_len, session_id
        
        Example: session [1, 2, 3, 4] creates 3 examples:
          - input_items=[1], target_item=2
          - input_items=[1, 2], target_item=3
          - input_items=[1, 2, 3], target_item=4
        
        Args:
            interactions: Clean interaction DataFrame grouped by session.
            vocab: Optional item vocabulary for encoding. Maps original -> sequential IDs.
            max_session_length: Optional cap on input sequence length.
        
        Returns:
            DataFrame with columns: input_items, target_item, seq_len, session_id, and time column.
        """
        examples = []
        
        # Group by session
        for session_id, group in interactions.groupby(self.session_col):
            seq = group[self.item_col].tolist()
            timestamps = group[self.timestamp_col].tolist()
            
            # Skip sessions too short for training
            if len(seq) < 2:
                continue
            
            # Apply max length cap if specified
            if max_session_length is not None and len(seq) > max_session_length:
                seq = seq[-max_session_length :]
                timestamps = timestamps[-max_session_length :]
            
            # Optionally remove items not seen in training vocab (legacy test behavior)
            if vocab is not None and drop_unknown_items:
                filtered_pairs = [
                    (item, ts) for item, ts in zip(seq, timestamps) if item in vocab
                ]
                if len(filtered_pairs) < 2:
                    continue
                seq = [item for item, _ in filtered_pairs]
                timestamps = [ts for _, ts in filtered_pairs]

            # Generate training examples from this session
            indices = range(1, len(seq))
            if sequence_order == "reverse":
                indices = range(len(seq) - 1, 0, -1)

            for i in indices:
                if sequence_order == "reverse":
                    input_seq = seq[:i]
                    target = seq[i]
                else:
                    input_seq = seq[:i]
                    target = seq[i]
                seq_len = len(input_seq)
                
                # Encode items if vocabulary provided
                if vocab is not None:
                    input_seq_encoded = [vocab.get(item, 0) for item in input_seq]
                    target_encoded = vocab.get(target, 0)
                else:
                    input_seq_encoded = input_seq
                    target_encoded = target
                
                examples.append({
                    "input_items": input_seq_encoded,
                    "target_item": target_encoded,
                    "seq_len": seq_len,
                    "session_id": session_id,
                    self.timestamp_col: timestamps[i],  # Use time of the predicted item
                })
        
        df_examples = pd.DataFrame(
            examples,
            columns=["input_items", "target_item", "seq_len", "session_id", self.timestamp_col],
        )
        logger.info(f"Generated {len(df_examples):,} training examples from {interactions[self.session_col].nunique():,} sessions")
        
        return df_examples

    def build_graph_examples(
        self,
        interactions: pd.DataFrame,
        vocab: dict[int, int] | None = None,
        max_session_length: int | None = None,
        sequence_order: str = "forward",
        drop_unknown_items: bool = False,
    ) -> pd.DataFrame:
        """Create SR-GNN style graph examples.

        Each row represents one training sample with fields:
        - x: unique node IDs in the sample graph
        - edge_index: [src_nodes, dst_nodes] directed edges by click transitions
        - alias_inputs: sequence indices that map the original sequence to x
        - item_seq_len: length of input sequence
        - pos_items: next-item label
        """
        graph_examples = []

        for session_id, group in interactions.groupby(self.session_col):
            seq = group[self.item_col].tolist()
            timestamps = group[self.timestamp_col].tolist()

            if len(seq) < 2:
                continue

            if max_session_length is not None and len(seq) > max_session_length:
                seq = seq[-max_session_length:]
                timestamps = timestamps[-max_session_length:]

            if vocab is not None and drop_unknown_items:
                filtered_pairs = [
                    (item, ts) for item, ts in zip(seq, timestamps) if item in vocab
                ]
                if len(filtered_pairs) < 2:
                    continue
                seq = [item for item, _ in filtered_pairs]
                timestamps = [ts for _, ts in filtered_pairs]

            indices = range(1, len(seq))
            if sequence_order == "reverse":
                indices = range(len(seq) - 1, 0, -1)

            for i in indices:
                input_seq = seq[:i]
                target = seq[i]

                if vocab is not None:
                    input_seq_encoded = [vocab.get(item, 0) for item in input_seq]
                    target_encoded = vocab.get(target, 0)
                else:
                    input_seq_encoded = input_seq
                    target_encoded = target

                nodes = list(dict.fromkeys(input_seq_encoded))
                node_index = {item_id: idx for idx, item_id in enumerate(nodes)}
                alias_inputs = [node_index[item_id] for item_id in input_seq_encoded]

                src_nodes = []
                dst_nodes = []
                for left, right in zip(input_seq_encoded[:-1], input_seq_encoded[1:]):
                    src_nodes.append(node_index[left])
                    dst_nodes.append(node_index[right])

                graph_examples.append(
                    {
                        "x": nodes,
                        "edge_index": [src_nodes, dst_nodes],
                        "alias_inputs": alias_inputs,
                        "item_seq_len": len(input_seq_encoded),
                        "pos_items": target_encoded,
                        "session_id": session_id,
                        self.timestamp_col: timestamps[i],
                    }
                )

        df_graph_examples = pd.DataFrame(
            graph_examples,
            columns=[
                "x",
                "edge_index",
                "alias_inputs",
                "item_seq_len",
                "pos_items",
                "session_id",
                self.timestamp_col,
            ],
        )
        logger.info(
            "Generated "
            f"{len(df_graph_examples):,} graph examples from "
            f"{interactions[self.session_col].nunique():,} sessions"
        )
        return df_graph_examples

    @staticmethod
    def compute_stats(
        interactions: pd.DataFrame,
        session_col: str = "session_id",
        item_col: str = "item_id",
    ) -> dict[str, Any]:
        """Compute statistics about the dataset.
        
        Args:
            interactions: Interaction DataFrame.
            session_col: Name of session ID column.
        
        Returns:
            Dictionary of statistics.
        """
        session_lengths = interactions.groupby(session_col).size()
        
        return {
            "n_interactions": len(interactions),
            "n_sessions": interactions[session_col].nunique(),
            "n_items": interactions[item_col].nunique() if item_col in interactions else 0,
            "avg_session_length": float(session_lengths.mean()),
            "min_session_length": int(session_lengths.min()),
            "max_session_length": int(session_lengths.max()),
            "median_session_length": int(session_lengths.median()),
        }
