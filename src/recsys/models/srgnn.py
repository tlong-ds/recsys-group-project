"""Build next-item training examples from clean interaction data.

Output DataFrame column contract
---------------------------------
context_items  : list[int]  — ordered item-ID history (the session so far)
target_item    : int        — the next item to predict
last_item_id   : int        — last element of context_items (convenience key)
seq_len        : int        — len(context_items)
session_id     : any
<timestamp_col>: timestamp of target_item
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


class SessionExampleBuilder:
    """Slide a window over each session to produce (context, target) pairs.

    For session [A, B, C, D] this yields:
        context=[A],       target=B
        context=[A,B],     target=C
        context=[A,B,C],   target=D
    """

    def __init__(
        self,
        session_col:        str      = "session_id",
        item_col:           str      = "item_id",
        timestamp_col:      str      = "eventdate",
        max_session_length: int | None = None,
    ) -> None:
        self.session_col        = session_col
        self.item_col           = item_col
        self.timestamp_col      = timestamp_col
        self.max_session_length = max_session_length

    # ------------------------------------------------------------------
    def build(
        self,
        interactions:       pd.DataFrame,
        max_session_length: int | None = None,
    ) -> pd.DataFrame:
        # constructor-level max_session_length wins if per-call value not given
        if max_session_length is None:
            max_session_length = self.max_session_length
        """Return a DataFrame of training examples.

        Parameters
        ----------
        interactions       : sorted interaction log (session_col, item_col, timestamp_col)
        max_session_length : if set, sessions are truncated to the *last* N items
        """
        rows: list[dict[str, Any]] = []

        for sid, grp in interactions.groupby(self.session_col, sort=False):
            seq   = grp[self.item_col].tolist()
            times = grp[self.timestamp_col].tolist()

            if len(seq) < 2:
                continue

            if max_session_length and len(seq) > max_session_length:
                seq   = seq[-max_session_length:]
                times = times[-max_session_length:]

            for i in range(1, len(seq)):
                ctx = seq[:i]
                rows.append(
                    {
                        "session_id":    sid,
                        "context_items": ctx,
                        "last_item_id":  ctx[-1],
                        "target_item":   seq[i],
                        "seq_len":       len(ctx),
                        self.timestamp_col: times[i],
                    }
                )

        df = pd.DataFrame(
            rows,
            columns=[
                "session_id", "context_items", "last_item_id",
                "target_item", "seq_len", self.timestamp_col,
            ],
        )
        logger.info(
            "Built {:,} examples from {:,} sessions",
            len(df),
            interactions[self.session_col].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    def build_graph_examples(
        self,
        interactions:       pd.DataFrame,
        max_session_length: int | None = None,
    ) -> pd.DataFrame:
        """Create SR-GNN style pre-built graph examples.

        Produces the same schema as the parquet training files:
            x            : unique node IDs (original item IDs) per example
            edge_index   : [src_local, dst_local] — local 0-based node indices
            alias_inputs : position → local node index sequence
            item_seq_len : sequence length
            pos_items    : target item ID (original)
            session_id   : session identifier
            <timestamp>  : timestamp of the target item
        """
        from collections import Counter as _Counter

        cap = max_session_length or self.max_session_length
        rows: list[dict[str, Any]] = []

        for sid, grp in interactions.groupby(self.session_col, sort=False):
            seq   = grp[self.item_col].tolist()
            times = grp[self.timestamp_col].tolist()

            if len(seq) < 2:
                continue
            if cap and len(seq) > cap:
                seq   = seq[-cap:]
                times = times[-cap:]

            for i in range(1, len(seq)):
                input_seq = seq[:i]
                target    = seq[i]

                # Build local node index
                node_map: dict[int, int] = {}
                nodes:    list[int] = []
                for item in input_seq:
                    if item not in node_map:
                        node_map[item] = len(nodes)
                        nodes.append(item)

                alias = [node_map[item] for item in input_seq]

                # Build COO edge lists
                src_list, dst_list = [], []
                for u, v in zip(alias[:-1], alias[1:]):
                    src_list.append(u)
                    dst_list.append(v)

                import numpy as np
                rows.append({
                    "x":             np.array(nodes, dtype=np.int64),
                    "edge_index":    np.array([src_list, dst_list], dtype=object),
                    "alias_inputs":  np.array(alias, dtype=np.int64),
                    "item_seq_len":  len(input_seq),
                    "pos_items":     target,
                    "session_id":    sid,
                    self.timestamp_col: times[i],
                })

        df = pd.DataFrame(
            rows,
            columns=["x", "edge_index", "alias_inputs", "item_seq_len",
                     "pos_items", "session_id", self.timestamp_col],
        )
        logger.info(
            "Built {:,} graph examples from {:,} sessions",
            len(df),
            interactions[self.session_col].nunique(),
        )
        return df

        # ------------------------------------------------------------------
    @staticmethod
    def compute_stats(interactions: pd.DataFrame, session_col: str = "session_id") -> dict:
        lens = interactions.groupby(session_col).size()
        return {
            "n_interactions":        len(interactions),
            "n_sessions":            interactions[session_col].nunique(),
            "n_items":               interactions["item_id"].nunique() if "item_id" in interactions.columns else 0,
            "avg_session_length":    float(lens.mean()),
            "min_session_length":    int(lens.min()),
            "max_session_length":    int(lens.max()),
            "median_session_length": int(lens.median()),
        }