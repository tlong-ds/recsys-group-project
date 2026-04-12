"""Feature generation for session-based next-item prediction."""

from __future__ import annotations

import pandas as pd


class SessionFeatureBuilder:
    """Convert ordered interactions into prefix/target training examples."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        max_session_length: int = 20,
    ) -> None:
        self.session_col = session_col
        self.item_col = item_col
        self.max_session_length = max_session_length

    def build_examples(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Generate next-item examples from each session."""
        columns = [
            self.session_col,
            "context_items",
            "target_item",
            "last_item_id",
            "context_length",
        ]
        records: list[dict[str, object]] = []
        if interactions.empty:
            return pd.DataFrame(columns=columns)

        for session_id, session_df in interactions.groupby(self.session_col, sort=False):
            items = [int(item) for item in session_df[self.item_col].tolist()]
            for index in range(1, len(items)):
                context = items[max(0, index - self.max_session_length) : index]
                target = items[index]
                records.append(
                    {
                        self.session_col: session_id,
                        "context_items": context,
                        "target_item": target,
                        "last_item_id": context[-1],
                        "context_length": len(context),
                    }
                )

        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame.from_records(records, columns=columns)
