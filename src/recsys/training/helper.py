"""Normalization helpers for processed training examples."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def _coerce_sequence(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, (list, tuple)):
            return [int(item) for item in converted]
        return [int(converted)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = [
                part.strip() for part in stripped.strip("[]").split(",") if part.strip()
            ]
        if isinstance(parsed, (list, tuple)):
            return [int(item) for item in parsed]
        return [int(parsed)]
    return [int(value)]


def normalize_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Normalize examples from supported schemas into trainer input columns."""
    rows: list[dict[str, Any]] = []
    if examples.empty:
        return pd.DataFrame(
            columns=[
                "input_items",
                "context_items",
                "target_item",
                "last_item_id",
                "context_length",
            ]
        )

    if {"context_items", "target_item", "last_item_id"}.issubset(examples.columns):
        for row in examples.itertuples(index=False):
            context = _coerce_sequence(getattr(row, "context_items"))
            if not context:
                continue
            rows.append(
                {
                    "input_items": context,
                    "context_items": context,
                    "target_item": int(getattr(row, "target_item")),
                    "last_item_id": int(getattr(row, "last_item_id")),
                    "context_length": len(context),
                }
            )
    elif {"input_items", "target_item"}.issubset(examples.columns):
        for row in examples.itertuples(index=False):
            context = _coerce_sequence(getattr(row, "input_items"))
            if not context:
                continue
            rows.append(
                {
                    "input_items": context,
                    "context_items": context,
                    "target_item": int(getattr(row, "target_item")),
                    "last_item_id": int(context[-1]),
                    "context_length": len(context),
                }
            )
    elif {"x", "alias_inputs", "pos_items"}.issubset(examples.columns):
        for row in examples.itertuples(index=False):
            nodes = _coerce_sequence(getattr(row, "x"))
            alias_inputs = _coerce_sequence(getattr(row, "alias_inputs"))
            context = [
                nodes[index] for index in alias_inputs if 0 <= index < len(nodes)
            ]
            if not context:
                continue
            rows.append(
                {
                    "input_items": context,
                    "context_items": context,
                    "target_item": int(getattr(row, "pos_items")),
                    "last_item_id": int(context[-1]),
                    "context_length": len(context),
                }
            )
    else:
        raise ValueError(
            "Unsupported training examples schema. Expected one of: "
            "{context_items,target_item,last_item_id}, "
            "{input_items,target_item}, "
            "{x,alias_inputs,pos_items}."
        )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "input_items",
            "context_items",
            "target_item",
            "last_item_id",
            "context_length",
        ],
    )
