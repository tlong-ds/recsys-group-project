"""Stage-oriented data pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from recsys.data.ingest import DataLoader
from recsys.data.preprocessor import SessionPreprocessor
from recsys.data.splitter import (
    split_by_days,
    split_by_session_time,
    split_by_time,
    split_diginetica_legacy,
)
from recsys.data.training_examples import ItemVocabBuilder, TrainingExampleBuilder
from recsys.data.validator import InteractionValidator
from recsys.utils.config import load_data_config_with_params

STAGE_ALL = "all"
STAGE_INGEST = "ingest"
STAGE_VALIDATE = "validate"
STAGE_PREPROCESS = "preprocess"
STAGE_SPLIT = "split"
STAGE_BUILD_EXAMPLES = "build_examples"
STAGE_NAMES = [
    STAGE_ALL,
    STAGE_INGEST,
    STAGE_VALIDATE,
    STAGE_PREPROCESS,
    STAGE_SPLIT,
    STAGE_BUILD_EXAMPLES,
]


def _load_data_config(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    return load_data_config_with_params(
        data_config_path=config_path,
        params_path=params_path,
    )


def _ensure_data_paths(data_cfg: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "raw": Path(data_cfg.get("raw_path", "data/raw")),
        "interim": Path(data_cfg.get("interim_path", "data/interim")),
        "processed": Path(data_cfg.get("processed_path", "data/processed")),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_column_name(
    df: pd.DataFrame,
    col_map: dict[str, Any],
    key: str,
    fallback: str,
) -> str:
    configured = col_map.get(key, fallback)
    if configured in df.columns:
        return configured
    if fallback in df.columns:
        return fallback
    return configured


def _canonicalize_for_dataset(
    df: pd.DataFrame,
    data_cfg: dict[str, Any],
) -> pd.DataFrame:
    col_map = data_cfg.get("columns", {})
    session_col = col_map.get("session_id", "session_id")
    item_col = col_map.get("item_id", "item_id")
    event_date_col = col_map.get("event_date")
    timeframe_col = col_map.get("timeframe")

    canonical = df.copy()

    if session_col not in canonical.columns and "session_id" in canonical.columns:
        canonical[session_col] = canonical["session_id"]
    if item_col not in canonical.columns and "item_id" in canonical.columns:
        canonical[item_col] = canonical["item_id"]

    if event_date_col and event_date_col in canonical.columns:
        canonical[event_date_col] = pd.to_datetime(canonical[event_date_col])
    if (
        timeframe_col
        and timeframe_col not in canonical.columns
        and "timeframe" in canonical.columns
    ):
        canonical[timeframe_col] = canonical["timeframe"]

    return canonical


def _save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    data_cfg: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    suffix = destination.suffix.lower()
    if suffix == ".parquet":
        compression = data_cfg.get("parquet_compression", "snappy")
        df.to_parquet(destination, compression=compression, index=False)
    elif suffix == ".csv":
        df.to_csv(destination, index=False)
    else:
        raise ValueError(f"Unsupported output format for path: {destination}")

    return destination


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported input format for path: {source}")


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return destination


def _build_ingest_params(data_cfg: dict[str, Any]) -> dict[str, Any]:
    ingest_defaults = {
        "item_views": "train-item-views.csv",
    }
    ingest_cfg = ingest_defaults | data_cfg.get("ingest", {})

    return {
        "data": {
            "raw_dir": data_cfg.get("raw_path", "data/raw"),
            "interim_dir": data_cfg.get("interim_path", "data/interim"),
            "processed_dir": data_cfg.get("processed_path", "data/processed"),
        },
        "ingest": ingest_cfg,
    }


def run_ingest_stage(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Execute ingest stage and return materialized artifacts."""
    data_cfg = _load_data_config(config_path, params_path)
    _ensure_data_paths(data_cfg)
    ingest_params = _build_ingest_params(data_cfg)

    loader = DataLoader(
        raw_path=data_cfg.get("raw_path", "data/raw"),
        interim_path=data_cfg.get("interim_path", "data/interim"),
    )
    loader.ingest(ingest_params)

    interim = Path(data_cfg.get("interim_path", "data/interim"))
    return {
        "interactions": str(interim / "interactions.parquet"),
        "ingest_report": "metrics/ingest_report.json",
    }


def run_validate_stage(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Execute schema/semantic validation on interim interactions."""
    data_cfg = _load_data_config(config_path, params_path)
    interim = Path(data_cfg.get("interim_path", "data/interim"))
    interactions_path = interim / "interactions.parquet"

    raw_df = _load_dataframe(interactions_path)
    canonical_df = _canonicalize_for_dataset(raw_df, data_cfg)

    col_map = data_cfg.get("columns", {})
    session_col = _resolve_column_name(canonical_df, col_map, "session_id", "sessionId")
    item_col = _resolve_column_name(canonical_df, col_map, "item_id", "itemId")
    event_date_col = _resolve_column_name(
        canonical_df,
        col_map,
        "event_date",
        "eventdate",
    )

    validator = InteractionValidator(
        session_col=session_col,
        item_col=item_col,
        timestamp_col=event_date_col,
    )

    val_cfg = data_cfg.get("validation", {})
    report = validator.generate_report(
        canonical_df,
        min_session_length=val_cfg.get("min_session_length", 1),
        max_session_length=val_cfg.get("max_session_length"),
        allow_duplicates=val_cfg.get("allow_duplicates", False),
    )

    report_path = (
        data_cfg.get("logging", {}).get("report_path")
        or "metrics/validation_report.json"
    )
    _write_json(report, report_path)

    return {"validation_report": str(report_path)}


def run_preprocess_stage(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Execute preprocessing and persist cleaned interactions."""
    data_cfg = _load_data_config(config_path, params_path)
    paths = _ensure_data_paths(data_cfg)

    raw_df = _load_dataframe(paths["interim"] / "interactions.parquet")
    canonical_df = _canonicalize_for_dataset(raw_df, data_cfg)

    col_map = data_cfg.get("columns", {})
    val_cfg = data_cfg.get("validation", {})

    session_col = _resolve_column_name(canonical_df, col_map, "session_id", "sessionId")
    item_col = _resolve_column_name(canonical_df, col_map, "item_id", "itemId")
    event_date_col = _resolve_column_name(
        canonical_df,
        col_map,
        "event_date",
        "eventdate",
    )
    timeframe_col = _resolve_column_name(
        canonical_df,
        col_map,
        "timeframe",
        "timeframe",
    )

    preprocessor = SessionPreprocessor(
        session_col=session_col,
        item_col=item_col,
        timestamp_col=event_date_col,
        event_date_col=event_date_col,
        timeframe_col=timeframe_col,
    )

    cleaned = preprocessor.transform(canonical_df)
    filtered = preprocessor.filter_sessions(
        cleaned,
        min_length=val_cfg.get("min_session_length", 2),
        max_length=val_cfg.get("max_session_length"),
    )

    filtered, _ = preprocessor.filter_items(
        filtered,
        min_frequency=val_cfg.get("min_item_frequency", 5),
    )

    filtered = preprocessor.filter_sessions(
        filtered,
        min_length=val_cfg.get("min_session_length", 2),
        max_length=val_cfg.get("max_session_length"),
    )

    if not val_cfg.get("allow_duplicates", False):
        filtered = preprocessor.remove_duplicate_items_in_session(filtered)

    clean_path = _save_dataframe(
        filtered,
        paths["interim"] / "clean_interactions.parquet",
        data_cfg,
    )
    logger.info("Saved cleaned interactions: {}", clean_path)

    return {"clean_interactions": str(clean_path)}


def run_split_stage(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Execute configured temporal split and persist split interactions."""
    data_cfg = _load_data_config(config_path, params_path)
    paths = _ensure_data_paths(data_cfg)

    clean_df = _load_dataframe(paths["interim"] / "clean_interactions.parquet")

    col_map = data_cfg.get("columns", {})
    split_cfg = data_cfg.get("temporal_split", {})

    session_col = _resolve_column_name(clean_df, col_map, "session_id", "sessionId")
    event_date_col = _resolve_column_name(clean_df, col_map, "event_date", "eventdate")

    strategy = split_cfg.get("strategy", "time_based")
    if strategy == "ratio_based":
        train_df, val_df, test_df = split_by_time(
            clean_df,
            timestamp_col=event_date_col,
            val_ratio=split_cfg.get("val_ratio", 0.15),
            test_ratio=split_cfg.get("test_ratio", 0.15),
        )
    elif strategy == "diginetica_legacy":
        train_df, val_df, test_df = split_diginetica_legacy(
            clean_df,
            session_col=session_col,
            timestamp_col=event_date_col,
            test_days=split_cfg.get("test_days", 7),
            val_days=split_cfg.get("val_days", 7),
        )
    elif strategy == "session_based":
        train_df, val_df, test_df = split_by_session_time(
            clean_df,
            session_col=session_col,
            timestamp_col=event_date_col,
            test_days=split_cfg.get("test_days", 7),
            val_days=split_cfg.get("val_days", 7),
        )
    else:
        train_df, val_df, test_df = split_by_days(
            clean_df,
            timestamp_col=event_date_col,
            test_days=split_cfg.get("test_days", 7),
            val_days=split_cfg.get("val_days", 7),
        )

    train_path = _save_dataframe(
        train_df,
        paths["interim"] / "train_interactions.parquet",
        data_cfg,
    )
    val_path = _save_dataframe(
        val_df,
        paths["interim"] / "val_interactions.parquet",
        data_cfg,
    )
    test_path = _save_dataframe(
        test_df,
        paths["interim"] / "test_interactions.parquet",
        data_cfg,
    )

    return {
        "train_interactions": str(train_path),
        "val_interactions": str(val_path),
        "test_interactions": str(test_path),
    }


def _build_examples_for_split(
    split_df: pd.DataFrame,
    example_builder: TrainingExampleBuilder,
    vocab: dict[int, int],
    sequence_order: str,
    output_format: str,
    drop_unknown_items: bool,
) -> pd.DataFrame:
    if output_format == "graph":
        return example_builder.build_graph_examples(
            split_df,
            vocab=vocab,
            sequence_order=sequence_order,
            drop_unknown_items=drop_unknown_items,
        )

    return example_builder.build_examples(
        split_df,
        vocab=vocab,
        sequence_order=sequence_order,
        drop_unknown_items=drop_unknown_items,
    )


def _stats_block(
    df: pd.DataFrame,
    examples: pd.DataFrame,
    session_col: str,
) -> dict[str, Any]:
    return {
        "interactions": TrainingExampleBuilder.compute_stats(df, session_col),
        "examples": len(examples),
    }


def run_build_examples_stage(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Build examples/vocabulary from split interactions and persist outputs."""
    data_cfg = _load_data_config(config_path, params_path)
    paths = _ensure_data_paths(data_cfg)

    train_df = _load_dataframe(paths["interim"] / "train_interactions.parquet")
    val_df = _load_dataframe(paths["interim"] / "val_interactions.parquet")
    test_df = _load_dataframe(paths["interim"] / "test_interactions.parquet")

    col_map = data_cfg.get("columns", {})
    compat_cfg = data_cfg.get("compatibility", {})

    session_col = _resolve_column_name(train_df, col_map, "session_id", "sessionId")
    item_col = _resolve_column_name(train_df, col_map, "item_id", "itemId")
    event_date_col = _resolve_column_name(train_df, col_map, "event_date", "eventdate")

    vocab_builder = ItemVocabBuilder(item_col=item_col)
    vocab_builder.build_from_interactions(
        train_df,
        sort_by=compat_cfg.get("vocab_order", "frequency"),
        timestamp_col=event_date_col,
        session_col=session_col,
    )

    example_builder = TrainingExampleBuilder(
        session_col=session_col,
        item_col=item_col,
        timestamp_col=event_date_col,
    )

    sequence_order = compat_cfg.get("example_order", "forward")
    output_format = data_cfg.get("training_example_format", "graph")

    train_examples = _build_examples_for_split(
        train_df,
        example_builder,
        vocab_builder.item2id,
        sequence_order,
        output_format,
        drop_unknown_items=False,
    )
    val_examples = _build_examples_for_split(
        val_df,
        example_builder,
        vocab_builder.item2id,
        sequence_order,
        output_format,
        drop_unknown_items=True,
    )
    test_examples = _build_examples_for_split(
        test_df,
        example_builder,
        vocab_builder.item2id,
        sequence_order,
        output_format,
        drop_unknown_items=True,
    )

    train_path = _save_dataframe(
        train_examples,
        paths["processed"] / "train_examples.parquet",
        data_cfg,
    )
    val_path = _save_dataframe(
        val_examples,
        paths["processed"] / "val_examples.parquet",
        data_cfg,
    )
    test_path = _save_dataframe(
        test_examples,
        paths["processed"] / "test_examples.parquet",
        data_cfg,
    )

    vocab_path = paths["processed"] / "item_vocab.json"
    stats_path = paths["processed"] / "data_stats.json"

    vocab_builder.save(vocab_path)

    stats = {
        "config_file": str(config_path),
        "train": _stats_block(train_df, train_examples, session_col),
        "val": _stats_block(val_df, val_examples, session_col),
        "test": _stats_block(test_df, test_examples, session_col),
        "vocab_size": len(vocab_builder.item2id),
    }
    _write_json(stats, stats_path)

    return {
        "train_examples": str(train_path),
        "val_examples": str(val_path),
        "test_examples": str(test_path),
        "vocab": str(vocab_path),
        "stats": str(stats_path),
    }


def run_full_data_pipeline(
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Run all data stages in order and return all output artifact paths."""
    outputs: dict[str, str] = {}
    outputs.update(run_ingest_stage(config_path=config_path, params_path=params_path))
    outputs.update(run_validate_stage(config_path=config_path, params_path=params_path))
    outputs.update(
        run_preprocess_stage(config_path=config_path, params_path=params_path)
    )
    outputs.update(run_split_stage(config_path=config_path, params_path=params_path))
    outputs.update(
        run_build_examples_stage(config_path=config_path, params_path=params_path)
    )
    return outputs


def run_stage(
    stage: str,
    config_path: str | Path = "configs/data_config.yaml",
    params_path: str | Path = "params.yaml",
) -> dict[str, str]:
    """Run a single stage or the full pipeline."""
    stage_runners: dict[str, Any] = {
        STAGE_INGEST: run_ingest_stage,
        STAGE_VALIDATE: run_validate_stage,
        STAGE_PREPROCESS: run_preprocess_stage,
        STAGE_SPLIT: run_split_stage,
        STAGE_BUILD_EXAMPLES: run_build_examples_stage,
    }

    if stage == STAGE_ALL:
        return run_full_data_pipeline(config_path=config_path, params_path=params_path)
    if stage not in stage_runners:
        supported = ", ".join(STAGE_NAMES)
        raise ValueError(f"Unsupported stage '{stage}'. Supported stages: {supported}")

    return stage_runners[stage](config_path=config_path, params_path=params_path)
