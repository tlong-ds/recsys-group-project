"""Main data processing pipeline orchestrating all steps."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd
import yaml
from loguru import logger

from recsys.data.ingest import DataIngestor
from recsys.data.preprocessor import SessionPreprocessor
from recsys.data.splitter import (
    split_by_days,
    split_by_session_time,
    split_by_time,
    split_diginetica_legacy,
)
from recsys.data.training_examples import ItemVocabBuilder, TrainingExampleBuilder
from recsys.data.validator import InteractionValidator

if TYPE_CHECKING:
    from typing import Any


class DataProcessingPipeline:
    """Complete pipeline: ingest → validate → preprocess → split → build examples."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file. Defaults to configs/data_config.yaml.
        """
        if config_path is None:
            config_path = Path("configs/data_config.yaml")
        
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_logging()

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.config = self._default_config()
        else:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Loaded config from {self.config_path}")

    def setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = self.config.get("data", {}).get("logging", {}).get("level", "INFO")
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_level,
            format="<level>[{level}]</level> {message}",
        )

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Return default configuration."""
        return {
            "data": {
                "raw_path": "data/raw",
                "interim_path": "data/interim",
                "processed_path": "data/processed",
                "interactions_file": "train-item-views.csv",
                "columns": {
                    "session_id": "session_id",
                    "item_id": "item_id",
                    "event_date": "eventdate",
                    "timeframe": "timeframe",
                },
                "csv_params": {"sep": ";", "encoding": "utf-8"},
                "validation": {
                    "min_session_length": 2,
                    "min_item_frequency": 5,
                    "allow_duplicates": True,
                },
                "temporal_split": {
                    "strategy": "diginetica_legacy",
                    "test_days": 7,
                    "val_days": 7,
                },
                "training_example_format": "graph",
                "compatibility": {
                    "vocab_order": "first_seen",
                    "example_order": "reverse",
                },
                "output_format": "parquet",
                "random_seed": 42,
                "logging": {"level": "INFO", "report_stats": True},
            }
        }

    def _get_paths(self) -> dict[str, Path]:
        """Create necessary directories and return path mapping."""
        data_cfg = self.config["data"]
        
        paths = {
            "raw": Path(data_cfg.get("raw_path", "data/raw")),
            "interim": Path(data_cfg.get("interim_path", "data/interim")),
            "processed": Path(data_cfg.get("processed_path", "data/processed")),
        }
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return paths

    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline.
        
        Returns:
            Dictionary with output file paths for each stage.
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("🚀 Starting Data Processing Pipeline")
        logger.info("=" * 80)
        
        paths = self._get_paths()
        outputs = {}
        
        try:
            # Step 1: Ingest
            logger.info("\n[STEP 1/6] Ingest raw data...")
            raw_interactions = self._ingest(paths["raw"])
            raw_interactions = self._canonicalize_for_dataset(raw_interactions)
            
            # Step 2: Validate
            logger.info("\n[STEP 2/6] Validate schema...")
            validation_report = self._validate(raw_interactions)
            outputs["validation_report"] = validation_report
            
            # Step 3: Preprocess
            logger.info("\n[STEP 3/6] Preprocess interactions...")
            clean_interactions = self._preprocess(raw_interactions)
            clean_path = paths["interim"] / "clean_interactions.parquet"
            self._save_parquet(clean_interactions, clean_path)
            outputs["clean_interactions"] = clean_path
            
            # Step 4: Split
            logger.info("\n[STEP 4/6] Temporal split...")
            train_df, val_df, test_df = self._split(clean_interactions)
            
            # Step 5: Build examples
            logger.info("\n[STEP 5/6] Build training examples...")
            (
                train_examples,
                val_examples,
                test_examples,
                vocab,
                stats,
            ) = self._build_examples(train_df, val_df, test_df)
            
            # Save training examples
            train_path = paths["processed"] / "train_examples.parquet"
            val_path = paths["processed"] / "val_examples.parquet"
            test_path = paths["processed"] / "test_examples.parquet"
            
            self._save_parquet(train_examples, train_path)
            self._save_parquet(val_examples, val_path)
            self._save_parquet(test_examples, test_path)
            
            outputs["train_examples"] = train_path
            outputs["val_examples"] = val_path
            outputs["test_examples"] = test_path
            
            # Step 6: Save vocab and stats
            logger.info("\n[STEP 6/6] Save vocabulary and statistics...")
            vocab_path = paths["processed"] / "item_vocab.json"
            stats_path = paths["processed"] / "data_stats.json"
            
            vocab.save(vocab_path)
            self._save_stats(stats, stats_path)
            
            outputs["vocab"] = vocab_path
            outputs["stats"] = stats_path
            
            # Summary
            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ Pipeline completed successfully in {elapsed:.1f}s")
            logger.info("=" * 80)
            logger.info("\n📊 Output files:")
            for name, path in outputs.items():
                if not isinstance(path, dict):
                    logger.info(f"  {name:25} → {path}")
            
            return outputs
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"\n❌ Pipeline failed after {elapsed:.1f}s: {e}", exc_info=True)
            raise

    def _ingest(self, raw_path: Path) -> pd.DataFrame:
        """Ingest raw CSV data."""
        data_cfg = self.config["data"]
        ingestor = DataIngestor(
            raw_path=raw_path,
            sep=data_cfg["csv_params"].get("sep", ";"),
            encoding=data_cfg["csv_params"].get("encoding", "utf-8"),
        )
        
        filename = data_cfg.get("interactions_file", "train-item-views.csv")
        df = ingestor.read_interactions(filename)
        
        return df

    def _validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data schema and semantics."""
        data_cfg = self.config["data"]
        col_map = data_cfg["columns"]

        session_col = self._resolve_column_name(df, col_map, "session_id", "sessionId")
        item_col = self._resolve_column_name(df, col_map, "item_id", "itemId")
        # Use event_date as the canonical time column for Diginetica pipelines.
        event_date_col = self._resolve_column_name(df, col_map, "event_date", "eventdate")

        validator = InteractionValidator(
            session_col=session_col,
            item_col=item_col,
            timestamp_col=event_date_col,
        )
        
        val_cfg = data_cfg["validation"]
        report = validator.generate_report(
            df,
            min_session_length=val_cfg.get("min_session_length", 1),
            max_session_length=val_cfg.get("max_session_length"),
            allow_duplicates=val_cfg.get("allow_duplicates", False),
        )
        
        if not report["valid"]:
            logger.warning("⚠️  Validation warnings present (will continue)")
            semantic_issues = report.get("semantics", {}).get("issues", [])
            if semantic_issues:
                logger.warning("Semantic validation issues:")
                for issue in semantic_issues:
                    logger.warning(f"  - {issue}")
        
        return report

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        data_cfg = self.config["data"]
        col_map = data_cfg["columns"]
        val_cfg = data_cfg["validation"]

        session_col = self._resolve_column_name(df, col_map, "session_id", "sessionId")
        item_col = self._resolve_column_name(df, col_map, "item_id", "itemId")
        event_date_col = self._resolve_column_name(df, col_map, "event_date", "eventdate")
        timeframe_col = self._resolve_column_name(df, col_map, "timeframe", "timeframe")
        
        preprocessor = SessionPreprocessor(
            session_col=session_col,
            item_col=item_col,
            timestamp_col=event_date_col,
            event_date_col=event_date_col,
            timeframe_col=timeframe_col,
        )
        
        # Step 1: Basic cleaning (nulls, type conversion, sorting)
        cleaned = preprocessor.transform(df)
        
        # Step 2: Filter sessions by length
        filtered = preprocessor.filter_sessions(
            cleaned,
            min_length=val_cfg.get("min_session_length", 2),
            max_length=val_cfg.get("max_session_length"),
        )
        
        # Step 3: Filter items by frequency
        filtered, _ = preprocessor.filter_items(
            filtered,
            min_frequency=val_cfg.get("min_item_frequency", 5),
        )

        # Original SR-GNN idea: after item-frequency filtering, drop short sessions again.
        filtered = preprocessor.filter_sessions(
            filtered,
            min_length=val_cfg.get("min_session_length", 2),
            max_length=val_cfg.get("max_session_length"),
        )
        
        # Step 4: Remove duplicate items in the same session
        if not val_cfg.get("allow_duplicates", False):
            filtered = preprocessor.remove_duplicate_items_in_session(filtered)
        
        logger.info(f"Final clean data: {len(filtered):,} rows")
        
        return filtered

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test."""
        data_cfg = self.config["data"]
        col_map = data_cfg["columns"]
        split_cfg = data_cfg["temporal_split"]
        session_col = self._resolve_column_name(df, col_map, "session_id", "sessionId")
        event_date_col = self._resolve_column_name(df, col_map, "event_date", "eventdate")
        
        strategy = split_cfg.get("strategy", "time_based")
        
        if strategy == "ratio_based":
            train_df, val_df, test_df = split_by_time(
                df,
                timestamp_col=event_date_col,
                val_ratio=split_cfg.get("val_ratio", 0.15),
                test_ratio=split_cfg.get("test_ratio", 0.15),
            )
        elif strategy == "diginetica_legacy":
            train_df, val_df, test_df = split_diginetica_legacy(
                df,
                session_col=session_col,
                timestamp_col=event_date_col,
                test_days=split_cfg.get("test_days", 7),
                val_days=split_cfg.get("val_days", 7),
            )
        elif strategy == "session_based":
            train_df, val_df, test_df = split_by_session_time(
                df,
                session_col=session_col,
                timestamp_col=event_date_col,
                test_days=split_cfg.get("test_days", 7),
                val_days=split_cfg.get("val_days", 7),
            )
        else:  # time_based (default when no have strategy in config)
            train_df, val_df, test_df = split_by_days(
                df,
                timestamp_col=event_date_col,
                test_days=split_cfg.get("test_days", 7),
                val_days=split_cfg.get("val_days", 7),
            )
        
        return train_df, val_df, test_df

    def _build_examples(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ItemVocabBuilder, dict[str, Any]]:
        """Build training examples and vocabulary."""
        data_cfg = self.config["data"]
        col_map = data_cfg["columns"]
        compat_cfg = data_cfg.get("compatibility", {})
        output_format = data_cfg.get("training_example_format", "graph")

        session_col = self._resolve_column_name(train_df, col_map, "session_id", "sessionId")
        item_col = self._resolve_column_name(train_df, col_map, "item_id", "itemId")
        event_date_col = self._resolve_column_name(train_df, col_map, "event_date", "eventdate")
        
        # Build vocabulary from training data
        vocab_builder = ItemVocabBuilder(item_col=item_col)
        vocab_builder.build_from_interactions(
            train_df,
            sort_by=compat_cfg.get("vocab_order", "frequency"),
            timestamp_col=event_date_col,
            session_col=session_col,
        )
        
        # Build examples for each split
        example_builder = TrainingExampleBuilder(
            session_col=session_col,
            item_col=item_col,
            timestamp_col=event_date_col,
        )

        if output_format == "graph":
            train_examples = example_builder.build_graph_examples(
                train_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=False,
            )
            val_examples = example_builder.build_graph_examples(
                val_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=True,
            )
            test_examples = example_builder.build_graph_examples(
                test_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=True,
            )
        else:
            train_examples = example_builder.build_examples(
                train_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=False,
            )
            val_examples = example_builder.build_examples(
                val_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=True,
            )
            test_examples = example_builder.build_examples(
                test_df,
                vocab=vocab_builder.item2id,
                sequence_order=compat_cfg.get("example_order", "forward"),
                drop_unknown_items=True,
            )
        
        # Compute statistics
        stats = {
            "build_date": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "train": {
                "interactions": TrainingExampleBuilder.compute_stats(train_df, session_col),
                "examples": len(train_examples),
            },
            "val": {
                "interactions": TrainingExampleBuilder.compute_stats(val_df, session_col),
                "examples": len(val_examples),
            },
            "test": {
                "interactions": TrainingExampleBuilder.compute_stats(test_df, session_col),
                "examples": len(test_examples),
            },
            "vocab_size": len(vocab_builder.item2id),
        }
        
        return train_examples, val_examples, test_examples, vocab_builder, stats

    def _canonicalize_for_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create canonical columns needed by the pipeline across dataset schemas."""
        data_cfg = self.config["data"]
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
        if timeframe_col and timeframe_col not in canonical.columns and "timeframe" in canonical.columns:
            canonical[timeframe_col] = canonical["timeframe"]

        return canonical

    @staticmethod
    def _resolve_column_name(
        df: pd.DataFrame,
        col_map: dict[str, Any],
        key: str,
        fallback: str,
    ) -> str:
        """Resolve configured column names with robust fallback to actual DataFrame columns."""
        configured = col_map.get(key, fallback)
        if configured in df.columns:
            return configured
        if fallback in df.columns:
            return fallback
        return configured

    def _save_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to Parquet file."""
        data_cfg = self.config["data"]
        compression = data_cfg.get("parquet_compression", "snappy")
        
        df.to_parquet(path, compression=compression, index=False)
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Saved to {path.name} ({size_mb:.2f} MB)")

    def _save_stats(self, stats: dict[str, Any], path: Path) -> None:
        """Save statistics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Statistics saved to {path.name}")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="configs/data_config.yaml",
    help="Path to configuration file.",
)
def main(config: str) -> None:
    """Run the data processing pipeline."""
    pipeline = DataProcessingPipeline(config_path=config)
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
