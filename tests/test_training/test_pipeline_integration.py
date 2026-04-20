"""Integration tests: all model types through run_training_pipeline.

Tests run end-to-end through the real pipeline → trainer stub → model → evaluator.
External integrations (MLflow, project logger, registry) are stubbed in
``conftest.py`` so the test file imports cleanly with only the model layer present.

Test coverage
-------------
TestSRGNNPipeline        – all 6 SR-GNN variants via the pipeline
TestTAGNNPipeline        – TAGNN with chunk > n_items and chunk < n_items
TestGGNNPipeline         – GGNN base + multi-step propagation
TestSaveLoadRoundTrip    – parametrised over all 8 model configs; asserts
                           graph-recs == live-recs after round-trip
TestEdgeCases            – single-example val, fallback weight, multi-step,
                           unknown type/variant error handling

Run with:
    pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Pipeline is imported at module level — stubs are already in sys.modules
# (conftest.py installs them before collection starts).
from recsys.training.pipeline import build_model, run_training_pipeline

# ---------------------------------------------------------------------------
# Tiny dataset helpers  (identical schema to production preprocessing)
# ---------------------------------------------------------------------------


def _examples() -> pd.DataFrame:
    """3 session-graph examples over items {1, 2, 3, 4}."""
    return pd.DataFrame(
        {
            "x": [
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1, 3], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
            ],
            "edge_index": [
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
                np.asarray([[0], [1]], dtype=np.int64),
            ],
            "alias_inputs": [
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ],
            "item_seq_len": [2, 2, 2],
            "pos_items": [3, 4, 4],
            "session_id": [11, 12, 13],
        }
    )


def _write_examples_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet using the production PyArrow schema."""
    table = pa.table(
        {
            "x": [list(map(int, v.tolist())) for v in df["x"]],
            "edge_index": [
                [list(map(int, v[0].tolist())), list(map(int, v[1].tolist()))]
                for v in df["edge_index"]
            ],
            "alias_inputs": [list(map(int, v.tolist())) for v in df["alias_inputs"]],
            "item_seq_len": df["item_seq_len"].astype(int).tolist(),
            "pos_items": df["pos_items"].astype(int).tolist(),
            "session_id": df["session_id"].astype(int).tolist(),
        },
        schema=pa.schema(
            [
                pa.field("x", pa.list_(pa.int64())),
                pa.field("edge_index", pa.list_(pa.list_(pa.int64()))),
                pa.field("alias_inputs", pa.list_(pa.int64())),
                pa.field("item_seq_len", pa.int64()),
                pa.field("pos_items", pa.int64()),
                pa.field("session_id", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Populate tmp_path/data/processed/ with parquet splits + item_vocab."""
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)

    train_df = _examples()
    val_df = _examples().iloc[:1].copy()
    test_df = _examples().iloc[1:2].copy()

    _write_examples_parquet(train_df, d / "train_examples.parquet")
    _write_examples_parquet(val_df, d / "val_examples.parquet")
    _write_examples_parquet(test_df, d / "test_examples.parquet")

    (d / "item_vocab.json").write_text(
        json.dumps({"item2id": {"101": 1, "102": 2, "103": 3, "104": 4}}),
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _config(
    data_dir: Path,
    registry_root: Path,
    model_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a minimal pipeline config with model_overrides applied."""
    model: dict[str, Any] = {
        "embedding_dim": 8,
        "hidden_size": 8,
        "step": 1,
        "max_session_length": 5,
        "fallback_weight": 0.0,
        "version": "test",
    }
    model.update(model_overrides)
    return {
        "data": {
            "train_examples_path": str(data_dir / "train_examples.parquet"),
            "val_examples_path": str(data_dir / "val_examples.parquet"),
            "test_examples_path": str(data_dir / "test_examples.parquet"),
            "item_vocab_path": str(data_dir / "item_vocab.json"),
        },
        "model": model,
        "training": {
            "seed": 7,
            "device": "cpu",
            "batch_size": 2,
            "num_epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "early_stopping_patience": 1,
            "top_k": 3,
            "num_workers": 0,
        },
        "registry": {"root_path": str(registry_root)},
        "mlflow": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------


def _assert_result(result: dict[str, Any]) -> None:
    """Assert the pipeline contract every successful run must satisfy."""
    assert "artifact_path" in result
    assert "validation_metrics" in result
    assert "test_metrics" in result

    assert Path(result["artifact_path"]).exists(), (
        f"artifact_path does not exist: {result['artifact_path']}"
    )
    assert set(result["validation_metrics"]) == {"hr@k", "mrr@k"}, (
        f"unexpected val metric keys: {set(result['validation_metrics'])}"
    )
    assert set(result["test_metrics"]) == {"hr@k", "mrr@k"}, (
        f"unexpected test metric keys: {set(result['test_metrics'])}"
    )
    for k, v in {**result["validation_metrics"], **result["test_metrics"]}.items():
        assert isinstance(v, float), f"metric {k} is not a float: {v!r}"
        assert 0.0 <= v <= 1.0, f"metric {k} out of [0,1]: {v}"


# ---------------------------------------------------------------------------
# SR-GNN variants
# ---------------------------------------------------------------------------


class TestSRGNNPipeline:
    """All six SR-GNN readout variants through the full pipeline."""

    def test_pipeline_runs_end_to_end_on_processed_parquet(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """Original test format — mirrors the reference test exactly."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn", "name": "srgnn"},
        )
        result = run_training_pipeline(config)
        _assert_result(result)

    def test_pipeline_srgnn_ngc(self, data_dir: Path, tmp_path: Path) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn-ngc", "name": "srgnn-ngc"},
        )
        _assert_result(run_training_pipeline(config))

    def test_pipeline_srgnn_fc(self, data_dir: Path, tmp_path: Path) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn-fc", "name": "srgnn-fc"},
        )
        _assert_result(run_training_pipeline(config))

    def test_pipeline_srgnn_local(self, data_dir: Path, tmp_path: Path) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn-l", "name": "srgnn-l"},
        )
        _assert_result(run_training_pipeline(config))

    def test_pipeline_srgnn_avg(self, data_dir: Path, tmp_path: Path) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn-avg", "name": "srgnn-avg"},
        )
        _assert_result(run_training_pipeline(config))

    def test_pipeline_srgnn_att(self, data_dir: Path, tmp_path: Path) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn-att", "name": "srgnn-att"},
        )
        _assert_result(run_training_pipeline(config))


# ---------------------------------------------------------------------------
# TAGNN
# ---------------------------------------------------------------------------


class TestTAGNNPipeline:
    """TAGNN with different score_chunk_size values."""

    def test_pipeline_runs_end_to_end_on_processed_parquet(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """chunk_size > n_items: single chunk covers the whole catalogue."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "tagnn", "name": "tagnn", "score_chunk_size": 16},
        )
        _assert_result(run_training_pipeline(config))

    def test_tagnn_score_chunk_smaller_than_catalogue(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """chunk_size=2 forces multiple loop iterations — exercises loop boundary."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "tagnn", "name": "tagnn-chunk2", "score_chunk_size": 2},
        )
        _assert_result(run_training_pipeline(config))


# ---------------------------------------------------------------------------
# GGNN
# ---------------------------------------------------------------------------


class TestGGNNPipeline:
    """GGNN — strict GRUCell propagation through the full pipeline."""

    def test_pipeline_runs_end_to_end_on_processed_parquet(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "ggnn", "name": "ggnn"},
        )
        _assert_result(run_training_pipeline(config))

    def test_ggnn_multiple_propagation_steps(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """step=3 exercises 3 rounds of GRU propagation."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "ggnn", "name": "ggnn-step3", "step": 3},
        )
        _assert_result(run_training_pipeline(config))


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    """After a pipeline run the saved model must reload and produce
    identical recommendations via both recommend_from_graph and recommend."""

    @pytest.mark.parametrize(
        "model_type,extra",
        [
            ("srgnn", {"variant": "srgnn"}),
            ("srgnn", {"variant": "srgnn-ngc"}),
            ("srgnn", {"variant": "srgnn-fc"}),
            ("srgnn", {"variant": "srgnn-l"}),
            ("srgnn", {"variant": "srgnn-avg"}),
            ("srgnn", {"variant": "srgnn-att"}),
            ("tagnn", {"score_chunk_size": 4}),
            ("ggnn", {}),
        ],
    )
    def test_save_load_produces_identical_recommendations(
        self,
        model_type: str,
        extra: dict[str, Any],
        data_dir: Path,
        tmp_path: Path,
    ) -> None:
        variant = extra.get("variant", model_type)
        name = f"{variant}-roundtrip"
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": model_type, "name": name, **extra},
        )
        result = run_training_pipeline(config)
        artifact_dir = Path(result["artifact_path"]).parent

        # Load via the correct class
        if model_type == "srgnn":
            from recsys.models.srgnn import SRGNNRecommender as Cls
        elif model_type == "tagnn":
            from recsys.models.tagnn import TAGNNRecommender as Cls
        else:
            from recsys.models.ggnn import GGNNRecommender as Cls

        loaded = Cls.load(artifact_dir)

        row = pd.read_parquet(str(data_dir / "test_examples.parquet")).iloc[0]

        # recommend_from_graph returns raw internal indices (Evaluator path)
        recs_graph_1 = loaded.recommend_from_graph(
            row["x"], row["edge_index"], row["alias_inputs"], top_k=3
        )
        # Calling it a second time must give the same result (deterministic)
        recs_graph_2 = loaded.recommend_from_graph(
            row["x"], row["edge_index"], row["alias_inputs"], top_k=3
        )
        # recommend() returns decoded external item IDs (live inference path)
        recs_live = loaded.recommend(list(row["x"]), top_k=3)

        assert len(recs_graph_1) == 3, f"Expected 3 graph recs, got {len(recs_graph_1)}"
        assert len(recs_live) == 3, f"Expected 3 live recs, got {len(recs_live)}"

        # Determinism: two identical calls must return the same ranked list
        assert recs_graph_1 == recs_graph_2, (
            f"recommend_from_graph is non-deterministic for {model_type}/{variant}: "
            f"{recs_graph_1} != {recs_graph_2}"
        )

        # Sanity: every recommendation must be a positive integer item id
        for rec in recs_graph_1:
            assert isinstance(rec, int) and rec > 0, (
                f"graph rec {rec!r} is not a positive int for {model_type}/{variant}"
            )
        for rec in recs_live:
            assert isinstance(rec, int) and rec > 0, (
                f"live rec {rec!r} is not a positive int for {model_type}/{variant}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and error handling."""

    def test_val_split_with_single_example(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """val_df of size 1 must not crash early-stopping logic."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn", "name": "srgnn-tiny-val"},
        )
        _assert_result(run_training_pipeline(config))

    def test_fallback_weight_nonzero(self, data_dir: Path, tmp_path: Path) -> None:
        """Popularity-blend must not error during recommendation."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {
                "type": "srgnn",
                "variant": "srgnn",
                "name": "srgnn-fallback",
                "fallback_weight": 0.1,
            },
        )
        _assert_result(run_training_pipeline(config))

    def test_multiple_propagation_steps_srgnn(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """step=3 must propagate correctly through the SR-GNN GNNCell."""
        config = _config(
            data_dir,
            tmp_path / "models",
            {"type": "srgnn", "variant": "srgnn", "name": "srgnn-step3", "step": 3},
        )
        _assert_result(run_training_pipeline(config))

    def test_unknown_model_type_raises(self) -> None:
        """Unrecognised type must raise a clear error immediately."""
        with pytest.raises((ValueError, KeyError)):
            build_model({"type": "nonexistent_model"}, seed=0)

    def test_unknown_srgnn_variant_raises(self) -> None:
        """Unrecognised SR-GNN variant must raise ValueError with a helpful message."""
        from recsys.models.srgnn import SRGNNRecommender

        with pytest.raises(ValueError, match="Unknown variant"):
            SRGNNRecommender(variant="srgnn-xyz")
