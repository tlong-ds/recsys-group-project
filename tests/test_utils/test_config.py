"""Tests for config loading and params overlay precedence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from recsys.utils.config import (
    load_data_config_with_params,
    load_training_runtime_config,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestConfigOverlay(unittest.TestCase):
    def test_data_config_uses_only_experiment_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_cfg_path = root / "data.yaml"
            params_path = root / "params.yaml"

            _write_yaml(
                data_cfg_path,
                {
                    "data": {
                        "raw_path": "data/raw",
                        "validation": {"min_session_length": 2},
                        "ingest": {"item_views": "views.csv"},
                        "logging": {"report_path": "metrics/validation_report.json"},
                    }
                },
            )
            _write_yaml(
                params_path,
                {
                    "data": {
                        "validation": {"min_session_length": 5},
                        "raw_path": "alt/raw",
                        "ingest": {"item_views": "views_override.csv"},
                        "logging": {"report_path": "data/validation_report.json"},
                    },
                },
            )

            merged = load_data_config_with_params(data_cfg_path, params_path)

            self.assertEqual(merged["raw_path"], "data/raw")
            self.assertEqual(merged["validation"]["min_session_length"], 5)
            self.assertEqual(merged["ingest"]["item_views"], "views.csv")
            self.assertEqual(
                merged["logging"]["report_path"],
                "metrics/validation_report.json",
            )

    def test_training_runtime_config_ignores_runtime_metadata_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_cfg_path = root / "data.yaml"
            model_cfg_path = root / "model.yaml"
            training_cfg_path = root / "training.yaml"
            params_path = root / "params.yaml"

            _write_yaml(data_cfg_path, {"data": {"processed_path": "data/processed"}})
            _write_yaml(model_cfg_path, {"model": {"hidden_size": 128}})
            _write_yaml(
                training_cfg_path,
                {
                    "training": {"top_k": 20},
                    "mlflow": {"enabled": False},
                    "registry": {"root_path": "models/trained"},
                },
            )
            _write_yaml(
                params_path,
                {
                    "model": {"hidden_size": 64},
                    "training": {"top_k": 50},
                    "mlflow": {"enabled": True},
                    "registry": {"root_path": "models/other"},
                },
            )

            merged = load_training_runtime_config(
                data_config_path=data_cfg_path,
                model_config_path=model_cfg_path,
                training_config_path=training_cfg_path,
                params_path=params_path,
            )

            self.assertEqual(merged["model"]["hidden_size"], 64)
            self.assertEqual(merged["training"]["top_k"], 50)
            self.assertFalse(merged["mlflow"]["enabled"])
            self.assertEqual(merged["registry"]["root_path"], "models/trained")
