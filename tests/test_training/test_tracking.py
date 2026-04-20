"""Tests for optional MLflow and DagsHub tracking integration."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from recsys.training.tracking import (
    configure_system_metrics,
    configure_tracking,
    dagshub_repo,
    log_evaluation_run,
    system_metrics_run_override,
)


class TestTracking(unittest.TestCase):
    def test_dagshub_repo_uses_defaults(self) -> None:
        owner, name = dagshub_repo({"mlflow": {"dagshub": {}}})
        self.assertEqual(owner, "lytlong.pers")
        self.assertEqual(name, "recsys-group-project")

    def test_configure_tracking_initializes_dagshub_when_enabled(self) -> None:
        config = {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "http://mlflow:5000",
                "experiment_name": "recsys",
                "dagshub": {
                    "enabled": True,
                    "repo_owner": "owner",
                    "repo_name": "repo",
                },
            }
        }
        fake_dagshub = SimpleNamespace(init=MagicMock())

        with patch("recsys.training.tracking.mlflow.set_experiment") as set_experiment:
            with patch(
                "recsys.training.tracking.mlflow.set_tracking_uri"
            ) as set_tracking_uri:
                with patch.dict("sys.modules", {"dagshub": fake_dagshub}):
                    configure_tracking(config)

        fake_dagshub.init.assert_called_once_with(
            repo_owner="owner",
            repo_name="repo",
            mlflow=True,
        )
        set_tracking_uri.assert_not_called()
        set_experiment.assert_called_once_with("recsys")

    def test_log_evaluation_run_noops_when_tracking_disabled(self) -> None:
        with patch("recsys.training.tracking.mlflow.start_run") as start_run:
            log_evaluation_run(
                config={"mlflow": {"enabled": False}},
                metrics={"hr@k": 0.3},
                model_path=Path("models/trained/latest/model.json"),
            )
        start_run.assert_not_called()

    def test_configure_tracking_maps_dagshub_token_to_mlflow_and_sdk_auth(self) -> None:
        config = {
            "mlflow": {
                "enabled": True,
                "experiment_name": "recsys",
                "dagshub": {
                    "enabled": True,
                    "repo_owner": "owner",
                    "repo_name": "repo",
                    "token_env_var": "DAGSHUB_USER_TOKEN",
                },
            }
        }
        fake_dagshub = SimpleNamespace(init=MagicMock())
        previous_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        previous_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

        try:
            with patch.dict(os.environ, {"DAGSHUB_USER_TOKEN": "abc123"}, clear=True):
                with patch("recsys.training.tracking.mlflow.set_experiment"):
                    with patch.dict("sys.modules", {"dagshub": fake_dagshub}):
                        configure_tracking(config)
                self.assertEqual(os.environ.get("MLFLOW_TRACKING_PASSWORD"), "abc123")
                self.assertEqual(os.environ.get("DAGSHUB_USER_TOKEN"), "abc123")
        finally:
            if previous_username is None:
                os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
            else:
                os.environ["MLFLOW_TRACKING_USERNAME"] = previous_username
            if previous_password is None:
                os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
            else:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = previous_password

    def test_configure_system_metrics_applies_settings(self) -> None:
        config = {
            "mlflow": {
                "system_metrics": {
                    "enabled": True,
                    "sampling_interval": 3,
                    "samples_before_logging": 2,
                }
            }
        }
        with patch(
            "recsys.training.tracking.mlflow.enable_system_metrics_logging"
        ) as enable_metrics:
            with patch(
                "recsys.training.tracking.mlflow.set_system_metrics_sampling_interval"
            ) as set_interval:
                with patch(
                    "recsys.training.tracking.mlflow.set_system_metrics_samples_before_logging"
                ) as set_samples:
                    configure_system_metrics(config)

        enable_metrics.assert_called_once()
        set_interval.assert_called_once_with(3)
        set_samples.assert_called_once_with(2)

    def test_system_metrics_run_override_uses_config_flag(self) -> None:
        self.assertTrue(
            system_metrics_run_override(
                {"mlflow": {"system_metrics": {"enabled": True}}}
            )
        )
        self.assertIsNone(system_metrics_run_override({"mlflow": {}}))
