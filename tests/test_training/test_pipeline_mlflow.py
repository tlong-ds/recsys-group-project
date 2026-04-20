from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from recsys.training.pipeline import (
    _log_config_to_mlflow,
    _mlflow_pt2_output_example,
    _mlflow_pt2_signature,
)


class _DummyCore(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def test_mlflow_pt2_output_example_from_model_core() -> None:
    model = SimpleNamespace(_core=_DummyCore())
    input_example = (torch.ones((1, 3), dtype=torch.float32),)

    output_example = _mlflow_pt2_output_example(model, input_example)

    assert len(output_example) == 1
    assert torch.equal(output_example[0], torch.full((1, 3), 2.0))


def test_mlflow_pt2_signature_includes_inputs_and_outputs() -> None:
    input_example = (torch.ones((1, 2), dtype=torch.float32),)
    output_example = (torch.ones((1, 4), dtype=torch.float32),)

    signature = _mlflow_pt2_signature(input_example, output_example)
    payload = signature.to_dict()
    inputs = json.loads(payload["inputs"])
    outputs = json.loads(payload["outputs"])

    assert inputs[0]["name"] == "input_0"
    assert outputs[0]["name"] == "output_0"


def test_log_config_to_mlflow_logs_hyperparameters(monkeypatch) -> None:
    mlflow = MagicMock()
    monkeypatch.setattr("recsys.training.pipeline._get_mlflow", lambda: mlflow)

    _log_config_to_mlflow(
        {
            "model": {"embedding_dim": 128, "hidden_size": 128},
            "training": {"lr": 1e-3, "num_epochs": 5},
            "mlflow": {"enabled": True},
        }
    )

    mlflow.log_param.assert_any_call("model.embedding_dim", "128")
    mlflow.log_param.assert_any_call("model.hidden_size", "128")
    mlflow.log_param.assert_any_call("training.lr", "0.001")
    mlflow.log_param.assert_any_call("training.num_epochs", "5")
