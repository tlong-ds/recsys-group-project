from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from recsys.training.mlflow_registry import register_model_version


def test_register_model_version_returns_none_when_disabled() -> None:
    result = register_model_version(
        config={"mlflow": {"registry": {"enabled": False}}},
        run_id="abc",
    )
    assert result is None


def test_register_model_version_registers_and_aliases() -> None:
    client = MagicMock()
    client.create_model_version.return_value = SimpleNamespace(version=3)

    with patch("recsys.training.mlflow_registry._mlflow_client", return_value=client):
        result = register_model_version(
            config={
                "mlflow": {
                    "registry": {
                        "enabled": True,
                        "model_name": "recsys-srgnn",
                        "register_alias": "Staging",
                    }
                }
            },
            run_id="run-123",
        )

    assert result is not None
    assert result["model_name"] == "recsys-srgnn"
    assert result["model_version"] == "3"
    client.create_model_version.assert_called_once()
    client.set_registered_model_alias.assert_called_once_with(
        name="recsys-srgnn",
        alias="Staging",
        version="3",
    )
