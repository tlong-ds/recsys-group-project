from __future__ import annotations

import pytest
import torch

from recsys.utils.device import resolve_torch_device


def test_resolve_torch_device_cpu_explicit() -> None:
    assert resolve_torch_device("cpu").type == "cpu"


def test_resolve_torch_device_auto_prefers_available_backends() -> None:
    device = resolve_torch_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}


def test_resolve_torch_device_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        resolve_torch_device("quantum")


def test_resolve_torch_device_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        return
    with pytest.raises(ValueError):
        resolve_torch_device("cuda")


def test_resolve_torch_device_rejects_unavailable_mps() -> None:
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    if mps_available:
        return
    with pytest.raises(ValueError):
        resolve_torch_device("mps")
