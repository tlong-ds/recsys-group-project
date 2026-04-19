"""Device selection helpers for torch-based training/inference."""

from __future__ import annotations

from typing import Any

import torch


def resolve_torch_device(device: Any = None) -> torch.device:
    """Resolve runtime device preference into a concrete ``torch.device``.

    Supported values:
    - ``None`` or ``"auto"``: use CUDA if available, then MPS, else CPU.
    - ``"cpu"``: force CPU.
    - ``"mps"``: force Apple Metal (errors if unavailable).
    - ``"cuda"`` / ``"cuda:N"``: force CUDA device (errors if unavailable).
    - ``torch.device``: returned as-is (with backend availability validation).
    """
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).strip().lower()
        if requested in ("", "auto"):
            if torch.cuda.is_available():
                resolved = torch.device("cuda")
            elif _mps_available():
                resolved = torch.device("mps")
            else:
                resolved = torch.device("cpu")
        elif requested == "cpu":
            resolved = torch.device("cpu")
        elif requested == "mps":
            resolved = torch.device("mps")
        elif requested.startswith("cuda"):
            resolved = torch.device(requested)
        else:
            raise ValueError(
                f"Unsupported device '{device}'. Use one of: auto, cpu, mps, cuda, cuda:N."
            )

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device '{resolved}' requested but CUDA is not available in this runtime."
        )
    if resolved.type == "mps" and not _mps_available():
        raise ValueError(
            "MPS device requested but MPS is not available in this runtime."
        )
    return resolved


def _mps_available() -> bool:
    return bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
