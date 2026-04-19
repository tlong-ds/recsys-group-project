"""Aggregate per-version training and evaluation metrics into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_comparison_report(
    specs: list[tuple[str, str | Path, str | Path]],
) -> dict[str, Any]:
    """Build a consolidated comparison report from version metric files."""
    versions: list[dict[str, Any]] = []
    for requested_name, train_metrics_path, eval_metrics_path in specs:
        train_payload = _read_json(train_metrics_path)
        eval_payload = _read_json(eval_metrics_path)
        versions.append(
            _build_version_entry(
                requested_name=requested_name,
                train_payload=train_payload,
                eval_payload=eval_payload,
                train_metrics_path=train_metrics_path,
                eval_metrics_path=eval_metrics_path,
            )
        )

    return {"versions": versions}


def write_comparison_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist the comparison payload to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _build_version_entry(
    *,
    requested_name: str,
    train_payload: dict[str, Any],
    eval_payload: dict[str, Any],
    train_metrics_path: str | Path,
    eval_metrics_path: str | Path,
) -> dict[str, Any]:
    train_version = _string_or_none(train_payload.get("data_version"))
    eval_version = _string_or_none(eval_payload.get("data_version"))
    if train_version and eval_version and train_version != eval_version:
        raise ValueError(
            "Mismatched data_version between training and evaluation metrics: "
            f"{train_version!r} != {eval_version!r}"
        )

    version_name = eval_version or train_version or requested_name
    data_params_path = (
        _string_or_none(eval_payload.get("data_params_path"))
        or _string_or_none(train_payload.get("data_params_path"))
    )

    return {
        "name": version_name,
        "data_version": version_name,
        "data_params_path": data_params_path,
        "artifact_path": _string_or_none(train_payload.get("artifact_path")),
        "model_path": _string_or_none(eval_payload.get("model_path")),
        "validation_metrics": _dict_or_empty(train_payload.get("validation_metrics")),
        "test_metrics": _dict_or_empty(eval_payload.get("test_metrics")),
        "training_metrics_path": str(train_metrics_path),
        "evaluation_metrics_path": str(eval_metrics_path),
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics payload must be a mapping: {path}")
    return payload


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _parse_spec(raw: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Each --spec must be 'version_name,train_metrics_path,evaluation_metrics_path'."
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-version training and evaluation metrics."
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        type=_parse_spec,
        help="version_name,train_metrics_path,evaluation_metrics_path",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_comparison_report(args.spec)
    output_path = write_comparison_report(payload, args.output)
    print({"output_path": str(output_path), "versions": len(payload["versions"])})


if __name__ == "__main__":
    main()
