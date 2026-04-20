"""Offline drift monitoring for benchmark replay windows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.monitoring.features import (
    MonitoringView,
    build_monitoring_view,
    inject_oov_items,
    load_interactions,
    load_item_vocab,
    resolve_interaction_columns,
    sample_sessions,
    shift_session_lengths,
)

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_CRITICAL = "critical"

NUMERICAL_DRIFT_FEATURES = [
    "session_length",
    "unique_items",
    "repeat_ratio",
    "duration_seconds",
    "oov_ratio",
]


def population_stability_index(
    reference: pd.Series | Sequence[float],
    current: pd.Series | Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI between two numeric distributions."""
    ref = _clean_numeric_array(reference)
    cur = _clean_numeric_array(current)
    if ref.size == 0 and cur.size == 0:
        return 0.0
    if ref.size == 0 or cur.size == 0:
        return 1.0

    edges = _psi_bin_edges(ref, cur, bins=bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = _safe_proportions(ref_counts)
    cur_pct = _safe_proportions(cur_counts)
    score = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(score, 0.0))


def jensen_shannon_divergence(
    reference_counts: dict[int, int],
    current_counts: dict[int, int],
) -> float:
    """Calculate Jensen-Shannon divergence for two count dictionaries."""
    if not reference_counts and not current_counts:
        return 0.0
    if not reference_counts or not current_counts:
        return 1.0

    keys = sorted(set(reference_counts) | set(current_counts))
    ref = np.asarray([reference_counts.get(key, 0) for key in keys], dtype=np.float64)
    cur = np.asarray([current_counts.get(key, 0) for key in keys], dtype=np.float64)
    ref = ref / max(ref.sum(), 1.0)
    cur = cur / max(cur.sum(), 1.0)
    midpoint = 0.5 * (ref + cur)
    score = 0.5 * _kl_divergence(ref, midpoint) + 0.5 * _kl_divergence(cur, midpoint)
    return float(max(score, 0.0))


def top_n_overlap(
    reference_counts: dict[int, int],
    current_counts: dict[int, int],
    *,
    top_n: int = 100,
) -> float:
    """Return top-N item overlap share between two windows."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    reference_top = _top_items(reference_counts, top_n)
    current_top = _top_items(current_counts, top_n)
    if not reference_top and not current_top:
        return 1.0
    denominator = max(1, min(top_n, len(reference_top), len(current_top)))
    return float(len(set(reference_top) & set(current_top)) / denominator)


def build_drift_report(
    *,
    reference_view: MonitoringView,
    current_view: MonitoringView,
    reference_path: str | Path,
    current_path: str | Path,
    top_n: int,
) -> dict[str, Any]:
    """Build a serializable drift report."""
    checks: dict[str, dict[str, Any]] = {}
    for feature in NUMERICAL_DRIFT_FEATURES:
        score = population_stability_index(
            reference_view.session_features[feature],
            current_view.session_features[feature],
        )
        checks[feature] = {
            "method": "psi",
            "score": score,
            "status": _status_for_high_score(score, warning=0.10, critical=0.25),
        }

    item_js = jensen_shannon_divergence(
        reference_view.item_counts,
        current_view.item_counts,
    )
    checks["item_popularity"] = {
        "method": "js_divergence",
        "score": item_js,
        "status": _status_for_high_score(item_js, warning=0.05, critical=0.15),
    }

    overlap = top_n_overlap(
        reference_view.item_counts,
        current_view.item_counts,
        top_n=top_n,
    )
    checks[f"top_{top_n}_item_overlap"] = {
        "method": "top_n_overlap",
        "score": overlap,
        "status": _status_for_low_score(overlap, warning=0.70, critical=0.50),
    }

    oov_ratio = _global_oov_ratio(current_view)
    checks["oov_ratio"] = {
        "method": "reference_vocab_coverage",
        "score": oov_ratio,
        "status": _status_for_high_score(oov_ratio, warning=0.05, critical=0.20),
    }

    status_counts = _status_counts(checks)
    overall_status = _overall_status(checks)
    return {
        "reference": {
            "path": str(reference_path),
            "rows": reference_view.total_interactions,
            "sessions": int(len(reference_view.session_features)),
            "unique_items": reference_view.unique_items,
        },
        "current": {
            "path": str(current_path),
            "rows": current_view.total_interactions,
            "sessions": int(len(current_view.session_features)),
            "unique_items": current_view.unique_items,
        },
        "status": overall_status,
        "summary": {
            "total_checks": len(checks),
            "ok_checks": status_counts[STATUS_OK],
            "warning_checks": status_counts[STATUS_WARNING],
            "critical_checks": status_counts[STATUS_CRITICAL],
            "oov_ratio": oov_ratio,
            f"top_{top_n}_item_overlap": overlap,
            "item_js_divergence": item_js,
        },
        "checks": checks,
    }


def run_drift_monitoring(
    *,
    reference_path: str | Path,
    current_path: str | Path,
    vocab_path: str | Path,
    output_path: str | Path,
    html_path: str | Path | None = None,
    top_n: int = 100,
    sample_size: int | None = None,
    random_seed: int = 42,
    inject_oov_rate: float = 0.0,
    inject_session_length_shift: float = 1.0,
) -> dict[str, Any]:
    """Run drift monitoring and persist JSON/optional HTML artifacts."""
    vocab_items = load_item_vocab(vocab_path)
    reference_df = load_interactions(reference_path)
    current_df = load_interactions(current_path)

    reference_cols = resolve_interaction_columns(reference_df)
    current_cols = resolve_interaction_columns(current_df)
    reference_df = sample_sessions(
        reference_df,
        columns=reference_cols,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    current_df = sample_sessions(
        current_df,
        columns=current_cols,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    current_df = shift_session_lengths(
        current_df,
        columns=current_cols,
        factor=inject_session_length_shift,
        random_seed=random_seed,
    )
    current_df = inject_oov_items(
        current_df,
        columns=current_cols,
        vocab_items=vocab_items,
        rate=inject_oov_rate,
        random_seed=random_seed,
    )

    reference_view = build_monitoring_view(
        reference_df,
        vocab_items=vocab_items,
        columns=reference_cols,
    )
    current_view = build_monitoring_view(
        current_df,
        vocab_items=vocab_items,
        columns=current_cols,
    )
    report = build_drift_report(
        reference_view=reference_view,
        current_view=current_view,
        reference_path=reference_path,
        current_path=current_path,
        top_n=top_n,
    )
    _write_json(report, output_path)

    if html_path:
        try:
            from recsys.monitoring.evidently_report import write_evidently_report

            write_evidently_report(
                reference_features=reference_view.session_features,
                current_features=current_view.session_features,
                output_path=html_path,
            )
        except Exception as exc:
            report["evidently_html_error"] = str(exc)
            _write_json(report, output_path)
            _write_html_fallback(html_path, str(exc))

    return report


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    report = run_drift_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        vocab_path=args.vocab,
        output_path=args.output,
        html_path=args.html,
        top_n=args.top_n,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        inject_oov_rate=args.inject_oov_rate,
        inject_session_length_shift=args.inject_session_length_shift,
    )
    print(
        {
            "output": str(args.output),
            "html": str(args.html) if args.html else None,
            "status": report["status"],
        }
    )
    if args.fail_on_critical and report["status"] == STATUS_CRITICAL:
        raise SystemExit(1)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline drift monitoring for benchmark replay windows."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--html", default=None, type=Path)
    parser.add_argument("--top-n", default=100, type=int)
    parser.add_argument(
        "--sample-size",
        default=None,
        type=int,
        help="Maximum number of sessions to sample from each window.",
    )
    parser.add_argument("--random-seed", default=42, type=int)
    parser.add_argument("--inject-oov-rate", default=0.0, type=float)
    parser.add_argument("--inject-session-length-shift", default=1.0, type=float)
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser.parse_args(argv)


def _clean_numeric_array(values: pd.Series | Sequence[float]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=np.float64
    )
    array = array[np.isfinite(array)]
    return array


def _psi_bin_edges(ref: np.ndarray, cur: np.ndarray, *, bins: int) -> np.ndarray:
    unique_ref = np.unique(ref)
    if unique_ref.size <= 1:
        center = float(unique_ref[0])
        span = max(0.5, abs(center) * 0.01, float(np.std(cur)))
        return np.asarray([-np.inf, center - span, center + span, np.inf])

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        low = min(float(ref.min()), float(cur.min()))
        high = max(float(ref.max()), float(cur.max()))
        if low == high:
            low -= 0.5
            high += 0.5
        edges = np.asarray([low, high])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _safe_proportions(counts: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    total = max(float(counts.sum()), 1.0)
    proportions = counts.astype(np.float64) / total
    return np.clip(proportions, eps, None)


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    mask = left > 0
    return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))


def _top_items(counts: dict[int, int], top_n: int) -> list[int]:
    return [
        item
        for item, _ in sorted(
            counts.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_n]
    ]


def _global_oov_ratio(view: MonitoringView) -> float:
    if view.total_interactions <= 0:
        return 0.0
    return float(view.oov_interactions / view.total_interactions)


def _status_for_high_score(score: float, *, warning: float, critical: float) -> str:
    if score >= critical:
        return STATUS_CRITICAL
    if score >= warning:
        return STATUS_WARNING
    return STATUS_OK


def _status_for_low_score(score: float, *, warning: float, critical: float) -> str:
    if score < critical:
        return STATUS_CRITICAL
    if score <= warning:
        return STATUS_WARNING
    return STATUS_OK


def _status_counts(checks: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        STATUS_OK: sum(1 for check in checks.values() if check["status"] == STATUS_OK),
        STATUS_WARNING: sum(
            1 for check in checks.values() if check["status"] == STATUS_WARNING
        ),
        STATUS_CRITICAL: sum(
            1 for check in checks.values() if check["status"] == STATUS_CRITICAL
        ),
    }


def _overall_status(checks: dict[str, dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks.values()}
    if STATUS_CRITICAL in statuses:
        return STATUS_CRITICAL
    if STATUS_WARNING in statuses:
        return STATUS_WARNING
    return STATUS_OK


def _write_json(payload: dict[str, Any], destination: str | Path) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _write_html_fallback(destination: str | Path, error: str) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_error = (
        error.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head><meta charset=\"utf-8\"><title>Drift Report</title></head>",
                "<body>",
                "<h1>Drift report visualization unavailable</h1>",
                "<p>The JSON drift report was generated successfully.</p>",
                f"<pre>{safe_error}</pre>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    main()
