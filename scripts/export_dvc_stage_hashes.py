#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stage_signature(stage: dict[str, Any]) -> dict[str, Any]:
    # Only include fields that should meaningfully represent stage state.
    sig: dict[str, Any] = {}

    cmd = stage.get("cmd")
    if cmd is not None:
        sig["cmd"] = " ".join(str(cmd).split())

    deps = stage.get("deps") or []
    norm_deps = []
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        item = {"path": dep.get("path")}
        if dep.get("hash") is not None:
            item["hash"] = dep.get("hash")
        if dep.get("md5") is not None:
            item["md5"] = dep.get("md5")
        if dep.get("etag") is not None:
            item["etag"] = dep.get("etag")
        if dep.get("size") is not None:
            item["size"] = dep.get("size")
        norm_deps.append(item)
    norm_deps.sort(key=lambda d: (str(d.get("path") or ""), str(d.get("md5") or ""), str(d.get("etag") or "")))
    sig["deps"] = norm_deps

    params = stage.get("params")
    if params is not None:
        sig["params"] = params

    outs = stage.get("outs") or []
    norm_outs = []
    for out in outs:
        if not isinstance(out, dict):
            continue
        item = {"path": out.get("path")}
        if out.get("hash") is not None:
            item["hash"] = out.get("hash")
        if out.get("md5") is not None:
            item["md5"] = out.get("md5")
        if out.get("etag") is not None:
            item["etag"] = out.get("etag")
        if out.get("size") is not None:
            item["size"] = out.get("size")
        if out.get("nfiles") is not None:
            item["nfiles"] = out.get("nfiles")
        norm_outs.append(item)
    norm_outs.sort(key=lambda d: (str(d.get("path") or ""), str(d.get("md5") or ""), str(d.get("etag") or "")))
    sig["outs"] = norm_outs

    return sig


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_matrix_id(stage_name: str) -> dict[str, str] | None:
    # Expect pattern: "<base>@<data_version>-<model_profile>"
    if "@" not in stage_name:
        return None
    base, rest = stage_name.split("@", 1)
    if "-" not in rest:
        return {"base": base, "id": rest}
    data_version, model_profile = rest.split("-", 1)
    return {"base": base, "id": rest, "data_version": data_version, "model_profile": model_profile}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export stable per-stage hashes from a DVC lockfile.")
    ap.add_argument("--lock", default="pipelines/training/dvc.lock", help="Path to dvc.lock (YAML).")
    ap.add_argument(
        "--out",
        default="reports/dvc_hashes/training_stage_hashes.json",
        help="Output JSON path (will be created).",
    )
    ap.add_argument(
        "--only-prefix",
        action="append",
        default=[],
        help="Include only stages whose name starts with this prefix. Can be repeated.",
    )
    ap.add_argument(
        "--include-matrix-meta",
        action="store_true",
        help="Add parsed data_version/model_profile metadata for matrix items.",
    )
    args = ap.parse_args()

    lock_path = Path(args.lock)
    out_path = Path(args.out)

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    stages: dict[str, Any] = lock.get("stages") or {}

    prefixes: list[str] = list(args.only_prefix or [])
    selected_names = sorted(stages.keys())
    if prefixes:
        selected_names = [n for n in selected_names if any(n.startswith(p) for p in prefixes)]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Any] = {}
    for name in selected_names:
        stage = stages.get(name) or {}
        sig = _stage_signature(stage)
        sig_json = _stable_json_dumps(sig)
        record: dict[str, Any] = {
            "stage_hash_sha256": _sha256_hex(sig_json),
            "signature": sig,
        }
        if args.include_matrix_meta:
            meta = _parse_matrix_id(name)
            if meta is not None:
                record["matrix"] = meta
        exported[name] = record

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lockfile": str(lock_path),
        "stage_count": len(exported),
        "stages": exported,
    }
    out_path.write_text(_stable_json_dumps(payload) + "\n", encoding="utf-8")

    print(f"Wrote {len(exported)} stage hashes to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

