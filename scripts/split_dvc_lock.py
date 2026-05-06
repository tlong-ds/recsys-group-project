#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT_LOCK = Path("dvc.lock")
PIPELINE_FILES = [
    Path("pipelines/data/dvc.yaml"),
    Path("pipelines/training/dvc.yaml"),
    Path("pipelines/monitoring/dvc.yaml"),
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def normalize_cmd(cmd: str) -> str:
    return " ".join(str(cmd).split())


def collect_stage_entries(root_stages: dict, pipeline_file: Path) -> tuple[dict, list[str]]:
    pipeline_doc = load_yaml(pipeline_file)
    pipeline_stages = pipeline_doc.get("stages", {})

    collected: dict[str, dict] = {}
    missing: list[str] = []

    for stage_name, stage_def in pipeline_stages.items():
        if stage_name in root_stages:
            root_stage = root_stages[stage_name]
            if normalize_cmd(stage_def.get("cmd", "")) == normalize_cmd(root_stage.get("cmd", "")):
                collected[stage_name] = root_stage
                continue

        if "matrix" in stage_def:
            matched = {
                root_name: root_entry
                for root_name, root_entry in root_stages.items()
                if root_name.startswith(f"{stage_name}@")
            }
            if matched:
                collected.update(matched)
                continue

        missing.append(stage_name)

    return collected, missing


def main() -> int:
    if not ROOT_LOCK.exists():
        print(f"missing root lockfile: {ROOT_LOCK}", file=sys.stderr)
        return 1

    root_doc = load_yaml(ROOT_LOCK)
    root_stages = root_doc.get("stages", {})
    if not root_stages:
        print(f"no stages found in {ROOT_LOCK}", file=sys.stderr)
        return 1

    all_missing: dict[str, list[str]] = {}

    for pipeline_file in PIPELINE_FILES:
        stage_entries, missing = collect_stage_entries(root_stages, pipeline_file)
        lock_doc = {
            "schema": root_doc.get("schema", "2.0"),
            "stages": stage_entries,
        }
        dump_yaml(pipeline_file.with_name("dvc.lock"), lock_doc)
        if missing:
            all_missing[str(pipeline_file)] = missing

    if all_missing:
        print("Split lockfiles created. Stages without compatible root-lock entries:")
        for pipeline_file, missing in all_missing.items():
            print(f"- {pipeline_file}: {', '.join(missing)}")
        print("These stages will need a first repro to populate their new split lock entries.")
    else:
        print("Split lockfiles created for all pipeline stages.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
