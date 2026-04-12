"""Local model registry for saved training artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys.models.srgnn import SRGNNRecommender


class ModelRegistry:
    """Persist model versions under a local filesystem registry."""

    def __init__(self, root_path: str | Path = "models/trained") -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        model: SRGNNRecommender,
        config: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Write a timestamped artifact and refresh the latest alias."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root_path / model.model_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model.save(run_dir)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics or {}, indent=2),
            encoding="utf-8",
        )

        latest_dir = self.root_path / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        model.save(latest_dir)
        (latest_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (latest_dir / "metrics.json").write_text(
            json.dumps(metrics or {}, indent=2),
            encoding="utf-8",
        )
        (latest_dir / "pointer.txt").write_text(str(artifact_path), encoding="utf-8")
        return artifact_path

    def latest_model_path(self) -> Path:
        """Return the latest model artifact path."""
        latest_model = self.root_path / "latest" / "model.json"
        if latest_model.exists():
            return latest_model
        raise FileNotFoundError("No latest model artifact is registered")
