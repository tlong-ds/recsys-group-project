"""conftest.py – install stubs for external project modules before collection.

The integration tests live inside the repo but the *full* project tree
(recsys.evaluation, recsys.training.trainer, recsys.training.registry, …)
is not required.  We insert lightweight stubs into sys.modules here, before
pytest collects any test module, so that ``from recsys.training.pipeline
import …`` succeeds even when only the model + helper layer is present.

What is stubbed
---------------
recsys.evaluation              – Evaluator.evaluate() returns fixed metrics
recsys.training.trainer        – Trainer.train() calls model.fit() + saves
recsys.training.registry       – ModelRegistry.register() / latest_model_path()
recsys.training.tracking       – all functions are no-ops; tracking_enabled→False
recsys.training.mlflow_registry – register_model_version is a no-op
recsys.utils.logger            – get_logger() returns stdlib Logger
recsys.utils.config            – load_training_runtime_config returns empty dict

What is NOT stubbed (runs real code)
--------------------------------------
recsys.models.*                – all model classes run as-is
recsys.training.pipeline       – the module under test
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# TrainingResult stub (mirrors trainer.TrainingResult interface)
# ---------------------------------------------------------------------------


@dataclass
class _TrainingResult:
    model: Any
    artifact_path: Path
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Trainer stub
# ---------------------------------------------------------------------------


class _Trainer:
    """Minimal Trainer: calls model.fit() and saves the artefact locally."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config  = config
        self._reg_cfg = config.get("registry", {})
        self._tr_cfg  = config.get("training", {})

    def train(
        self,
        model: Any,
        train_df: Any,
        val_df: Any,
        item_vocab: dict[str, Any] | None = None,
    ) -> _TrainingResult:
        fit_kwargs: dict[str, Any] = {
            "num_epochs":               int(self._tr_cfg.get("num_epochs",  1)),
            "batch_size":               int(self._tr_cfg.get("batch_size",  32)),
            "lr":                       float(self._tr_cfg.get("lr",        1e-3)),
            "weight_decay":             float(self._tr_cfg.get("weight_decay", 0.0)),
            "early_stopping_patience":  int(self._tr_cfg.get("early_stopping_patience", 3)),
            "val_df":                   val_df if not val_df.empty else None,
            "item_vocab":               item_vocab,
            "num_workers":              int(self._tr_cfg.get("num_workers", 0)),
        }
        model.fit(train_df, **fit_kwargs)

        # Evaluate on val split
        metrics: dict[str, float] = {}
        if val_df is not None and not val_df.empty:
            from recsys.evaluation import Evaluator
            ev      = Evaluator(top_k=int(self._tr_cfg.get("top_k", 20)))
            metrics = ev.evaluate(model, val_df)

        # Save to registry root (versioned subdir named after model_name)
        root = Path(
            self._reg_cfg.get("root_path")
            or self._tr_cfg.get("registry_path")
            or "models/trained"
        )
        save_dir  = root / "latest"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = model.save(save_dir)

        return _TrainingResult(model=model, artifact_path=save_path, metrics=metrics)


# ---------------------------------------------------------------------------
# ModelRegistry stub
# ---------------------------------------------------------------------------


class _ModelRegistry:
    def __init__(self, root_path: str | Path, **_kw: Any) -> None:
        self._root = Path(root_path)

    def latest_model_path(self) -> Path:
        # The Trainer stub always saves to <root>/latest/
        return self._root / "latest"

    def register(self, model: Any, config: Any, metrics: Any) -> Path:
        return self.latest_model_path()


# ---------------------------------------------------------------------------
# Build stub modules and inject into sys.modules
# ---------------------------------------------------------------------------


def _make_stub(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _build_evaluator_module() -> types.ModuleType:
    """Real-ish Evaluator that calls recommend_from_graph and computes HR/MRR."""
    from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k  # type: ignore

    class _Evaluator:
        def __init__(self, top_k: int = 10) -> None:
            self.top_k = top_k

        def evaluate(self, model: Any, examples: Any) -> dict[str, float]:
            if examples.empty:
                return {"hr@k": 0.0, "mrr@k": 0.0}
            hr_scores: list[float]  = []
            mrr_scores: list[float] = []
            for row in examples.itertuples(index=False):
                recs   = model.recommend_from_graph(
                    row.x, row.edge_index, row.alias_inputs, top_k=self.top_k
                )
                target = int(row.pos_items)
                hr_scores.append(hit_rate_at_k(recs, [target], self.top_k))
                mrr_scores.append(mrr_at_k(recs, [target], self.top_k))
            n = len(hr_scores)
            return {"hr@k": sum(hr_scores) / n, "mrr@k": sum(mrr_scores) / n}

    mod = types.ModuleType("recsys.evaluation")
    mod.Evaluator = _Evaluator  # type: ignore[attr-defined]
    return mod


def _install_stubs() -> None:
    """Register all stubs into sys.modules (idempotent)."""
    already = set(sys.modules)

    # Only install what isn't already present (real modules take precedence)
    def maybe(name: str, mod: types.ModuleType) -> None:
        if name not in already:
            sys.modules[name] = mod

    # recsys.evaluation — try the real metrics layer first, fall back to stub
    try:
        import recsys.evaluation.metrics  # noqa: F401  -- may already be present

        maybe("recsys.evaluation", _build_evaluator_module())
    except ModuleNotFoundError:
        # Fully synthetic fallback: always returns 0.5 for both metrics
        class _FakeEvaluator:
            def __init__(self, top_k: int = 10) -> None:
                pass

            def evaluate(self, model: Any, examples: Any) -> dict[str, float]:
                return {"hr@k": 0.5, "mrr@k": 0.5}

        maybe(
            "recsys.evaluation",
            _make_stub("recsys.evaluation", Evaluator=_FakeEvaluator),
        )

    # Trainer
    maybe(
        "recsys.training.trainer",
        _make_stub(
            "recsys.training.trainer",
            Trainer=_Trainer,
            TrainingResult=_TrainingResult,
        ),
    )

    # ModelRegistry
    maybe(
        "recsys.training.registry",
        _make_stub("recsys.training.registry", ModelRegistry=_ModelRegistry),
    )

    # Tracking — all no-ops; tracking_enabled returns False
    maybe(
        "recsys.training.tracking",
        _make_stub(
            "recsys.training.tracking",
            tracking_enabled            = lambda cfg: False,
            configure_tracking          = lambda cfg: None,
            configure_system_metrics    = lambda cfg: None,
            log_evaluation_run          = lambda **kw: None,
            sanitize_metric_key         = lambda k: k,
            system_metrics_run_override = lambda cfg: None,
        ),
    )

    # MLflow registry — no-op
    maybe(
        "recsys.training.mlflow_registry",
        _make_stub(
            "recsys.training.mlflow_registry",
            register_model_version=lambda **kw: None,
        ),
    )

    # Utils
    maybe(
        "recsys.utils.logger",
        _make_stub(
            "recsys.utils.logger",
            get_logger=lambda: logging.getLogger("test.pipeline"),
        ),
    )
    maybe(
        "recsys.utils.config",
        _make_stub(
            "recsys.utils.config",
            load_training_runtime_config=lambda **kw: {},
        ),
    )


# Run immediately at import time so stubs are present before any test module
# tries to import recsys.training.pipeline.
_install_stubs()