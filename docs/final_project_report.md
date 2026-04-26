# Final Project Report

This report is the grading-facing project summary. It intentionally lives under
`docs/` so the root `README.md` can stay as the operational quick start.

## 1. Problem Definition

The project builds a session-based next-item recommender for anonymous or
weakly identified traffic. Given the ordered item events in the current session,
the model ranks likely next items.

Primary success metrics:
- `HR@K`: target item appears in the top-K recommendations.
- `MRR@K`: target item appears earlier in the ranked list.
- `NDCG@K`: recommended future extension for position-discounted ranking
  quality.

Operational success criteria:
- data and model artifacts can be reproduced from source-controlled configs
- serving exposes health/readiness and rejects invalid requests
- monitoring covers API health, prediction latency, error rate, and OOV input
  rate
- model promotion is gated by evaluation metrics

## 2. Data

The data pipeline is stage-based and DVC-tracked:

```text
raw interactions
  -> ingest
  -> validate
  -> preprocess
  -> temporal split
  -> graph training examples + vocabulary
```

Key artifacts:
- `configs/data_config.yaml`: default data pipeline configuration
- `configs/data_versions/*.yaml`: versioned data experiment overlays
- `dvc.yaml`: reproducible data, training, evaluation, and drift stages
- `data/versions/*/processed/data_stats.json`: post-filter split statistics
- `docs/data_contract.md`: schema, quality, lineage, and rollback contract

Validation note:
The raw validation report can show `"valid": false` because raw sessions may be
shorter or longer than the accepted training range. The final training evidence
is the processed `data_stats.json` for each data version after filtering and
example generation.

## 3. Modeling

Implemented model families:
- SR-GNN with multiple readout/adjacency variants
- TAGNN with chunked candidate scoring for memory control
- GGNN with a standard GRU-cell propagation update

All models consume the same graph-example parquet schema and share the same
training/evaluation pipeline.

Latest checked local artifact metrics:

| Artifact | HR@K | MRR@K | Source |
|---|---:|---:|---|
| `models/trained/v1_strict_filter/latest` | 0.5395 | 0.1782 | `metrics.json` |
| `models/trained/v2_sliding_window/latest` | 0.4570 | 0.1467 | `metrics.json` |
| `models/trained/v3_train_plus_val/latest` | n/a | n/a | validation intentionally empty |

For final benchmark reporting, regenerate the official metrics from DVC stages
and cite the generated DVC metrics files, not temporary pytest smoke outputs.

## 4. MLOps Pipeline

Implemented pipeline components:
- DVC stages for data versioning, training, evaluation, comparison, and drift
- MLflow tracking and registry integration hooks
- model artifact registry layout under `models/trained/`
- FastAPI serving with Docker packaging
- GitHub Actions CI with Ruff, pytest, compile check, Bandit, pip-audit, and
  Gitleaks
- GHCR image publishing after successful CI on `main`
- EKS deployment workflow for published images
- Kubernetes manifests for API, HPA, PDB, NetworkPolicy, Prometheus, and Grafana
- Terraform scaffold for EKS and supporting AWS add-ons

Current CI/CD workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/publish-image.yml`
- `.github/workflows/deploy-eks.yml`

CT status:
The repository includes the `recsys-ct-promote` helper for metric-gated MLflow
alias promotion. A scheduled CT workflow is not currently checked in; it can be
added by calling the helper after train/evaluate stages finish.

## 5. Monitoring And Robustness

Online serving metrics:
- request count and status
- prediction latency histogram
- input sequence length histogram
- requested `top_k` histogram
- OOV item count and OOV ratio inputs
- model readiness gauge

Alert scenarios:
- API target down
- model not ready
- high OOV ratio
- high recommendation error rate
- high p95 prediction latency

Offline drift monitoring:
- PSI for session-level features
- Jensen-Shannon divergence for item popularity
- top-N item overlap
- OOV ratio against reference vocabulary
- optional Evidently HTML report for visualization

Security and robustness controls:
- API-key authentication for protected endpoints
- request body size limit
- per-key in-memory rate limit
- strict Pydantic request schema with extra fields rejected
- sanitized health/readiness responses
- Kubernetes non-root runtime, resource bounds, health probes, HPA/PDB, and
  NetworkPolicy

## 6. Reproducibility

Reproducibility mechanisms:
- source-controlled configs under `configs/`
- DVC-tracked data/model stages in `dvc.yaml` and `dvc.lock`
- version-specific artifact directories
- deterministic seed in training config
- model/data contracts under `docs/`
- automated test suite

Local verification performed on 2026-04-20:

```bash
python -m pytest -q
# 94 passed, 1 skipped
```

## 7. Demo Checklist

Recommended grading demo sequence:

```bash
python -m pytest -q
dvc stage list
export RECSYS_API_KEYS=local-dev-key
python -m recsys.serving.api \
  --config configs/serving_config.yaml \
  --model-path models/trained/v1_strict_filter/latest/
```

In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"item_sequence":[101,205,330],"top_k":10}' \
  http://localhost:8000/recommend
```

For observability:

```bash
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker compose --profile observability up -d prometheus grafana
```

## 8. Known Limitations

- The project does not include production traffic logs, so drift monitoring is
  implemented as benchmark replay rather than real production drift detection.
- Grafana datasource provisioning is checked in, but a production dashboard JSON
  export is not currently included.
- The Docker Compose frontend service is currently a commented placeholder; the
  active local demo is API-first.
- CT promotion logic exists as a CLI helper, but no scheduled CT GitHub Actions
  workflow is currently checked in.
- In-memory rate limiting is sufficient for the demo service process but should
  be replaced with a distributed limiter for multi-replica production abuse
  protection.
