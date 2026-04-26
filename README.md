# recsys-group-project

Session-based recommendation system scaffold aligned to an SR-GNN style
workflow and packaged for MLOps demonstration.

## Repository layout

```text
recsys-group-project/
├── .github/workflows/      # CI: lint, test, package checks
├── configs/                # YAML configs for data, model, training, serving
├── data/
│   ├── raw/                # source datasets
│   ├── interim/            # cleaned / intermediate tables
│   └── processed/          # train-ready examples and splits
├── deployment/             # K8s, MLflow, monitoring assets
├── docs/                   # problem statement and team contracts
├── models/trained/         # local model registry / exported artifacts
├── notebooks/              # EDA only
├── src/recsys/             # application package
├── tests/                  # unit and smoke tests
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Technology Stack

### Backend (Python 3.10+)
- **Core Framework:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Machine Learning:** [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (SR-GNN, GGNN, TAGNN), [Scikit-learn](https://scikit-learn.org/)
- **Data Processing:** [Polars](https://pola.rs/), [Pandas](https://pandas.pydata.org/), [PyArrow](https://arrow.apache.org/docs/python/), [Pandera](https://pandera.readthedocs.io/) (Schema Validation)
- **Experiment Tracking:** [MLflow](https://mlflow.org/), [DagsHub](https://dagshub.com/)
- **Data Versioning:** [DVC](https://dvc.org/) (Data Version Control) with S3 backend
- **Utilities:** [Loguru](https://github.com/Delgan/loguru) (Logging), [Pydantic](https://docs.pydantic.dev/) (Data Validation), [PyYAML](https://pyyaml.org/)

### DevOps & Infrastructure
- **Containerization:** [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)
- **Orchestration:** [Kubernetes](https://kubernetes.io/) (AWS EKS), [Kustomize](https://kustomize.io/)
- **Infrastructure as Code:** [Terraform](https://www.terraform.io/)
- **CI/CD:** [GitHub Actions](https://github.com/features/actions), [GitHub Container Registry](https://github.com/features/packages) (GHCR)
- **Monitoring:** [Prometheus](https://prometheus.io/), [Grafana](https://grafana.com/), [Evidently](https://www.evidentlyai.com/) (Model Drift & Quality)

### Testing & Quality
- **Testing:** [Pytest](https://docs.pytest.org/), [Unittest](https://docs.python.org/3/library/unittest.html)
- **Linting & Formatting:** [Ruff](https://beta.ruff.rs/docs/), [ESLint](https://eslint.org/)
- **Static Analysis:** [Mypy](http://mypy-lang.org/), [Bandit](https://bandit.readthedocs.io/) (Security), [Pip-audit](https://github.com/pypa/pip-audit)

## Local commands

```bash
pip install -e .[dev]
python -m unittest discover -s tests -p "test_*.py"
recsys-process-data --stage all --config configs/data_config.yaml --params params.yaml
recsys-train --data-config configs/data_config.yaml --model-config configs/model_config.yaml --training-config configs/training_config.yaml --params params.yaml
recsys-serve --config configs/serving_config.yaml
# Closed-loop: build data versions + train/eval + compare in one target
dvc repro compare_data_versions
# Data-only builds (no training)
dvc repro data_version_v1
dvc repro data_version_v2
# Full matrix: 3 data versions × 8 model profiles (24 train + 24 evaluate stages)
dvc repro train_matrix evaluate_matrix
# Select winner, retrain winner profile on train+val, then promote
dvc repro select_best_model retrain_selected_model promote_retrained_model
```

## Versioned data pipelines (V1/V2)

Each data version has its own params file and artifact directory so it can be
tracked independently in DVC. The repo also provides a closed-loop comparison
target that runs data build + train/eval + drift monitoring + aggregation in one
pipeline.

```bash
# Run the full version comparison pipeline
dvc repro compare_data_versions

# Build data versions independently (no training)
dvc repro data_version_v1
dvc repro data_version_v2

# Or run one version branch explicitly
dvc repro eval_v1
dvc repro eval_v2
```

Version definitions:

- V1 strict filter: `configs/data_versions/v1_strict_filter.yaml`
- V2 sliding-window safety: `configs/data_versions/v2_sliding_window.yaml`

Generated data artifacts:

- `data/versions/v1_strict_filter/*`
- `data/versions/v2_sliding_window/*`

Generated model and metrics namespaces:

- `models/trained/v1_strict_filter/latest`
- `models/trained/v2_sliding_window/latest`
- `metrics/v1_strict_filter/*.json`
- `metrics/v2_sliding_window/*.json`
- `metrics/monitoring/drift_*.json`
- `metrics/data_version_comparison.json`

Ad hoc train/evaluate with a specific data version:

```bash
python -m recsys.training.pipeline --stage train --dvc-mode --data-config configs/data_config.yaml --model-config configs/model_profiles/srgnn.yaml --training-config configs/training_config.yaml --params configs/data_versions/v1_strict_filter.yaml --registry-root models/trained/v1_strict_filter --train-metrics-path metrics/v1_strict_filter/training_metrics.json
python -m recsys.training.pipeline --stage evaluate --dvc-mode --data-config configs/data_config.yaml --model-config configs/model_profiles/srgnn.yaml --training-config configs/training_config.yaml --params configs/data_versions/v1_strict_filter.yaml --registry-root models/trained/v1_strict_filter --evaluation-metrics-path metrics/v1_strict_filter/evaluation_metrics.json
```

## Model matrix training with DVC

Model profiles are defined in `configs/model_profiles/*.yaml`:

- `srgnn`, `srgnn_ngc`, `srgnn_fc`, `srgnn_l`, `srgnn_avg`, `srgnn_att`, `tagnn`, `ggnn`

Data versions are defined in `configs/data_versions/*.yaml`:

- `baseline`, `v1_strict_filter`, `v2_sliding_window`

Run fast profiles (default iteration loop):

```bash
dvc repro train_matrix evaluate_matrix
```

Run long-running profiles (`srgnn_ngc`, `tagnn`):

```bash
dvc repro train_matrix_slow evaluate_matrix_slow
```

Run one specific job (fast or slow):

```bash
dvc repro train_matrix@v2_sliding_window-tagnn evaluate_matrix@v2_sliding_window-tagnn
dvc repro train_matrix_slow@v2_sliding_window-tagnn evaluate_matrix_slow@v2_sliding_window-tagnn
```

Artifacts and metrics are separated per job:

- `models/experiments/<data_version>/<model_profile>/latest/`
- `metrics/experiments/<data_version>/<model_profile>/training_metrics.json`
- `metrics/experiments/<data_version>/<model_profile>/evaluation_metrics.json`

Final production flow retrains the selected winner on merged train+val data:

```bash
dvc repro select_best_model retrain_selected_model promote_retrained_model
```

This retrain uses `metrics/best_model.json` to resolve both `data_version` and
`model_profile`, builds `data/retrained_selected/trainval_examples.parquet`, and
promotes from `metrics/retrained_selected/training_metrics.json`.

## Artifact commit hygiene and DVC push checks

Training outputs are intentionally split into:

- DVC-managed model artifacts: `models/trained/**/latest/`, `models/experiments/**/latest/`
- Local-only experiment metrics: `metrics/experiments/**` (git-ignored)

Before committing:

```bash
git status --short
```

Expected: source/config/lock changes only (`dvc.yaml`, `dvc.lock`, code/config/docs).

After training and before `dvc push`, confirm whether there are new cache objects:

```bash
dvc status -c
```

Then push:

```bash
dvc push
```

If push reports `Everything is up to date`, that means the remote already has the
object hashes referenced by your current `dvc.lock` (or `dvc.lock` did not change
for those stage outputs).

If you see warnings like `Output ... is missing version info. Cache for it will not be collected`,
run reproduce for the relevant train stage(s) first so `dvc.lock` gets output hashes:

```bash
dvc repro train_matrix@v1_strict_filter-srgnn
git diff -- dvc.lock
dvc status -c
dvc push
```

## Config ownership and precedence

Runtime config is merged in this order:
1. `configs/*.yaml` defaults
2. `params.yaml` overrides (**experiment knobs only**)

Use `params.yaml` for DVC experiment tuning only (model/data/training hyperparameters). Keep metadata and runtime settings (paths, registry, MLflow URI, DagsHub repo, report destinations) in `configs/*_config.yaml`.

## Optional DagsHub MLflow tracking

```yaml
# configs/training_config.yaml
mlflow:
  enabled: true
  dagshub:
    enabled: true
    token_env_var: DAGSHUB_USER_TOKEN
    repo_owner: lytlong.pers
    repo_name: recsys-group-project
```

```bash
export DAGSHUB_USER_TOKEN=<your_personal_access_token>
```

## CI/CD image publishing

GitHub Actions publishes container images to GHCR after successful `ci` runs
triggered by pushes to `main`.

- Image: `ghcr.io/<owner>/<repo>`
- Tags: `main` and `sha-<short_sha>`

## MLflow system metrics

Enable MLflow system metrics collection by config:

```yaml
mlflow:
  system_metrics:
    enabled: true
    sampling_interval: 10
    samples_before_logging: 1
```

Install host dependencies for system metrics:

```bash
pip install psutil
# optional GPU metrics:
pip install nvidia-ml-py
```

## Model scope

The repo is structured for a session-based recommender. The production-facing
model wrapper is `SRGNNRecommender`, with a transition-graph baseline behind
the same interfaces so the repository stays usable before a full neural SR-GNN
implementation is dropped in.
