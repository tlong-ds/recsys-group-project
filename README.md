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

## Local commands

```bash
pip install -e .[dev]
python -m unittest discover -s tests -p "test_*.py"
recsys-process-data --stage all --config configs/data_config.yaml --params params.yaml
recsys-train --data-config configs/data_config.yaml --model-config configs/model_config.yaml --training-config configs/training_config.yaml --params params.yaml
recsys-serve --config configs/serving_config.yaml
dvc repro train evaluate
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
    token_env_var: DAGSHUB_TOKEN
    repo_owner: lytlong.pers
    repo_name: recsys-group-project
```

```bash
export DAGSHUB_TOKEN=<your_personal_access_token>
```

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
