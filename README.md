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
├── streamlit_app/          # lightweight demo UI
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
recsys-train --data-config configs/data_config.yaml --model-config configs/model_config.yaml --training-config configs/training_config.yaml
recsys-serve --config configs/serving_config.yaml
```

## Model scope

The repo is structured for a session-based recommender. The production-facing
model wrapper is `SRGNNRecommender`, with a transition-graph baseline behind
the same interfaces so the repository stays usable before a full neural SR-GNN
implementation is dropped in.
