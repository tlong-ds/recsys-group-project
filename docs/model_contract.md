# Model Contract - Training and Model Artifact Interface

##  1. Ownership

- Model interface owner: `src/recsys/models/srgnn.py`
- Training orchestration owner: `src/recsys/training/pipeline.py`
- Local artifact registry owner: `src/recsys/training/registry.py`
- Serving consumer: `src/recsys/serving/predictor.py`

Any breaking change to training inputs, output artifact layout, or serving load
behavior must update this file in the same pull request.

## 2. Runtime Entry Points

Primary supported entrypoints:
- CLI script: `recsys-train`
- Module: `python -m recsys.training.pipeline`


Current canonical command:

```bash
python -m recsys.training.pipeline \
  --data-config configs/data_config.yaml \
  --model-config configs/model_config.yaml \
  --training-config configs/training_config.yaml
```

## 3. Training Flow Overview

Training is a downstream consumer of processed feature artifacts. It does not
run the data pipeline end to end on its own.

Current flow:
1. Load and merge `data_config.yaml`, `model_config.yaml`, and `training_config.yaml`
2. Read processed parquet examples and vocabulary from `data/processed`
3. Instantiate `SRGNNRecommender`
4. Fit the model on `train_examples.parquet`
5. Evaluate on validation examples
6. Save the trained model into the local model registry
7. Evaluate on test examples
8. Optionally log params, metrics, and artifacts to MLflow



## 4. Required Upstream Inputs

Training consumes the following artifacts from the same processed-data build:
- `data/processed/train_examples.parquet`
- `data/processed/val_examples.parquet`
- `data/processed/test_examples.parquet`
- `data/processed/item_vocab.json`



Training also consumes configuration from:
- `configs/data_config.yaml`
- `configs/model_config.yaml`
- `configs/training_config.yaml`

Contract requirement:
- the train, val, test parquet files and `item_vocab.json` must come from the same preprocessing run


## 5. Input Schema Expected by the Model

The current model implementation expects graph-style examples.

Required columns in each processed parquet file:
- `x`: list of encoded item IDs representing unique nodes in the session graph
- `edge_index`: graph edges with shape `[2, n_edges]`
- `alias_inputs`: sequence-to-node index mapping
- `item_seq_len`: integer sequence length
- `pos_items`: encoded next-item label



## 6. Configuration Contract

### 6.1 Model Configuration

Current model config keys:
- `model.name`
- `model.version`
- `model.embedding_dim`
- `model.hidden_size`
- `model.step`
- `model.max_session_length`
- `model.fallback_weight`

Behavior notes:
- `hidden_size` currently must effectively match `embedding_dim`
- if they differ, the implementation logs a warning and uses `embedding_dim`

### 6.2 Training Configuration

Current training config keys:
- `training.seed`
- `training.batch_size`
- `training.num_epochs`
- `training.lr`
- `training.weight_decay`
- `training.early_stopping_patience`
- `training.top_k`
- `training.num_workers`

Registry configuration:
- `registry.root_path`

MLflow configuration:
- `mlflow.enabled`
- `mlflow.tracking_uri`
- `mlflow.experiment_name`
- `mlflow.run_name`



## 6. Training Outputs

Training returns and/or writes the following outputs.

### 6.1 Local Model Registry

Default registry root:
- `models/trained`

Per-run artifact layout:

```text
models/trained/
  srgnn/
    <UTC timestamp>/
      model.pt
      model.json
      config.json
      metrics.json
  latest/
    model.pt
    model.json
    config.json
    metrics.json
    pointer.txt
```

Semantics:
- `srgnn/<timestamp>/` is the immutable saved run artifact
- `latest/` is a mutable alias for the most recently registered model
- `pointer.txt` stores the path to the timestamped artifact file returned by training

Current serving default:
- serving loads from `models/trained/latest/`

Implication for MLOps and deployment:
- anything promoting or packaging a model for inference can safely consume the
  `latest/` directory as the default local handoff
- anything implementing versioned release behavior should prefer the timestamped
  run directory, not the mutable alias

### 6.2 Artifact File Contract

Required files in a loadable model artifact directory:
- `model.pt`
- `model.json`

Additional files written by training:
- `config.json`
- `metrics.json`

#### `model.json`

`model.json` must contain enough metadata to reconstruct the model for serving.

Current required keys:
- `model_name`
- `model_version`
- `embedding_dim`
- `step`
- `max_session_length`
- `fallback_weight`
- `seed`
- `n_items`
- `item_to_idx`
- `idx_to_item`
- `popularity`

#### `config.json`

Contains the merged runtime config used for the training run.

Minimum use:
- reproducibility
- debugging deployment mismatches
- later promotion to a stronger registry system

#### `metrics.json`

Contains offline validation metrics returned by the evaluator.

Current note:
- test metrics are produced by the pipeline output and MLflow logging, but are
  not currently written into `metrics.json`
- if downstream automation expects test metrics on disk, that needs an explicit
  contract change

## 7. MLflow Logging Contract

When `mlflow.enabled` is true, training additionally logs:
- flattened config parameters
- validation metrics with `val_` prefix
- test metrics with `test_` prefix
- registered local artifact directory under MLflow artifacts
- PyTorch core model under `model_core` when available

Current MLflow contract is additive:
- MLflow is observability and lineage support
- the local filesystem artifact remains the serving source of truth in this repo
