# Data Contract - Data Processing Pipeline

## 1. Purpose and Scope

This document defines the authoritative data contract for the data pipeline in this repository.

Scope:
- Input interaction data consumed by the processing pipeline
- Intermediate and processed artifacts produced by the pipeline
- Quality rules and acceptance criteria for model training readiness
- Lineage and reproducibility requirements for MLOps operations

Out of scope:
- Online feature store contracts
- Serving request/response schema (covered in API/model contracts)

## 2. Contract Ownership

- Data producer: Data Processing Pipeline (recsys.data.pipeline)
- Data consumers: Training pipeline, evaluation workflow, and downstream analysis
- Contract owner: ML Platform/Data Engineering maintainers for this repository

Any schema or semantic change must update this file and be versioned in source control.

## 3. Runtime Interface

Entry points:
- CLI: recsys-process-data
- Module: python -m recsys.data.pipeline

Primary config file:
- configs/data_config.yaml
- Optional overlay for DVC experiment knobs only: params.yaml

Ownership rule:
- `params.yaml` must not own runtime metadata such as paths, external URLs/endpoints, or registry/tracking destinations.
- Those values are defined in `configs/*_config.yaml`.

Default paths:
- Raw: data/raw
- Interim: data/interim
- Processed: data/processed

## 4. Input Data Contract

### 4.1 Required Logical Fields

The pipeline resolves actual column names from config key mapping.

Required logical fields:
- session_id
- item_id
- event_date

Default configured physical names in current config:
- sessionId
- itemId
- eventdate

Optional logical field:
- timeframe

### 4.2 Input Constraints

- Dataset must be non-empty
- Required columns must exist after mapping resolution
- event_date column must be datetime-parseable
- session_id, item_id, event_date must not contain nulls for accepted quality status

Note on enforcement:
- Missing required columns or empty dataset is a hard failure
- Semantic issues are reported as warnings and pipeline currently continues

## 5. Processing Stages and Guarantees

The pipeline executes the following stages:
1. Ingest raw interactions
2. Validate schema and semantics
3. Preprocess interactions
4. Temporal split into train/val/test
5. Build examples and vocabulary
6. Persist vocabulary and stats

Current preprocessing behavior:
- basic cleaning and sorting by session/time
- session length filtering
- item frequency filtering
- second session length filtering after item pruning
- duplicate removal only when allow_duplicates is false

## 6. Output Artifact Contract

### 6.1 Interim Artifact

File:
- data/interim/clean_interactions.parquet

Schema:
- Includes mapped session, item, and event_date columns
- May include timeframe when present in source/config
- Item IDs are original item IDs at this stage (not vocabulary encoded)

### 6.2 Processed Example Artifacts

Files:
- data/processed/train_examples.parquet
- data/processed/val_examples.parquet
- data/processed/test_examples.parquet

The schema depends on data.training_example_format.

#### A) Graph format (current default)

Required columns:
- x: List[int], unique node IDs in sample graph
- edge_index: List[List[int]], shape [2, num_edges], directed transitions
- alias_inputs: List[int], index mapping from sequence positions to x
- item_seq_len: int, length of encoded input sequence
- pos_items: int, next-item label (encoded ID)
- session_id: original session identifier
- eventdate (or configured event_date name): timestamp of predicted item

Constraints:
- item_seq_len equals len(alias_inputs)
- pos_items is in vocabulary ID range [start_id, start_id + size - 1]
- x values are encoded item IDs produced from training vocabulary

#### B) Sequence format

Required columns:
- input_items: List[int], encoded prefix sequence
- target_item: int, encoded next item
- seq_len: int, length of input_items
- session_id: original session identifier
- eventdate (or configured event_date name): timestamp of predicted item

Constraints:
- seq_len equals len(input_items)
- target_item is in vocabulary ID range [start_id, start_id + size - 1]

### 6.3 Vocabulary Artifact

File:
- data/processed/item_vocab.json

Schema:
- item2id: mapping from original item ID (serialized string key) to encoded ID (int)
- id2item: reverse mapping from encoded ID (serialized string key) to original item ID
- size: vocabulary size
- start_id: first valid encoded ID (default 1)

Invariants:
- item2id and id2item are bijective
- ID 0 is reserved by convention for padding/unknown handling
- Vocabulary is built from training split only

### 6.4 Statistics Artifact

File:
- data/processed/data_stats.json

Guaranteed keys:
- build_date
- config_file
- train, val, test blocks containing:
  - interactions stats (n_interactions, n_sessions, n_items, avg/min/max/median session length)
  - examples count
- vocab_size

## 7. Split and Leakage Rules

Supported split strategies:
- ratio_based
- diginetica_legacy
- session_based
- time_based 

Contract requirement:
- Train/validation/test splits must be temporally ordered per selected strategy
- No future interaction from validation/test may influence vocabulary fitting

Implementation note:
- Vocabulary is built only from train_df
- Validation/test example generation drops unknown items when configured in pipeline (current behavior: true for val/test)

## 8. Quality Gates for MLOps

### 8.1 Blocking Gates (must pass)

- Input file readable
- Required columns present after mapping
- Input dataset non-empty
- Output artifacts written successfully

### 8.2 Non-blocking Gates (warn and track)

- sessions shorter than min_session_length found
- sessions longer than max_session_length found (if configured)
- duplicate items within session found when duplicates disallowed
- null values detected in key columns

Recommendation for production hardening:
- Promote selected semantic checks to blocking in CI/CD for scheduled retraining

## 9. Reproducibility and Lineage

Minimum reproducibility metadata:
- exact config path recorded in data_stats.json
- build_date recorded in data_stats.json
- source-controlled code revision (Git commit) tracked by CI/CD metadata
- dataset artifacts versioned with DVC and/or immutable object storage

Operational rule:
- The training pipeline must consume train/val/test, item_vocab.json, and data_stats.json from the same build run

## 10. Retention and Rollback

- Keep at least one prior successful processed dataset version for rollback
- Rollback unit is the full artifact set, not individual files
- Do not mix split files and vocabulary from different versions

## 11. Change Management

Any change to:
- input field names
- output schema
- split strategy defaults
- filtering thresholds that alter semantics

must include:
- contract version bump
- release note in PR
- backward compatibility statement

## 12. Contract Version

- Effective date: 2026-04-16
- Version: 1.1
- Status: Active
