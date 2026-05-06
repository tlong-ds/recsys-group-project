# Problem Definition

Build a session-based recommender for anonymous or weakly identified traffic.
The product task is next-item prediction: given the ordered events inside the
current session, rank the next likely item.

## Motivation

Many commerce and content journeys happen before a user logs in or accumulates
stable long-term history. In that setting, a session-based recommender is useful
because it can react to the current click sequence and still produce ranked
next-item candidates. The expected product value is better item discovery,
higher click-through on recommended items, and fewer empty or irrelevant
recommendation responses for anonymous traffic.

Primary offline objectives:

- improve `HR@K`
- improve `MRR@K`
- optionally add `NDCG@K` for deeper ranking-quality analysis across
  evaluation windows

Operational objectives:

- keep the data, training, evaluation, and serving steps reproducible
- version data and model artifacts so a previous model can be recovered
- expose health, readiness, and prediction-quality signals in serving
- keep model promotion behind explicit metric gates

## Success Metrics

The core offline task is top-K next-item ranking. The current evaluator tracks:

- `HR@K`: whether the held-out next item appears in the top-K list
- `MRR@K`: how early the held-out next item appears when it is recommended

Recommended extension:

- `NDCG@K`: ranking quality with position discounting, useful as a stability
  metric when comparing evaluation windows

For the current training and evaluation configuration, `K` is controlled by
`training.top_k` in `configs/training_config.yaml`.

## Constraints

- Input traffic is treated as session-level event history, not long-term user
  profile history.
- Validation and test examples must not influence the training vocabulary.
- Serving must reject malformed requests and avoid exposing local artifact paths
  or raw model-loading exceptions.
- Monitoring is split between online Prometheus metrics and offline benchmark
  drift replay because this repository does not contain production traffic logs.

Non-goals for this scaffold:

- feature-complete online feature store
- distributed training
- real-time drift automation
