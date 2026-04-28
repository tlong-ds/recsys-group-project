# Monitoring

This document details the monitoring stack used to track the health, performance, and usage of the recommendation system.

## Stack Overview
The monitoring stack relies on **Prometheus** for metrics collection and **Grafana** for visualization.

## Metrics Collection (Prometheus)
- **Library**: `prometheus-fastapi-instrumentator` is used in `src/recsys/serving/api.py` to automatically instrument FastAPI endpoints, collecting default metrics such as request latency, HTTP status codes, and request count.
- **Custom Metrics**: A custom Prometheus counter, `recsys_recommendations_total`, tracks the total number of successful recommendation requests served.
- **Online Input Quality Metrics**: `/recommend` records request outcomes,
  model prediction latency, `item_sequence` length, requested `top_k`, and
  unknown/OOV item ratio against the currently loaded model catalog.
- **Configuration**: `deployment/monitoring/prometheus.yml` configures Prometheus to scrape metrics from the `api` service on port `8000` every 15 seconds.
- **Authentication**: `/metrics` requires the same bearer API key as other protected API routes. Docker Compose mounts `deployment/secrets/recsys-api-key` into Prometheus as a bearer token file.
- **Alerts**: Alerting rules are defined in `deployment/monitoring/alerts.yml` to trigger notifications when metrics cross specific thresholds.

Configured alert scenarios:
- API target is unreachable.
- Model readiness gauge reports unavailable.
- Online OOV item ratio is above threshold.
- Recommendation error rate is above threshold.
- Prediction p95 latency is above threshold.

### Online Serving Signals
The online layer is intentionally lightweight and does not run Evidently,
Pandas, or parquet-based drift reports in the API request path.

Key Prometheus metrics:
- `recsys_recommendation_requests_total{status}`: request outcomes.
- `recsys_prediction_latency_seconds`: model inference latency histogram.
- `recsys_input_sequence_length`: input session length histogram.
- `recsys_requested_top_k`: requested recommendation count histogram.
- `recsys_input_items_total`: total request items received.
- `recsys_oov_items_total`: items unknown to the loaded model catalog.
- `recsys_model_ready`: readiness gauge for the configured model.

Useful Prometheus queries:
```promql
# Online OOV ratio
rate(recsys_oov_items_total[5m])
/
clamp_min(rate(recsys_input_items_total[5m]), 1e-9)

# Recommendation p95 latency
histogram_quantile(
  0.95,
  sum(rate(recsys_prediction_latency_seconds_bucket[5m])) by (le)
)

# Non-success recommendation request rate
sum(rate(recsys_recommendation_requests_total{status!="success"}[5m]))
```

## Visualization (Grafana)
Grafana is used to build dashboards visualizing the metrics collected by Prometheus.
- **Datasource**: Prometheus is provisioned as a default datasource via `deployment/monitoring/grafana/provisioning/datasources/prometheus.yml`.
- **Access**: The Grafana UI is bound to localhost in Docker Compose. Set `GRAFANA_ADMIN_PASSWORD` before starting the stack.

Dashboard status:
- The repo provisions the Prometheus datasource.
- Dashboard panels are expected to be created from the PromQL queries above or
  exported after a demo run.
- No production traffic dashboard JSON is checked in because the project does
  not include production traffic logs.

## Offline Drift Monitoring
This repository does not have production traffic logs. Drift monitoring is
therefore implemented as a **benchmark replay** workflow rather than a claim of
real production drift detection.

The drift workflow compares:
- a reference window, usually `v1_strict_filter` train interactions
- a current window, such as the same baseline window for sanity checks,
  `v2_sliding_window` test interactions, or a synthetic OOV scenario
- the reference item vocabulary used to detect unknown items

Run a baseline replay:
```bash
python -m recsys.monitoring.drift \
  --reference data/versions/v1_strict_filter/interim/train_interactions.parquet \
  --current data/versions/v1_strict_filter/interim/train_interactions.parquet \
  --vocab data/versions/v1_strict_filter/processed/item_vocab.json \
  --output metrics/monitoring/drift_baseline.json \
  --html reports/monitoring/evidently_baseline.html
```

Run a natural benchmark shift:
```bash
python -m recsys.monitoring.drift \
  --reference data/versions/v1_strict_filter/interim/train_interactions.parquet \
  --current data/versions/v2_sliding_window/interim/test_interactions.parquet \
  --vocab data/versions/v1_strict_filter/processed/item_vocab.json \
  --output metrics/monitoring/drift_v1_vs_v2.json \
  --html reports/monitoring/evidently_v1_vs_v2.html
```

Run a controlled synthetic OOV scenario:
```bash
python -m recsys.monitoring.drift \
  --reference data/versions/v1_strict_filter/interim/train_interactions.parquet \
  --current data/versions/v1_strict_filter/interim/train_interactions.parquet \
  --vocab data/versions/v1_strict_filter/processed/item_vocab.json \
  --inject-oov-rate 0.30 \
  --output metrics/monitoring/drift_synthetic_oov.json \
  --html reports/monitoring/evidently_synthetic_oov.html
```

The JSON report is the primary artifact for DVC, tests, and report writing. It
contains PSI checks for session-level features, Jensen-Shannon divergence for
item popularity, top-N item overlap, OOV ratio, and an overall `ok`,
`warning`, or `critical` status.

Retraining trigger interpretation:
- `ok`: no retraining action needed.
- `warning`: inspect drift details and compare offline metrics before promotion.
- `critical`: block automatic promotion and run a retraining/evaluation cycle
  before serving a new alias.

Evidently is optional and is used only for HTML visualization. Install it with:
```bash
pip install -e .[monitoring]
```

The same scenarios are available as DVC stages:
```bash
dvc repro pipelines/monitoring/dvc.yaml:drift_baseline
dvc repro pipelines/monitoring/dvc.yaml:drift_v1_vs_v2
dvc repro pipelines/monitoring/dvc.yaml:drift_synthetic_oov
```

They are also integrated into the closed-loop `compare_data_versions` target, so
`dvc repro pipelines/monitoring/dvc.yaml:compare_data_versions` now generates both the version comparison and
drift summaries in one run.

## Running the Monitoring Stack
The monitoring stack can be launched independently or as part of the full stack via Docker Compose:
```bash
export RECSYS_API_KEYS=<api-key>
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker compose --profile observability up -d prometheus grafana
```

Grafana is provisioned with the `RecSys API` dashboard under the `RecSys`
folder. If the dashboard only shows scrape/process metrics, send traffic to
`POST /recommend`; the application-specific recommendation and latency metrics
are emitted only after the endpoint has handled requests.

## Monitoring on EKS

EKS manifests for Prometheus and Grafana are included under `deployment/kubernetes/`
and are applied via:

```bash
kubectl apply -k deployment/kubernetes/
```

Prometheus scrapes `recsys-api.recsys.svc.cluster.local:80` and authenticates to
`/metrics` using the `recsys-monitoring-secrets` secret key `recsys-api-key`.
