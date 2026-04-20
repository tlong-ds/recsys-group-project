# Monitoring

This document details the monitoring stack used to track the health, performance, and usage of the recommendation system.

## Stack Overview
The monitoring stack relies on **Prometheus** for metrics collection and **Grafana** for visualization.

## Metrics Collection (Prometheus)
- **Library**: `prometheus-fastapi-instrumentator` is used in `src/recsys/serving/api.py` to automatically instrument FastAPI endpoints, collecting default metrics such as request latency, HTTP status codes, and request count.
- **Custom Metrics**: A custom Prometheus counter, `recsys_recommendations_total`, tracks the total number of successful recommendation requests served.
- **Configuration**: `deployment/monitoring/prometheus.yml` configures Prometheus to scrape metrics from the `api` service on port `8000` every 15 seconds.
- **Authentication**: `/metrics` requires the same bearer API key as other protected API routes. Docker Compose mounts `deployment/secrets/recsys-api-key` into Prometheus as a bearer token file.
- **Alerts**: Alerting rules are defined in `deployment/monitoring/alerts.yml` to trigger notifications when metrics cross specific thresholds.

## Visualization (Grafana)
Grafana is used to build dashboards visualizing the metrics collected by Prometheus.
- **Datasource**: Prometheus is provisioned as a default datasource via `deployment/monitoring/grafana/provisioning/datasources/prometheus.yml`.
- **Access**: The Grafana UI is bound to localhost in Docker Compose. Set `GRAFANA_ADMIN_PASSWORD` before starting the stack.

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

Evidently is optional and is used only for HTML visualization. Install it with:
```bash
pip install -e .[monitoring]
```

The same scenarios are available as DVC stages:
```bash
dvc repro drift_baseline
dvc repro drift_v1_vs_v2
dvc repro drift_synthetic_oov
```

## Running the Monitoring Stack
The monitoring stack can be launched independently or as part of the full stack via Docker Compose:
```bash
export RECSYS_API_KEYS=<api-key>
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker-compose up -d prometheus grafana
```
