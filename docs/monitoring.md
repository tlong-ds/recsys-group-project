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

## Running the Monitoring Stack
The monitoring stack can be launched independently or as part of the full stack via Docker Compose:
```bash
export RECSYS_API_KEYS=<api-key>
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker-compose up -d prometheus grafana
```
