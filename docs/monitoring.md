# Monitoring

This document details the monitoring stack used to track the health, performance, and usage of the recommendation system.

## Stack Overview
The monitoring stack relies on **Prometheus** for metrics collection and **Grafana** for visualization.

## Metrics Collection (Prometheus)
- **Library**: `prometheus-fastapi-instrumentator` is used in `src/recsys/serving/api.py` to automatically instrument FastAPI endpoints, collecting default metrics such as request latency, HTTP status codes, and request count.
- **Custom Metrics**: A custom Prometheus counter, `recsys_recommendations_total`, tracks the total number of successful recommendation requests served.
- **Configuration**: `deployment/monitoring/prometheus.yml` configures Prometheus to scrape metrics from the `api` service on port `8000` every 15 seconds.
- **Alerts**: Alerting rules are defined in `deployment/monitoring/alerts.yml` to trigger notifications when metrics cross specific thresholds.

## Visualization (Grafana)
Grafana is used to build dashboards visualizing the metrics collected by Prometheus.
- **Datasource**: Prometheus is provisioned as a default datasource via `deployment/monitoring/grafana/provisioning/datasources/prometheus.yml`.
- **Access**: The Grafana UI is available on port `3000` (default credentials configured via `GF_SECURITY_ADMIN_PASSWORD`).

## Running the Monitoring Stack
The monitoring stack can be launched independently or as part of the full stack via Docker Compose:
```bash
docker-compose up -d prometheus grafana
```