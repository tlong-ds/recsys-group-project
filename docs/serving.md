# Serving the Recommendation System

This document outlines how the RecSys model is served using FastAPI.

## Overview
The recommendation system uses **FastAPI** to provide a RESTful API for session-based recommendations. The API loads a trained model and serves predictions over HTTP.

## Key Components
- **Framework**: FastAPI
- **Main Application**: `src/recsys/serving/api.py`
- **Predictor**: `src/recsys/serving/predictor.py` wraps inference and auto-dispatches to SRGNN/TAGNN/GGNN loaders based on artifact metadata (`model.json`).
- **Schemas**: `src/recsys/serving/schemas.py` defines the Pydantic models for request and response validation.
- **Server**: Uvicorn is used as the ASGI web server.

## Configuration
Serving is configured via `configs/serving_config.yaml` or CLI arguments.
Key settings:
- `host`: Host IP (default: `0.0.0.0`)
- `port`: Serving port (default: `8000`)
- `model_path`: Path to the trained model directory or file (default: `models/trained/latest/`)
- `preload_model_on_startup`: If `true`, loads the predictor during API startup to reduce first-request latency.
- `default_top_k`: Default number of recommendations to return.
- `model_registry.enabled`: Use MLflow Model Registry for model resolution.
- `model_registry.model_name`: Registered model name in MLflow.
- `model_registry.model_alias` or `model_registry.model_version`: Selector for deployable model.
- `model_registry.artifact_path`: Artifact directory downloaded from selected run (default: `registered_model`).
- `model_registry.local_cache_dir`: Optional persistent cache directory for downloaded registry artifacts.
- `model_registry.fallback_to_filesystem`: If `true`, serving falls back to `model_path` when registry resolution fails.
- `security.enabled`: Require API-key auth for protected endpoints.
- `security.api_keys_env_var`: Environment variable containing comma-separated API keys.
- `security.public_paths`: Paths that stay public. Default: `/health`.
- `security.rate_limit_per_minute`: Per-key in-memory request limit.
- `security.max_body_bytes`: Maximum declared request body size.
- `security.docs_enabled`: Expose or disable FastAPI docs/OpenAPI routes.

## Endpoints
- `GET /health`: Public health endpoint with sanitized model status.
- `POST /recommend`: Authenticated recommendation endpoint.
- `GET /metrics`: Authenticated Prometheus metrics endpoint.

When model registry loading is enabled, `/health` exposes only non-sensitive
source metadata. It does not return local artifact paths, run IDs, or raw
exceptions.

## Running Locally
To run the server locally, you can use the provided CLI entrypoint or Docker Compose:
```bash
# Using Python
export RECSYS_API_KEYS=local-dev-key
python -m recsys.serving.api --config configs/serving_config.yaml

# Using Docker Compose
export RECSYS_API_KEYS=local-dev-key
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker-compose up api
```

Send protected requests with:

```bash
curl -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"item_sequence":[101,205,330],"top_k":10}' \
  http://localhost:8000/recommend
```
