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
- `preload_model_on_startup`: If `true`, loads the predictor during API startup to reduce first-request latency.
- `default_top_k`: Default number of recommendations to return.
- `model_registry.enabled`: Use MLflow Model Registry for model resolution.
- `model_registry.model_name`: Registered model name in MLflow.
- `model_registry.model_alias` or `model_registry.model_version`: Selector for deployable model.
- `model_registry.artifact_path`: Artifact directory downloaded from selected run (default: `registered_model`).
- `model_registry.local_cache_dir`: Optional persistent cache directory for downloaded registry artifacts.
- `model_registry.fallback_to_filesystem`: Defaults to `false` for registry-only serving.
- `model_path`: Optional legacy fallback path used only if filesystem fallback is explicitly enabled.
- `cors.allowed_origins`: Browser origins allowed to call the API. Defaults include `http://0.0.0.0:5173`, `http://localhost:5173`, and `http://127.0.0.1:5173` for local Vite development.
- `security.enabled`: Require API-key auth for protected endpoints.
- `security.api_keys_env_var`: Environment variable containing comma-separated API keys.
- `security.public_paths`: Paths that stay public. Default: `/health` and `/ready`.
- `security.rate_limit_per_minute`: Per-key in-memory request limit.
- `security.max_body_bytes`: Maximum declared request body size.
- `security.docs_enabled`: Expose or disable FastAPI docs/OpenAPI routes.

Model source note:
- Production serving should resolve only from MLflow Registry (`recsys-serving`).
- `metrics/promotion_result.json` is the deployment pin contract (`model_name`, `model_version`, `run_id`).
- Local filesystem loading is supported only as an explicit compatibility fallback.

## Endpoints
- `GET /health`: Public liveness endpoint with sanitized model status.
- `GET /ready`: Public readiness endpoint that returns `503` when the model is unavailable.
- `POST /recommend`: Authenticated recommendation endpoint.
- `GET /metrics`: Authenticated Prometheus metrics endpoint.
- `GET /products`: Authenticated product catalog endpoint.
- `POST /views`: Authenticated user view logging endpoint.
- `GET /evaluations`: Authenticated offline evaluation summary endpoint.

When model registry loading is enabled, `/health` exposes only source metadata
used for deployment checks (`model_source`, `model_name`, `model_version`,
`run_id`). It does not return local artifact paths or raw exceptions.

`/recommend` also records online monitoring signals for Prometheus, including
request outcome, prediction latency, input sequence length, requested `top_k`,
and OOV item counts against the loaded model catalog. These online signals are
separate from the offline benchmark replay drift reports in `recsys.monitoring`.

For CT-driven deployments, the release should pin `RECSYS_DEPLOY_MODEL_NAME`,
`RECSYS_DEPLOY_MODEL_VERSION`, and `RECSYS_DEPLOY_RUN_ID` from
`metrics/promotion_result.json` so prewarm and API resolve the same model.

## Running Locally
To run the server locally, you can use the provided CLI entrypoint or Docker Compose:
```bash
# Using Python
export RECSYS_API_KEYS=<your-api-key>
export RECSYS_MODEL_CACHE_ROOT=models/cache
python -m recsys.serving.api --config configs/serving_config.yaml

# Serve an explicit versioned artifact
python -m recsys.serving.api \
  --config configs/serving_config.yaml \
  --model-path models/trained/v1_strict_filter/latest/

# Using Docker Compose
export RECSYS_API_KEYS=<your-api-key>
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "$RECSYS_API_KEYS" > deployment/secrets/recsys-api-key
docker compose up api
```

For local Python runs, `RECSYS_MODEL_CACHE_ROOT` must point to a writable path.
The deployment config keeps `/app/models/cache` as the container path; Kubernetes
mounts the shared model-cache PVC there, and Docker Compose mounts a local named
volume at the same path.

Send protected requests with:

```bash
curl -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"item_sequence":[101,205,330],"top_k":10}' \
  http://localhost:8000/recommend
```
