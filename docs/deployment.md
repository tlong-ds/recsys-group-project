# Deployment

This document describes the deployment strategy for the recommendation system, focusing on Docker and Kubernetes.

## Docker Deployment
The project includes a `Dockerfile` and a `docker-compose.yaml` file to containerize the application and its dependencies.

### Services in Docker Compose
- **api**: The FastAPI application serving the recommendations (Port `8000`). It mounts the models and config directories.
- **mlflow**: MLflow tracking server for model registry and experiment tracking (Port `5000`).
- **prometheus**: Prometheus server for scraping and storing metrics (Port `9090`).
- **grafana**: Grafana dashboard for visualizing metrics (Port `3000`).

To start the full stack:
```bash
export RECSYS_API_KEYS=<comma-separated-api-keys>
export GRAFANA_ADMIN_PASSWORD=<strong-password>
printf '%s' "${RECSYS_API_KEYS%%,*}" > deployment/secrets/recsys-api-key
docker-compose up -d
```

Compose binds MLflow, Prometheus, and Grafana to `127.0.0.1` by default.
The API still binds to port `8000` for local clients, but `/recommend` and
`/metrics` require the bearer token configured in `RECSYS_API_KEYS`.

## Kubernetes Deployment
Kubernetes manifests are located in the `deployment/kubernetes/` directory.

### Components
- **Deployment**: `api-deployment.yaml` defines the deployment for the `recsys-api`, ensuring a replica is running and configured with the correct environment variables (e.g., `RECSYS_MODEL_PATH`).
- **Service**: `api-service.yaml` exposes the FastAPI deployment to be accessible within or outside the cluster.

## CI/CD container publishing (GHCR)

The repository includes a dedicated publish workflow:

- `.github/workflows/publish-image.yml`

Behavior:

1. Runs after workflow `ci` completes successfully on `main`.
2. Builds Docker image from `Dockerfile`.
3. Pushes to `ghcr.io/<owner>/<repo>` with tags:
   - `main`
   - `sha-<short_sha>`

To pull images from private GHCR repositories, authenticate first:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
docker pull ghcr.io/<owner>/<repo>:main
```

## Production model registry flow

For production, prefer MLflow Model Registry as the deployment source of truth:
1. Train and log run artifacts with MLflow.
2. Register the run model in MLflow Model Registry.
3. Promote the selected version by alias (for example `Production`).
4. Configure serving with `serving.model_registry.enabled=true` and the target alias/version.

Required runtime secrets/environment:
- `RECSYS_API_KEYS` (comma-separated API keys for serving auth)
- `DAGSHUB_USER_TOKEN` (or explicit MLflow credentials env vars)
- any DVC/data storage credentials used by your environment

The Kubernetes `recsys-secrets` Secret must contain:
- `api-keys`
- `dagshub-user-token`

To deploy to a Kubernetes cluster:
```bash
kubectl apply -f deployment/kubernetes/
```

The Kubernetes deployment runs as a non-root user, defines health probes and
resource bounds, and includes a NetworkPolicy that only allows selected
in-cluster clients such as Prometheus or pods labeled `app=recsys-api-client`.
