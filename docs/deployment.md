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
docker-compose up -d
```

## Kubernetes Deployment
Kubernetes manifests are located in the `deployment/kubernetes/` directory.

### Components
- **Deployment**: `api-deployment.yaml` defines the deployment for the `recsys-api`, ensuring a replica is running and configured with the correct environment variables (e.g., `RECSYS_MODEL_PATH`).
- **Service**: `api-service.yaml` exposes the FastAPI deployment to be accessible within or outside the cluster.

## Production model registry flow

For production, prefer MLflow Model Registry as the deployment source of truth:
1. Train and log run artifacts with MLflow.
2. Register the run model in MLflow Model Registry.
3. Promote the selected version by alias (for example `Production`).
4. Configure serving with `serving.model_registry.enabled=true` and the target alias/version.

Required runtime secrets/environment:
- `DAGSHUB_TOKEN` (or explicit MLflow credentials env vars)
- any DVC/data storage credentials used by your environment

To deploy to a Kubernetes cluster:
```bash
kubectl apply -f deployment/kubernetes/
```
