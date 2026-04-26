# Deployment

This document describes the deployment strategy for the recommendation system, focusing on Docker and Kubernetes.

## Docker Deployment
The project includes a `Dockerfile` and a `docker-compose.yaml` file to containerize the application and its dependencies.

### Services in Docker Compose
- **api**: The FastAPI application serving recommendations (Port `8000`) with health checks.
- **frontend**: Optional Vite UI placeholder. The service is currently commented out in `docker-compose.yaml`, so the active local stack is API-first.
- **mlflow** *(profile: `tracking`)*: Local MLflow tracking server (Port `5000`).
- **prometheus** *(profile: `observability`)*: Metrics collection (Port `9090`).
- **grafana** *(profile: `observability`)*: Metrics visualization (Port `3000`).

To start local-first core services:
```bash
cp .env.example .env
# set at minimum: RECSYS_API_KEYS
docker compose up -d --build api
```

The default serving config is registry-first and expects a promoted canonical
model in MLflow (`recsys-serving`). Filesystem model paths are only a legacy
fallback option and are disabled by default.

To enable local observability:
```bash
# set GRAFANA_ADMIN_PASSWORD in .env
printf '%s' "${RECSYS_API_KEYS%%,*}" > deployment/secrets/recsys-api-key
docker compose --profile observability up -d prometheus grafana
```

To enable local MLflow:
```bash
docker compose --profile tracking up -d mlflow
```

Compose binds MLflow, Prometheus, and Grafana to `127.0.0.1` by default.
The API still binds to port `8000` for local clients, but `/recommend` and
`/metrics` require the bearer token configured in `RECSYS_API_KEYS`.

## Kubernetes Deployment
Kubernetes manifests are located in the `deployment/kubernetes/` directory.

### Components
- **Deployment**: `api-deployment.yaml` defines the deployment for the `recsys-api`, ensuring a replica is running and configured with the correct environment variables (e.g., `RECSYS_MODEL_PATH`).
- **Service**: `api-service.yaml` exposes the FastAPI deployment to be accessible within or outside the cluster.
- **Ingress**: `api-ingress.yaml` is an AWS ALB ingress example. The checked-in API ingress is HTTP-only; add host, certificate ARN, HTTPS listener, and ExternalDNS annotations before using it as a public production endpoint.
- **Namespace/Account/Config**: `namespace.yaml`, `api-service-account.yaml`, and `recsys-serving-configmap.yaml` provide EKS-ready runtime defaults.

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
3. Promote the selected winner into canonical model `recsys-serving` and point alias `Production`.
4. Write `metrics/promotion_result.json` with deploy pin triplet (`model_name`, `model_version`, `run_id`).
5. Configure serving with `serving.model_registry.enabled=true` and `fallback_to_filesystem=false`.

Required runtime secrets/environment:
- `RECSYS_API_KEYS` (comma-separated API keys for serving auth)
- `DAGSHUB_USER_TOKEN` (or explicit MLflow credentials env vars)
- any DVC/data storage credentials used by your environment

The Kubernetes `recsys-secrets` Secret must contain:
- `api-keys`
- `dagshub-user-token`

Create the secret (replace placeholders):
```bash
kubectl -n recsys create secret generic recsys-secrets \
  --from-literal=api-keys='<comma-separated-api-keys>' \
  --from-literal=dagshub-user-token='<dagshub-token>'
```

Deploy with kustomize:
```bash
kubectl apply -k deployment/kubernetes/
```

The Kubernetes deployment runs as a non-root user, defines health probes and
resource bounds, and includes a NetworkPolicy that only allows selected
in-cluster clients such as Prometheus or pods labeled `app=recsys-api-client`.

Before applying ingress in EKS, update:

- `deployment/kubernetes/api-ingress.yaml` host and ACM certificate ARN
- `deployment/kubernetes/grafana-ingress.example.yaml` host/certificate when Grafana service is deployed

## AWS EKS deployment (Terraform)

EKS bootstrap Terraform is provided in `deployment/terraform/` to provision:

1. VPC + subnets + NAT
2. EKS cluster + managed node group
3. IRSA roles for AWS Load Balancer Controller, ExternalDNS, cert-manager
4. Helm add-ons: ALB controller, ExternalDNS, cert-manager, metrics-server

Quick start:

```bash
cd deployment/terraform
cp terraform.tfvars.example terraform.tfvars
# update terraform.tfvars with your AWS region, subnet CIDRs, Route53 zone, and domain filters
terraform init
terraform plan
terraform apply
```

Deployment target decisions for this repository:

- **MLflow remains external on DagsHub** (no in-cluster MLflow deployment).
- **Ingress strategy**: public ALB with HTTPS.
- **DNS strategy**: Route53 records managed by ExternalDNS.

After cluster bootstrap, the next step is adapting `deployment/kubernetes/` manifests
for ingress hosts, certificates, and secret management in EKS.

## EKS runtime manifests included

The `deployment/kubernetes/` kustomization now includes:

- API runtime: deployment, service, network policy, ALB ingress
- Availability controls: `api-hpa.yaml`, `api-pdb.yaml`, zone spread/anti-affinity
- Monitoring runtime: Prometheus + Grafana deployments/services/pdbs and datasource/config maps

Create/update required secrets before applying:

```bash
kubectl -n recsys create secret generic recsys-secrets \
  --from-literal=api-keys='<comma-separated-api-keys>' \
  --from-literal=dagshub-user-token='<dagshub-token>' \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n recsys create secret generic recsys-monitoring-secrets \
  --from-literal=recsys-api-key='<single-api-key-for-prometheus>' \
  --from-literal=grafana-admin-password='<grafana-admin-password>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Apply everything:

```bash
kubectl apply -k deployment/kubernetes/
```

## EKS deployment CI/CD

Workflow `.github/workflows/deploy-eks.yml` performs EKS deployment after `publish-image` succeeds on `main`, and also supports manual `workflow_dispatch`.

Required GitHub repository variables:

- `AWS_REGION`
- `EKS_CLUSTER_NAME`

Required GitHub repository secrets:

- `AWS_ROLE_TO_ASSUME`
- `GHCR_USERNAME`
- `GHCR_TOKEN`
- `RECSYS_API_KEYS`
- `DAGSHUB_USER_TOKEN`
- `RECSYS_METRICS_API_KEY`
- `GRAFANA_ADMIN_PASSWORD`

## Continuous Training (CT) with model promotion gates

The repository includes the CT promotion helper `recsys-ct-promote`
(`src/recsys/training/ct_promote.py`). It is designed to be called after a
training/evaluation run has produced metrics JSON files.

The helper:

1. Reads the candidate training and evaluation metrics JSON outputs.
2. Applies quality gates (minimum `hr@k`, optional improvement over a baseline).
3. Promotes an MLflow registry alias (default `Production`) when gates pass.

Default baseline for the improvement gate is `metrics/evaluation_metrics.json`.
For a production schedule, wire this helper into a GitHub Actions workflow or an
external scheduler after the train/evaluate command finishes.

Example manual promotion command:

```bash
recsys-ct-promote \
  --training-config configs/training_config.yaml \
  --train-metrics-path metrics/v1_strict_filter/training_metrics.json \
  --evaluation-metrics-path metrics/v1_strict_filter/evaluation_metrics.json \
  --metric-key 'hr@k' \
  --min-threshold 0.45 \
  --target-alias Production
```

If CT is wired into GitHub Actions, the scheduler needs these repository secrets:

- `DAGSHUB_USER_TOKEN`
- `DAGSHUB_USERNAME`

If you want serving to load promoted registry models directly, enable
`serving.model_registry.enabled` and set `serving.model_registry.model_name` /
`model_alias` in serving config.

## Prewarm rollout verification harness

Use the built-in verifier to confirm the prewarm job and a multi-pod rollout
are consistent for one pinned model release:

```bash
scripts/verify_model_prewarm_rollout.sh
```

What it checks:

1. `deployment/recsys-api` rollout is complete.
2. Deployment env vars are pinned (`RECSYS_DEPLOY_MODEL_NAME`,
   `RECSYS_DEPLOY_MODEL_VERSION`, `RECSYS_DEPLOY_RUN_ID`,
   `RECSYS_MODEL_CACHE_ROOT`).
3. The latest prewarm Job (`app=recsys-model-prewarm`) completed and logged the
   same pinned model/version/run ID.
4. At least two ready API pods have `model-downloader` init logs with
   `'status': 'hit'` and matching model/version/run ID.

Optional flags:

```bash
scripts/verify_model_prewarm_rollout.sh \
  --namespace recsys \
  --deployment recsys-api \
  --label-selector app=recsys-api \
  --prewarm-job recsys-model-prewarm-<version>-<runid> \
  --required-pods 2 \
  --timeout-seconds 900
```

## Operations runbook (EKS + CT)

1. **Bootstrap**
   - Apply Terraform: `terraform -chdir=deployment/terraform apply`
   - Seed Kubernetes secrets (`recsys-secrets`, `recsys-monitoring-secrets`, `ghcr-pull-secret`)
   - Deploy manifests: `kubectl apply -k deployment/kubernetes/`

2. **Release / rollback**
   - Deploy via GitHub Actions `deploy-eks` workflow.
   - Rollback app quickly with:
     ```bash
     kubectl -n recsys rollout undo deployment/recsys-api
     kubectl -n recsys rollout status deployment/recsys-api
     ```
   - If infra changes caused issues, revert Terraform module inputs and re-apply.

3. **CT promotion controls**
   - Run `recsys-ct-promote` after ad-hoc retraining, or call it from an external schedule.
   - Promotion is blocked when quality gates fail (`min_hr_at_k`, optional baseline improvement).
   - To stop auto-promotion, disable the scheduler that calls the helper or set a stricter threshold.

4. **Secret rotation**
   - Rotate GitHub secrets first (`DAGSHUB_USER_TOKEN`, API keys, Grafana password).
   - Re-run `deploy-eks` so Kubernetes secrets are reconciled from GitHub Actions.
   - Confirm API and metrics auth health (`/health`, Prometheus target status).
