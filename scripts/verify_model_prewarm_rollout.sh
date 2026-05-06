#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="recsys"
DEPLOYMENT="recsys-api"
LABEL_SELECTOR="app=recsys-api"
PREWARM_JOB=""
REQUIRED_PODS=2
TIMEOUT_SECONDS=600

usage() {
  cat <<'EOF'
Verify prewarmed model rollout consistency and cache hits across API pods.

Usage:
  scripts/verify_model_prewarm_rollout.sh [options]

Options:
  --namespace <name>         Kubernetes namespace (default: recsys)
  --deployment <name>        Deployment name (default: recsys-api)
  --label-selector <expr>    Pod label selector (default: app=recsys-api)
  --prewarm-job <name>       Explicit prewarm Job name (default: newest recsys-model-prewarm job)
  --required-pods <count>    Number of ready pods to verify (default: 2)
  --timeout-seconds <secs>   Wait timeout for rollout/job checks (default: 600)
  --help                     Show this help
EOF
}

log() {
  printf '[verify-rollout] %s\n' "$*"
}

die() {
  printf '[verify-rollout] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --namespace)
        NAMESPACE="$2"
        shift 2
        ;;
      --deployment)
        DEPLOYMENT="$2"
        shift 2
        ;;
      --label-selector)
        LABEL_SELECTOR="$2"
        shift 2
        ;;
      --prewarm-job)
        PREWARM_JOB="$2"
        shift 2
        ;;
      --required-pods)
        REQUIRED_PODS="$2"
        shift 2
        ;;
      --timeout-seconds)
        TIMEOUT_SECONDS="$2"
        shift 2
        ;;
      --help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

extract_deploy_env() {
  python3 - "$NAMESPACE" "$DEPLOYMENT" <<'PY'
import json
import subprocess
import sys

namespace, deployment = sys.argv[1], sys.argv[2]
payload = json.loads(
    subprocess.check_output(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "deployment",
            deployment,
            "-o",
            "json",
        ],
        text=True,
    )
)
containers = payload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
api = next((c for c in containers if c.get("name") == "api"), None)
if not api:
    raise SystemExit("missing api container in deployment spec")

env_map = {}
for item in api.get("env", []):
    name = item.get("name")
    if name:
        env_map[name] = item.get("value", "")

for key in (
    "RECSYS_DEPLOY_MODEL_NAME",
    "RECSYS_DEPLOY_MODEL_VERSION",
    "RECSYS_DEPLOY_RUN_ID",
    "RECSYS_MODEL_CACHE_ROOT",
):
    print(f"{key}={env_map.get(key, '')}")
PY
}

find_latest_prewarm_job() {
  python3 - "$NAMESPACE" <<'PY'
import json
import subprocess
import sys

namespace = sys.argv[1]
payload = json.loads(
    subprocess.check_output(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "jobs",
            "-l",
            "app=recsys-model-prewarm",
            "-o",
            "json",
        ],
        text=True,
    )
)
items = payload.get("items", [])
if not items:
    raise SystemExit(1)

items.sort(key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
print(items[-1].get("metadata", {}).get("name", ""))
PY
}

choose_ready_pods() {
  local required="$1"
  python3 - "$NAMESPACE" "$LABEL_SELECTOR" "$required" <<'PY'
import json
import subprocess
import sys

namespace, label_selector, required = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.loads(
    subprocess.check_output(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            label_selector,
            "-o",
            "json",
        ],
        text=True,
    )
)

ready_running = []
for pod in payload.get("items", []):
    if pod.get("status", {}).get("phase") != "Running":
        continue
    conds = pod.get("status", {}).get("conditions", [])
    is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds)
    if not is_ready:
        continue
    ready_running.append(pod)

ready_running.sort(key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""), reverse=True)
if len(ready_running) < required:
    raise SystemExit(2)

for pod in ready_running[:required]:
    print(pod.get("metadata", {}).get("name", ""))
PY
}

check_logs_for_marker() {
  local logs="$1"
  local needle="$2"
  grep -Fq "$needle" <<<"$logs"
}

main() {
  parse_args "$@"

  require_cmd kubectl
  require_cmd python3

  log "Waiting for rollout to complete: deployment/$DEPLOYMENT"
  kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout="${TIMEOUT_SECONDS}s" >/dev/null

  local model_name=""
  local model_version=""
  local run_id=""
  local cache_root=""

  while IFS='=' read -r key value; do
    case "$key" in
      RECSYS_DEPLOY_MODEL_NAME) model_name="$value" ;;
      RECSYS_DEPLOY_MODEL_VERSION) model_version="$value" ;;
      RECSYS_DEPLOY_RUN_ID) run_id="$value" ;;
      RECSYS_MODEL_CACHE_ROOT) cache_root="$value" ;;
    esac
  done < <(extract_deploy_env)

  [[ -n "$model_name" ]] || die "Deployment is missing RECSYS_DEPLOY_MODEL_NAME"
  [[ -n "$model_version" ]] || die "Deployment is missing RECSYS_DEPLOY_MODEL_VERSION"
  [[ -n "$run_id" ]] || die "Deployment is missing RECSYS_DEPLOY_RUN_ID"
  [[ -n "$cache_root" ]] || die "Deployment is missing RECSYS_MODEL_CACHE_ROOT"

  log "Pinned rollout target: model=$model_name version=$model_version run_id=$run_id cache_root=$cache_root"

  if [[ -z "$PREWARM_JOB" ]]; then
    PREWARM_JOB="$(find_latest_prewarm_job)" || die "No prewarm jobs found with label app=recsys-model-prewarm"
  fi

  log "Checking prewarm job completion: job/$PREWARM_JOB"
  kubectl -n "$NAMESPACE" wait --for=condition=Complete "job/$PREWARM_JOB" --timeout="${TIMEOUT_SECONDS}s" >/dev/null

  local prewarm_logs
  prewarm_logs="$(kubectl -n "$NAMESPACE" logs "job/$PREWARM_JOB" --tail=500)"

  check_logs_for_marker "$prewarm_logs" "'model_name': '$model_name'" || die "Prewarm logs do not match pinned model_name"
  check_logs_for_marker "$prewarm_logs" "'model_version': '$model_version'" || die "Prewarm logs do not match pinned model_version"
  check_logs_for_marker "$prewarm_logs" "'run_id': '$run_id'" || die "Prewarm logs do not match pinned run_id"

  if check_logs_for_marker "$prewarm_logs" "'status': 'downloaded'"; then
    log "Prewarm job downloaded artifacts into shared cache"
  elif check_logs_for_marker "$prewarm_logs" "'status': 'hit'"; then
    log "Prewarm job found warmed artifacts already present"
  else
    die "Prewarm job logs did not report cache status"
  fi

  local -a pods
  mapfile -t pods < <(choose_ready_pods "$REQUIRED_PODS") || die "Not enough ready running pods found for selector: $LABEL_SELECTOR"

  log "Verifying init-container cache hits on ${#pods[@]} pods"
  local pod
  for pod in "${pods[@]}"; do
    local init_logs
    init_logs="$(kubectl -n "$NAMESPACE" logs "$pod" -c model-downloader --tail=500)"

    check_logs_for_marker "$init_logs" "'status': 'hit'" || die "Pod $pod did not report cache hit in model-downloader logs"
    check_logs_for_marker "$init_logs" "'model_name': '$model_name'" || die "Pod $pod logs do not match pinned model_name"
    check_logs_for_marker "$init_logs" "'model_version': '$model_version'" || die "Pod $pod logs do not match pinned model_version"
    check_logs_for_marker "$init_logs" "'run_id': '$run_id'" || die "Pod $pod logs do not match pinned run_id"
    log "Pod $pod passed cache-hit + pin consistency checks"
  done

  log "PASS: Prewarm and ${#pods[@]}-pod rollout consistency verified"
}

main "$@"
