#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="recsys-local"
COMPOSE=(docker compose --profile observability)
SECRETS_DIR="deployment/secrets"
API_KEY_FILE="${SECRETS_DIR}/recsys-api-key"
LOCAL_MODEL_DIR="models/trained/v1_strict_filter/latest"

usage() {
  cat <<'USAGE'
Usage: scripts/local_compose.sh {up|down|reset|logs|status}

Commands:
  up      Validate local env, write the Prometheus API key file, and start API, frontend, Prometheus, and Grafana.
  down    Stop the local compose stack without deleting volumes.
  reset   Stop the stack and remove only this compose project's named volumes.
  logs    Follow logs for the local compose stack.
  status  Show local compose service status.
USAGE
}

load_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
}

require_env() {
  local missing=()
  for name in RECSYS_API_KEYS GRAFANA_ADMIN_PASSWORD; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'Missing required env vars: %s\n' "${missing[*]}" >&2
    printf 'Set them in .env or export them before running this script.\n' >&2
    exit 1
  fi
}

require_paths() {
  if [[ ! -f "${LOCAL_MODEL_DIR}/model.pt" ]]; then
    printf 'Missing local serving model artifact: %s/model.pt\n' "${LOCAL_MODEL_DIR}" >&2
    printf 'Run or fetch the local training artifacts before starting compose.\n' >&2
    exit 1
  fi
}

write_api_key_file() {
  local first_key
  first_key="${RECSYS_API_KEYS%%,*}"
  first_key="${first_key#"${first_key%%[![:space:]]*}"}"
  first_key="${first_key%"${first_key##*[![:space:]]}"}"

  if [[ -z "${first_key}" ]]; then
    printf 'RECSYS_API_KEYS is set but does not contain a usable key.\n' >&2
    exit 1
  fi

  mkdir -p "${SECRETS_DIR}"
  printf '%s' "${first_key}" > "${API_KEY_FILE}"
  chmod 0600 "${API_KEY_FILE}"
}

compose_volumes() {
  docker volume ls \
    --filter "label=com.docker.compose.project=${PROJECT_NAME}" \
    --format '{{.Name}}'
}

cmd="${1:-}"
case "${cmd}" in
  up)
    load_env
    require_env
    require_paths
    write_api_key_file
    "${COMPOSE[@]}" config --quiet
    "${COMPOSE[@]}" up --build -d api frontend prometheus grafana
    "${COMPOSE[@]}" ps
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  reset)
    "${COMPOSE[@]}" down
    volumes="$(compose_volumes)"
    if [[ -n "${volumes}" ]]; then
      printf '%s\n' "${volumes}" | xargs docker volume rm
    fi
    ;;
  logs)
    "${COMPOSE[@]}" logs -f "${@:2}"
    ;;
  status)
    "${COMPOSE[@]}" ps
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
