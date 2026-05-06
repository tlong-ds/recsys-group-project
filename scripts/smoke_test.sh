#!/usr/bin/env bash
set -euo pipefail

# Post-deploy smoke test for RecSys API
# Verifies health, readiness, and authenticated endpoints.

URL=""
API_KEY=""
TIMEOUT=30

usage() {
  cat <<EOF
Usage: $0 --url <api_url> --api-key <key> [--timeout <seconds>]

Options:
  --url       Base URL of the RecSys API (e.g., http://localhost:8000)
  --api-key   Valid API key for authenticated routes
  --timeout   Maximum time in seconds to wait for each check (default: 30)
EOF
  exit 1
}

log() {
  echo "[smoke-test] $*"
}

die() {
  echo "[smoke-test] ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="${2%/}"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$URL" ]] && die "Missing --url"
[[ -z "$API_KEY" ]] && die "Missing --api-key"

check_endpoint() {
  local path="$1"
  local method="${2:-GET}"
  local auth="${3:-false}"
  local data="${4:-}"
  
  local full_url="${URL}${path}"
  local curl_opts=("-s" "-o" "/dev/null" "-w" "%{http_code}" "--max-time" "$TIMEOUT")
  
  if [[ "$auth" == "true" ]]; then
    curl_opts+=("-H" "Authorization: Bearer $API_KEY")
  fi
  
  if [[ -n "$data" ]]; then
    curl_opts+=("-X" "POST" "-H" "Content-Type: application/json" "-d" "$data")
  else
    curl_opts+=("-X" "$method")
  fi

  log "Checking $method $path (Auth: $auth)..."
  local status
  status=$(curl "${curl_opts[@]}" "$full_url")

  if [[ "$status" -eq 200 || "$status" -eq 202 ]]; then
    log "  PASS: $path returned $status"
  else
    die "  FAIL: $path returned $status"
  fi
}

log "Starting smoke tests for $URL"

# 1. Public Health Check
check_endpoint "/health"

# 2. Public Readiness Check
check_endpoint "/ready"

# 3. Authenticated Recommendation Check
RECOMMEND_DATA='{"item_sequence": [1, 2, 3], "top_k": 5}'
check_endpoint "/recommend" "POST" "true" "$RECOMMEND_DATA"

# 4. Authenticated Metrics Check
check_endpoint "/metrics" "GET" "true"

log "ALL SMOKE TESTS PASSED"
