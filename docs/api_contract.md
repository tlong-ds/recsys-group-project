# API Contract

## `GET /health`

Public readiness endpoint. Returns service status without exposing local model
paths, run IDs, or raw exception details.

Response body:

```json
{
  "status": "ok",
  "model_source": "filesystem",
  "model_name": "",
  "model_version": "",
  "model_alias": ""
}
```

## Authentication

All endpoints except `/health` require an API key:

```http
Authorization: Bearer <api-key>
```

The service reads comma-separated keys from `RECSYS_API_KEYS` by default.

## `POST /recommend`

Requires authentication.

Request body:

```json
{
  "session_id": "optional-session-id",
  "item_sequence": [101, 205, 330],
  "top_k": 10
}
```

Constraints:
- `session_id`: optional, at most 128 characters
- `item_sequence`: 1-100 positive integer item IDs
- `top_k`: 1-100
- extra fields are rejected

Response body:

```json
{
  "session_id": "optional-session-id",
  "item_sequence": [101, 205, 330],
  "recommendations": [411, 412, 413]
}
```

## `GET /metrics`

Requires authentication. Prometheus must send the same bearer token as other
API clients.
