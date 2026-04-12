# API Contract

## `GET /health`

Returns service status and the resolved model path.

## `POST /recommend`

Request body:

```json
{
  "session_id": "optional-session-id",
  "item_sequence": [101, 205, 330],
  "top_k": 10
}
```

Response body:

```json
{
  "session_id": "optional-session-id",
  "item_sequence": [101, 205, 330],
  "recommendations": [411, 412, 413]
}
```
