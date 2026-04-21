# Recommendation API Input Dictionary

The RecSys API accepts a JSON dictionary representing a session's interaction history. This dictionary is validated using Pydantic.

## RecommendRequest Schema

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `session_id` | `string` | No | Optional session identifier echoed in the response for client-side correlation. |
| `item_sequence` | `list[integer]` | **Yes** | An ordered list of item IDs clicked in the current session. At least one item is required. |
| `top_k` | `integer` | No | Number of recommendations to return. Default: `10`. Range: `1-100`. |

### Example Request
```json
{
  "session_id": "session_12345",
  "item_sequence": [101, 102, 105, 101],
  "top_k": 5
}
```

## RecommendResponse Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `session_id` | `string` | The session identifier provided in the request. |
| `item_sequence` | `list[integer]` | The input item sequence reflected back. |
| `recommendations` | `list[integer]` | A ranked list of predicted next-item IDs. |

### Example Response
```json
{
  "session_id": "session_12345",
  "item_sequence": [101, 102, 105, 101],
  "recommendations": [201, 150, 99, 305, 42]
}
```

## Internal Transformation
Internally, the `item_sequence` is converted into the same graph tensor shape
used by offline evaluation: unique node IDs, directed transition edges, and an
alias map from sequence positions to graph nodes. The predictor loads the model
family described by artifact metadata (`model.json`) and dispatches to the
matching SR-GNN, TAGNN, or GGNN recommender implementation.

The API records request-level monitoring signals before returning the response:
input sequence length, requested `top_k`, number of unknown/OOV items, request
outcome, and prediction latency.
