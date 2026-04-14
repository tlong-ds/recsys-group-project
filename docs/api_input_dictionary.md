# Recommendation API Input Dictionary

The RecSys API accepts a JSON dictionary representing a session's interaction history. This dictionary is validated using Pydantic.

## RecommendRequest Schema

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `session_id` | `string` | No | A unique identifier for the user session. Used for tracking. |
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
Internally, the `item_sequence` is converted into a session graph using **PyTorch Geometric**. The graph is directed, where an edge `(u, v)` exists if item `v` was clicked immediately after item `u`. This graph is then processed by the SR-GNN (Gated Graph Convolution) model to generate embeddings for the session and produce next-item scores via dot-product with the item catalog.
