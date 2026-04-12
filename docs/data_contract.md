# Data Contract

Required interaction schema:

| column | type | notes |
| --- | --- | --- |
| `session_id` | string or int | session boundary |
| `item_id` | int | catalog identifier |
| `timestamp` | datetime-compatible | ordering column |

Optional tables:

- `items.csv`
- `users.csv`

Expected raw file names are configured in `configs/data_config.yaml`.
