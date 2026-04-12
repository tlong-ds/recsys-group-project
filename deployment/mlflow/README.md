# MLflow Setup

The local compose stack exposes MLflow on port `5000`.

Default assumptions:

- backend store: `sqlite:///mlflow.db`
- artifact root: `/mlflow/artifacts`
- experiment name: `recsys_srgnn`

This folder is the mounted persistence location for the local service.
