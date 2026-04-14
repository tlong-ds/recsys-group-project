# Serving the Recommendation System

This document outlines how the RecSys model is served using FastAPI.

## Overview
The recommendation system uses **FastAPI** to provide a RESTful API for session-based recommendations. The API loads a trained model and serves predictions over HTTP.

## Key Components
- **Framework**: FastAPI
- **Main Application**: `src/recsys/serving/api.py`
- **Predictor**: `src/recsys/serving/predictor.py` wraps the model inference logic.
- **Schemas**: `src/recsys/serving/schemas.py` defines the Pydantic models for request and response validation.
- **Server**: Uvicorn is used as the ASGI web server.

## Configuration
Serving is configured via `configs/serving_config.yaml` or CLI arguments.
Key settings:
- `host`: Host IP (default: `0.0.0.0`)
- `port`: Serving port (default: `8000`)
- `model_path`: Path to the trained model directory or file (default: `models/trained/latest/`)
- `default_top_k`: Default number of recommendations to return.

## Endpoints
- `GET /health`: Returns the health status of the API and the loaded model path.
- `POST /recommend`: Accepts a sequence of items (`item_sequence`) and returns a list of recommended items for the session.

## Running Locally
To run the server locally, you can use the provided CLI entrypoint or Docker Compose:
```bash
# Using Python
python -m recsys.serving.api --config configs/serving_config.yaml

# Using Docker Compose
docker-compose up api
```