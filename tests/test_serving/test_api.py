"""Tests for the FastAPI API with Prometheus instrumentation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from recsys.serving.api import create_app


class TestApiMonitoring(unittest.TestCase):
    def setUp(self) -> None:
        # Mock the predictor and model path resolver
        self.mock_predictor = MagicMock()
        self.mock_predictor.get_recommendations.return_value = [1, 2, 3]

        with patch("recsys.serving.api._resolve_model_path", return_value="/tmp/model"):
            with patch("recsys.serving.predictor.Predictor.from_path", return_value=self.mock_predictor):
                self.app = create_app()
                self.client = TestClient(self.app)

    def test_metrics_endpoint_is_available(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("http_requests_total", response.text)

    def test_recommend_increments_custom_metric(self) -> None:
        # 1. Check initial state of the metric (if possible, but usually just check after increment)
        # 2. Call recommend
        payload = {
            "session_id": "test_session",
            "item_sequence": [1, 2, 3],
            "top_k": 5
        }
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 200)

        # 3. Check /metrics for our custom metric
        metrics_response = self.client.get("/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        self.assertIn("recsys_recommendations_total_total", metrics_response.text)
