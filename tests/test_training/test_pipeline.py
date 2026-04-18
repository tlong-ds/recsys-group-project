"""Tests for training pipeline example normalization."""

from __future__ import annotations

import unittest

import pandas as pd

from recsys.training.helper import normalize_examples


class TestTrainingPipelineNormalization(unittest.TestCase):
    def test_normalize_sequence_examples(self) -> None:
        raw = pd.DataFrame(
            {
                "input_items": [[11, 12], [21]],
                "target_item": [13, 22],
            }
        )

        normalized = normalize_examples(raw)

        self.assertEqual(list(normalized["input_items"]), [[11, 12], [21]])
        self.assertEqual(list(normalized["context_items"]), [[11, 12], [21]])
        self.assertEqual(list(normalized["last_item_id"]), [12, 21])
        self.assertEqual(list(normalized["target_item"]), [13, 22])

    def test_normalize_graph_examples(self) -> None:
        raw = pd.DataFrame(
            {
                "x": [[4, 7, 9]],
                "alias_inputs": [[0, 1, 2]],
                "pos_items": [10],
            }
        )

        normalized = normalize_examples(raw)

        self.assertEqual(list(normalized["input_items"]), [[4, 7, 9]])
        self.assertEqual(list(normalized["context_items"]), [[4, 7, 9]])
        self.assertEqual(list(normalized["last_item_id"]), [9])
        self.assertEqual(list(normalized["target_item"]), [10])
