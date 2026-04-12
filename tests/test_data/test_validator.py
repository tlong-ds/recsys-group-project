"""Tests for DataValidator."""

from __future__ import annotations

import unittest

import pandas as pd

from recsys.data.validator import DataValidator


class TestDataValidator(unittest.TestCase):
    def test_validate_interactions_raises_on_missing_columns(self) -> None:
        validator = DataValidator()
        interactions = pd.DataFrame({"session_id": [1], "item_id": [10]})

        with self.assertRaises(ValueError):
            validator.validate_interactions(interactions)
