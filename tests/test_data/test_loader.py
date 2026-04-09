"""Tests for DataLoader."""

import pytest

from recsys.data.loader import DataLoader


class TestDataLoader:
    def test_load_interactions_not_implemented(self, tmp_path):
        loader = DataLoader(raw_path=tmp_path)
        with pytest.raises(NotImplementedError):
            loader.load_interactions()

    def test_load_items_not_implemented(self, tmp_path):
        loader = DataLoader(raw_path=tmp_path)
        with pytest.raises(NotImplementedError):
            loader.load_items()

    def test_load_users_not_implemented(self, tmp_path):
        loader = DataLoader(raw_path=tmp_path)
        with pytest.raises(NotImplementedError):
            loader.load_users()
