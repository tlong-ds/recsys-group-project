"""Tests for FeatureEngineer."""

import pytest

from recsys.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    def test_build_user_features_not_implemented(self):
        fe = FeatureEngineer()
        with pytest.raises(NotImplementedError):
            fe.build_user_features(None)  # type: ignore[arg-type]

    def test_build_item_features_not_implemented(self):
        fe = FeatureEngineer()
        with pytest.raises(NotImplementedError):
            fe.build_item_features(None, None)  # type: ignore[arg-type]
