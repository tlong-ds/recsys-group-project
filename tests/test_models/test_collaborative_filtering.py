"""Tests for CollaborativeFilteringModel."""

import pytest

from recsys.models.collaborative_filtering import CollaborativeFilteringModel


class TestCollaborativeFilteringModel:
    def test_fit_not_implemented(self):
        model = CollaborativeFilteringModel()
        with pytest.raises(NotImplementedError):
            model.fit(None)  # type: ignore[arg-type]

    def test_predict_not_implemented(self):
        model = CollaborativeFilteringModel()
        with pytest.raises(NotImplementedError):
            model.predict([], [])

    def test_recommend_not_implemented(self):
        model = CollaborativeFilteringModel()
        with pytest.raises(NotImplementedError):
            model.recommend(user_id=1)
