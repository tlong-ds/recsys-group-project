"""Tests for ContentBasedModel."""

import pytest

from recsys.models.content_based import ContentBasedModel


class TestContentBasedModel:
    def test_fit_not_implemented(self):
        model = ContentBasedModel()
        with pytest.raises(NotImplementedError):
            model.fit(None)  # type: ignore[arg-type]

    def test_predict_not_implemented(self):
        model = ContentBasedModel()
        with pytest.raises(NotImplementedError):
            model.predict([], [])

    def test_recommend_not_implemented(self):
        model = ContentBasedModel()
        with pytest.raises(NotImplementedError):
            model.recommend(user_id=1)
