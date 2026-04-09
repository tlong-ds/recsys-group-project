"""Tests for Preprocessor."""

import pytest

from recsys.data.preprocessor import Preprocessor


class TestPreprocessor:
    def test_fit_not_implemented(self):
        prep = Preprocessor()
        with pytest.raises(NotImplementedError):
            prep.fit(None)  # type: ignore[arg-type]

    def test_transform_not_implemented(self):
        prep = Preprocessor()
        with pytest.raises(NotImplementedError):
            prep.transform(None)  # type: ignore[arg-type]
