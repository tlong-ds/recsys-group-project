from __future__ import annotations

from recsys.serving.predictor import Predictor


class _ModelWithVocab:
    _item_to_idx = {1: 1, 2: 2}


class _ModelWithNItemsOnly:
    n_items = 3


class _ModelWithoutCatalog:
    pass


def test_input_quality_uses_explicit_item_vocab() -> None:
    predictor = Predictor(_ModelWithVocab())

    quality = predictor.input_quality([1, 2, 999])

    assert quality["sequence_length"] == 3
    assert quality["known_items"] == 2
    assert quality["unknown_items"] == 1
    assert quality["oov_ratio"] == 1 / 3
    assert quality["known_catalog_items"] == 2


def test_input_quality_falls_back_to_n_items_range() -> None:
    predictor = Predictor(_ModelWithNItemsOnly())

    quality = predictor.input_quality([1, 3, 4])

    assert quality["known_items"] == 2
    assert quality["unknown_items"] == 1
    assert quality["known_catalog_items"] == 3


def test_input_quality_treats_items_as_unknown_without_catalog_metadata() -> None:
    predictor = Predictor(_ModelWithoutCatalog())

    quality = predictor.input_quality([1, 2])

    assert quality["known_items"] == 0
    assert quality["unknown_items"] == 2
    assert quality["known_catalog_items"] == 0

