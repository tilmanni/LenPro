from unittest.mock import Mock

import pytest

from lenpro.measures.global_measures import (
    EmptyOutputProbabilityMeasure,
    GlobalFractionRetainedEventsMeasure,
    GlobalFractionSyntheticEventsMeasure,
    GlobalMeasure,
    LengthUnchangedProbabilityMeasure,
    count_events,
)


def test_global_measure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        GlobalMeasure()


def test_count_events_ignores_none_values():
    data = ["a", None, "b", None, "c"]

    assert count_events(data) == 3


def test_count_events_returns_zero_for_empty_data():
    assert count_events([]) == 0
    assert count_events([None, None]) == 0


def test_empty_output_probability():
    measure = EmptyOutputProbabilityMeasure()

    original_dataset = [
        ["a", "b"],       # nonempty
        ["c"],            # nonempty
        [None, None],     # empty
        ["d"],            # nonempty
    ]
    perturbed_dataset = [
        [None, None],     # became empty: counted
        ["c"],            # remains nonempty
        [None],           # originally empty: not counted
        [],               # became empty: counted
    ]

    result = measure.process_dataset(
        original_dataset,
        perturbed_dataset,
    )

    assert result == pytest.approx(2 / 4)


def test_empty_output_probability_returns_zero_when_none_are_empty():
    measure = EmptyOutputProbabilityMeasure()

    original_dataset = [
        ["a"],
        ["b"],
    ]
    perturbed_dataset = [
        ["x"],
        ["y"],
    ]

    result = measure.process_dataset(
        original_dataset,
        perturbed_dataset,
    )

    assert result == 0.0


def test_empty_output_probability_returns_zero_for_empty_dataset():
    measure = EmptyOutputProbabilityMeasure()

    assert measure.process_dataset([], []) == 0.0


def test_empty_output_probability_rejects_different_dataset_sizes():
    measure = EmptyOutputProbabilityMeasure()

    with pytest.raises(ValueError, match="same length"):
        measure.process_dataset(
            original_dataset=[["a"], ["b"]],
            perturbed_dataset=[["a"]],
        )


def test_empty_output_probability_to_dict():
    measure = EmptyOutputProbabilityMeasure()

    assert measure.to_dict() == {
        "name": "EmptyOutputProbabilityMeasure",
    }


def test_length_unchanged_probability():
    measure = LengthUnchangedProbabilityMeasure()

    original_dataset = [
        ["a", "b"],       # length 2
        ["c"],            # length 1
        [None, "d"],      # length 1
        [],               # length 0
    ]
    perturbed_dataset = [
        ["x", "y"],       # length 2: unchanged
        ["x", "y"],       # length 2: changed
        ["z", None],      # length 1: unchanged
        [None],           # length 0: unchanged
    ]

    result = measure.process_dataset(
        original_dataset,
        perturbed_dataset,
    )

    assert result == pytest.approx(3 / 4)


def test_length_unchanged_probability_returns_zero_for_empty_dataset():
    measure = LengthUnchangedProbabilityMeasure()

    assert measure.process_dataset([], []) == 0.0


def test_length_unchanged_rejects_different_dataset_sizes():
    measure = LengthUnchangedProbabilityMeasure()

    with pytest.raises(ValueError, match="same length"):
        measure.process_dataset(
            original_dataset=[["a"]],
            perturbed_dataset=[],
        )


def test_length_unchanged_to_dict():
    measure = LengthUnchangedProbabilityMeasure()

    assert measure.to_dict() == {
        "name": "LengthUnchangedProbabilityMeasure",
    }


def test_global_fraction_retained_events():
    measure = GlobalFractionRetainedEventsMeasure(
        is_original=lambda event: event.startswith("original"),
    )

    original_dataset = [
        ["original-1", "original-2"],
        ["original-3", None],
    ]
    perturbed_dataset = [
        ["original-1", "synthetic-1"],
        [None, "original-3"],
    ]

    result = measure.process_dataset(
        original_dataset,
        perturbed_dataset,
    )

    # 3 original events in total, 2 retained
    assert result == pytest.approx(2 / 3)


def test_global_fraction_retained_returns_zero_without_original_events():
    measure = GlobalFractionRetainedEventsMeasure(
        is_original=lambda event: True,
    )

    result = measure.process_dataset(
        original_dataset=[
            [None],
            [],
        ],
        perturbed_dataset=[
            ["event"],
            [],
        ],
    )

    assert result == 0.0


def test_retained_predicate_is_not_called_for_none():
    predicate = Mock(return_value=True)
    measure = GlobalFractionRetainedEventsMeasure(predicate)

    measure.process_dataset(
        original_dataset=[["original"]],
        perturbed_dataset=[[None, "original", None]],
    )

    predicate.assert_called_once_with("original")


def test_global_fraction_retained_to_dict():
    measure = GlobalFractionRetainedEventsMeasure(
        is_original=lambda event: True,
    )

    assert measure.to_dict() == {
        "name": "GlobalFractionRetainedEventsMeasure",
    }


def test_global_fraction_synthetic_events():
    measure = GlobalFractionSyntheticEventsMeasure(
        is_synthetic=lambda event: event.startswith("synthetic"),
    )

    original_dataset = [
        ["original-1"],
        ["original-2"],
    ]
    perturbed_dataset = [
        ["original-1", "synthetic-1"],
        [None, "synthetic-2", "original-2"],
    ]

    result = measure.process_dataset(
        original_dataset,
        perturbed_dataset,
    )

    # 4 non-None perturbed events, 2 synthetic
    assert result == pytest.approx(2 / 4)


def test_global_fraction_synthetic_returns_zero_for_empty_output():
    measure = GlobalFractionSyntheticEventsMeasure(
        is_synthetic=lambda event: True,
    )

    result = measure.process_dataset(
        original_dataset=[
            ["original-1"],
            ["original-2"],
        ],
        perturbed_dataset=[
            [None],
            [],
        ],
    )

    assert result == 0.0


def test_synthetic_predicate_is_not_called_for_none():
    predicate = Mock(return_value=True)
    measure = GlobalFractionSyntheticEventsMeasure(predicate)

    measure.process_dataset(
        original_dataset=[["original"]],
        perturbed_dataset=[[None, "synthetic", None]],
    )

    predicate.assert_called_once_with("synthetic")


def test_global_fraction_synthetic_to_dict():
    measure = GlobalFractionSyntheticEventsMeasure(
        is_synthetic=lambda event: True,
    )

    assert measure.to_dict() == {
        "name": "GlobalFractionSyntheticEventsMeasure",
    }