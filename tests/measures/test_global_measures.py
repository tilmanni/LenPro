from unittest.mock import Mock

import pytest
from itertools import zip_longest
from types import SimpleNamespace

from lenpro.protocols.ldp_protocol import LDPProtocol

from lenpro.measures.global_measures import (
    EmptyOutputProbabilityMeasure,
    GlobalFractionRetainedEventsMeasure,
    GlobalFractionSyntheticEventsMeasure,
    GlobalMeasure,
    LengthUnchangedProbabilityMeasure,
    MeanRelativeErrorFrequencyMeasure,
    count_events,
)

from lenpro.statistics.statistics import (
    Statistic,
    CountStatistic,
    EstimatorStatistic,
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

def make_count_statistic():
# Counts non-None in a column
    return CountStatistic()

def test_mean_relative_error_frequency_basic():
    # Two columns: col1 original has 2 events; col2 original has 1 event
    original = [
        ["a", "b"],
        ["x", None], 
    ]
    perturbed = [
        ["a", None],
        [None, None],
    ]
    measure = MeanRelativeErrorFrequencyMeasure(
        statistic=make_count_statistic(),
        estimator_name=None,
        sanity_bound=1,
    )
    result = measure.process_dataset(original, perturbed, protocol=None) 
    assert result == pytest.approx(0.75)

def test_mean_relative_error_frequency_handles_padding_with_zip_longest():
    # Different row lengths; padding with None should not inflate counts.
    original = [
        ["a", None, "b"], # length 3
        ["x"], # length 1
    ]
    perturbed = [
        [None, "p"], # length 2
        [None, None, "q"], # length 3
    ]
    # Columns (zipped longest):
    # c1: ("a","x") -> orig 2, pert 0
    # c2: (None,None) -> orig 0, pert 1
    # c3: ("b",None) -> orig 1, pert 1
    # REs (sanity_bound=1):
    # c1: |2-0|/max(2,1)=1
    # c2: |0-1|/max(0,1)=1
    # c3: |1-1|/max(1,1)=0
    # mean = (1+1+0)/3 = 2/3
    measure = MeanRelativeErrorFrequencyMeasure(
        statistic=make_count_statistic(),
        estimator_name=None,
        sanity_bound=1,
    )
    result = measure.process_dataset(original, perturbed, protocol=None)
    assert result == pytest.approx(2/3)

def test_mean_relative_error_frequency_uses_estimator_when_name_and_protocol_given(monkeypatch):
    # Use a dummy protocol with an estimator attribute; validate_estimator returns it.
    class DummyEstimator:
        def estimate(self, data):
            # returns number of non-None, same as CountStatistic, but exercise estimator path
            return sum(x is not None for x in data)
        def to_dict(self):
            return {"name": "DummyEstimator"}
    protocol = Mock(spec=LDPProtocol) 
    protocol.length_estimator = DummyEstimator()
    # Patch validate_estimator import used in measure to return our estimator 
    import lenpro.measures.global_measures as gm 
    monkeypatch.setattr( gm, "validate_estimator", lambda estimator_name, protocol: getattr(protocol, estimator_name), ) 
    original = [ ["a", "b"], ["x", None], ] 
    perturbed = [ ["a", None], [None, None], ] 
    measure = MeanRelativeErrorFrequencyMeasure(statistic=make_count_statistic(), estimator_name="length_estimator", sanity_bound=1) 
    result = measure.process_dataset(original, perturbed, protocol=protocol) # Same scenario as basic test → 0.75 
    assert result == pytest.approx(0.75)

def test_mean_relative_error_frequency_rejects_non_positive_sanity_bound():
    with pytest.raises(ValueError, match="sanity_bound must be greater than zero"):
        MeanRelativeErrorFrequencyMeasure(
            statistic=make_count_statistic(),
            estimator_name=None,
            sanity_bound=0,
        )

def test_to_dict_includes_statistic_and_params():
    stat = make_count_statistic()
    measure = MeanRelativeErrorFrequencyMeasure(
        statistic=stat,
        estimator_name="length_estimator",
        sanity_bound=2.0,
    )
    assert measure.to_dict() == { 
        "name": "MeanRelativeErrorFrequencyMeasure", 
        "statistic": {
            "name": "Count"
            }, 
        "estimator_name": "length_estimator", 
        "sanity_bound": 2.0, 
        }