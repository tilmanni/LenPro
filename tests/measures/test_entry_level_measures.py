from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lenpro.estimators.estimator import Estimator
from lenpro.protocols.ldp_protocol import LDPProtocol
from lenpro.measures.entry_level_measures import (
    EntryLevelMeasure,
    FractionRetainedEventsMeasure,
    FractionSyntheticEventsMeasure,
    Metric,
    RelativeErrorMeasure,
    RelativeErrorMeasureProtocolDependent,
    PrefixMeanRelativeErrorMeasureProtocolDependent,
)


def test_entry_level_measure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        EntryLevelMeasure()


def test_relative_error_uses_number_of_non_none_events():
    measure = RelativeErrorMeasure()

    original = ["a", None, "b", "c"]
    perturbed = ["a", None]

    result = measure.process_data(original, perturbed)

    # Original length = 3, perturbed length = 1
    assert result == pytest.approx(2 / 3)


def test_relative_error_uses_sanity_bound():
    measure = RelativeErrorMeasure(sanity_bound=2)

    result = measure.process_data(
        original_data=[],
        perturbed_data=["a"],
    )

    assert result == pytest.approx(0.5)


def test_relative_error_accepts_custom_metric():
    metric = Mock(side_effect=[10, 7])
    measure = RelativeErrorMeasure(metric=metric)

    original = ["original"]
    perturbed = ["perturbed"]

    result = measure.process_data(original, perturbed)

    assert result == pytest.approx(0.3)
    assert metric.call_count == 2
    metric.assert_any_call(original)
    metric.assert_any_call(perturbed)


def test_relative_error_rejects_non_positive_sanity_bound():
    with pytest.raises(
        ValueError,
        match="sanity_bound must be greater than zero",
    ):
        RelativeErrorMeasure(sanity_bound=0)


def test_relative_error_to_dict():
    measure = RelativeErrorMeasure(sanity_bound=2)

    assert measure.to_dict() == {
        "name": "RelativeErrorMeasure",
        "sanity_bound": 2,
    }


def test_protocol_dependent_relative_error():
    estimator = Mock(spec=Estimator)
    estimator.estimate.side_effect = [7]

    metric = Mock(spec=Metric)
    metric.side_effect = [10]

    protocol = Mock(spec=LDPProtocol)
    protocol.length_estimator = estimator

    measure = RelativeErrorMeasureProtocolDependent(
        estimator_name="length_estimator",
        metric=metric
    )

    original = ["original"]
    perturbed = ["perturbed"]

    result = measure.process_data(
        original,
        perturbed,
        protocol=protocol,
    )

    assert result == pytest.approx(0.3)
    assert estimator.estimate.call_count == 1
    assert metric.call_count == 1
    metric.assert_any_call(original)
    estimator.estimate.assert_any_call(perturbed)


def test_protocol_dependent_measure_requires_protocol():
    metric = Mock(spec=Metric)

    measure = RelativeErrorMeasureProtocolDependent(
        estimator_name="length_estimator",
        metric=metric
    )

    with pytest.raises(
        ValueError,
        match="A protocol is required",
    ):
        measure.process_data([], [], protocol=None)


def test_protocol_dependent_measure_requires_estimator():
    protocol = Mock(spec=LDPProtocol)
    metric = Mock(spec=Metric)


    measure = RelativeErrorMeasureProtocolDependent(
        estimator_name="missing_estimator",
        metric=metric
    )

    with pytest.raises(
        AttributeError,
        match="missing_estimator",
    ):
        measure.process_data([], [], protocol=protocol)



def test_protocol_dependent_measure_to_dict():
    metric = Mock(spec=Metric)


    measure = RelativeErrorMeasureProtocolDependent(
        estimator_name="length_estimator",
        sanity_bound=2,
        metric=metric
    )

    assert measure.to_dict() == {
        "name": "RelativeErrorMeasureProtocolDependent",
        "sanity_bound": 2,
        "estimator_name": "length_estimator",
    }


def test_prefix_mean_relative_error():
    estimator = Mock(spec=Estimator)
    estimator.estimate.side_effect = [0, 0, 1, 2, 3, 3, 4, 7]

    metric = Mock(spec=Metric)
    metric.side_effect = [0, 1, 2, 3, 3, 3, 4, 5]

    protocol = Mock(spec=LDPProtocol)
    protocol.prefix_length_estimator = estimator

    measure = PrefixMeanRelativeErrorMeasureProtocolDependent(
        estimator_name="prefix_length_estimator",
        metric=metric
    )

    original = ["original"] * 8
    perturbed = ["perturbed"] * 8

    result = measure.process_data(
        original,
        perturbed,
        protocol=protocol,
    )

    assert result == pytest.approx(67 / 240)
    assert estimator.estimate.call_count == 8
    assert metric.call_count == 8
    metric.assert_any_call(original)
    estimator.estimate.assert_any_call(perturbed)


def test_prefix_mean_relative_error_different_lengths():
    estimator = Mock(spec=Estimator)
    estimator.estimate.side_effect = [4, 5, 5, 6]

    metric = Mock(spec=Metric)
    metric.side_effect = [4, 5, 6, 6]

    protocol = Mock(spec=LDPProtocol)
    protocol.prefix_length_estimator = estimator

    measure = PrefixMeanRelativeErrorMeasureProtocolDependent(
        estimator_name="prefix_length_estimator",
        metric=metric
    )

    original = ["original"] * 3
    perturbed = ["perturbed"] * 4

    result = measure.process_data(
        original,
        perturbed,
        protocol=protocol,
    )

    assert result == pytest.approx(1 / 24)
    assert estimator.estimate.call_count == 4
    assert metric.call_count == 4
    metric.assert_any_call(original)
    estimator.estimate.assert_any_call(perturbed)


def test_fraction_retained_events():
    measure = FractionRetainedEventsMeasure(
        is_original=lambda event: event.startswith("original"),
    )

    original = [
        "original-1",
        "original-2",
        "original-3",
        None,
    ]
    perturbed = [
        None,
        "original-1",
        "synthetic-1",
        "original-3",
        None,
    ]

    result = measure.process_data(original, perturbed)

    assert result == pytest.approx(2 / 3)


def test_fraction_retained_returns_zero_for_empty_original_data():
    measure = FractionRetainedEventsMeasure(
        is_original=lambda event: True,
    )

    result = measure.process_data(
        original_data=[None, None],
        perturbed_data=["event"],
    )

    assert result == 0.0


def test_retained_predicate_is_not_called_for_none():
    predicate = Mock(return_value=True)
    measure = FractionRetainedEventsMeasure(predicate)

    measure.process_data(
        original_data=["event"],
        perturbed_data=[None, "event", None],
    )

    predicate.assert_called_once_with("event")


def test_fraction_retained_to_dict():
    measure = FractionRetainedEventsMeasure(
        is_original=lambda event: True,
    )

    assert measure.to_dict() == {
        "name": "FractionRetainedEventsMeasure",
    }


def test_fraction_synthetic_events():
    measure = FractionSyntheticEventsMeasure(
        is_synthetic=lambda event: event.startswith("synthetic"),
    )

    perturbed = [
        "original-1",
        "synthetic-1",
        None,
        "synthetic-2",
    ]

    result = measure.process_data([], perturbed)

    assert result == pytest.approx(2 / 3)


def test_fraction_synthetic_returns_zero_for_empty_output():
    measure = FractionSyntheticEventsMeasure(
        is_synthetic=lambda event: True,
    )

    result = measure.process_data(
        original_data=["event"],
        perturbed_data=[None, None],
    )

    assert result == 0.0


def test_synthetic_predicate_is_not_called_for_none():
    predicate = Mock(return_value=True)
    measure = FractionSyntheticEventsMeasure(predicate)

    measure.process_data(
        original_data=[],
        perturbed_data=[None, "event", None],
    )

    predicate.assert_called_once_with("event")


def test_fraction_synthetic_to_dict():
    measure = FractionSyntheticEventsMeasure(
        is_synthetic=lambda event: True,
    )

    assert measure.to_dict() == {
        "name": "FractionSyntheticEventsMeasure",
    }

