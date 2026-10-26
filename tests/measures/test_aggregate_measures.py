import math

import numpy as np
import pytest

from lenpro.measures.aggregate_measures import (
    AggregateMeasure,
    ConfidenceIntervalMeasure,
    MeanMeasure,
)


def test_aggregate_measure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AggregateMeasure()


@pytest.mark.parametrize(
    ("measurements", "expected"),
    [
        ([1, 2, 3], 2.0),
        ([5], 5.0),
        ([-2, 0, 2], 0.0),
        ([1.5, 2.5], 2.0),
        ((2, 4, 6), 4.0),
    ],
)
def test_mean_measure(measurements, expected):
    measure = MeanMeasure()

    result = measure.process_measures(measurements)

    assert result == pytest.approx(expected)


def test_mean_measure_accepts_numpy_array():
    measure = MeanMeasure()

    result = measure.process_measures(
        np.array([1.0, 2.0, 3.0]),
    )

    assert result == pytest.approx(2.0)


def test_mean_measure_returns_python_float():
    measure = MeanMeasure()

    result = measure.process_measures([1, 2, 3])

    assert type(result) is float


def test_mean_measure_rejects_empty_measurements():
    measure = MeanMeasure()

    with pytest.raises(
        ValueError,
        match="At least one measurement is required",
    ):
        measure.process_measures([])


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_mean_measure_rejects_non_finite_values(invalid_value):
    measure = MeanMeasure()

    with pytest.raises(
        ValueError,
        match="only contain finite values",
    ):
        measure.process_measures([1.0, invalid_value])


def test_mean_measure_to_dict():
    measure = MeanMeasure()

    assert measure.to_dict() == {
        "name": "MeanMeasure",
    }


def test_confidence_interval_contains_sample_mean():
    measure = ConfidenceIntervalMeasure(
        confidence_level=0.95,
    )
    measurements = [1, 2, 3, 4, 5]

    lower, upper = measure.process_measures(measurements)

    sample_mean = np.mean(measurements)

    assert lower < sample_mean < upper


def test_confidence_interval_known_result():
    measure = ConfidenceIntervalMeasure(
        confidence_level=0.95,
    )

    lower, upper = measure.process_measures(
        [1, 2, 3, 4, 5],
    )

    assert lower == pytest.approx(1.0367568)
    assert upper == pytest.approx(4.9632432)


def test_confidence_interval_is_symmetric_around_mean():
    measure = ConfidenceIntervalMeasure(
        confidence_level=0.95,
    )

    lower, upper = measure.process_measures(
        [1, 2, 3, 4, 5],
    )

    mean = 3.0

    assert mean - lower == pytest.approx(upper - mean)


def test_constant_measurements_produce_point_interval():
    measure = ConfidenceIntervalMeasure(
        confidence_level=0.95,
    )

    result = measure.process_measures(
        [5, 5, 5, 5],
    )

    assert result == (5.0, 5.0)


def test_confidence_interval_returns_python_floats():
    measure = ConfidenceIntervalMeasure()

    lower, upper = measure.process_measures(
        [1, 2, 3],
    )

    assert type(lower) is float
    assert type(upper) is float


def test_confidence_interval_requires_two_measurements():
    measure = ConfidenceIntervalMeasure()

    with pytest.raises(
        ValueError,
        match="At least two measurements",
    ):
        measure.process_measures([1])


def test_confidence_interval_rejects_empty_measurements():
    measure = ConfidenceIntervalMeasure()

    with pytest.raises(
        ValueError,
        match="At least one measurement",
    ):
        measure.process_measures([])


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_confidence_interval_rejects_non_finite_values(
    invalid_value,
):
    measure = ConfidenceIntervalMeasure()

    with pytest.raises(
        ValueError,
        match="only contain finite values",
    ):
        measure.process_measures([1.0, invalid_value])


@pytest.mark.parametrize(
    "confidence_level",
    [
        -0.1,
        0,
        1,
        1.1,
    ],
)
def test_invalid_confidence_level_is_rejected(
    confidence_level,
):
    with pytest.raises(
        ValueError,
        match="confidence_level must be between 0 and 1",
    ):
        ConfidenceIntervalMeasure(
            confidence_level=confidence_level,
        )


def test_lower_confidence_produces_narrower_interval():
    measurements = [1, 2, 3, 4, 5]

    interval_80 = ConfidenceIntervalMeasure(
        confidence_level=0.80,
    ).process_measures(measurements)

    interval_95 = ConfidenceIntervalMeasure(
        confidence_level=0.95,
    ).process_measures(measurements)

    width_80 = interval_80[1] - interval_80[0]
    width_95 = interval_95[1] - interval_95[0]

    assert width_80 < width_95


def test_confidence_interval_to_dict():
    measure = ConfidenceIntervalMeasure(
        confidence_level=0.90,
    )

    assert measure.to_dict() == {
        "name": "ConfidenceIntervalMeasure",
        "confidence_level": 0.90,
    }