import sys
from unittest.mock import Mock

import pytest

from lenpro.distributions.distributions import Distribution
from lenpro.stream_generation.start_point_generators import (
    CenteredStreamStartPointGenerator,
    DistributionStartPointGenerator,
    StartPointGenerator,
)


def test_start_point_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        StartPointGenerator()


def make_distribution(sample):
    distribution = Mock(spec=Distribution)
    distribution.sample.return_value = sample
    return distribution


def test_distribution_generator_returns_sample_inside_bounds():
    distribution = make_distribution(sample=4)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    assert result == 4
    distribution.sample.assert_called_once_with()


def test_distribution_generator_clamps_negative_sample_to_zero():
    distribution = make_distribution(sample=-5)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    assert result == 0


def test_distribution_generator_clamps_sample_to_maximum_start():
    distribution = make_distribution(sample=100)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    # A stream of length 3 can start at 7 and occupy
    # positions 7, 8, and 9.
    assert result == 7


def test_distribution_generator_accepts_maximum_start():
    distribution = make_distribution(sample=7)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    assert result == 7


def test_distribution_generator_uses_default_horizon():
    distribution = make_distribution(sample=42)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point()

    assert result == 42


def test_distribution_generator_returns_python_int():
    distribution = make_distribution(sample=4)

    generator = DistributionStartPointGenerator(distribution)

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    assert type(result) is int


def test_distribution_generator_to_dict():
    distribution = Mock(spec=Distribution)
    distribution.to_dict.return_value = {
        "name": "ExampleDistribution",
        "parameter": 42,
    }

    generator = DistributionStartPointGenerator(distribution)

    assert generator.to_dict() == {
        "name": "DistributionStartPointGenerator",
        "distribution": {
            "name": "ExampleDistribution",
            "parameter": 42,
        },
    }

    distribution.to_dict.assert_called_once_with()


def test_centered_generator_returns_zero_without_fixed_horizon():
    generator = CenteredStreamStartPointGenerator()

    result = generator.generate_start_point()

    assert result == 0


def test_centered_generator_centers_stream_with_even_difference():
    generator = CenteredStreamStartPointGenerator()

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=4,
    )

    # (10 - 4) / 2 = 3
    assert result == 3


def test_centered_generator_floors_odd_difference():
    generator = CenteredStreamStartPointGenerator()

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=3,
    )

    # floor((10 - 3) / 2) = floor(3.5) = 3
    assert result == 3


def test_centered_generator_returns_zero_when_stream_fills_horizon():
    generator = CenteredStreamStartPointGenerator()

    result = generator.generate_start_point(
        fixed_horizon=10,
        length=10,
    )

    assert result == 0


def test_centered_generator_to_dict():
    generator = CenteredStreamStartPointGenerator()

    assert generator.to_dict() == {
        "name": "CenteredStreamStartPointGenerator",
    }


@pytest.mark.parametrize(
    ("fixed_horizon", "length", "error_message"),
    [
        (-1, 0, "fixed_horizon must be non-negative"),
        (10, -1, "length must be non-negative"),
    ],
)
@pytest.mark.parametrize(
    "generator_factory",
    [
        lambda: DistributionStartPointGenerator(
            make_distribution(sample=1)
        ),
        CenteredStreamStartPointGenerator,
    ],
    ids=["distribution", "centered"],
)
def test_generator_rejects_invalid_arguments(
    generator_factory,
    fixed_horizon,
    length,
    error_message,
):
    generator = generator_factory()

    with pytest.raises(ValueError, match=error_message):
        generator.generate_start_point(
            fixed_horizon=fixed_horizon,
            length=length,
        )