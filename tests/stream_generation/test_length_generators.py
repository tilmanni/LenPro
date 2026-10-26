from numbers import Integral
from unittest.mock import Mock

import numpy as np
import pytest

from lenpro.distributions.distributions import (
    Distribution,
    GaussianDistribution,
    UniformDistribution,
    ZipfDistribution,
)
from lenpro.stream_generation.length_generators import (
    DistributionLengthGenerator,
    LengthGenerator,
)


def test_length_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LengthGenerator()


def test_generate_length_uses_distribution_sample():
    distribution = Mock(spec=Distribution)
    distribution.sample.return_value = 7

    generator = DistributionLengthGenerator(distribution)

    result = generator.generate_length()

    assert result == 7
    distribution.sample.assert_called_once_with()


def test_generate_length_clamps_negative_values_to_zero():
    distribution = Mock(spec=Distribution)
    distribution.sample.return_value = -5

    generator = DistributionLengthGenerator(distribution)

    assert generator.generate_length() == 0


@pytest.mark.parametrize(
    "distribution",
    [
        UniformDistribution(
            low=1,
            high=10,
            rng=np.random.default_rng(42),
        ),
        GaussianDistribution(
            mean=5,
            scale=2,
            rng=np.random.default_rng(42),
        ),
        ZipfDistribution(
            a=2,
            rng=np.random.default_rng(42),
        ),
    ],
    ids=["uniform", "gaussian", "zipf"],
)
def test_generate_length_with_different_distributions(distribution):
    generator = DistributionLengthGenerator(distribution)

    lengths = [generator.generate_length() for _ in range(100)]

    assert all(isinstance(length, Integral) for length in lengths)
    assert all(length >= 0 for length in lengths)


def test_generate_length_with_uniform_distribution_respects_range():
    distribution = UniformDistribution(
        low=5,
        high=10,
        rng=np.random.default_rng(42),
    )
    generator = DistributionLengthGenerator(distribution)

    lengths = [generator.generate_length() for _ in range(100)]

    assert all(5 <= length < 10 for length in lengths)


def test_to_dict():
    distribution = UniformDistribution(
        low=1,
        high=10,
        rng=np.random.default_rng(42),
    )
    generator = DistributionLengthGenerator(distribution)

    assert generator.to_dict() == {
        "name": "DistributionLengthGenerator",
        "distribution": {
            "name": "UniformDistribution",
            "high": 10,
            "low": 1,
        },
    }