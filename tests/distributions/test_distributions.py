import numpy as np
import pytest

from lenpro.distributions.distributions import (
    Distribution,
    UniformDistribution,
    GaussianDistribution,
    ZipfDistribution,
)


def test_distribution_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Distribution()


def test_uniform_sample_is_integer_and_in_range():
    distribution = UniformDistribution(
        low=5,
        high=10,
        rng=np.random.default_rng(42),
    )

    samples = [distribution.sample() for _ in range(100)]

    assert all(isinstance(sample, int) for sample in samples)
    assert all(5 <= sample < 10 for sample in samples)


def test_uniform_to_dict():
    distribution = UniformDistribution(low=5, high=10)

    assert distribution.to_dict() == {
        "name": "UniformDistribution",
        "high": 10,
        "low": 5,
    }


def test_gaussian_sample_is_integer():
    distribution = GaussianDistribution(
        mean=10,
        scale=2,
        rng=np.random.default_rng(42),
    )

    sample = distribution.sample()

    assert isinstance(sample, (int, np.integer))


def test_gaussian_to_dict():
    distribution = GaussianDistribution(mean=10, scale=2)

    assert distribution.to_dict() == {
        "name": "GaussianDistribution",
        "mean": 10,
        "scale": 2,
    }


def test_zipf_samples_are_positive():
    distribution = ZipfDistribution(
        a=2,
        rng=np.random.default_rng(42),
    )

    samples = [distribution.sample() for _ in range(100)]

    assert all(sample >= 1 for sample in samples)


def test_zipf_to_dict():
    distribution = ZipfDistribution(a=2)

    assert distribution.to_dict() == {
        "name": "ZipfDistribution",
        "a": 2,
    }


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (UniformDistribution, {"low": 1, "high": 10}),
        (GaussianDistribution, {"mean": 5, "scale": 2}),
        (ZipfDistribution, {"a": 2}),
    ],
)
def test_seeded_distributions_are_reproducible(factory, arguments):
    first = factory(**arguments, rng=np.random.default_rng(42))
    second = factory(**arguments, rng=np.random.default_rng(42))

    first_samples = [first.sample() for _ in range(10)]
    second_samples = [second.sample() for _ in range(10)]

    assert first_samples == second_samples