import numpy as np
import pytest

from lenpro.distributions.distributions import (
    Distribution,
    TruncatedGaussianDistribution,
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

def test_uniform_pmf():
    distribution = UniformDistribution(low=5, high=10)

    assert distribution.pmf(5) == 1 / 5
    assert distribution.pmf(0) == 0


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


def test_truncated_gaussian_is_integer():
    distribution = TruncatedGaussianDistribution(
        mean= 10,
        scale= 2,
        rng=np.random.default_rng(42),
        low=1,
        high=20,
    )

    sample = distribution.sample()

    assert isinstance(sample, (int, np.integer))


def test_truncated_gaussian_truncation_low():
    distribution = TruncatedGaussianDistribution(
        mean= 10,
        scale= 1,
        rng=np.random.default_rng(42),
        low=100,
        high=120,
    )

    sample = distribution.sample()

    assert sample == 100

def test_truncated_gaussian_truncation_high():
    distribution = TruncatedGaussianDistribution(
        mean= 100,
        scale= 1,
        rng=np.random.default_rng(42),
        low=0,
        high=10,
    )

    sample = distribution.sample()

    assert sample == 10


def test_truncated_gaussian_pmf():
    distribution = TruncatedGaussianDistribution(
        mean= 10,
        scale=1,
        rng=np.random.default_rng(42),
        low=8,
        high=12,
    )

    assert distribution.pmf(10) == pytest.approx(0.3829249225)
    assert distribution.pmf(8) == pytest.approx(0.0668072013)
    assert distribution.pmf(12) == pytest.approx(0.0668072013)
    assert distribution.pmf(0) == 0
    assert distribution.pmf(20) == 0

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