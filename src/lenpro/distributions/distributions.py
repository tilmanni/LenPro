from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm, zipf, uniform
from typing import Callable, Mapping, Any


class Distribution(ABC):

    @abstractmethod
    def sample(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def pmf(self, x: int) -> float:
        pass

class UniformDistribution(Distribution):

    def __init__(self, low, high, rng: np.random.Generator | None = None,) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.high = high
        self.low = low

    def sample(self) -> int:
        return int(self.rng.integers(self.low, self.high))

    def to_dict(self) -> dict:
        return {
            "name": "UniformDistribution",
            "high": self.high,
            "low": self.low,
        }

    def pmf(self, x: int) -> float:
        if x >= self.low and x < self.high:
            return float(1 / (self.high - self.low))
        return 0.0


class GaussianDistribution(Distribution):

    def __init__(self, mean, scale, rng: np.random.Generator | None = None,) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mean = mean
        self.scale = scale

    def sample(self) -> int:
        return np.round(self.rng.normal(loc=self.mean, scale=self.scale)).astype(int)

    def to_dict(self) -> dict:
        return {
            "name": "GaussianDistribution",
            "mean": self.mean,
            "scale": self.scale,
        }

    def pmf(self, x: int) -> float:
        # PMF of rounded normal distribution
        return float(norm.cdf(x = x  + 0.5, loc=self.mean, scale=self.scale) - norm.cdf(x = x  - 0.5, loc=self.mean, scale=self.scale))
        #return float(norm.pdf(x=x, loc=self.mean, scale=self.scale))

class ZipfDistribution(Distribution):

    def __init__(self, a, rng: np.random.Generator | None = None,) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.a = a

    def sample(self) -> int:
        return int(self.rng.zipf(self.a))

    def to_dict(self) -> dict:
        return {
            "name": "ZipfDistribution",
            "a": self.a,
        }

    def pmf(self, x: int) -> float:
        return float(zipf.pmf(k = x, a=self.a))



def distribution_from_dict(distribution_config: Mapping[str, Any], rng: np.random.Generator | None = None) -> Distribution:
    if distribution_config["name"] == "UniformDistribution":
        return UniformDistribution(low=distribution_config["low"] , high=distribution_config["high"], rng=rng)
    elif distribution_config["name"] == "GaussianDistribution":
        return GaussianDistribution(mean=distribution_config["mean"], scale=distribution_config["scale"], rng=rng)
    elif distribution_config["name"] == "ZipfDistribution":
        return ZipfDistribution(a=distribution_config["a"], rng=rng)
    else:
        raise ValueError(f"Unknown distribution: {distribution_config["name"]}")
