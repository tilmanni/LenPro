from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):

    @abstractmethod
    def sample(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class UniformDistribution(Distribution):

    def __init__(self, low, high, rng = None) -> None:
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

class GaussianDistribution(Distribution):

    def __init__(self, mean, scale, rng = None) -> None:
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

class ZipfDistribution(Distribution):

    def __init__(self, a, rng = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.a = a

    def sample(self) -> int:
        return int(self.rng.zipf(self.a))

    def to_dict(self) -> dict:
        return {
            "name": "ZipfDistribution",
            "a": self.a,
        }