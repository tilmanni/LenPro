from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import zipf

class LengthGenerator(ABC):

    @abstractmethod
    def generate_length(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class UniformLengthGenerator(LengthGenerator):

    def __init__(self, low, high) -> None:
        self.rng = np.random.default_rng()
        self.high = high
        self.low = low

    def generate_length(self) -> int:
        return int(self.rng.integers(self.low, self.high))

    def __str__(self) -> str:
        return f"Uniform Distribution(low = '{self.low}', high = '{self.high}')"

    def to_dict(self) -> dict:
        return {
            "name": "UniformDistribution",
            "high": self.high,
            "low": self.low,
        }



class GaussLengthGenerator(LengthGenerator):

    def __init__(self, mean, scale) -> None:
        self.rng = np.random.default_rng()
        self.mean = mean
        self.scale = scale

    def generate_length(self) -> int:
        return np.round(np.random.normal(loc=self.mean, scale=self.scale)).astype(int)

    def __str__(self) -> str:
        return f"GaussDistribution(mean = '{self.mean}', scale = '{self.scale}')"

    def to_dict(self) -> dict:
        return {
            "name": "GaussDistribution",
            "mean": self.mean,
            "scale": self.scale,
        }

class ZipfLengthGenerator(LengthGenerator):

    def __init__(self, a) -> None:
        self.rng = np.random.default_rng()
        self.a = a

    def generate_length(self) -> int:
        return self.rng.zipf(self.a)

    def __str__(self) -> str:
        return f"Zipf Distribution(a = '{self.a}')"

    def to_dict(self) -> dict:
        return {
            "name": "ZipfDistribution",
            "a": self.a,
        }