from abc import ABC, abstractmethod
from lenpro.distributions import distributions

class LengthGenerator(ABC):

    @abstractmethod
    def generate_length(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class DistributionLengthGenerator(LengthGenerator):

    def __init__(self, distribution: distributions.Distribution):
        self.distribution = distribution

    def generate_length(self) -> int:
        return max(0, self.distribution.sample())

    def to_dict(self) -> dict:
        return {
            "name": "DistributionLengthGenerator",
            "distribution": self.distribution.to_dict(),
        }