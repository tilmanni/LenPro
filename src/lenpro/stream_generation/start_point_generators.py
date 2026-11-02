from abc import ABC, abstractmethod
from lenpro.distributions import distributions
import sys
import numpy as np


def validate_arguments(fixed_horizon: int, length: int) -> None:
    if fixed_horizon < 0:
        raise ValueError("fixed_horizon must be non-negative")

    if length < 0:
        raise ValueError("length must be non-negative")

    # Should just be truncated
    #if length > fixed_horizon:
    #    raise ValueError(
    #        "length cannot be greater than fixed_horizon"
    #    )


class StartPointGenerator(ABC):

    @abstractmethod
    def generate_start_point(self, fixed_horizon: int = sys.maxsize, length: int = 0) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class DistributionStartPointGenerator(StartPointGenerator):

    def __init__(self, distribution: distributions.Distribution):
        self.distribution = distribution

    def generate_start_point(self, fixed_horizon: int = sys.maxsize, length: int = 0) -> int:
        # Should truncate sampled value into [0, fixed_horizon - length], i.e. any start point 
        # guarantees that the entire stream is in the horizon. If fixed horizon or length are not 
        # explicitly set, the default initializations guarantee that they do not affect sampling.

        validate_arguments(fixed_horizon=fixed_horizon, length=length)

        maximum_start = fixed_horizon - length
        sampled_start = self.distribution.sample()

        return int(min(max(0, sampled_start), maximum_start))

    def to_dict(self) -> dict:
        return {
            "name": "DistributionStartPointGenerator",
            "distribution": self.distribution.to_dict(),
        }


class CenteredStreamStartPointGenerator(StartPointGenerator):

    def generate_start_point(self, fixed_horizon: int = sys.maxsize, length: int = 0) -> int:
        validate_arguments(fixed_horizon=fixed_horizon, length=length)

        if (fixed_horizon == sys.maxsize):
            return 0
        return int(np.floor((fixed_horizon - length) / 2))

    def to_dict(self) -> dict:
        return {
            "name": "CenteredStreamStartPointGenerator",
        }