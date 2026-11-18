from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from lenpro.estimators.estimator import Estimator

Data = Sequence[Any]

class Statistic(ABC):

    @abstractmethod
    def evaluate(self, data: Data) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class CountStatistic(Statistic):

    def evaluate(self, data: Data) -> float:
        return float(sum(value is not None for value in data))

    def to_dict(self) -> dict:
        return {
            "name": "Count",
        }


class EstimatorStatistic(Statistic):
    
    def __init__(self, estimator: Estimator):
        self.estimator = estimator

    def evaluate(self, data: Data) -> float:
        return float(self.estimator.estimate(data))

    def to_dict(self) -> dict:
        return {
            "name": "Estimator",
            "estimator": self.estimator.to_dict(),
        }