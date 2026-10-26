from abc import ABC, abstractmethod
from typing import Any


class Estimator(ABC):

    @abstractmethod
    def estimate(self, stream) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass