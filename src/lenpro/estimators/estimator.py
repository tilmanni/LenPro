from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Sequence

Data = Sequence[Any]

class Estimator(ABC):

    @abstractmethod
    def estimate(self, data: Data) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass