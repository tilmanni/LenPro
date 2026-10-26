from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from numpy import isfinite

Stream = Sequence[Any]


class LDPProtocol(ABC):

    # Add init that forces setting self.epsilon?
    def __init__(self, epsilon: float) -> None:
        if not isfinite(epsilon):
            raise ValueError("epsilon must be finite")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon

    @abstractmethod
    def process_event(self, event):
        pass

    @abstractmethod
    def process_stream(self, stream: Stream) -> Stream:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass