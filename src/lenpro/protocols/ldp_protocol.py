from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from numpy import isfinite

Stream = Sequence[Any]


class LDPProtocol(ABC):

    # Add init that forces setting self.epsilon?
    def __init__(self, epsilon: float, delta_length: int) -> None:
        if not isfinite(epsilon):
            raise ValueError("epsilon must be finite")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")        
        if not isfinite(delta_length):
            raise ValueError("delta_length must be finite")
        if delta_length <= 0:
            raise ValueError("delta_length must be greater than zero")
        self.epsilon = epsilon
        self.delta_length = delta_length

    @abstractmethod
    def process_event(self, event):
        pass

    @abstractmethod
    def process_stream(self, stream: Stream) -> Stream:
        
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass