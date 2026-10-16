from abc import ABC, abstractmethod
from typing import List

class LDPProtocol(ABC):

    # Add init that forces setting self.epsilon?

    @abstractmethod
    def process_event(self, event):
        pass

    @abstractmethod
    def process_stream(self, stream) -> List:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass