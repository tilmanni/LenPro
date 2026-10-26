from abc import ABC, abstractmethod
from typing import Any

# Includes previous event to account for markov model 
class EventGenerator(ABC):

    @abstractmethod
    def generate_event(self, previous_event = None) -> Any: # Change return type to be more generic
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class ConstantEventGenerator(EventGenerator):

    def __init__(self, constant_event) -> None:
        super().__init__()
        self.constant_event = constant_event

    def generate_event(self, previous_event = None) -> Any:
        return self.constant_event

    def to_dict(self) -> dict:
        return {
            "name": "ConstantEventGenerator",
            "constant_event": self.constant_event,
        }
    