from abc import ABC, abstractmethod

# We generally assume adaptive Mechanisms
class Mechanism(ABC):

    @abstractmethod
    def perturb_event(self, event, previous_outputs = None):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass



class IdentityMechanism(Mechanism):

    def perturb_event(self, event, previous_outputs = None):
        return event

    def to_dict(self) -> dict:
        return {
            "name": "Identity",
        }