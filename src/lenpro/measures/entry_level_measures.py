from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from typing import Any

# Good practice, should adapt for other modules
Data = Sequence[Any]
Metric = Callable[[Data], float]
Predicate = Callable[[Any], bool]

def count_events(data: Data) -> int:
    return sum(event is not None for event in data)


def relative_error(original_value: float, perturbed_value: float, sanity_bound: float) -> float:
    return abs(original_value - perturbed_value) / max(abs(original_value), sanity_bound)



class EntryLevelMeasure(ABC):
    # Default not protocol dependent
    is_protocol_dependent = False 

    @abstractmethod
    def process_data(self, original_data: Data, perturbed_data: Data, protocol = None) -> float:
        # Could also use ...
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class RelativeErrorMeasure(EntryLevelMeasure):

    # Define Metric?
    # Default values assigned as if length is metric
    def __init__(self, sanity_bound: float = 1, metric: Metric | None = None) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        
        self.metric = metric if metric is not None else count_events
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol = None) -> float:
        original_value = self.metric(original_data)
        perturbed_value = self.metric(perturbed_data)
        return relative_error(original_value=original_value, perturbed_value=perturbed_value, sanity_bound=self.sanity_bound)
    
    def to_dict(self) -> dict:
        #return f"Relative Error(sanity bound = '{self.sanity_bound}', metric = '{self.metric}')"
        # Have to implement to_dict() for metric for above return value.
        return {
            "name": "RelativeErrorMeasure",
            "sanity_bound": self.sanity_bound,
        }

class RelativeErrorMeasureProtocolDependent(EntryLevelMeasure):
    is_protocol_dependent = True

    def __init__(self, estimator_name: str, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol = None) -> float:
        if protocol is None:
            raise ValueError("A protocol is required for this measure")
        estimator = getattr(protocol, self.estimator_name, None)
        if estimator is None:
            raise AttributeError(f"Protocol has no estimator named {self.estimator_name}")
        
        original_value = estimator.estimate(original_data)
        perturbed_value = estimator.estimate(perturbed_data)
        
        return relative_error(original_value=original_value, perturbed_value=perturbed_value, sanity_bound=self.sanity_bound)

    def to_dict(self) -> dict:
        return {
            "name": "RelativeErrorMeasureProtocolDependent",
            "sanity_bound": self.sanity_bound,
            "estimator_name": self.estimator_name
        }

class FractionRetainedEventsMeasure(EntryLevelMeasure):

    def __init__(self, is_original: Predicate) -> None:
        self.is_original = is_original

    def process_data(self, original_data, perturbed_data, protocol = None) -> float:
        number_original_events = count_events(original_data)

        if (number_original_events == 0):
            return 0.0

        number_retained_original_events = sum(event is not None and self.is_original(event) for event in perturbed_data)
        
        return number_retained_original_events / number_original_events

    def to_dict(self) -> dict:
        #return f"Fraction of Retained Events(is_original = '{self.is_original}')"
        # Have to implement to_dict() for is_original
        return {
            "name": "FractionRetainedEventsMeasure",
        }


class FractionSyntheticEventsMeasure(EntryLevelMeasure):

    def __init__(self, is_synthetic: Predicate) -> None:
        self.is_synthetic = is_synthetic

    # Does not need original data
    def process_data(self, original_data, perturbed_data, protocol = None) -> float:
        number_perturbed_events = count_events(perturbed_data)
        # Prevent division by zero
        if (number_perturbed_events == 0):
            return 0.0
        
        number_synthetic_events = sum(event is not None and self.is_synthetic(event) for event in perturbed_data)
        return number_synthetic_events / number_perturbed_events

    def to_dict(self) -> dict:
        #return f"Fraction of Synthetic Events(is_synthetic = '{self.is_synthetic}')"
        # Have to implement to_dict() for is_synthethic
        return {
            "name": "FractionSyntheticEventsMeasure",
        }


#class LengthsMeasure(EntryLevelMeasure):
    # Local Measure version of metrics of length for both perturbed and original data
    # Does not mesh well with generic global measures, maybe rethink
#    def __init__(self) -> None:
#        super().__init__()

#    def process_data(self, original_data, perturbed_data) -> Any:
#        number_original_events = sum(x is not None for x in original_data)
#        number_perturbed_events = sum(x is not None for x in perturbed_data)
#        return (number_original_events, number_perturbed_events)

#    def to_dict(self) -> dict:
#        return {
#            "name": "LengthsMeasure",
#        }

    

# Should always record length of original and perturbed stream.
# How to avoid doing this multiple times?
# Could make property of original / perturbed data, so it is only initialized once?
