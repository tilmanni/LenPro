from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from lenpro.protocols.ldp_protocol import LDPProtocol
from lenpro.estimators.estimator import Estimator


# Good practice, should adapt for other modules
Data = Sequence[Any]
Metric = Callable[[Data], float]
Predicate = Callable[[Any], bool]

def count_events(data: Data) -> int:
    return sum(event is not None for event in data)


def relative_error(original_value: float, perturbed_value: float, sanity_bound: float) -> float:
    return abs(original_value - perturbed_value) / max(abs(original_value), sanity_bound)

def validate_estimator(estimator_name: str, protocol: LDPProtocol | None) -> Estimator:
    if protocol is None:
            raise ValueError("A protocol is required for this measure")
    estimator = getattr(protocol, estimator_name, None)
    if estimator is None:
        raise AttributeError(f"Protocol has no estimator named {estimator_name}")
    if not isinstance(estimator, Estimator):
        raise AttributeError(f"Atrribute named {estimator_name} is not Estimator")
    return estimator



class EntryLevelMeasure(ABC):
    # Default not protocol dependent
    is_protocol_dependent = False 

    @abstractmethod
    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None) -> float:
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

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:
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

    def __init__(self, estimator_name: str, metric: Metric, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.metric = metric
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:

        estimator = validate_estimator(estimator_name=self.estimator_name, protocol=protocol)

        original_value = self.metric(original_data)
        perturbed_value = estimator.estimate(perturbed_data)
        
        return relative_error(original_value=original_value, perturbed_value=perturbed_value, sanity_bound=self.sanity_bound)

    def to_dict(self) -> dict:
        return {
            "name": "RelativeErrorMeasureProtocolDependent",
            "estimator_name": self.estimator_name,
            "sanity_bound": self.sanity_bound,
        }

class PrefixMeanRelativeErrorMeasureProtocolDependent(EntryLevelMeasure):

    is_protocol_dependent = True 

    def __init__(self, estimator_name: str, metric: Metric, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.metric = metric
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:

        estimator = validate_estimator(estimator_name=self.estimator_name, protocol=protocol)

        max_len = max(len(original_data), len(perturbed_data))
    
        original_data = list(original_data) + [None] * (max_len - len(original_data)) if len(original_data) < max_len else original_data
        perturbed_data = list(perturbed_data) + [None] * (max_len - len(perturbed_data)) if len(perturbed_data) < max_len else perturbed_data


        return np.mean([relative_error(
            original_value=self.metric(original_data[:i]),
            perturbed_value=estimator.estimate(perturbed_data[:i]),
            sanity_bound=self.sanity_bound
            ) for i in range(1, len(original_data) + 1)])

    def to_dict(self) -> dict:
        return {
            "name": "PrefixMeanRelativeErrorMeasureProtocolDependent",
            "estimator_name": self.estimator_name,
            "sanity_bound": self.sanity_bound,
        }

class FractionRetainedEventsMeasure(EntryLevelMeasure):

    def __init__(self, is_original: Predicate) -> None:
        self.is_original = is_original

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:
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
    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:
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
