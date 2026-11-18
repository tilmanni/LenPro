from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from lenpro.protocols.ldp_protocol import LDPProtocol
from lenpro.estimators.estimator import Estimator
from lenpro.statistics.statistics import EstimatorStatistic, Statistic, CountStatistic

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

    @abstractmethod
    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None) -> float:
        # Could also use ...
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class RelativeErrorMeasure(EntryLevelMeasure):

    def __init__(self, statistic: Statistic | None = None, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.statistic = statistic if statistic is not None else CountStatistic()
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:
        original_value = self.statistic.evaluate(data=original_data)
        perturbed_value = self.statistic.evaluate(data=perturbed_data)
        return relative_error(original_value=original_value, perturbed_value=perturbed_value, sanity_bound=self.sanity_bound)
    
    def to_dict(self) -> dict:
        return {
            "name": "RelativeErrorMeasure",
            "statistic": self.statistic.to_dict(),
            "sanity_bound": self.sanity_bound,
        }

class RelativeErrorMeasureProtocolDependent(EntryLevelMeasure):

    # Wrap estimator into statistic?
    # Could also set estimator to be statistic as default if unset
    def __init__(self, statistic: Statistic, estimator_name: str | None = None, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.statistic = statistic
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:

        if self.estimator_name is not None:
            perturbed_data_statistic = EstimatorStatistic(estimator=validate_estimator(estimator_name=self.estimator_name, protocol=protocol))
        else: 
            perturbed_data_statistic = self.statistic

        original_value = self.statistic.evaluate(original_data)
        perturbed_value = perturbed_data_statistic.evaluate(perturbed_data)
        
        return relative_error(original_value=original_value, perturbed_value=perturbed_value, sanity_bound=self.sanity_bound)

    def to_dict(self) -> dict:
        return {
            "name": "RelativeErrorMeasureProtocolDependent",
            "statistic": self.statistic.to_dict(),
            "estimator_name": self.estimator_name,
            "sanity_bound": self.sanity_bound,
        }

class PrefixMeanRelativeErrorMeasureProtocolDependent(EntryLevelMeasure):

    def __init__(self, statistic: Statistic, estimator_name: str | None = None, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.statistic = statistic
        self.sanity_bound = sanity_bound

    def process_data(self, original_data: Data, perturbed_data: Data, protocol: LDPProtocol | None = None) -> float:

        if self.estimator_name is not None:
            perturbed_data_statistic = EstimatorStatistic(estimator=validate_estimator(estimator_name=self.estimator_name, protocol=protocol))
        else: 
            perturbed_data_statistic = self.statistic

        max_len = max(len(original_data), len(perturbed_data))
    
        original_data = list(original_data) + [None] * (max_len - len(original_data)) if len(original_data) < max_len else original_data
        perturbed_data = list(perturbed_data) + [None] * (max_len - len(perturbed_data)) if len(perturbed_data) < max_len else perturbed_data


        return np.mean([relative_error(
            original_value=self.statistic.evaluate(original_data[:i]),
            perturbed_value=perturbed_data_statistic.evaluate(perturbed_data[:i]),
            sanity_bound=self.sanity_bound
            ) for i in range(1, len(original_data) + 1)])

    def to_dict(self) -> dict:
        return {
            "name": "PrefixMeanRelativeErrorMeasureProtocolDependent",
            "statistic": self.statistic.to_dict(),
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
