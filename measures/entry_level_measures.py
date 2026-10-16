from abc import ABC, abstractmethod
import numpy as np

from typing import Any

class EntryLevelMeasure(ABC):

    @abstractmethod
    def process_data(self, original_data, perturbed_data) -> Any:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class RelativeErrorMeasure(EntryLevelMeasure):

    # Define Metric?
    # Default values assigned as if length is metric
    def __init__(self, sanity_bound = 1, metric = None) -> None:
        super().__init__()
        if metric is None:
            self.metric = lambda data : sum(x is not None for x in data)
        else: 
            self.metric = metric
        self.sanity_bound = sanity_bound

    def process_data(self, original_data, perturbed_data) -> Any:
        return abs(self.metric(original_data) - self.metric(perturbed_data)) / max(self.metric(original_data), self.sanity_bound)

    def __str__(self) -> str:
        #return f"Relative Error(sanity bound = '{self.sanity_bound}', metric = '{self.metric}')"
        # Have to implement __str__ for metric for above return value.
        return f"Relative Error(sanity bound = '{self.sanity_bound}')"
    
    def to_dict(self) -> dict:
        return {
            "name": "RelativeErrorMeasure",
            "sanity_bound": self.sanity_bound,
        }

class FractionRetainedEventsMeasure(EntryLevelMeasure):

    def __init__(self, is_original) -> None:
        super().__init__()
        self.is_original = is_original

    def process_data(self, original_data, perturbed_data) -> Any:
        number_original_events = sum(x is not None for x in original_data)
        # Have to verify if this works
        # Prevent division by zero
        # What to return in that case? 1 because all (none) original events are retained?
        # Or zero, because there is nothing to retain? Or NaN, and then later remove from 
        # further processing?
        number_retained_original_events = sum(self.is_original(x) is True for x in perturbed_data)
        if (number_original_events == 0):
            return 0
        return number_retained_original_events / number_original_events
    
    def __str__(self) -> str:
        #return f"Fraction of Retained Events(is_original = '{self.is_original}')"
        # Have to implement __str__ for is_original
        return "Fraction of Retained Events"

    def to_dict(self) -> dict:
        return {
            "name": "FractionRetainedEventsMeasure",
        }


class FractionSyntheticEventsMeasure(EntryLevelMeasure):

    def __init__(self, is_synthetic) -> None:
        super().__init__()
        self.is_synthetic = is_synthetic

    # Does not need original data
    def process_data(self, original_data, perturbed_data) -> Any:
        number_perturbed_events = sum(x is not None for x in perturbed_data)
        # Have to also verify if this works
        # Prevent division by zero
        if (number_perturbed_events == 0):
            return 0
        number_synthetic_events = sum(self.is_synthetic(x) is True for x in perturbed_data)
        return number_synthetic_events / number_perturbed_events

    def __str__(self) -> str:
        #return f"Fraction of Synthetic Events(is_synthetic = '{self.is_synthetic}')"
        # Have to implement __str__ for is_synthethic
        return "Fraction of Synthetic Events"

    def to_dict(self) -> dict:
        return {
            "name": "FractionSyntheticEventsMeasure",
        }


class LengthsMeasure(EntryLevelMeasure):
    # Local Measure version of metrics of length for both perturbed and original data
    # Does not mesh well with generic global measures, maybe rethink
    def __init__(self) -> None:
        super().__init__()

    def process_data(self, original_data, perturbed_data) -> Any:
        number_original_events = sum(x is not None for x in original_data)
        number_perturbed_events = sum(x is not None for x in perturbed_data)
        return (number_original_events, number_perturbed_events)
    
    def __str__(self) -> str:
        return "Original and Perturbed Lengths"

    def to_dict(self) -> dict:
        return {
            "name": "LengthsMeasure",
        }

    

# Should always record length of original and perturbed stream.
# How to avoid doing this multiple times?
# Could make property of original / perturbed data, so it is only initialized once?
