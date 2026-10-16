from abc import ABC, abstractmethod
from typing import Any


class GlobalMeasure(ABC):

    @abstractmethod
    def process_dataset(self, original_dataset, perturbed_dataset) -> Any:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class EmptyOutputProbabilityMeasure(GlobalMeasure):

    def process_dataset(self, original_dataset, perturbed_dataset) -> Any:
        assert(len(original_dataset) == len(perturbed_dataset))
        return sum(sum(x is not None for x in original_dataset[i]) != 0 and sum(x is not None for x in perturbed_dataset[i]) == 0 for i in range(0, len(original_dataset))) / len(original_dataset)

    def __str__(self) -> str:
        return "Empty Output Probability"

    def to_dict(self) -> dict:
        return {
            "name": "EmptyOutputProbabilityMeasure",
        }

class LengthUnchangedProbabilityMeasure(GlobalMeasure):

    def process_dataset(self, original_dataset, perturbed_dataset) -> Any:
        assert(len(original_dataset) == len(perturbed_dataset))
        
        return sum(sum(x is not None for x in original_dataset[i]) == sum(x is not None for x in perturbed_dataset[i]) for i in range(0, len(original_dataset))) / len(original_dataset)

    def __str__(self) -> str:
        return "Probability of unchanged Length"

    def to_dict(self) -> dict:
        return {
            "name": "LengthUnchangedProbabilityMeasure",
        }
    

class GlobalFractionRetainedEventsMeasure(GlobalMeasure):

    def __init__(self, is_original) -> None:
        super().__init__()
        self.is_original = is_original

    def process_dataset(self, original_dataset, perturbed_dataset) -> Any:
        total_number_of_original_events = 0
        total_number_of_retained_original_events = 0
        for original_entry in original_dataset:
            total_number_of_original_events += sum(x is not None for x in original_entry)
        if (total_number_of_original_events == 0):
            return 0
        for perturbed_entry in perturbed_dataset:
            total_number_of_retained_original_events += sum(self.is_original(x) is True for x in perturbed_entry)

        return total_number_of_retained_original_events / total_number_of_original_events
    
    def __str__(self) -> str:
        #return f"Global Fraction of Retained Events(is_original = '{self.is_original}')"
        # Have to implement __str__ for is_original
        return "Global Fraction of Retained Events"

    def to_dict(self) -> dict:
        return {
            "name": "GlobalFractionRetainedEventsMeasure",
        }

class GlobalFractionSyntheticEventsMeasure(GlobalMeasure):

    def __init__(self, is_synthetic) -> None:
        super().__init__()
        self.is_synthetic = is_synthetic

    def process_dataset(self, original_dataset, perturbed_dataset) -> Any:
        total_number_of_perturbed_events = 0
        total_number_of_synthetic_events = 0
        for perturbed_entry in perturbed_dataset:
            total_number_of_perturbed_events += sum(x is not None for x in perturbed_entry)
            total_number_of_synthetic_events += sum(self.is_synthetic(x) is True for x in perturbed_entry)
        if (total_number_of_perturbed_events == 0):
            return 0
        return total_number_of_synthetic_events / total_number_of_perturbed_events

    def __str__(self) -> str:
        #return f"Global Fraction of Synthetic Events(is_synthetic = '{self.is_synthetic}')"
        # Have to implement __str__ for is_synthethic
        return "Global Fraction of Synthetic Events"

    def to_dict(self) -> dict:
        return {
            "name": "GlobalFractionSyntheticEventsMeasure",
        }