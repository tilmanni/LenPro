from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable, Sequence



Data = Sequence[Any]
Dataset = Sequence[Data]

Predicate = Callable[[Any], bool]

def count_events(data: Data) -> int:
    return sum(event is not None for event in data)

def validate_dataset(original_dataset: Dataset, perturbed_dataset: Dataset) -> None:
    if (len(original_dataset) != len(perturbed_dataset)):
        raise ValueError("original_dataset and perturbed_dataset must have the same length")


class GlobalMeasure(ABC):

    @abstractmethod
    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class EmptyOutputProbabilityMeasure(GlobalMeasure):

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset) -> float:

        validate_dataset(original_dataset=original_dataset, perturbed_dataset=perturbed_dataset)
        dataset_size = len(original_dataset)
        if dataset_size == 0:
            return 0.0

        return sum(count_events(original_dataset[i]) != 0 and count_events(perturbed_dataset[i]) == 0 for i in range(0, dataset_size)) / dataset_size

    def to_dict(self) -> dict:
        return {
            "name": "EmptyOutputProbabilityMeasure",
        }

class LengthUnchangedProbabilityMeasure(GlobalMeasure):

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset) -> float:
        validate_dataset(original_dataset=original_dataset, perturbed_dataset=perturbed_dataset)
        dataset_size = len(original_dataset)
        if dataset_size == 0:
            return 0.0
        
        return sum(count_events(original_dataset[i]) == count_events(perturbed_dataset[i]) for i in range(0, dataset_size)) / dataset_size

    def to_dict(self) -> dict:
        return {
            "name": "LengthUnchangedProbabilityMeasure",
        }
    

class GlobalFractionRetainedEventsMeasure(GlobalMeasure):

    def __init__(self, is_original: Predicate) -> None:
        self.is_original = is_original

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset) -> float:
        validate_dataset(original_dataset=original_dataset, perturbed_dataset=perturbed_dataset)
        total_number_of_original_events = 0
        total_number_of_retained_original_events = 0
        for original_data in original_dataset:
            total_number_of_original_events += count_events(original_data)
        if (total_number_of_original_events == 0):
            return 0.0
        for perturbed_entry in perturbed_dataset:
            total_number_of_retained_original_events += sum(event is not None and self.is_original(event) for event in perturbed_entry)

        return total_number_of_retained_original_events / total_number_of_original_events

    def to_dict(self) -> dict:
        return {
            "name": "GlobalFractionRetainedEventsMeasure",
        }

class GlobalFractionSyntheticEventsMeasure(GlobalMeasure):

    def __init__(self, is_synthetic: Predicate) -> None:
        self.is_synthetic = is_synthetic

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset) -> float:
        validate_dataset(original_dataset=original_dataset, perturbed_dataset=perturbed_dataset)
        total_number_of_perturbed_events = 0
        total_number_of_synthetic_events = 0
        for perturbed_data in perturbed_dataset:
            total_number_of_perturbed_events += count_events(perturbed_data)
            total_number_of_synthetic_events += sum(event is not None and self.is_synthetic(event) for event in perturbed_data)
        if (total_number_of_perturbed_events == 0):
            return 0.0
        return total_number_of_synthetic_events / total_number_of_perturbed_events

    def to_dict(self) -> dict:
        return {
            "name": "GlobalFractionSyntheticEventsMeasure",
        }