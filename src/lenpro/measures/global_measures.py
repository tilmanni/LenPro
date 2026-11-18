from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable, Sequence
from itertools import zip_longest
import numpy as np
from lenpro.protocols.ldp_protocol import LDPProtocol


from lenpro.statistics.statistics import Statistic, EstimatorStatistic
from lenpro.measures.entry_level_measures import validate_estimator



Data = Sequence[Any]
Dataset = Sequence[Data]

Predicate = Callable[[Any], bool]

def count_events(data: Data) -> int:
    return sum(event is not None for event in data)

def relative_error(original_value: float, perturbed_value: float, sanity_bound: float) -> float:
    return abs(original_value - perturbed_value) / max(abs(original_value), sanity_bound)

def validate_dataset(original_dataset: Dataset, perturbed_dataset: Dataset) -> None:
    if (len(original_dataset) != len(perturbed_dataset)):
        raise ValueError("original_dataset and perturbed_dataset must have the same length")


class GlobalMeasure(ABC):

    @abstractmethod
    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class EmptyOutputProbabilityMeasure(GlobalMeasure):

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:

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

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:
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

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:
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

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:
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

class MeanRelativeErrorFrequencyMeasure(GlobalMeasure):
    def __init__(self, statistic: Statistic, estimator_name: str | None = None, sanity_bound: float = 1) -> None:
        if sanity_bound <= 0:
            raise ValueError("sanity_bound must be greater than zero")
        self.estimator_name = estimator_name
        self.statistic = statistic
        self.sanity_bound = sanity_bound

    def process_dataset(self, original_dataset: Dataset, perturbed_dataset: Dataset, protocol: LDPProtocol | None = None) -> float:
        if self.estimator_name is not None and protocol is not None:
            perturbed_data_statistic = EstimatorStatistic(estimator=validate_estimator(estimator_name=self.estimator_name, protocol=protocol))
        else: 
            perturbed_data_statistic = self.statistic
        
        validate_dataset(original_dataset=original_dataset, perturbed_dataset=perturbed_dataset)
        original_values = []
        for column in zip_longest(*original_dataset, fillvalue=None):
            original_values.append(self.statistic.evaluate(data=column))

        perturbed_values = []
        for column in zip_longest(*perturbed_dataset, fillvalue=None):
            perturbed_values.append(perturbed_data_statistic.evaluate(data=column))
        assert len(original_values) == len(perturbed_values)
        return np.mean(
            [relative_error(original_value=original_value, 
                            perturbed_value=perturbed_value, 
                            sanity_bound=self.sanity_bound) 
            for original_value, perturbed_value in zip(original_values, perturbed_values)]
            )

    def to_dict(self) -> dict:
        return {
            "name": "MeanRelativeErrorFrequencyMeasure",
            "statistic": self.statistic.to_dict(),
            "estimator_name": self.estimator_name,
            "sanity_bound": self.sanity_bound,
        }