from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.stats as stats

EntryLevelMeasurements = Sequence[float]

def validate_measurements(entry_level_measurements: EntryLevelMeasurements) -> np.ndarray:
    values = np.asarray(entry_level_measurements, dtype=float)
    if values.ndim != 1:
        raise ValueError("Measurements must be one-dimensional")
    if values.size == 0:
        raise ValueError("At least one measurement is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Measurements must only contain finite values")
    return values

class AggregateMeasure(ABC):

    @abstractmethod
    def process_measures(self, entry_level_measurements: EntryLevelMeasurements) -> Any:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class MeanMeasure(AggregateMeasure):

    def process_measures(self, entry_level_measurements: EntryLevelMeasurements) -> float:
        values = validate_measurements(entry_level_measurements=entry_level_measurements)
        return float(np.mean(values))


    def to_dict(self) -> dict:
        return {
            "name": "MeanMeasure",
        }

class ConfidenceIntervalMeasure(AggregateMeasure):

    def __init__(self, confidence_level: float = 0.95) -> None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        self.confidence_level = confidence_level

    def process_measures(self, entry_level_measurements: EntryLevelMeasurements) -> tuple[float, float]:
        values = validate_measurements(entry_level_measurements=entry_level_measurements)
        if values.size < 2:
            raise ValueError("At least two measurements required for confidence interval")
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        if standard_error == 0.0:
            return mean, mean
        lower_bound, upper_bound = stats.t.interval(
            confidence=self.confidence_level,
            df = values.size - 1,
            loc = mean,
            scale = standard_error
        )
        return float(lower_bound), float(upper_bound)

    def to_dict(self) -> dict:
        return {
            "name": "ConfidenceIntervalMeasure",
            "confidence_level": self.confidence_level,
        }