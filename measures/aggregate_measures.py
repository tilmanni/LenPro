from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from typing import Any

from measures.entry_level_measures import EntryLevelMeasure

class AggregrateMeasure(ABC):

    @abstractmethod
    def process_measures(self, entry_level_measurements) -> Any:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class MeanMeasure(AggregrateMeasure):

    def __init__(self) -> None:
        super().__init__()

    def process_measures(self, entry_level_measurements) -> Any:
        return np.mean(entry_level_measurements)

    def __str__(self) -> str:
        return "Mean"

    def to_dict(self) -> dict:
        return {
            "name": "MeanMeasure",
        }

class ConfidenceIntervalMeasure(AggregrateMeasure):

    def __init__(self, confidence_level = 0.95) -> None:
        super().__init__()
        self.confidence_level = confidence_level

    def process_measures(self, entry_level_measurements) -> Any:
        return stats.t.interval(self.confidence_level, df = len(entry_level_measurements) - 1, loc = np.mean(entry_level_measurements), scale = np.std(entry_level_measurements, ddof = 1) / np.sqrt(len(entry_level_measurements)))

    def __str__(self) -> str:
        return f"Confidence Interval(confidence level = '{self.confidence_level}')"

    def to_dict(self) -> dict:
        return {
            "name": "ConfidenceIntervalMeasure",
            "confidence_level": self.confidence_level,
        }