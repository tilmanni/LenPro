import numpy as np
from typing import Any


#from lenpro.estimators import estimator
from lenpro.estimators.estimator import Estimator


class LenProEstimator(Estimator):
    
    def estimate(self, stream) -> float:
        return sum(x is not None for x in stream)

    def to_dict(self) -> dict:
        return {
            "name": "LenProEstimator",
        }


# For Prefix sum 
class LenProMedianEstimator(Estimator):
    def __init__(self, p) -> None:
        self.expected_median = ((1 - np.pow((1-p), (1 / p) + 1)) / p) - 1


    def estimate(self, stream) -> float:
        non_nones = sum(x is not None for x in stream)
        #Only use median estimate if trajectory currently being streamed
        if stream[len(stream) - 1] is not None:
            return non_nones + self.expected_median
        return non_nones

    def to_dict(self) -> dict:
        return {
            "name": "LenProMedianEstimator",
            "expected_median": self.expected_median,
        }