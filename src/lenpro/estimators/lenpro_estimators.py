import numpy as np

from lenpro.estimators.estimator import Estimator, Data

# Could also reuse for frequency estimation?
class LenProCountEstimator(Estimator):
    
    def estimate(self, data: Data) -> float:
        # basically count_events, could maybe generalize?
        return sum(x is not None for x in data)

    def to_dict(self) -> dict:
        return {
            "name": "LenProCountEstimator",
        }


# For Prefix sum 
class LenProPrefixLengthMedianEstimator(Estimator):
    def __init__(self, p) -> None:
        self.expected_median = ((1 - np.pow((1-p), (1 / p) + 1)) / p) - 1

    def estimate(self, data: Data) -> float:
        # See above for count_events
        non_nones = sum(x is not None for x in data)
        #Only use median estimate if trajectory currently being streamed
        if data[len(data) - 1] is not None:
            return non_nones + self.expected_median
        return non_nones

    def to_dict(self) -> dict:
        return {
            "name": "LenProPrefixLengthMedianEstimator",
            "expected_median": self.expected_median,
        }