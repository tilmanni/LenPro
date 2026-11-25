import numpy as np

from lenpro.distributions.distributions import Distribution
from lenpro.estimators.estimator import Estimator, Data


def calculate_likelihood_function(m: int, n: int, p: float) -> float:
    return p * np.pow((1-p), np.abs(m - n)) / (2 - p) 

# Could also reuse for frequency estimation?
class LenProCountEstimator(Estimator):
    
    def estimate(self, data: Data) -> float:
        # basically count_events, could maybe generalize?
        return sum(x is not None for x in data)

    def to_dict(self) -> dict:
        return {
            "name": "LenProCountEstimator",
        }

class LenProBayesianCountEstimator(Estimator):

    # Uses expectation value instead of arg max

    # Maybe only use p? 
    def __init__(self, p: float, prior: Distribution, max_length: int) -> None:
        self.p = p
        self.prior = prior
        self.max_length = max_length

    def estimate(self, data: Data) -> float:
        m = sum(x is not None for x in data)
        # Should be equal to "standard" for uniform distribution (i.e. just m=n)
        # Input can only be in [1, max_length], so estimator overestimates!
        return sum(n * calculate_likelihood_function(m=m, n=n, p=self.p) * self.prior.pmf(x=n) for n in range(1, self.max_length + 1))

    def to_dict(self) -> dict:
        return super().to_dict()

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