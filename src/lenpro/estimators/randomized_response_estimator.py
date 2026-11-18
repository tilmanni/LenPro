import numpy as np
from lenpro.estimators.estimator import Estimator, Data
from lenpro.protocols import randomized_response

# Can reuse estimator for count frequency estimation, 
# should change name to "RandomizedResponseCountEstimator"?

class RandomizedResponseCountEstimator(Estimator):

    def __init__(self, randomized_response: randomized_response.RandomizedResponse):
        self.p = randomized_response.p
        self.q = randomized_response.q


    def estimate(self, data: Data) -> float:
        # n = H, could (maybe should) pass explicitly
        n = len(data)
        reported_number = sum(x is not None for x in data) 
        return (reported_number - self.q * n) / (self.p - self.q)

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponseCountEstimator",
            "p": self.p,
            "q": self.q,
        }