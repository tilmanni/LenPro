import numpy as np
from lenpro.estimators import estimator
from lenpro.protocols import randomized_response

class RandomizedResponseEstimator(estimator.Estimator):

    def __init__(self, randomized_response: randomized_response.RandomizedResponse):
        self.p = randomized_response.p
        self.q = randomized_response.q


    def estimate(self, stream) -> float:
        # n = H, could (maybe should) pass explicitly
        n = len(stream)
        reported_number = sum(x is not None for x in stream) 
        return (reported_number - self.q * n) / (self.p - self.q)

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponseEstimator",
            "p": self.p,
            "q": self.q,
        }
