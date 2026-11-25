import numpy as np
import math
from lenpro.estimators.estimator import Estimator, Data
from lenpro.protocols.randomized_response import RandomizedResponse
from lenpro.distributions.distributions import Distribution

def calculate_likelihood_function(s: int, k: int, H: int, p: float, q: float) -> float:
    return sum(
        (math.comb(k, j) * np.pow(p, j) * np.pow(q, (k-j))) - 
        (math.comb(H-k, s-j) * np.pow(q, s-j) * np.pow(p, (H-k-s+j)))
        #TODO verify min 
        for j in range(min(s+1, k+1))
    )



class RandomizedResponseCountEstimator(Estimator):

    def __init__(self, randomized_response: RandomizedResponse):
        self.p = randomized_response.p
        self.q = randomized_response.q


    def estimate(self, data: Data) -> float:
        # n = H, could (maybe should) pass explicitly
        n = len(data)
        reported_number = sum(x is not None for x in data) 
        #return (reported_number - self.q * n) / (self.p - self.q)
        return max(0, min((reported_number - self.q * n) / (self.p - self.q), n))

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponseCountEstimator",
            "p": self.p,
            "q": self.q,
        }

class RandomizedResponseMLECountEstimator(Estimator):

    def __init__(self, randomized_response: RandomizedResponse):
        self.p = randomized_response.p
        self.q = randomized_response.q

    def estimate(self, data: Data) -> float:
        H = len(data)
        s = sum(x is not None for x in data) 
        # Takes first value with maximum likelihood, should return at random
        # Adjust so that it start at 1 ?
        likelihoods = np.array(calculate_likelihood_function(s=s, k=k, H=H, p=self.p, q=self.q) for k in range(H + 1))
        # Would also have to adjust this? (+1?)
        return float(np.argmax(likelihoods))

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponseMLECountEstimator",
            "p": self.p,
            "q": self.q,
        }

class RandomizedResponseBayesianCountEstimator(Estimator):

    def __init__(self, randomized_response: RandomizedResponse, prior: Distribution, max_length: int):
        self.p = randomized_response.p
        self.q = randomized_response.q
        self.prior = prior
        self.max_length = max_length


    def estimate(self, data: Data) -> float:
        H = len(data)
        s = sum(x is not None for x in data) 
        # Expectation value for posterior probability, requires prior
        # Similarly, input k can only be in [1, max_length], so estimator overestimates
        return sum(k * calculate_likelihood_function(s=s, k=k, H=H, p=self.p, q=self.q) * self.prior.pmf(x=k) for k in range(1, self.max_length + 1))

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponseBayesianCountEstimator",
            "p": self.p,
            "q": self.q,
            "prior": self.prior.to_dict(),
        }  
