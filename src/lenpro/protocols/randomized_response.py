import numpy as np
from collections.abc import Sequence
from typing import Any


from lenpro.protocols import ldp_protocol
from lenpro.mechanisms import mechanisms
from lenpro.stream_generation import event_generators
from lenpro.estimators.randomized_response_estimator import RandomizedResponseEstimator


Stream = Sequence[Any]


class RandomizedResponse(ldp_protocol.LDPProtocol):

    # could also do rng: np.random.Generator | None = None,

    def __init__(self, epsilon, delta_length, perturbing_mechanism: mechanisms.Mechanism, synthetic_event_generator: event_generators.EventGenerator, rng = None) -> None:
        super().__init__(epsilon=epsilon)

        if not np.isfinite(delta_length):
            raise ValueError("delta_length must be finite")

        if delta_length <= 0:
            raise ValueError("delta_length must be greater than zero")


        self.rng = rng if rng is not None else np.random.default_rng()
        self.delta_length = delta_length
        privacy_per_event = epsilon / delta_length

        exp_negative = np.exp(-privacy_per_event)

        self.p = 1.0 / (1.0 + exp_negative)
        self.q = exp_negative / (1.0 + exp_negative)
        self.perturbing_mechanism = perturbing_mechanism
        self.synthetic_event_generator = synthetic_event_generator
        self.length_estimator = RandomizedResponseEstimator(self)

    def process_event(self, event):
        if event is not None:
            if  self.rng.uniform(0, 1) <= self.p:
                # Assume non-adaptive for now
                return self.perturbing_mechanism.perturb_event(event = event)
            else: 
                return None
        else:
            if self.rng.uniform(0, 1) <= self.q:
                # Assume independent generator for now
                return self.synthetic_event_generator.generate_event()
            else:
                return None

    def process_stream(self, stream: Stream) -> Stream:
        return [self.process_event(event) for event in stream]


    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponse",
            "epsilon": self.epsilon,
            "delta_length": self.delta_length,
            "perturbing_mechanism": self.perturbing_mechanism.to_dict(),
            "synthetic_event_generator": self.synthetic_event_generator.to_dict(),
        }