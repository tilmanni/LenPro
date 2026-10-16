import numpy as np
from protocols import ldp_protocol
from mechanisms import mechanisms
from stream_generation import event_generators


class RandomizedResponse(ldp_protocol.LDPProtocol):

    def __init__(self, epsilon, perturbing_mechanism: mechanisms.Mechanism, synthetic_event_generator: event_generators.EventGenerator) -> None:
        self.p = np.pow(np.e, epsilon) / (np.pow(np.e, epsilon) + 1.0)
        self.q = 1 / (np.pow(np.e, epsilon) + 1.0)
        self.epsilon = epsilon
        self.perturbing_mechanism = perturbing_mechanism
        self.synthetic_event_generator = synthetic_event_generator

    #@classmethod
    #def from_epsilon(cls, epsilon, mechanism, generator):
    #    p = np.pow(np.e, epsilon) / (np.pow(np.e, epsilon) + 1.0)
    #    q = 1 / (np.pow(np.e, epsilon) + 1.0)
    #    return RandomizedResponse(p=p, q=q, mechanism=mechanism, generator=generator)

    def process_event(self, event):
        if event is not None:
            if np.random.uniform(0, 1) <= self.p:
                # Assume non-adaptive for now
                return self.perturbing_mechanism.perturb_event(event = event)
            else: 
                return None
        else:
            if np.random.uniform(0, 1) <= self.q:
                # Assume independent generator for now
                return self.synthetic_event_generator.generate_event()
            else:
                return None

    def process_stream(self, stream):
        processed_stream = []
        for event in stream:
            processed_stream.append(self.process_event(event))
        return processed_stream

    def __str__(self) -> str:
        return f"Randomized Response(epsilon = '{self.epsilon}', perturbing mechanism = '{self.perturbing_mechanism}', synthetic event generator = '{self.synthetic_event_generator}')"

    def to_dict(self) -> dict:
        return {
            "name": "RandomizedResponse",
            "epsilon": self.epsilon,
            "perturbing_mechanism": self.perturbing_mechanism.to_dict(),
            "synthetic_event_generator": self.synthetic_event_generator.to_dict(),
        }