import numpy as np
from collections.abc import Sequence
from typing import Any

from lenpro.protocols import ldp_protocol
from lenpro.mechanisms import mechanisms
from lenpro.stream_generation import event_generators


Stream = Sequence[Any]

class UniformLength(ldp_protocol.LDPProtocol):

    def __init__(self, epsilon: float, uniform_length: int, delta_length: int, perturbing_mechanism: mechanisms.Mechanism, synthetic_event_generator: event_generators.EventGenerator, rng: np.random.Generator | None = None, extendable: bool = False):
        super().__init__(epsilon=epsilon, delta_length=delta_length)
        if not np.isfinite(uniform_length):
            raise ValueError("uniform_length must be finite")
        if uniform_length < 0: 
            raise ValueError("uniform_length must be non-negative")
        self.extendable = extendable
        self.uniform_length = uniform_length
        self.perturbing_mechanism = perturbing_mechanism
        self.synthetic_event_generator = synthetic_event_generator
        self._reset_connection()

    def _reset_connection(self) -> None:
        self.k = 0
        self.connected = False


    def process_event(self, event):
        if event is not None:
            if not self.connected:
                self.connected = True
            if self.k  < self.uniform_length:
                self.k += 1
                # Assume non-adaptive for now
                return self.perturbing_mechanism.perturb_event(event=event)
            # Cut events if already returned uniform_length events
            return None 
        # event is None
        if not self.connected:
            return None
        if self.k  < self.uniform_length:
            # Generate fake events until length is reached
            self.k += 1
            # Assume independent generator for now
            return self.synthetic_event_generator.generate_event()
        self._reset_connection()
        return None

    def process_stream(self, stream: Stream) -> Stream:
        processed_stream = [self.process_event(event=event) for event in stream]
        if (self.extendable):
            while (self.connected and self.k < self.uniform_length):
                # Assumes stream can be extended over its length. Not necessarily true
                processed_stream.append(self.process_event(None))
        self._reset_connection()
        return processed_stream

    def to_dict(self) -> dict:
        return {
            "name": "UniformLength",
            "epsilon": self.epsilon,
            "delta_length": self.delta_length,
            "uniform_length": self.uniform_length,
            "perturbing_mechanism": self.perturbing_mechanism.to_dict(),
            "synthetic_event_generator": self.synthetic_event_generator.to_dict(),
            "extendable": self.extendable,
        }