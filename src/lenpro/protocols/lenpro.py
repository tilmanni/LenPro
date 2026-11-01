import numpy as np
from collections.abc import Sequence
from typing import Any

from lenpro.protocols import ldp_protocol
from lenpro.mechanisms import mechanisms
from lenpro.stream_generation import event_generators
from lenpro.estimators.lenpro_estimators import LenProLengthEstimator, LenProPrefixLengthMedianEstimator

Stream = Sequence[Any]


class LenPro(ldp_protocol.LDPProtocol):

    def __init__(self, epsilon, delta_length, perturbing_mechanism: mechanisms.Mechanism, synthetic_event_generator: event_generators.EventGenerator, rng: np.random.Generator | None = None,) -> None:
        super().__init__(epsilon=epsilon)
        if epsilon == 0:
            raise ValueError(
                "epsilon must be greater than zero for LenPro"
            )
        if not np.isfinite(delta_length):
            raise ValueError("delta_length must be finite")

        if delta_length <= 0:
            raise ValueError(
                "delta_length must be greater than zero"
            )
        self.rng = rng if rng is not None else np.random.default_rng()
        self.delta_length = delta_length
        self.p = float(-np.expm1(-epsilon / delta_length))
        #self.p = (1 - np.pow(np.e, (-epsilon / delta_length)))
        self.perturbing_mechanism = perturbing_mechanism
        self.synthetic_event_generator = synthetic_event_generator
        self.length_estimator = LenProLengthEstimator()
        self.prefix_length_estimator = LenProLengthEstimator()
        self._reset_connection()

    def _reset_connection(self) -> None:
        self.received_events: list[Any] = []
        self.returned_events: list[Any] = []
        self.k = 0
        self.connected = False
        self.a: int | None = None
        self.r: int | None = None 

    def _start_connection(self) -> None:
        self.a = int(self.rng.geometric(p=self.p) - 1)
        self.r = int(self.rng.geometric(p=self.p) - 1)
        self.connected = True 

    def _adjusted_length(self) -> int:
        if not self.connected:
            return 0
        
        assert self.a is not None
        assert self.r is not None

        return max(0, len(self.received_events) - self.r + self.a)

    def _last_returned_event(self) -> Any:
        return next((event for event in reversed(self.returned_events) if event is not None), None)


    def process_event(self, event):
        if event is not None: # New connection
            if not self.connected:
                self._start_connection()
            self.received_events.append(event)
            assert self.r is not None 
            if len(self.received_events) <= self.r: # Could be done in fewer lines, kept this way for readability
                self.returned_events.append(None)
                return None 
            perturbed_event = self.perturbing_mechanism.perturb_event(event = event, previous_outputs = self.returned_events.copy()) # Could also add received events?
            self.returned_events.append(perturbed_event)
            self.k += 1
            return perturbed_event

        # event is None 
        if not self.connected:
            return None 
        adjusted_length = self._adjusted_length()
        # Check if fake event generation still necessary
        if self.k < adjusted_length:
            generated_event = self.synthetic_event_generator.generate_event(previous_event=self._last_returned_event()) # Only take last event
            self.returned_events.append(generated_event)
            self.k += 1
            return generated_event
        self._reset_connection()
        return None 

    def process_stream(self, stream: Stream) -> Stream:
        processed_stream = [self.process_event(event) for event in stream]

        # Generate trailing synthetic events until 
        # adjusted output length is reached
        while (self.connected and self.k < self._adjusted_length()):
            processed_stream.append(self.process_event(None))
        # Reset connection so that another stream can be safely processed
        self._reset_connection()
        return processed_stream

    def to_dict(self) -> dict:
        return {
            "name": "LenPro",
            "epsilon": self.epsilon,
            "delta_length": self.delta_length,
            "perturbing_mechanism": self.perturbing_mechanism.to_dict(),
            "synthetic_event_generator": self.synthetic_event_generator.to_dict(),
        }



    # None signifies that no connection took place, i.e. if received that user was truly not connected, 
    # and if returned, that user was truly or pretends to not be connected.
    # Once a non-None event appears, LenPro has to be re-initialised, until a None event appears once again
    # Assume for now that a fake generated connection does not overlap with a new one arriving.
    # If overlap right now, generate fake events until new connection, post new perturbed events until 
    # second connection stops, then continue generating fake events until 'a' used up. No new 'r' or 'a' 
    # generated

"""
    def process_stream_static(self, stream):
        k = 0
        a = self.rng.geometric(p=self.p) - 1 # unsure if a,r >= 1 or a,r >= 0 for all p, but this ensures the latter, which makes more sense in my opinion
        r = self.rng.geometric(p=self.p) - 1
        if len(stream) - r + a  <= 0: # only the case when r >= len(list_x) + a, meaning output is empty
            return []
        processed_stream = []
        for _ in range(r, len(stream)):
            processed_stream.append(self.perturbing_mechanism.perturb_event(None if k == 0 else processed_stream[k - 1], stream[k])) # replace mechanism with adaptive version, taking all previous inputs? unsage since k = 0
            k = k + 1
        for _ in range(k, len(stream) - r + a):
            processed_stream.append(self.synthetic_event_generator.generate_event(stream[k - 1]))
        return processed_stream
"""