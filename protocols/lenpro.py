import numpy as np

from protocols import ldp_protocol
from mechanisms import mechanisms
from stream_generation import event_generators

def lenpro_static(list_x, p, markov_model_m_0, mechanism_m_1):
    k = 0
    a = np.random.geometric(p=p) - 1 # unsure if a,r >= 1 or a,r >= 0 for all p, but this ensures the latter, which makes more sense in my opinion
    r = np.random.geometric(p=p) - 1
    if len(list_x) - r + a  <= 0: # only the case when r >= len(list_x) + a, meaning output is empty
        return []
    output_list_x = []
    for i in range(r, len(list_x)):
        output_list_x.append(mechanism_m_1(None if k == 0 else output_list_x[k - 1], list_x[k])) # replace mechanism with adaptive version, taking all previous inputs? unsage since k = 0
        k = k + 1
    for i in range(k, len(list_x) - r + a):
        output_list_x.append(markov_model_m_0(list_x[k - 1]))
    return output_list_x


class LenPro(ldp_protocol.LDPProtocol):

    def __init__(self, epsilon, delta_length, perturbing_mechanism: mechanisms.Mechanism, synthetic_event_generator: event_generators.EventGenerator) -> None:
        self.received_events = [] # Maybe only length n is necessary
        self.returned_events = []
        self.k = 0
        self.connected = False
        self.epsilon = epsilon
        self.delta_length = delta_length
        self.p = (1 - np.pow(np.e, (-epsilon / delta_length)))
        self.perturbing_mechanism = perturbing_mechanism
        self.synthetic_event_generator = synthetic_event_generator

    #@classmethod
    #def from_epsilon(cls, epsilon, mechanism, markov_model, delta_length=1):
        #return LenPro(p=(1 - np.pow(np.e, (-epsilon / delta_length))), mechanism=mechanism, markov_model=markov_model)

    # None signifies that no connection took place, i.e. if received that user was truly not connected, 
    # and if returned, that user was truly or pretends to not be connected.
    # Once a non-None event appears, LenPro has to be re-initialised, until a None event appears once again
    # Assume for now that a fake generated connection does not overlap with a new one arriving.
    # If overlap right now, generate fake events until new connection, post new perturbed events until 
    # second connection stops, then continue generating fake events until 'a' used up. No new 'r' or 'a' 
    # generated
    def process_event(self, event):
        if event is not None: # New connection
            if len(self.received_events) == 0:
                self.a = np.random.geometric(p = self.p) - 1 # unsure if a,r >= 1 or a,r >= 0 for all p, but this ensures the latter, which makes more sense in my opinion
                self.r = np.random.geometric(p = self.p) - 1
            self.received_events.append(event)
            if len(self.received_events) <= self.r: # Could be done in fewer lines, kept this way for readability
                self.returned_events.append(None)
                return None 
            perturbed_event = self.perturbing_mechanism.perturb_event(event = event, previous_outputs = self.returned_events) # Could also add received events?
            self.returned_events.append(perturbed_event)
            self.k = self.k + 1
            return perturbed_event
        else:
            if self.k == 0: # No connection, None received
                self.returned_events.append(None)
                return None 
            adjusted_length = len(self.received_events) - self.r + self.a
            if adjusted_length > 0 and self.k < adjusted_length:
                generated_event = self.synthetic_event_generator.generate_event(self.returned_events[self.k - 1]) # Only take last event
                self.returned_events.append(generated_event)
                self.k = self.k + 1
                return generated_event
            else: # Reset connection
                self.k = 0
                self.received_events = []
                self.returned_events = [None]
                return None

    def process_stream(self, stream):
        processed_stream = []
        for event in stream:
            processed_stream.append(self.process_event(event))
        # Reset connection 
        self.k = 0
        self.received_events = []
        self.returned_events = [None]
        return processed_stream

    def __str__(self) -> str:
        return f"LenPro(epsilon = '{self.epsilon}', perturbing mechanism = '{self.perturbing_mechanism}', synthetic event generator = '{self.synthetic_event_generator}')"

    def to_dict(self) -> dict:
        return {
            "name": "LenPro",
            "epsilon": self.epsilon,
            "delta_length": self.delta_length,
            "perturbing_mechanism": self.perturbing_mechanism.to_dict(),
            "synthetic_event_generator": self.synthetic_event_generator.to_dict(),
        }