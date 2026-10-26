import math
import pytest


from lenpro.protocols.ldp_protocol import LDPProtocol


class ConcreteProtocol(LDPProtocol):

    def process_event(self, event):
        return event

    def process_stream(self, stream):
        return list(stream)

    def to_dict(self):
        return {
            "name": "ConcreteProtocol",
            "epsilon": self.epsilon,
        }


def test_ldp_protocol_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LDPProtocol(epsilon=1.0)


def test_ldp_protocol_stores_epsilon():
    protocol = ConcreteProtocol(epsilon=1.5)

    assert protocol.epsilon == 1.5


def test_ldp_protocol_accepts_zero_epsilon():
    protocol = ConcreteProtocol(epsilon=0)

    assert protocol.epsilon == 0


@pytest.mark.parametrize(
    "epsilon",
    [
        -1,
        -0.001,
    ],
)
def test_ldp_protocol_rejects_negative_epsilon(epsilon):
    with pytest.raises(
        ValueError,
        match="epsilon must be non-negative",
    ):
        ConcreteProtocol(epsilon=epsilon)


@pytest.mark.parametrize(
    "epsilon",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_ldp_protocol_rejects_non_finite_epsilon(epsilon):
    with pytest.raises(
        ValueError,
        match="epsilon must be finite",
    ):
        ConcreteProtocol(epsilon=epsilon)