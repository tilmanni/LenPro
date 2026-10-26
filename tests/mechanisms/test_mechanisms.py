import pytest

from lenpro.mechanisms.mechanisms import Mechanism, IdentityMechanism


def test_mechanism_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Mechanism()


def test_identity_mechanism_returns_event_unchanged():
    mechanism = IdentityMechanism()
    event = {"value": 42}

    result = mechanism.perturb_event(event)

    assert result is event


def test_identity_mechanism_accepts_previous_outputs():
    mechanism = IdentityMechanism()
    event = 42
    previous_outputs = [1, 2, 3]

    result = mechanism.perturb_event(
        event,
        previous_outputs=previous_outputs,
    )

    assert result == event


def test_identity_mechanism_to_dict():
    mechanism = IdentityMechanism()

    assert mechanism.to_dict() == {
        "name": "Identity",
    }