from unittest.mock import Mock, call

import pytest

from lenpro.mechanisms.mechanisms import Mechanism
from lenpro.protocols.uniform_length import UniformLength
from lenpro.stream_generation.event_generators import EventGenerator


def make_protocol(uniform_length=3, extendable = True):
    mechanism = Mock(spec=Mechanism)
    mechanism.perturb_event.side_effect = (
        lambda event: f"perturbed-{event}"
    )

    event_generator = Mock(spec=EventGenerator)
    event_generator.generate_event.return_value = "synthetic"

    protocol = UniformLength(
        epsilon=1.0,
        uniform_length=uniform_length,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        delta_length=5,
        extendable=extendable
    )

    return protocol, mechanism, event_generator


def test_initial_state():
    protocol, _, _ = make_protocol()

    assert protocol.k == 0
    assert protocol.connected is False


def test_first_real_event_starts_connection():
    protocol, mechanism, event_generator = make_protocol()

    result = protocol.process_event("a")

    assert result == "perturbed-a"
    assert protocol.connected is True
    assert protocol.k == 1

    mechanism.perturb_event.assert_called_once_with(event="a")
    event_generator.generate_event.assert_not_called()


def test_none_before_connection_returns_none():
    protocol, mechanism, event_generator = make_protocol()

    result = protocol.process_event(None)

    assert result is None
    assert protocol.connected is False
    assert protocol.k == 0
    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_none_during_connection_generates_synthetic_event():
    protocol, _, event_generator = make_protocol(uniform_length=3)

    protocol.process_event("a")
    result = protocol.process_event(None)

    assert result == "synthetic"
    assert protocol.k == 2
    assert protocol.connected is True
    event_generator.generate_event.assert_called_once_with()


def test_events_beyond_uniform_length_are_removed():
    protocol, mechanism, _ = make_protocol(uniform_length=2)

    results = [
        protocol.process_event("a"),
        protocol.process_event("b"),
        protocol.process_event("c"),
    ]

    assert results == [
        "perturbed-a",
        "perturbed-b",
        None,
    ]
    assert mechanism.perturb_event.call_args_list == [
        call(event="a"),
        call(event="b"),
    ]


def test_none_resets_completed_connection():
    protocol, _, event_generator = make_protocol(uniform_length=1)

    protocol.process_event("a")
    result = protocol.process_event(None)

    assert result is None
    assert protocol.connected is False
    assert protocol.k == 0
    event_generator.generate_event.assert_not_called()


def test_process_event_multiple_connections():
    protocol, mechanism, _ = make_protocol(uniform_length=3)

    events = [None, None, "a", None, None, None, "a", "b", "c", "d"]
    results = [protocol.process_event(event=event) for event in events]
    assert  results == [None, None, "perturbed-a", "synthetic", "synthetic", None, "perturbed-a", "perturbed-b", "perturbed-c", None]



def test_process_stream_fills_missing_events():
    protocol, mechanism, event_generator = make_protocol(
        uniform_length=3
    )

    result = protocol.process_stream(["a"])

    assert result == [
        "perturbed-a",
        "synthetic",
        "synthetic",
    ]
    mechanism.perturb_event.assert_called_once_with(event="a")
    assert event_generator.generate_event.call_count == 2


def test_process_stream_cuts_excess_events():
    protocol, mechanism, event_generator = make_protocol(
        uniform_length=2
    )

    result = protocol.process_stream(["a", "b", "c"])

    assert result == [
        "perturbed-a",
        "perturbed-b",
        None,
    ]
    assert mechanism.perturb_event.call_count == 2
    event_generator.generate_event.assert_not_called()


def test_process_stream_preserves_padding_before_connection():
    protocol, _, event_generator = make_protocol(uniform_length=2)

    result = protocol.process_stream([None, "a"])

    assert result == [
        None,
        "perturbed-a",
        "synthetic",
    ]
    event_generator.generate_event.assert_called_once_with()


def test_empty_stream_returns_empty_list():
    protocol, mechanism, event_generator = make_protocol()

    result = protocol.process_stream([])

    assert result == []
    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_process_stream_resets_protocol_state():
    protocol, _, _ = make_protocol(uniform_length=2)

    protocol.process_stream(["a"])

    assert protocol.connected is False
    assert protocol.k == 0


def test_protocol_can_process_multiple_streams():
    protocol, _, _ = make_protocol(uniform_length=2)

    first = protocol.process_stream(["a"])
    second = protocol.process_stream(["b"])

    assert first == ["perturbed-a", "synthetic"]
    assert second == ["perturbed-b", "synthetic"]


def test_negative_uniform_length_is_rejected():
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(ValueError, match="uniform_length"):
        UniformLength(
            epsilon=1.0,
            uniform_length=-1,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
            delta_length=5,
        )


def test_to_dict():
    mechanism = Mock(spec=Mechanism)
    mechanism.to_dict.return_value = {
        "name": "ExampleMechanism",
    }

    event_generator = Mock(spec=EventGenerator)
    event_generator.to_dict.return_value = {
        "name": "ExampleEventGenerator",
    }

    protocol = UniformLength(
        epsilon=1.0,
        uniform_length=5,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        extendable=False,
        delta_length=5,

    )

    assert protocol.to_dict() == {
        "name": "Fixed Length",
        "epsilon": 1.0,
        "delta_length": 5,
        "uniform_length": 5,
        "perturbing_mechanism": {
            "name": "ExampleMechanism",
        },
        "synthetic_event_generator": {
            "name": "ExampleEventGenerator",
        },
        "extendable": False,
    }