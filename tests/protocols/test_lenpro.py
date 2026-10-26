import math
from unittest.mock import Mock, call

import numpy as np
import pytest

from lenpro.mechanisms.mechanisms import Mechanism
from lenpro.protocols.lenpro import LenPro
from lenpro.stream_generation.event_generators import EventGenerator


def make_protocol(
    *,
    epsilon=1.0,
    delta_length=1.0,
    a=0,
    r=0,
):
    mechanism = Mock(spec=Mechanism)
    mechanism.perturb_event.return_value = "perturbed"

    event_generator = Mock(spec=EventGenerator)
    event_generator.generate_event.return_value = "synthetic"

    rng = Mock(spec=np.random.Generator)

    # geometric() returns values starting at 1.
    # LenPro subtracts one to obtain a and r.
    rng.geometric.side_effect = [a + 1, r + 1]

    protocol = LenPro(
        epsilon=epsilon,
        delta_length=delta_length,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        rng=rng,
    )

    return protocol, mechanism, event_generator, rng


def test_probability_is_calculated_correctly():
    protocol, _, _, _ = make_protocol(
        epsilon=2,
        delta_length=4,
    )

    expected = 1 - math.exp(-0.5)

    assert protocol.p == pytest.approx(expected)


def test_probability_is_between_zero_and_one():
    protocol, _, _, _ = make_protocol(
        epsilon=2,
        delta_length=4,
    )

    assert 0 < protocol.p <= 1


def test_initial_state():
    protocol, _, _, _ = make_protocol()

    assert protocol.received_events == []
    assert protocol.returned_events == []
    assert protocol.k == 0
    assert protocol.connected is False
    assert protocol.a is None
    assert protocol.r is None


def test_zero_epsilon_is_rejected():
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="epsilon must be greater than zero",
    ):
        LenPro(
            epsilon=0,
            delta_length=1,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


def test_negative_epsilon_is_rejected():
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="epsilon must be non-negative",
    ):
        LenPro(
            epsilon=-1,
            delta_length=1,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_epsilon_is_rejected(epsilon):
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(ValueError, match="epsilon must be finite"):
        LenPro(
            epsilon=epsilon,
            delta_length=1,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


@pytest.mark.parametrize(
    "delta_length",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_non_positive_delta_length_is_rejected(delta_length):
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="delta_length must be greater than zero",
    ):
        LenPro(
            epsilon=1,
            delta_length=delta_length,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


@pytest.mark.parametrize(
    "delta_length",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_delta_length_is_rejected(delta_length):
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="delta_length must be finite",
    ):
        LenPro(
            epsilon=1,
            delta_length=delta_length,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


def test_first_event_starts_connection():
    protocol, mechanism, _, rng = make_protocol(
        a=0,
        r=0,
    )

    result = protocol.process_event("original")

    assert result == "perturbed"
    assert protocol.connected is True
    assert protocol.received_events == ["original"]
    assert protocol.returned_events == ["perturbed"]
    assert protocol.k == 1

    assert rng.geometric.call_args_list == [
        call(p=protocol.p),
        call(p=protocol.p),
    ]

    mechanism.perturb_event.assert_called_once_with(
        event="original",
        previous_outputs=[],
    )


def test_initial_events_are_suppressed_according_to_r():
    protocol, mechanism, _, _ = make_protocol(
        a=0,
        r=2,
    )

    first = protocol.process_event("event-1")
    second = protocol.process_event("event-2")
    third = protocol.process_event("event-3")

    assert first is None
    assert second is None
    assert third == "perturbed"

    assert protocol.returned_events == [
        None,
        None,
        "perturbed",
    ]
    assert protocol.k == 1

    mechanism.perturb_event.assert_called_once_with(
        event="event-3",
        previous_outputs=[None, None],
    )


def test_none_before_connection_returns_none():
    protocol, mechanism, event_generator, rng = make_protocol()

    result = protocol.process_event(None)

    assert result is None
    assert protocol.connected is False
    assert protocol.received_events == []
    assert protocol.returned_events == []

    rng.geometric.assert_not_called()
    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_none_generates_synthetic_event_when_output_is_too_short():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=1,
        r=0,
    )

    assert protocol.process_event("original") == "perturbed"

    result = protocol.process_event(None)

    assert result == "synthetic"
    assert protocol.k == 2

    event_generator.generate_event.assert_called_once_with(
        previous_event="perturbed",
    )


def test_none_resets_completed_connection():
    protocol, _, event_generator, _ = make_protocol(
        a=0,
        r=0,
    )

    protocol.process_event("original")
    result = protocol.process_event(None)

    assert result is None
    assert protocol.connected is False
    assert protocol.received_events == []
    assert protocol.returned_events == []
    assert protocol.k == 0

    event_generator.generate_event.assert_not_called()



def test_process_stream_without_length_adjustment():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=0,
        r=0,
    )
    mechanism.perturb_event.side_effect = [
        "perturbed-1",
        "perturbed-2",
    ]

    result = protocol.process_stream(
        ["original-1", "original-2"]
    )

    assert result == [
        "perturbed-1",
        "perturbed-2",
    ]

    assert mechanism.perturb_event.call_count == 2
    event_generator.generate_event.assert_not_called()


def test_process_stream_suppresses_first_r_events():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=0,
        r=2,
    )

    result = protocol.process_stream(
        ["event-1", "event-2", "event-3"]
    )

    assert result == [
        None,
        None,
        "perturbed",
    ]

    mechanism.perturb_event.assert_called_once_with(
        event="event-3",
        previous_outputs=[None, None],
    )
    event_generator.generate_event.assert_not_called()


def test_process_stream_adds_synthetic_events():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=2,
        r=0,
    )

    mechanism.perturb_event.return_value = "perturbed"
    event_generator.generate_event.side_effect = [
        "synthetic-1",
        "synthetic-2",
    ]

    result = protocol.process_stream(["original"])

    assert result == [
        "perturbed",
        "synthetic-1",
        "synthetic-2",
    ]

    assert event_generator.generate_event.call_args_list == [
        call(previous_event="perturbed"),
        call(previous_event="synthetic-1"),
    ]


def test_synthetic_events_are_generated_even_when_all_originals_suppressed():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=3,
        r=2,
    )

    event_generator.generate_event.side_effect = [
        "synthetic-1",
        "synthetic-2",
    ]

    # Adjusted length = 1 - 2 + 3 = 2.
    # The original event is suppressed, but two synthetic
    # events must still be generated.
    result = protocol.process_stream(["original"])

    assert result == [
        None,
        "synthetic-1",
        "synthetic-2",
    ]

    mechanism.perturb_event.assert_not_called()

    assert event_generator.generate_event.call_args_list == [
        call(previous_event=None),
        call(previous_event="synthetic-1"),
    ]


def test_negative_adjusted_length_is_treated_as_zero():
    protocol, mechanism, event_generator, _ = make_protocol(
        a=0,
        r=3,
    )

    result = protocol.process_stream(["original"])

    assert result == [None]
    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_empty_stream_returns_empty_list():
    protocol, mechanism, event_generator, rng = make_protocol()

    result = protocol.process_stream([])

    assert result == []
    rng.geometric.assert_not_called()
    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_process_stream_resets_state():
    protocol, _, _, _ = make_protocol(
        a=1,
        r=0,
    )

    protocol.process_stream(["original"])

    assert protocol.connected is False
    assert protocol.received_events == []
    assert protocol.returned_events == []
    assert protocol.k == 0
    assert protocol.a is None
    assert protocol.r is None

def test_protocol_can_process_multiple_streams():
    mechanism = Mock(spec=Mechanism)
    mechanism.perturb_event.side_effect = [
        "first-result",
        "second-result",
    ]

    event_generator = Mock(spec=EventGenerator)

    rng = Mock(spec=np.random.Generator)

    # First stream: a=0, r=0.
    # Second stream: a=0, r=0.
    rng.geometric.side_effect = [1, 1, 1, 1]

    protocol = LenPro(
        epsilon=1,
        delta_length=1,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        rng=rng,
    )

    first = protocol.process_stream(["first"])
    second = protocol.process_stream(["second"])

    assert first == ["first-result"]
    assert second == ["second-result"]
    assert rng.geometric.call_count == 4

def test_to_dict():
    mechanism = Mock(spec=Mechanism)
    mechanism.to_dict.return_value = {
        "name": "ExampleMechanism",
    }

    event_generator = Mock(spec=EventGenerator)
    event_generator.to_dict.return_value = {
        "name": "ExampleEventGenerator",
    }

    protocol = LenPro(
        epsilon=2,
        delta_length=4,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        rng=np.random.default_rng(42),
    )

    assert protocol.to_dict() == {
        "name": "LenPro",
        "epsilon": 2,
        "delta_length": 4,
        "perturbing_mechanism": {
            "name": "ExampleMechanism",
        },
        "synthetic_event_generator": {
            "name": "ExampleEventGenerator",
        },
    }

    mechanism.to_dict.assert_called_once_with()
    event_generator.to_dict.assert_called_once_with()