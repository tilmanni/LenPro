import math
from unittest.mock import Mock

import numpy as np
import pytest

from lenpro.mechanisms.mechanisms import Mechanism
from lenpro.protocols.randomized_response import RandomizedResponse
from lenpro.stream_generation.event_generators import EventGenerator


def make_protocol(
    *,
    epsilon=1.0,
    delta_length=1.0,
    random_value=None,
):
    mechanism = Mock(spec=Mechanism)
    mechanism.perturb_event.return_value = "perturbed"

    event_generator = Mock(spec=EventGenerator)
    event_generator.generate_event.return_value = "synthetic"

    rng = Mock(spec=np.random.Generator)

    if random_value is not None:
        rng.uniform.return_value = random_value

    protocol = RandomizedResponse(
        epsilon=epsilon,
        delta_length=delta_length,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        rng=rng,
    )

    return protocol, mechanism, event_generator, rng


def test_probabilities_at_zero_epsilon():
    protocol, _, _, _ = make_protocol(
        epsilon=0,
        delta_length=1,
    )

    assert protocol.p == pytest.approx(0.5)
    assert protocol.q == pytest.approx(0.5)


def test_probabilities_sum_to_one():
    protocol, _, _, _ = make_protocol(
        epsilon=2,
        delta_length=4,
    )

    assert protocol.p + protocol.q == pytest.approx(1.0)


def test_probability_values_are_calculated_correctly():
    protocol, _, _, _ = make_protocol(
        epsilon=2,
        delta_length=4,
    )

    expected_p = math.exp(0.5) / (math.exp(0.5) + 1)
    expected_q = 1 / (math.exp(0.5) + 1)

    assert protocol.p == pytest.approx(expected_p)
    assert protocol.q == pytest.approx(expected_q)


def test_larger_epsilon_increases_p():
    low_epsilon, _, _, _ = make_protocol(
        epsilon=1,
        delta_length=1,
    )
    high_epsilon, _, _, _ = make_protocol(
        epsilon=10,
        delta_length=1,
    )

    assert high_epsilon.p > low_epsilon.p
    assert high_epsilon.q < low_epsilon.q


def test_large_epsilon_does_not_overflow():
    protocol, _, _, _ = make_protocol(
        epsilon=1_000_000,
        delta_length=1,
    )

    assert protocol.p == pytest.approx(1.0)
    assert protocol.q == pytest.approx(0.0)


def test_non_none_event_is_perturbed_when_random_value_is_below_p():
    protocol, mechanism, event_generator, rng = make_protocol(
        random_value=0.0,
    )

    result = protocol.process_event("original")

    assert result == "perturbed"

    mechanism.perturb_event.assert_called_once_with(
        event="original"
    )
    event_generator.generate_event.assert_not_called()
    rng.uniform.assert_called_once_with(0, 1)


def test_non_none_event_is_removed_when_random_value_is_above_p():
    protocol, mechanism, event_generator, rng = make_protocol(
        random_value=1.0,
    )

    result = protocol.process_event("original")

    assert result is None

    mechanism.perturb_event.assert_not_called()
    event_generator.generate_event.assert_not_called()
    rng.uniform.assert_called_once_with(0, 1)


def test_none_becomes_synthetic_when_random_value_is_below_q():
    protocol, mechanism, event_generator, rng = make_protocol(
        random_value=0.0,
    )

    result = protocol.process_event(None)

    assert result == "synthetic"

    event_generator.generate_event.assert_called_once_with()
    mechanism.perturb_event.assert_not_called()
    rng.uniform.assert_called_once_with(0, 1)


def test_none_remains_none_when_random_value_is_above_q():
    protocol, mechanism, event_generator, rng = make_protocol(
        random_value=1.0,
    )

    result = protocol.process_event(None)

    assert result is None

    event_generator.generate_event.assert_not_called()
    mechanism.perturb_event.assert_not_called()
    rng.uniform.assert_called_once_with(0, 1)


def test_probability_boundary_is_inclusive_for_original_event():
    protocol, mechanism, _, rng = make_protocol()

    rng.uniform.return_value = protocol.p

    result = protocol.process_event("original")

    assert result == "perturbed"
    mechanism.perturb_event.assert_called_once_with(
        event="original"
    )


def test_probability_boundary_is_inclusive_for_synthetic_event():
    protocol, _, event_generator, rng = make_protocol()

    rng.uniform.return_value = protocol.q

    result = protocol.process_event(None)

    assert result == "synthetic"
    event_generator.generate_event.assert_called_once_with()


def test_process_stream_processes_every_event():
    protocol, _, _, _ = make_protocol()

    protocol.process_event = Mock(
        side_effect=["result-1", None, "result-3"]
    )

    stream = ["event-1", None, "event-3"]

    result = protocol.process_stream(stream)

    assert result == ["result-1", None, "result-3"]
    assert protocol.process_event.call_count == 3
    assert protocol.process_event.call_args_list == [
        (("event-1",),),
        ((None,),),
        (("event-3",),),
    ]


def test_process_stream_returns_empty_list_for_empty_stream():
    protocol, _, _, _ = make_protocol()

    result = protocol.process_stream([])

    assert result == []


@pytest.mark.parametrize(
    "delta_length",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_invalid_delta_length_is_rejected(delta_length):
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="delta_length must be greater than zero",
    ):
        RandomizedResponse(
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
        RandomizedResponse(
            epsilon=1,
            delta_length=delta_length,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        -1,
        -0.1,
    ],
)
def test_negative_epsilon_is_rejected(epsilon):
    mechanism = Mock(spec=Mechanism)
    event_generator = Mock(spec=EventGenerator)

    with pytest.raises(
        ValueError,
        match="epsilon must be non-negative",
    ):
        RandomizedResponse(
            epsilon=epsilon,
            delta_length=1,
            perturbing_mechanism=mechanism,
            synthetic_event_generator=event_generator,
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

    protocol = RandomizedResponse(
        epsilon=2,
        delta_length=4,
        perturbing_mechanism=mechanism,
        synthetic_event_generator=event_generator,
        rng=np.random.default_rng(42),
    )

    assert protocol.to_dict() == {
        "name": "RandomizedResponse",
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