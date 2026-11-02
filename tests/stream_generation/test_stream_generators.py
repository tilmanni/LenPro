from unittest.mock import Mock

import pytest

from lenpro.stream_generation.event_generators import EventGenerator
from lenpro.stream_generation.length_generators import LengthGenerator
from lenpro.stream_generation.start_point_generators import StartPointGenerator, CenteredStreamStartPointGenerator
from lenpro.stream_generation.stream_generators import (
    StandardStreamGenerator,
    StreamGenerator,
)


def make_generator(length, events, fixed_horizon=None, start_point_generator=None):
    """Create a stream generator with mocked dependencies."""
    length_generator = Mock(spec=LengthGenerator)
    length_generator.generate_length.return_value = length

    event_generator = Mock(spec=EventGenerator)
    event_generator.generate_event.side_effect = events

    generator = StandardStreamGenerator(
        length_generator=length_generator,
        event_generator=event_generator,
        fixed_horizon=fixed_horizon,
        start_point_generator=start_point_generator
    )

    return generator, length_generator, event_generator


def test_stream_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        StreamGenerator()


def test_generate_stream_without_fixed_length():
    generator, length_generator, event_generator = make_generator(
        length=3,
        events=["a", "b", "c"],
    )

    stream = generator.generate_stream()

    assert stream == ["a", "b", "c"]
    length_generator.generate_length.assert_called_once_with()
    assert event_generator.generate_event.call_count == 3


def test_generate_empty_stream():
    generator, _, event_generator = make_generator(
        length=0,
        events=[],
    )

    stream = generator.generate_stream()

    assert stream == []
    event_generator.generate_event.assert_not_called()

# Centered Stream
def test_short_stream_is_padded_evenly():
    generator, _, event_generator = make_generator(
        length=2,
        events=["a", "b"],
        fixed_horizon=6,
        start_point_generator=CenteredStreamStartPointGenerator()
    )

    stream = generator.generate_stream()

    assert stream == [None, None, "a", "b", None, None]
    assert len(stream) == 6
    assert event_generator.generate_event.call_count == 2

# Centered Stream 
def test_short_stream_with_odd_padding_difference():
    generator, _, event_generator = make_generator(
        length=2,
        events=["a", "b"],
        fixed_horizon=5,
        start_point_generator=CenteredStreamStartPointGenerator()
    )

    stream = generator.generate_stream()

    # floor(3 / 2) = 1 None before
    # ceil(3 / 2) = 2 Nones after
    assert stream == [None, "a", "b", None, None]
    assert len(stream) == 5
    assert event_generator.generate_event.call_count == 2

# Centered Stream
def test_stream_longer_than_fixed_length_is_truncated():
    generator, _, event_generator = make_generator(
        length=5,
        events=["a", "b", "c"],
        fixed_horizon=3,
        start_point_generator=CenteredStreamStartPointGenerator()
    )

    stream = generator.generate_stream()

    assert stream == ["a", "b", "c"]
    assert len(stream) == 3
    assert event_generator.generate_event.call_count == 3

# Centered Stream
def test_stream_equal_to_fixed_length_is_not_padded():
    generator, _, event_generator = make_generator(
        length=3,
        events=["a", "b", "c"],
        fixed_horizon=3,
    )

    stream = generator.generate_stream()

    assert stream == ["a", "b", "c"]
    assert None not in stream
    assert event_generator.generate_event.call_count == 3


def test_generate_multiple_streams():
    length_generator = Mock(spec=LengthGenerator)
    length_generator.generate_length.return_value = 2

    event_generator = Mock(spec=EventGenerator)
    event_generator.generate_event.return_value = "event"

    generator = StandardStreamGenerator(
        length_generator=length_generator,
        event_generator=event_generator,
    )

    streams = generator.generate_streams(amount=3)

    assert streams == [
        ["event", "event"],
        ["event", "event"],
        ["event", "event"],
    ]
    assert length_generator.generate_length.call_count == 3
    assert event_generator.generate_event.call_count == 6


def test_generate_zero_streams():
    length_generator = Mock(spec=LengthGenerator)
    event_generator = Mock(spec=EventGenerator)

    generator = StandardStreamGenerator(
        length_generator=length_generator,
        event_generator=event_generator,
    )

    streams = generator.generate_streams(amount=0)

    assert streams == []
    length_generator.generate_length.assert_not_called()
    event_generator.generate_event.assert_not_called()


def test_to_dict():
    length_generator = Mock(spec=LengthGenerator)
    length_generator.to_dict.return_value = {
        "name": "ExampleLengthGenerator",
    }

    event_generator = Mock(spec=EventGenerator)
    event_generator.to_dict.return_value = {
        "name": "ExampleEventGenerator",
    }

    generator = StandardStreamGenerator(
        length_generator=length_generator,
        event_generator=event_generator,
        fixed_horizon=10,
    )

    assert generator.to_dict() == {
        "name": "StandardStreamGenerator",
        "length_generator": {
            "name": "ExampleLengthGenerator",
        },
        "event_generator": {
            "name": "ExampleEventGenerator",
        },
        "fixed_horizon": 10,
        "start_point_generator": None,
    }

    length_generator.to_dict.assert_called_once_with()
    event_generator.to_dict.assert_called_once_with()