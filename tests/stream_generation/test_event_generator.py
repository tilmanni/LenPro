import pytest

from lenpro.stream_generation.event_generators import (
    EventGenerator,
    ConstantEventGenerator,
)


def test_event_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        EventGenerator()


def test_constant_event_generator_returns_constant():
    generator = ConstantEventGenerator(constant_event=42)

    assert generator.generate_event() == 42
    assert generator.generate_event() == 42


def test_constant_event_generator_ignores_previous_event():
    generator = ConstantEventGenerator(constant_event="constant")

    result = generator.generate_event(previous_event="previous")

    assert result == "constant"


def test_constant_event_generator_preserves_object_identity():
    event = {"type": "example", "value": 42}
    generator = ConstantEventGenerator(constant_event=event)

    result = generator.generate_event()

    assert result is event


def test_constant_event_generator_to_dict():
    generator = ConstantEventGenerator(constant_event="example")

    assert generator.to_dict() == {
        "name": "ConstantEventGenerator",
        "constant_event": "example",
    }