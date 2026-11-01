from collections import Counter
from unittest.mock import Mock, call

import pytest

from lenpro.aggregator import Aggregator
from lenpro.measures.aggregate_measures import AggregateMeasure
from lenpro.measures.entry_level_measures import EntryLevelMeasure
from lenpro.measures.global_measures import GlobalMeasure
from lenpro.protocols.ldp_protocol import LDPProtocol


def test_aggregator_starts_with_empty_registries():
    aggregator = Aggregator()

    assert aggregator.protocols == []
    assert aggregator.entry_level_measures == []
    assert aggregator.aggregate_measures == []
    assert aggregator.global_measures == []


def test_register_protocol():
    aggregator = Aggregator()
    protocol = Mock(spec=LDPProtocol)

    aggregator.register_protocol(protocol)

    assert aggregator.protocols == [protocol]


def test_register_entry_level_measure():
    aggregator = Aggregator()
    measure = Mock(spec=EntryLevelMeasure)

    aggregator.register_entry_level_measure(measure)

    assert aggregator.entry_level_measures == [measure]


def test_register_aggregate_measure():
    aggregator = Aggregator()
    measure = Mock(spec=AggregateMeasure)

    aggregator.register_aggregate_measure(measure)

    assert aggregator.aggregate_measures == [measure]


def test_register_global_measure():
    aggregator = Aggregator()
    measure = Mock(spec=GlobalMeasure)

    aggregator.register_global_measure(measure)

    assert aggregator.global_measures == [measure]

def test_run_simulation_processes_stream_with_each_protocol():
    aggregator = Aggregator()

    first_protocol = Mock(spec=LDPProtocol)
    first_protocol.process_stream.return_value = [
        "first-result",
        None,
    ]

    second_protocol = Mock(spec=LDPProtocol)
    second_protocol.process_stream.return_value = [
        "second-result",
        "second-result",
    ]

    aggregator.register_protocol(first_protocol)
    aggregator.register_protocol(second_protocol)

    input_stream = ["a", "b"]

    result = aggregator.run_simulation(input_stream)

    assert result == [
        ["first-result", None],
        ["second-result", "second-result"],
    ]

    first_protocol.process_stream.assert_called_once_with(
        input_stream
    )
    second_protocol.process_stream.assert_called_once_with(
        input_stream
    )


def test_run_simulation_without_protocols_returns_empty_list():
    aggregator = Aggregator()

    result = aggregator.run_simulation(["a", "b"])

    assert result == []


def test_measure_run():
    aggregator = Aggregator()

    protocol = Mock(spec=LDPProtocol)
    protocol.process_stream.side_effect = [
        ["perturbed-1", None],
        ["perturbed-2"],
    ]
    protocol.to_dict.return_value = {
        "name": "TestProtocol",
    }

    entry_measure = Mock(spec=EntryLevelMeasure)
    entry_measure.process_data.side_effect = [0.5, 1.0]
    entry_measure.to_dict.return_value = {
        "name": "TestEntryMeasure",
    }

    aggregate_measure = Mock(spec=AggregateMeasure)
    aggregate_measure.process_measures.return_value = 0.75
    aggregate_measure.to_dict.return_value = {
        "name": "TestAggregateMeasure",
    }

    global_measure = Mock(spec=GlobalMeasure)
    global_measure.process_dataset.return_value = 0.25
    global_measure.to_dict.return_value = {
        "name": "TestGlobalMeasure",
    }

    aggregator.register_protocol(protocol)
    aggregator.register_entry_level_measure(entry_measure)
    aggregator.register_aggregate_measure(aggregate_measure)
    aggregator.register_global_measure(global_measure)

    inputs = [
        ["original-1", "original-2"],
        ["original-3"],
    ]

    result = aggregator.measure_run(inputs)

    assert result == [
        {
            "protocol": {
                "name": "TestProtocol",
            },
            "aggregate": [
                {
                    "measure": {
                        "name": "TestAggregateMeasure",
                    },
                    "entry_level_measure": {
                        "name": "TestEntryMeasure",
                    },
                    "value": 0.75,
                }
            ],
            "global": [
                {
                    "measure": {
                        "name": "TestGlobalMeasure",
                    },
                    "value": 0.25,
                }
            ],
        }
    ]

    assert protocol.process_stream.call_args_list == [
        call(["original-1", "original-2"]),
        call(["original-3"]),
    ]

    assert entry_measure.process_data.call_args_list == [
        call(
            original_data=["original-1", "original-2"],
            perturbed_data=["perturbed-1", None],
            protocol=protocol,
        ),
        call(
            original_data=["original-3"],
            perturbed_data=["perturbed-2"],
            protocol=protocol,
        ),
    ]

    aggregate_measure.process_measures.assert_called_once_with(
        [0.5, 1.0]
    )

    global_measure.process_dataset.assert_called_once_with(
        original_dataset=inputs,
        perturbed_dataset=[
            ["perturbed-1", None],
            ["perturbed-2"],
        ],
    )

def test_every_aggregate_measure_is_applied_to_every_entry_measure():
    aggregator = Aggregator()

    protocol = Mock(spec=LDPProtocol)
    protocol.process_stream.return_value = ["output"]
    protocol.to_dict.return_value = {
        "name": "Protocol",
    }

    first_entry_measure = Mock(spec=EntryLevelMeasure)
    first_entry_measure.process_data.return_value = 1.0
    first_entry_measure.to_dict.return_value = {
        "name": "FirstEntryMeasure",
    }

    second_entry_measure = Mock(spec=EntryLevelMeasure)
    second_entry_measure.process_data.return_value = 2.0
    second_entry_measure.to_dict.return_value = {
        "name": "SecondEntryMeasure",
    }

    first_aggregate_measure = Mock(spec=AggregateMeasure)
    first_aggregate_measure.process_measures.side_effect = [
        10.0,
        20.0,
    ]
    first_aggregate_measure.to_dict.return_value = {
        "name": "FirstAggregateMeasure",
    }

    second_aggregate_measure = Mock(spec=AggregateMeasure)
    second_aggregate_measure.process_measures.side_effect = [
        100.0,
        200.0,
    ]
    second_aggregate_measure.to_dict.return_value = {
        "name": "SecondAggregateMeasure",
    }

    aggregator.register_protocol(protocol)
    aggregator.register_entry_level_measure(
        first_entry_measure
    )
    aggregator.register_entry_level_measure(
        second_entry_measure
    )
    aggregator.register_aggregate_measure(
        first_aggregate_measure
    )
    aggregator.register_aggregate_measure(
        second_aggregate_measure
    )

    result = aggregator.measure_run([["input"]])

    aggregate_results = result[0]["aggregate"]

    assert len(aggregate_results) == 4

    assert [
        entry["value"]
        for entry in aggregate_results
    ] == [
        10.0,
        20.0,
        100.0,
        200.0,
    ]

    assert first_aggregate_measure.process_measures.call_args_list == [
        call([1.0]),
        call([2.0]),
    ]

    assert second_aggregate_measure.process_measures.call_args_list == [
        call([1.0]),
        call([2.0]),
    ]


def test_measure_run_without_protocols_returns_empty_list():
    aggregator = Aggregator()

    result = aggregator.measure_run([["input"]])

    assert result == []


def test_measure_run_without_measures():
    aggregator = Aggregator()

    protocol = Mock(spec=LDPProtocol)
    protocol.process_stream.return_value = ["output"]
    protocol.to_dict.return_value = {
        "name": "Protocol",
    }

    aggregator.register_protocol(protocol)

    result = aggregator.measure_run([["input"]])

    assert result == [
        {
            "protocol": {
                "name": "Protocol",
            },
            "aggregate": [],
            "global": [],
        }
    ]
