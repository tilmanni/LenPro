
# execution.py
import json
from pathlib import Path

import numpy as np


def execution_from_data(
    aggregator,
    input_data,
    stream_generator,
    experiment_metadata=None,
):
    return {
        "stream_generator": stream_generator.to_dict(),
        "data_set_size": len(input_data),
        "experiment": experiment_metadata or {},
        "results": aggregator.measure_run(input_data),
    }


def execution_from_generator(
    aggregator,
    stream_generator,
    data_set_size,
    experiment_metadata=None,
):
    input_data = list(stream_generator.generate_streams(data_set_size))

    return execution_from_data(
        aggregator=aggregator,
        input_data=input_data,
        stream_generator=stream_generator,
        experiment_metadata=experiment_metadata,
    )


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def store_result(result, file_name="run", directory="results"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{file_name}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            default=to_builtin,
            indent=2,
            allow_nan=False,
        )

    return output_path
