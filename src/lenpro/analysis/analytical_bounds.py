from dataclasses import dataclass
from typing import Callable, Mapping, Any

from lenpro.distributions.distributions import distribution_from_dict

import numpy as np


@dataclass(frozen=True)
class AnalyticalResult:
    value: float | None = None
    lower: float | None = None
    upper: float | None = None


BoundFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    AnalyticalResult,
]


def lenpro_empty_output_probability(
    protocol: Mapping[str, Any],
    stream_generator: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> AnalyticalResult:
    epsilon = protocol["epsilon"]
    delta_length = protocol["delta_length"]


    reconstructed_distribution = distribution_from_dict(distribution_config=distribution)


    p = float(-np.expm1(-epsilon / delta_length))
    # Sum over fixed horizon could turn into problem if horizon and actual sampling divorced
    # -> use delta length?
    value = sum(reconstructed_distribution.pmf(x=x) * np.pow(1 - p, x) / (2 - p) for x in range(1, delta_length + 1))

    return AnalyticalResult(
        value=float(value),
    )

def randomized_response_empty_output_probability(
    protocol: Mapping[str, Any],
    stream_generator: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> AnalyticalResult:
    epsilon = protocol["epsilon"]
    delta_length = protocol["delta_length"]
    fixed_horizon = stream_generator["fixed_horizon"]

    privacy_per_event = epsilon / delta_length

    exp_negative = np.exp(-privacy_per_event)

    # same as exp / (1.0 + exp)
    p = 1.0 / (1.0 + exp_negative)
    q = exp_negative / (1.0 + exp_negative)

    reconstructed_distribution = distribution_from_dict(distribution_config=distribution)

    value = sum(reconstructed_distribution.pmf(x=x) * np.pow(q, x) * np.pow(p, fixed_horizon - x) for x in range(1, delta_length + 1))

    return AnalyticalResult(
        value=float(value),
    )

def lenpro_length_unchanged_probability(
    protocol: Mapping[str, Any],
    stream_generator: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> AnalyticalResult:
    epsilon = protocol["epsilon"]
    delta_length = protocol["delta_length"]


    reconstructed_distribution = distribution_from_dict(distribution_config=distribution)

    zero_prob = reconstructed_distribution.pmf(x = 0)

    p = float(-np.expm1(-epsilon / delta_length))

    # Replace with the derived formulas.
    value = zero_prob + (1 - zero_prob) * (p / (2 - p))

    return AnalyticalResult(
        value=float(value),
    )


ANALYTICAL_FUNCTIONS: dict[
    tuple[str, str],
    BoundFunction,
] = {
    (
        "LenPro",
        "LengthUnchangedProbabilityMeasure",
    ): lenpro_length_unchanged_probability,


    (
        "LenPro",
        "EmptyOutputProbabilityMeasure",
    ): lenpro_empty_output_probability,

    (
        "RandomizedResponse",
        "EmptyOutputProbabilityMeasure",
    ): randomized_response_empty_output_probability,
}

def calculate_analytical_result(
    protocol: Mapping[str, Any],
    stream_generator: Mapping[str, Any],
    distribution: Mapping[str, Any],
    measure_name: str,
) -> AnalyticalResult:
    key = (
        protocol["name"],
        measure_name,
    )

    function = ANALYTICAL_FUNCTIONS.get(key)

    if function is None:
        return AnalyticalResult()

    result = function(
        protocol,
        stream_generator,
        distribution,
    )

    for name, value in (
        ("value", result.value),
        ("lower", result.lower),
        ("upper", result.upper),
    ):
        if value is not None and not np.isfinite(value):
            raise ValueError(
                f"Analytical {name} for {key} is not finite: "
                f"{value}"
            )

    return result