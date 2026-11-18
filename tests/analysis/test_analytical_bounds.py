import math
import numpy as np
import pytest

from lenpro.analysis.analytical_bounds import (
AnalyticalResult,
calculate_analytical_result,
lenpro_empty_output_probability,
lenpro_length_unchanged_probability,
randomized_response_empty_output_probability,
)

LENPRO_PROTOCOL = {"name": "LenPro", "epsilon": 1.0, "delta_length": 10}
RR_PROTOCOL = {"name": "RandomizedResponse", "epsilon": 1.0, "delta_length": 10}

def uniform_length_distribution(low, high):
    return {"name": "UniformDistribution", "low": low, "high": high}

def stream_generator_with_horizon(h):
    return {
        "fixed_horizon": h,
        # included for completeness; code doesn’t use length_generator here
        "length_generator": {
            "name": "DistributionLengthGenerator",
            "distribution": uniform_length_distribution(1, h),
        },
    }

@pytest.mark.parametrize("protocol, measure_name, expect_function",
    [
        (LENPRO_PROTOCOL, "EmptyOutputProbabilityMeasure", lenpro_empty_output_probability),
        (LENPRO_PROTOCOL, "LengthUnchangedProbabilityMeasure", lenpro_length_unchanged_probability),
        (RR_PROTOCOL, "EmptyOutputProbabilityMeasure", randomized_response_empty_output_probability),
    ],
)
def test_dispatch_returns_finite_value(protocol, measure_name, expect_function):
    H = 10
    sg = stream_generator_with_horizon(H)
    dist = uniform_length_distribution(1, H)
    result = calculate_analytical_result(protocol=protocol, stream_generator=sg, distribution=dist, measure_name=measure_name) 
    assert isinstance(result, AnalyticalResult) 
    assert result.lower is None and result.upper is None 
    assert result.value is not None 
    assert math.isfinite(result.value) # Probabilities should be in [0, 1] assert 0.0 <= result.value <= 1.0

def test_unknown_mapping_returns_empty_result():
    sg = stream_generator_with_horizon(10)
    dist = uniform_length_distribution(1, 10)
    protocol = {"name": "UnknownProtocol", "epsilon": 1.0, "delta_length": 10}
    result = calculate_analytical_result(protocol=protocol, stream_generator=sg, distribution=dist, measure_name="EmptyOutputProbabilityMeasure") 
    assert isinstance(result, AnalyticalResult) 
    assert result.value is None and result.lower is None and result.upper is None

def test_lenpro_length_unchanged_probability_matches_formula():
# value = p / (2 - p), p = 1 - exp(-epsilon/delta)
    eps, dl = 2.0, 5.0
    protocol = {"name": "LenPro", "epsilon": eps, "delta_length": dl}
    sg = stream_generator_with_horizon(10)
    dist = uniform_length_distribution(1, 10)
    result = calculate_analytical_result(protocol=protocol, stream_generator=sg, distribution=dist, measure_name="LengthUnchangedProbabilityMeasure") 
    p = float(-np.expm1(-eps / dl)) 
    expected = p / (2 - p) 
    assert result.value == pytest.approx(expected, rel=1e-12, abs=1e-12)

@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0, 5.0])
def test_monotonicity_lenpro_empty_output_decreases_with_epsilon(epsilon):
# As epsilon increases, q decreases and probability of empty output should not increase.
    sg = stream_generator_with_horizon(10)
    dist = uniform_length_distribution(1, 10)
    proto_low = {"name": "LenPro", "epsilon": 0.0, "delta_length": 10} 
    proto_hi = {"name": "LenPro", "epsilon": epsilon, "delta_length": 10} 
    res_low = calculate_analytical_result(proto_low, sg, dist, "EmptyOutputProbabilityMeasure").value 
    res_hi = calculate_analytical_result(proto_hi, sg, dist, "EmptyOutputProbabilityMeasure").value 
    assert res_low is not None
    assert res_hi is not None
    assert res_low >= res_hi
"""
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0, 5.0])
def test_monotonicity_rr_empty_output_decreases_with_epsilon(epsilon):
# As epsilon increases, q decreases and probability of empty output should not increase.
    sg = stream_generator_with_horizon(10)
    dist = uniform_length_distribution(1, 10)
    proto_low = {"name": "RandomizedResponse", "epsilon": 0.0, "delta_length": 10} 
    proto_hi = {"name": "RandomizedResponse", "epsilon": epsilon, "delta_length": 10} 
    res_low = calculate_analytical_result(proto_low, sg, dist, "EmptyOutputProbabilityMeasure").value 
    res_hi = calculate_analytical_result(proto_hi, sg, dist, "EmptyOutputProbabilityMeasure").value 
    assert res_low is not None
    assert res_hi is not None
    assert res_low >= res_hi
"""
