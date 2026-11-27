from dataclasses import dataclass, field
from typing import Any
import numpy as np

from lenpro.distributions.distributions import UniformDistribution, TruncatedGaussianDistribution
from lenpro.stream_generation.start_point_generators import CenteredStreamStartPointGenerator, DistributionStartPointGenerator, ConstantStreamStartPointGenerator

@dataclass(frozen=True)
class StartPointVariant:
    id: str
    label: str
    kind: str 
    parameters: dict[str, Any] = field(default_factory=dict)


def make_start_point_generator(variant: StartPointVariant, maximum_length: int, fixed_horizon: int, rng: np.random.Generator):

    if maximum_length < 1:
        raise ValueError("maximum_length must be >= 1")

    if "rng" in variant.parameters:
        raise ValueError("rng is controlled by the experiment runner")

    # Assumes starts must be nonnegative and the longest stream
    # must fit entirely within fixed_horizon.
    #maximum_start = fixed_horizon - maximum_length
    
    # Assumes starts can be anywhere in fixed_horizon
    maximum_start = fixed_horizon - 1

    if maximum_start < 0:
        raise ValueError("fixed_horizon must be >= maximum_length")

    if variant.kind == "centered":
        return CenteredStreamStartPointGenerator()
    elif variant.kind == "uniform":
        distribution_class = UniformDistribution
        defaults = {
            "low": 0,
            "high": maximum_start,
        }
    elif variant.kind == "gaussian":
        distribution_class = TruncatedGaussianDistribution
        defaults = {
            "low": 0,
            "high": maximum_start,
            "mean": maximum_start / 2,
            "scale": 1,
        }
    else:
        raise ValueError(f"Unknown start-point kind: {variant.kind}")
    
    parameters = {**defaults, **variant.parameters}
    
    distribution = distribution_class(
        **parameters,
        rng=rng,
    )
    
    return DistributionStartPointGenerator(distribution=distribution)