from dataclasses import dataclass, field
from typing import Any
import numpy as np

from lenpro.stream_generation.length_generators import DistributionLengthGenerator, LengthGenerator
from lenpro.distributions.distributions import UniformDistribution, TruncatedGaussianDistribution

@dataclass(frozen=True)
class LengthVariant:
    id: str
    label: str
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)


def make_length_generator(variant: LengthVariant, rng: np.random.Generator, maximum_length: int) -> LengthGenerator:
    
    if maximum_length < 1:
        raise ValueError("maximum_length must be >= 1")

    if "rng" in variant.parameters:
        raise ValueError("rng is controlled by the experiment runner")


    if variant.kind == "uniform":
        distribution_class = UniformDistribution
        defaults = {
            "low": 1,
            # Inclusive upper bound
            "high": maximum_length,
        }
    elif variant.kind == "gaussian":
        distribution_class = TruncatedGaussianDistribution
        defaults = {
            # Assumes mean in middle by default
            "mean": maximum_length / 2,
            "scale": 1,
            "low": 1,
            "high": maximum_length,
        }
    else: 
        raise ValueError(f"Unknown length kind: {variant.kind}")

    # Allows passing default parameters as variant parameters as well.
    parameters = {**defaults, **variant.parameters}
    distribution = distribution_class(
        **parameters,
        rng=rng,
    )

    return DistributionLengthGenerator(distribution=distribution)