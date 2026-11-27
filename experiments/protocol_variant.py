from dataclasses import dataclass, field
from typing import Any
import numpy as np

from lenpro.protocols.ldp_protocol import LDPProtocol
from lenpro.protocols.lenpro import LenPro
from lenpro.protocols.randomized_response import RandomizedResponse
from lenpro.protocols.uniform_length import UniformLength

@dataclass(frozen=True)
class ProtocolVariant:
    id: str
    label: str
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)


def make_protocol(variant: ProtocolVariant, epsilon: float, maximum_length: int, rng: np.random.Generator) -> LDPProtocol:
    
    if epsilon < 0:
        raise ValueError("epsilon must be positive")

    if maximum_length < 1:
        raise ValueError("maximum_length must be >= 1")

    reserved = {"epsilon", "delta_length", "rng"}
    conflicting = reserved.intersection(variant.parameters)

    if conflicting:
        raise ValueError(
            f"{variant.id}: parameters controlled by the runner: "
            f"{sorted(conflicting)}"
        )
    
    if variant.kind == "lenpro":
        protocol_class = LenPro
        defaults = {
            "extendable": False,
        }
    elif variant.kind == "lenpro_extendable":
        protocol_class = LenPro
        defaults = {
            "extendable": True,
        }
    elif variant.kind == "randomized_response":
        protocol_class = RandomizedResponse
        defaults = {}
    elif variant.kind == "uniform_length":
        protocol_class = UniformLength
        defaults = {
            "extendable": False,
            "uniform_length": maximum_length // 2,
        }
    else: 
        raise ValueError(f"Unknown protocol variant: {variant.kind}")

    parameters = {**defaults, **variant.parameters}
    return protocol_class(
        **parameters,
        epsilon=epsilon,
        delta_length=maximum_length,
        rng=rng,
    )