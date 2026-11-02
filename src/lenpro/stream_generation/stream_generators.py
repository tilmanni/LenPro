from abc import ABC, abstractmethod
from typing import List
import numpy as np

from lenpro.stream_generation import start_point_generators
from lenpro.stream_generation.event_generators import EventGenerator
from lenpro.stream_generation.length_generators import LengthGenerator
from lenpro.stream_generation.start_point_generators import StartPointGenerator

class StreamGenerator(ABC):

    @abstractmethod
    def generate_stream(self) -> List:
        pass 

    @abstractmethod
    def generate_streams(self, amount: int) -> List[List]:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

# This stream generator applies Nones to fixed length streams in equal before and after the 
# actual generated stream
class StandardStreamGenerator(StreamGenerator):

    def __init__(self, length_generator: LengthGenerator, event_generator: EventGenerator, start_point_generator: StartPointGenerator | None = None, fixed_horizon = None) -> None:
        if fixed_horizon is not None and fixed_horizon < 0:
            raise ValueError("fixed_horizon must be non-negative")

        if (fixed_horizon is None) != (start_point_generator is None):
            raise ValueError(
                "fixed_horizon and start_point_generator must either "
                "both be provided or both be None"
            )
        self.length_generator = length_generator
        self.event_generator = event_generator
        self.start_point_generator = start_point_generator
        self.fixed_horizon = fixed_horizon

    def generate_stream(self):
        length = self.length_generator.generate_length()
        stream = []
        if self.fixed_horizon is None:
            for _ in range(length):
                stream.append(self.event_generator.generate_event())
        assert self.start_point_generator is not None and self.fixed_horizon is not None 
        length = min(length, self.fixed_horizon)
        start_point = self.start_point_generator.generate_start_point(fixed_horizon=self.fixed_horizon, length=length)
        for _ in range(start_point):
            stream.append(None)
        for _ in range(min(self.fixed_horizon, length)):
            stream.append(self.event_generator.generate_event())
        for _ in range(max(0, self.fixed_horizon - (start_point + length))):
            stream.append(None)
            
        return stream 

    def generate_streams(self, amount: int):
        if amount < 0:
            raise ValueError("amount must be non-negative")
        streams = []
        for _ in range(amount):
            streams.append(self.generate_stream())
        return streams

    def to_dict(self) -> dict:
        return {
            "name": "StandardStreamGenerator",
            "length_generator": self.length_generator.to_dict(),
            "event_generator": self.event_generator.to_dict(),
            "fixed_horizon": self.fixed_horizon,
            "start_point_generator": self.start_point_generator.to_dict() if self.start_point_generator is not None else None,
        }

        