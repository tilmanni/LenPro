from abc import ABC, abstractmethod
from typing import List
import numpy as np

from lenpro.stream_generation.event_generators import EventGenerator
from lenpro.stream_generation.length_generators import LengthGenerator
from lenpro.stream_generation.start_point_generators import StartPointGenerator


DEFAULT_START_POINT = 0

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

        self.length_generator = length_generator
        self.event_generator = event_generator
        self.start_point_generator = start_point_generator
        self.fixed_horizon = fixed_horizon

    def generate_stream(self):
        length = self.length_generator.generate_length()
        stream = []
        # default start value
        start_point = DEFAULT_START_POINT

        # Fixed horizon not specified
        if self.fixed_horizon is None:
            if self.start_point_generator is not None:
                start_point = max(DEFAULT_START_POINT, self.start_point_generator.generate_start_point(length=length))
            # Pad with None's until start point
            stream.extend([None] * start_point)
            stream.extend([self.event_generator.generate_event() for _ in range(length)])
        # Fixed horizon specified
        else:
            if self.start_point_generator is not None:
                start_point = max(DEFAULT_START_POINT, self.start_point_generator.generate_start_point(fixed_horizon=self.fixed_horizon, length=length))
            # Start point may not be smaller than 0
            padded_start_point = min(start_point, self.fixed_horizon)
            # Ensures length + start point does not exceed horizon
            effective_length = min(length, self.fixed_horizon - padded_start_point)
            
            stream.extend([None] * padded_start_point)
            
            stream.extend([self.event_generator.generate_event() for _ in range(effective_length)])
            
            remaining_padding = self.fixed_horizon - (padded_start_point + effective_length)
            if remaining_padding > 0:
                stream.extend([None] * remaining_padding)
            
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

        