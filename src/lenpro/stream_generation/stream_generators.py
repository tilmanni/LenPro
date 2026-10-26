from abc import ABC, abstractmethod
from typing import List
import numpy as np

from lenpro.stream_generation import event_generators, length_generators


class StreamGenerator(ABC):

    @abstractmethod
    def generate_stream(self) -> List:
        pass 

    @abstractmethod
    def generate_streams(self, amount) -> List[List]:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

# This stream generator applies Nones to fixed length streams in equal before and after the 
# actual generated stream
class StandardStreamGenerator(StreamGenerator):

    def __init__(self, length_generator: length_generators.LengthGenerator, event_generator: event_generators.EventGenerator, fixed_length = None) -> None:
        self.length_generator = length_generator
        self.event_generator = event_generator
        self.fixed_length = fixed_length

    def generate_stream(self):
        length = self.length_generator.generate_length()
        stream = []
        if self.fixed_length is not None:
            if self.fixed_length <= length:
                for _ in range(self.fixed_length):
                    stream.append(self.event_generator.generate_event())
            else:
                diff = self.fixed_length - length # Pad equally with None's before and after
                for _ in range(int(np.floor(diff / 2))):
                    stream.append(None)
                for _ in range(length):
                    stream.append(self.event_generator.generate_event())
                for _ in range(int(np.ceil(diff / 2))):
                    stream.append(None)
        else:
            for _ in range(length):
                stream.append(self.event_generator.generate_event())
        return stream 

    def generate_streams(self, amount):
        streams = []
        for _ in range(amount):
            streams.append(self.generate_stream())
        return streams

    def to_dict(self) -> dict:
        return {
            "name": "StandardStreamGenerator",
            "length_generator": self.length_generator.to_dict(),
            "event_generator": self.event_generator.to_dict(),
            "fixed_length": self.fixed_length,
        }
        

        