
from typing import List
import json
import numpy as np
import datetime
from pathlib import Path


import aggregator
from stream_generation import stream_generators
from protocols import lenpro, randomized_response
from mechanisms import mechanisms
from stream_generation import stream_generators, event_generators, length_generators
from measures import entry_level_measures, aggregate_measures, global_measures

def execution_from_generator(aggregator, stream_generator, data_set_size):
    return {
        "stream_generator": stream_generator.to_dict(),
        "data_set_size": data_set_size,
        "results": aggregator.measure_run(stream_generator.generate_streams(data_set_size)),
    }


original_event = 1
synthetic_event = 2

aggregator_instantiation = aggregator.Aggregator()

lenpro_instantiation_1 = lenpro.LenPro(epsilon=1, delta_length=1, perturbing_mechanism=mechanisms.IdentityMechanism(), synthetic_event_generator=event_generators.ConstantEventGenerator(synthetic_event))
lenpro_instantiation_2 = lenpro.LenPro(epsilon=0.1, delta_length=1, perturbing_mechanism=mechanisms.IdentityMechanism(), synthetic_event_generator=event_generators.ConstantEventGenerator(synthetic_event))
randomized_response_instantiation = randomized_response.RandomizedResponse(epsilon=1, perturbing_mechanism=mechanisms.IdentityMechanism(), synthetic_event_generator=event_generators.ConstantEventGenerator(synthetic_event))

aggregator_instantiation.register_protocol(lenpro_instantiation_1)
aggregator_instantiation.register_protocol(lenpro_instantiation_2)
aggregator_instantiation.register_protocol(randomized_response_instantiation)

aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.RelativeErrorMeasure())
aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.MeanMeasure())
aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.ConfidenceIntervalMeasure())
aggregator_instantiation.register_global_measure(global_measure=global_measures.EmptyOutputProbabilityMeasure())
aggregator_instantiation.register_global_measure(global_measure=global_measures.LengthUnchangedProbabilityMeasure())


stream_generator_instantiation = stream_generators.StandardStreamGenerator(length_generator=length_generators.UniformLengthGenerator(10, 20), event_generator=event_generators.ConstantEventGenerator(original_event), fixed_length=40)

data_set_size = 10


result = execution_from_generator(aggregator=aggregator_instantiation, stream_generator=stream_generator_instantiation, data_set_size=data_set_size)

print(result)

def to_builtin(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


Path("results").mkdir(exist_ok=True)

with open(f"results/run_{datetime.datetime.now().isoformat()}.json", "w") as f:
    json.dump(result, f, default=to_builtin, indent=2)


# This just passes input on to aggregator
def execution_from_data(aggregator: aggregator.Aggregator, input_data: List[List]):
    #result = aggregator.measure_run(input_data)
    pass
