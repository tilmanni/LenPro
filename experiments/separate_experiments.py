from os import name
from lenpro.execution import execution_from_generator, store_result
from lenpro.aggregator import Aggregator
from lenpro.protocols import lenpro, randomized_response
from lenpro.mechanisms import mechanisms
from lenpro.measures import entry_level_measures, aggregate_measures, global_measures
from lenpro.distributions import distributions
from lenpro.stream_generation import event_generators, length_generators, stream_generators
from typing import List

ORIGINAL_EVENT = 1
SYNTHETIC_EVENT = 2
EPSILONS = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
DATA_SET_SIZES = [500, 5000, 50000]
MAXIMUM_LENGTHS = [20, 60, 80]


def lenpro_epsilon_evaluation(epsilons, delta_length, stream_generator, data_set_size):
    aggregator_instantiation = Aggregator()
  
    for epsilon in epsilons:
        lenpro_instantiation = lenpro.LenPro(epsilon=epsilon, delta_length=delta_length, perturbing_mechanism=mechanisms.IdentityMechanism(), synthetic_event_generator=event_generators.ConstantEventGenerator(SYNTHETIC_EVENT))
        aggregator_instantiation.register_protocol(ldp_protocol=lenpro_instantiation)
    
    is_original = lambda event : event == ORIGINAL_EVENT
    is_synthethic = lambda event : event == SYNTHETIC_EVENT

    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.RelativeErrorMeasure())
    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.FractionRetainedEventsMeasure(is_original=is_original))
    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.FractionSyntheticEventsMeasure(is_synthetic=is_synthethic))
    aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.MeanMeasure())
    aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.ConfidenceIntervalMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.EmptyOutputProbabilityMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.LengthUnchangedProbabilityMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.GlobalFractionRetainedEventsMeasure(is_original=is_original))
    aggregator_instantiation.register_global_measure(global_measure=global_measures.GlobalFractionSyntheticEventsMeasure(is_synthetic=is_synthethic))

    result = execution_from_generator(aggregator=aggregator_instantiation, stream_generator=stream_generator, data_set_size=data_set_size)
    store_result(result=result, file_name=f"lenpro_{stream_generator.to_dict()["length_generator"]["distribution"]["name"]}_{data_set_size}_{delta_length}")


def randomized_response_epsilon_evaluation(epsilons, delta_length, stream_generator, data_set_size):
    aggregator_instantiation = Aggregator()
  
    for epsilon in epsilons:
        rr_instantiation = randomized_response.RandomizedResponse(epsilon=epsilon, delta_length=delta_length, perturbing_mechanism=mechanisms.IdentityMechanism(), synthetic_event_generator=event_generators.ConstantEventGenerator(SYNTHETIC_EVENT))
        aggregator_instantiation.register_protocol(ldp_protocol=rr_instantiation)
    
    is_original = lambda event : event == ORIGINAL_EVENT
    is_synthethic = lambda event : event == SYNTHETIC_EVENT

    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.RelativeErrorMeasure())
    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.FractionRetainedEventsMeasure(is_original=is_original))
    aggregator_instantiation.register_entry_level_measure(entry_level_measure=entry_level_measures.FractionSyntheticEventsMeasure(is_synthetic=is_synthethic))
    aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.MeanMeasure())
    aggregator_instantiation.register_aggregate_measure(aggregate_measure=aggregate_measures.ConfidenceIntervalMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.EmptyOutputProbabilityMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.LengthUnchangedProbabilityMeasure())
    aggregator_instantiation.register_global_measure(global_measure=global_measures.GlobalFractionRetainedEventsMeasure(is_original=is_original))
    aggregator_instantiation.register_global_measure(global_measure=global_measures.GlobalFractionSyntheticEventsMeasure(is_synthetic=is_synthethic))

    result = execution_from_generator(aggregator=aggregator_instantiation, stream_generator=stream_generator, data_set_size=data_set_size)
    store_result(result=result, file_name=f"randomized_response_{stream_generator.to_dict()["length_generator"]["distribution"]["name"]}_{data_set_size}_{delta_length}")



stream_generator_instantiations = []

for maximum_length in MAXIMUM_LENGTHS:
    uniform_distribution_instantiation = distributions.UniformDistribution(low = 0, high=maximum_length + 1)
    gaussian_distribution_instantiation = distributions.GaussianDistribution(mean = maximum_length / 2, scale=1)
    uniform_length_generator_instantiation = length_generators.DistributionLengthGenerator(uniform_distribution_instantiation)
    gaussian_length_generator_instantiation = length_generators.DistributionLengthGenerator(gaussian_distribution_instantiation)
    stream_generator_instantiations.append((maximum_length, stream_generators.StandardStreamGenerator(length_generator=uniform_length_generator_instantiation, event_generator=event_generators.ConstantEventGenerator(ORIGINAL_EVENT), fixed_length=maximum_length)))
    stream_generator_instantiations.append((maximum_length, stream_generators.StandardStreamGenerator(length_generator=gaussian_length_generator_instantiation, event_generator=event_generators.ConstantEventGenerator(ORIGINAL_EVENT), fixed_length=maximum_length)))



for data_set_size in DATA_SET_SIZES:
    for stream_generator_instantiation in stream_generator_instantiations:
        lenpro_epsilon_evaluation(epsilons=EPSILONS, delta_length=stream_generator_instantiation[0], stream_generator=stream_generator_instantiation[1], data_set_size=data_set_size)
        randomized_response_epsilon_evaluation(epsilons=EPSILONS, delta_length=stream_generator_instantiation[0], stream_generator=stream_generator_instantiation[1], data_set_size=data_set_size)
