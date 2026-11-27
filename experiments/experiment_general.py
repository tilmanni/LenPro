import numpy as np
import datetime


from lenpro.aggregator import Aggregator
from lenpro.execution import execution_from_data, store_result

from lenpro.estimators.lenpro_estimators import LenProBayesianCountEstimator, LenProCountEstimator
from lenpro.estimators.randomized_response_estimator import RandomizedResponseBayesianCountEstimator, RandomizedResponseCountEstimator, RandomizedResponseMLECountEstimator
from lenpro.protocols.uniform_length import UniformLength
from lenpro.protocols.lenpro import LenPro
from lenpro.protocols.randomized_response import RandomizedResponse
from lenpro.mechanisms.mechanisms import IdentityMechanism

from lenpro.measures import (
    entry_level_measures,
    aggregate_measures,
    global_measures,
)
from lenpro.distributions.distributions import Distribution
from lenpro.statistics.statistics import CountStatistic
from lenpro.stream_generation import (
    event_generators,
    length_generators,
    stream_generators,
)

from protocol_variant import ProtocolVariant, make_protocol
from start_point_variant import StartPointVariant, make_start_point_generator
from length_variant import LengthVariant, make_length_generator

ORIGINAL_EVENT = 1
SYNTHETIC_EVENT = 2

EVENT_GENERATOR = event_generators.ConstantEventGenerator(constant_event=ORIGINAL_EVENT)
PERTURBING_MECHANISM = IdentityMechanism()
SYNTHETIC_EVENT_GENERATOR = event_generators.ConstantEventGenerator(SYNTHETIC_EVENT)

# Concrete variants for one experiment
PROTOCOL_VARIANTS = [
    ProtocolVariant(
        id="lenpro",
        label="LenPro",
        kind="lenpro",
        parameters={
            "extendable": False,
            "perturbing_mechanism": PERTURBING_MECHANISM,
            "synthetic_event_generator": SYNTHETIC_EVENT_GENERATOR,
        },
    ),
    ProtocolVariant(
        id="lenpro_extendable",
        label="LenPro, extendable",
        kind="lenpro_extendable",
        parameters={
            "extendable": True,
            "perturbing_mechanism": PERTURBING_MECHANISM,
            "synthetic_event_generator": SYNTHETIC_EVENT_GENERATOR,
        },
    ),
    ProtocolVariant(
        id="randomized_response",
        label="Randomized response",
        kind="randomized_response",
        parameters={
            "perturbing_mechanism": PERTURBING_MECHANISM,
            "synthetic_event_generator": SYNTHETIC_EVENT_GENERATOR,
        },
    ),
    ProtocolVariant(
        id="uniform_length",
        label="Uniform Length",
        kind="uniform_length",
        parameters={
            "extendable": False,
            "perturbing_mechanism": PERTURBING_MECHANISM,
            "synthetic_event_generator": SYNTHETIC_EVENT_GENERATOR,
        },
    )
]


START_POINT_VARIANTS = [
    StartPointVariant(
        id="centered", 
        label="Centered", 
        kind="centered",
    ),
    #StartPointVariant(
    #    id="uniform", 
    #    label="Uniform start", 
    #    kind="uniform",
    #),
    #StartPointVariant(
    #    id="gaussian", 
    #    label="Gaussian start", 
    #    kind="gaussian"
    #),
]

LENGTH_VARIANTS = [
    LengthVariant(
        id="uniform",
        label="Uniform",
        kind="uniform",
    ),
    LengthVariant(
        id="gaussian",
        label="Gaussian",
        kind="gaussian",
    ),
]

#EPSILONS = [0.1, 1, 2, 5, 10, 20]
EPSILONS = [1, 2]
#DATA_SET_SIZES = [500]
DATA_SET_SIZES = [10, 50]
#MAXIMUM_LENGTHS = [5, 10, 20, 40]
MAXIMUM_LENGTHS = [5, 10]
FIXED_HORIZON_FACTOR = 2

REPETITIONS = 2
MASTER_SEED = 12345


def register_measures(agg):
    is_original = lambda event: event == ORIGINAL_EVENT
    is_synthetic = lambda event: event == SYNTHETIC_EVENT

    for measure in [
        entry_level_measures.RelativeErrorMeasure(statistic=CountStatistic(), sanity_bound=1),
        #entry_level_measures.RelativeErrorMeasureProtocolDependent(estimator_name="standard_estimator", statistic=CountStatistic(), sanity_bound=1),
        #entry_level_measures.RelativeErrorMeasureProtocolDependent(estimator_name="mle_estimator", statistic=CountStatistic(), sanity_bound=1),
        entry_level_measures.RelativeErrorMeasureProtocolDependent(estimator_name="bayesian_estimator", statistic=CountStatistic(), sanity_bound=1),
        entry_level_measures.PrefixMeanRelativeErrorMeasureProtocolDependent(estimator_name="standard_estimator", statistic=CountStatistic(), sanity_bound=1),
        entry_level_measures.FractionRetainedEventsMeasure(
            is_original=is_original
        ),
        entry_level_measures.FractionSyntheticEventsMeasure(
            is_synthetic=is_synthetic
        ),
    ]:
        agg.register_entry_level_measure(entry_level_measure=measure)

    for measure in [
        aggregate_measures.MeanMeasure(),
        aggregate_measures.ConfidenceIntervalMeasure(),
    ]:
        agg.register_aggregate_measure(aggregate_measure=measure)

    for measure in [
        global_measures.EmptyOutputProbabilityMeasure(),
        global_measures.LengthUnchangedProbabilityMeasure(),
        global_measures.GlobalFractionRetainedEventsMeasure(
            is_original=is_original
        ),
        global_measures.GlobalFractionSyntheticEventsMeasure(
            is_synthetic=is_synthetic
        ),
        global_measures.MeanRelativeErrorFrequencyMeasure(statistic=CountStatistic(), estimator_name="mle_estimator"),
    ]:
        agg.register_global_measure(global_measure=measure)

def attach_estimators(protocol, prior, maximum_length):
    if isinstance(protocol, LenPro):
        protocol.standard_estimator = LenProCountEstimator()
        protocol.mle_estimator = LenProCountEstimator()
        protocol.bayesian_estimator = LenProBayesianCountEstimator(
            p=protocol.p,
            prior=prior,
            max_length=maximum_length,
        )

    elif isinstance(protocol, RandomizedResponse):
        protocol.standard_estimator = RandomizedResponseCountEstimator(
            randomized_response=protocol,
        )
        protocol.mle_estimator = RandomizedResponseMLECountEstimator(
            randomized_response=protocol,
        )
        protocol.bayesian_estimator = RandomizedResponseBayesianCountEstimator(
            randomized_response=protocol,
            prior=prior,
            max_length=maximum_length,
        )
    #TODO implement
    elif isinstance(protocol, UniformLength):
        protocol.standard_estimator = LenProCountEstimator()
        protocol.mle_estimator = LenProCountEstimator()
        protocol.bayesian_estimator = LenProCountEstimator()

    else:
        raise TypeError(
            f"No estimator configuration for {type(protocol).__name__}"
        )

def make_aggregator(maximum_length: int, protocol_seeds, prior: Distribution):
    agg = Aggregator()
    register_measures(agg)

    for variant in PROTOCOL_VARIANTS:
        for epsilon in EPSILONS:
            protocol = make_protocol(
                variant=variant, 
                epsilon=epsilon, 
                maximum_length=maximum_length, 
                rng=np.random.default_rng(
                    protocol_seeds[(variant.id, epsilon)]
                ),
            )

            attach_estimators(
                protocol=protocol,
                prior=prior,
                maximum_length=maximum_length,
            )
            agg.register_protocol(
                ldp_protocol=protocol, 
                protocol_meta_data=(variant.id, variant.label),
            )
    return agg


def make_stream_generator(length_variant:LengthVariant, start_point_variant:StartPointVariant, maximum_length: int, length_rng: np.random.Generator, start_point_rng: np.random.Generator):
    fixed_horizon = maximum_length * FIXED_HORIZON_FACTOR

    length_generator = make_length_generator(
        variant=length_variant, 
        rng=length_rng, 
        maximum_length=maximum_length,
    )

    start_point_generator = make_start_point_generator(
        variant=start_point_variant, 
        rng=start_point_rng, 
        maximum_length=maximum_length, 
        fixed_horizon=fixed_horizon,
    )

    return stream_generators.StandardStreamGenerator(
        length_generator=length_generator,
        start_point_generator=start_point_generator,
        event_generator=EVENT_GENERATOR,
        fixed_horizon=fixed_horizon
    )


def seed_as_int(seed_sequence):
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def run_experiments():
    master = np.random.SeedSequence(MASTER_SEED)

    directory_name = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    for length_variant in LENGTH_VARIANTS:
        for start_point_variant in START_POINT_VARIANTS:
            for maximum_length in MAXIMUM_LENGTHS:
                for data_set_size in DATA_SET_SIZES:

                    experiment_seed = master.spawn(1)[0]

                    # One independent length seed, 
                    # One independent start point seed,
                    # plus one seed for every
                    # protocol/epsilon combination times repetitions.
                    children = experiment_seed.spawn(
                        2 + len(PROTOCOL_VARIANTS) * len(EPSILONS) * REPETITIONS
                    )

                    length_seed = seed_as_int(children[0])
                    start_point_seed = seed_as_int(children[1])
                    seed_index = 2

                    generator = make_stream_generator(
                        length_variant=length_variant,
                        start_point_variant=start_point_variant,
                        maximum_length=maximum_length,
                        length_rng=np.random.default_rng(length_seed),
                        start_point_rng=np.random.default_rng(start_point_seed),
                    )

                    # Inputs shared across repetitions
                    inputs = list(
                        generator.generate_streams(data_set_size)
                    )

                    #TODO dirty fix, change (maybe take uniform prior if not DistributionLengthGenerator)
                    length_generator = generator.length_generator
                    assert isinstance(length_generator, length_generators.DistributionLengthGenerator)
                    input_length_distribution = length_generator.distribution

                    for repetition in range(REPETITIONS):

                        #Generate input seed for protocols
                        protocol_seeds = {}
                        seed_records = []

                        for protocol_variant in PROTOCOL_VARIANTS:
                            for epsilon in EPSILONS:
                                seed = seed_as_int(children[seed_index])
                                seed_index += 1

                                protocol_seeds[
                                    (protocol_variant.id, epsilon)
                                ] = seed

                                seed_records.append({
                                    "protocol": protocol_variant.id,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                })

                        agg = make_aggregator(
                            maximum_length=maximum_length,
                            protocol_seeds=protocol_seeds,
                            prior=input_length_distribution,
                        )

                        scenario_id = (
                            f"general_length-{length_variant.id}"
                            f"_start-{start_point_variant.id}"
                            f"_n{data_set_size}"
                            f"_max{maximum_length}"
                        )

                        observation_id = f"{scenario_id}_rep{repetition:02d}"

                        result = execution_from_data(
                            aggregator=agg,
                            input_data=inputs,
                            stream_generator=generator,
                            experiment_metadata={
                                "scenario_id": scenario_id,
                                "observation_id": observation_id,
                                "length_variant_id": length_variant.id,
                                "length_variant_label": length_variant.label,
                                "start_point_variant_id": start_point_variant.id,
                                "start_point_variant_label": start_point_variant.label,
                                "repetition": repetition,
                                "master_seed": MASTER_SEED,
                                "length_seed": length_seed,
                                "start_point_seed": start_point_seed,
                                "protocol_seeds": seed_records,
                                "shared_inputs_across_protocols": True,
                                "shared_inputs_across_epsilons": True,
                                "shared_inputs_across_repetitions": True,
                            },
                        )

                        path = store_result(result=result, file_name=observation_id, directory="./results/" + directory_name)
                        print(f"Saved {path}")


if __name__ == "__main__":
    run_experiments()