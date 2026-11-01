# run_experiments.py

import numpy as np
import datetime


from lenpro import aggregator
from lenpro import execution

from lenpro.protocols import lenpro, randomized_response
from lenpro.mechanisms import mechanisms
from lenpro.measures import (
    entry_level_measures,
    aggregate_measures,
    global_measures,
)
from lenpro.distributions import distributions
from lenpro.stream_generation import (
    event_generators,
    length_generators,
    stream_generators,
)


ORIGINAL_EVENT = 1
SYNTHETIC_EVENT = 2

EPSILONS = [0.1, 1, 2, 5, 10, 20]
DATA_SET_SIZES = [5000]
MAXIMUM_LENGTHS = [5, 10, 20, 40]


# Additional experimental settings; your original grids stay unchanged.
REPETITIONS = 10
MASTER_SEED = 12345


def register_measures(agg):
    is_original = lambda event: event == ORIGINAL_EVENT
    is_synthetic = lambda event: event == SYNTHETIC_EVENT
    naive_length_measure = lambda data : sum(x is not None for x in data)



    for measure in [
        entry_level_measures.RelativeErrorMeasure(metric=naive_length_measure, sanity_bound=1),
        entry_level_measures.RelativeErrorMeasureProtocolDependent(estimator_name="length_estimator", metric=naive_length_measure, sanity_bound=1),
        entry_level_measures.PrefixMeanRelativeErrorMeasureProtocolDependent(estimator_name="prefix_length_estimator", metric=naive_length_measure, sanity_bound=1),
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
    ]:
        agg.register_global_measure(global_measure=measure)




def make_stream_generator(distribution_name, maximum_length, rng):
    if distribution_name == "UniformDistribution":
        distribution = distributions.UniformDistribution(
            low=0,
            high=maximum_length + 1,
            rng=rng,
        )
    elif distribution_name == "GaussianDistribution":
        distribution = distributions.GaussianDistribution(
            mean=maximum_length / 2,
            scale=1,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    return stream_generators.StandardStreamGenerator(
        length_generator=length_generators.DistributionLengthGenerator(
            distribution
        ),
        event_generator=event_generators.ConstantEventGenerator(
            ORIGINAL_EVENT
        ),
        fixed_length=maximum_length,
    )


def make_aggregator(delta_length, protocol_seeds):
    agg = aggregator.Aggregator()
    register_measures(agg)

    protocol_classes = [
        lenpro.LenPro,
        randomized_response.RandomizedResponse,
    ]

    for protocol_class in protocol_classes:
        for epsilon in EPSILONS:
            seed = protocol_seeds[(protocol_class.__name__, epsilon)]

            protocol = protocol_class(
                epsilon=epsilon,
                delta_length=delta_length,
                perturbing_mechanism=mechanisms.IdentityMechanism(),
                synthetic_event_generator=(
                    event_generators.ConstantEventGenerator(SYNTHETIC_EVENT)
                ),
                rng=np.random.default_rng(seed),
            )

            agg.register_protocol(ldp_protocol=protocol)

    return agg


def seed_as_int(seed_sequence):
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def run_experiments():
    master = np.random.SeedSequence(MASTER_SEED)

    directory_name = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    for distribution_name in [
        "UniformDistribution",
        "GaussianDistribution",
    ]:
        for maximum_length in MAXIMUM_LENGTHS:
            for data_set_size in DATA_SET_SIZES:
                for repetition in range(REPETITIONS):
                    experiment_seed = master.spawn(1)[0]

                    # One independent input seed, plus one seed for every
                    # protocol/epsilon combination.
                    children = experiment_seed.spawn(
                        1 + 2 * len(EPSILONS)
                    )

                    input_seed = seed_as_int(children[0])

                    protocol_seeds = {}
                    seed_records = []
                    seed_index = 1

                    for protocol_class in [
                        lenpro.LenPro,
                        randomized_response.RandomizedResponse,
                    ]:
                        for epsilon in EPSILONS:
                            seed = seed_as_int(children[seed_index])
                            seed_index += 1

                            protocol_seeds[
                                (protocol_class.__name__, epsilon)
                            ] = seed

                            seed_records.append({
                                "protocol": protocol_class.__name__,
                                "epsilon": epsilon,
                                "seed": seed,
                            })

                    generator = make_stream_generator(
                        distribution_name,
                        maximum_length,
                        rng=np.random.default_rng(input_seed),
                    )

                    # Generate once: both protocols and all epsilons
                    # receive exactly the same original streams.
                    inputs = list(
                        generator.generate_streams(data_set_size)
                    )

                    agg = make_aggregator(
                        delta_length=maximum_length,
                        protocol_seeds=protocol_seeds,
                    )

                    result = execution.execution_from_data(
                        aggregator=agg,
                        input_data=inputs,
                        stream_generator=generator,
                        experiment_metadata={
                            "master_seed": MASTER_SEED,
                            "repetition": repetition,
                            "input_seed": input_seed,
                            "protocol_seeds": seed_records,
                            "shared_inputs_across_protocols": True,
                            "shared_inputs_across_epsilons": True,
                        },
                    )

                    file_name = (
                        f"comparison_{distribution_name}"
                        f"_n{data_set_size}"
                        f"_max{maximum_length}"
                        f"_rep{repetition:02d}"
                    )

                    path = execution.store_result(result=result, file_name=file_name, directory="./results/" + directory_name)
                    print(f"Saved {path}")


if __name__ == "__main__":
    run_experiments()