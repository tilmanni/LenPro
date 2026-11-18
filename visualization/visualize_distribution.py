from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import re

from lenpro.distributions.distributions import (
    GaussianDistribution,
    UniformDistribution,
    ZipfDistribution,
)
from lenpro.stream_generation.length_generators import (
    DistributionLengthGenerator,
)
from lenpro.stream_generation.start_point_generators import (
    DistributionStartPointGenerator,
)



def distribution_from_dict(
    configuration: dict,
    rng: np.random.Generator,
):
    """Reconstruct a distribution from its serialized configuration."""
    name = configuration["name"]

    if name == "UniformDistribution":
        return UniformDistribution(
            low=configuration["low"],
            high=configuration["high"],
            rng=rng,
        )

    if name == "GaussianDistribution":
        return GaussianDistribution(
            mean=configuration["mean"],
            scale=configuration["scale"],
            rng=rng,
        )

    if name == "ZipfDistribution":
        return ZipfDistribution(
            a=configuration["a"],
            rng=rng,
        )

    raise ValueError(f"Unsupported distribution: {name!r}")

def plot_discrete_samples(
    ax: plt.Axes,
    samples,
    *,
    title: str,
    x_label: str,
    max_value: int | None = None,
) -> None:
    samples = np.asarray(samples, dtype=int)
    total_samples = len(samples)

    cut_off = 0

    if max_value is not None:
        cut_off = int(np.sum(samples > max_value))
        visible_samples = samples[samples <= max_value]
    else:
        visible_samples = samples

    if visible_samples.size == 0:
        ax.text(
            0.5,
            0.5,
            "No samples in displayed range",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        values, counts = np.unique(
            visible_samples,
            return_counts=True,
        )

        ax.bar(
            values,
            counts / total_samples,
            width=0.9,
            edgecolor="black",
        )

    if cut_off:
        title += (
            f"\n{cut_off} samples greater than "
            f"{max_value} not shown"
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Relative frequency")
    ax.grid(axis="y", alpha=0.3)

def sample_stream_generator_configuration(
    generator_configuration: dict,
    amount_of_samples: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Resample lengths and start points from a serialized stream-generator
    configuration.

    These are new samples and are not necessarily the samples used in the
    original experiment.
    """
    seed_sequence = np.random.SeedSequence(seed)
    length_seed, start_seed = seed_sequence.spawn(2)

    length_configuration = generator_configuration.get(
        "length_generator"
    )

    if (
        length_configuration is None
        or length_configuration.get("name")
        != "DistributionLengthGenerator"
    ):
        return None, None

    length_distribution_configuration = (
        length_configuration["distribution"]
    )

    length_distribution = distribution_from_dict(
        length_distribution_configuration,
        np.random.default_rng(length_seed),
    )

    length_generator = DistributionLengthGenerator(
        length_distribution
    )

    sampled_lengths = np.array(
        [
            length_generator.generate_length()
            for _ in range(amount_of_samples)
        ],
        dtype=int,
    )

    start_configuration = generator_configuration.get(
        "start_point_generator"
    )
    fixed_horizon = generator_configuration.get(
        "fixed_horizon"
    )

    if (
        start_configuration is None
        or start_configuration.get("name")
        != "DistributionStartPointGenerator"
        or fixed_horizon is None
    ):
        return sampled_lengths, None

    start_distribution = distribution_from_dict(
        start_configuration["distribution"],
        np.random.default_rng(start_seed),
    )

    start_generator = DistributionStartPointGenerator(
        start_distribution
    )

    # This mirrors StandardStreamGenerator: lengths exceeding the horizon
    # are restricted to the horizon before generating a start point.
    effective_lengths = np.minimum(
        sampled_lengths,
        fixed_horizon,
    )

    sampled_start_points = np.array(
        [
            start_generator.generate_start_point(
                fixed_horizon=fixed_horizon,
                length=int(length),
            )
            for length in effective_lengths
        ],
        dtype=int,
    )

    return sampled_lengths, sampled_start_points

def safe_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_sampling_distributions(
    results_directory: Path,
    output_directory: Path,
    *,
    amount_of_samples: int = 10_000,
    seed: int = 42,
) -> None:
    """
    Plot each unique distribution-based length/start-point configuration
    found in the result files.
    """
    configurations: dict[str, dict] = {}

    for file_path in sorted(results_directory.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            run = json.load(file)

        generator_configuration = run.get(
            "stream_generator"
        )

        if generator_configuration is None:
            continue

        key = json.dumps(
            generator_configuration,
            sort_keys=True,
        )

        configurations.setdefault(
            key,
            generator_configuration,
        )

    sampling_directory = (
        output_directory / "sampling_distributions"
    )
    sampling_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, configuration in enumerate(
        configurations.values(),
        start=1,
    ):
        sampled_lengths, sampled_start_points = (
            sample_stream_generator_configuration(
                configuration,
                amount_of_samples=amount_of_samples,
                seed=seed + index,
            )
        )

        if sampled_lengths is None:
            continue

        number_of_plots = (
            2 if sampled_start_points is not None else 1
        )

        fig, axes = plt.subplots(
            1,
            number_of_plots,
            figsize=(7 * number_of_plots, 4.5),
            squeeze=False,
            layout="constrained",
        )
        axes = axes[0]

        fixed_horizon = configuration.get(
            "fixed_horizon"
        )

        length_configuration = configuration[
            "length_generator"
        ]["distribution"]

        length_description = json.dumps(
            length_configuration,
            sort_keys=True,
        )

        plot_discrete_samples(
            axes[0],
            sampled_lengths,
            title=(
                "Sampled stream lengths\n"
                f"{length_description}"
            ),
            x_label="Sampled length",
            max_value=fixed_horizon,
        )

        if fixed_horizon is not None:
            axes[0].axvline(
                fixed_horizon,
                color="tab:red",
                linestyle="--",
                label="Fixed horizon",
            )
            axes[0].legend()

        if sampled_start_points is not None:
            start_configuration = configuration[
                "start_point_generator"
            ]["distribution"]

            start_description = json.dumps(
                start_configuration,
                sort_keys=True,
            )

            plot_discrete_samples(
                axes[1],
                sampled_start_points,
                title=(
                    "Effective sampled start points\n"
                    f"{start_description}"
                ),
                x_label="Start point",
                max_value=fixed_horizon,
            )

        fig.suptitle(
            f"Stream-generator sampling distributions\n"
            f"{amount_of_samples:,} newly generated samples",
            fontsize=13,
        )

        distribution_name = length_configuration[
            "name"
        ]

        output_path = sampling_directory / safe_file_name(
            f"generator_{index:02d}_{distribution_name}.png"
        )

        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved {output_path}")