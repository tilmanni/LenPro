from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

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


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# insert run directory for run here. 
RUN_DIRECTORY = Path("20261102T121149056624Z")
RESULTS_DIRECTORY = Path("results") / RUN_DIRECTORY
OUTPUT_DIRECTORY = Path("plots/comparisons") / RUN_DIRECTORY
CONFIDENCE_LEVEL = 0.95
DISTRIBUTION_SAMPLES = 10_000
DISTRIBUTION_SEED = 42

# Set to {"RelativeErrorMeasure"} to plot only relative error.
# None means plot every aggregate mean and global measure.
SELECTED_MEASURES = None

MEASURE_LABELS = {
    "RelativeErrorMeasure": "Mean relative error length",
    "RelativeErrorMeasureProtocolDependent": "Mean relative error length estimator",
    "PrefixMeanRelativeErrorMeasureProtocolDependent": "Mean relative error of prefix length estimatior",
    "FractionRetainedEventsMeasure": "Mean fraction of retained events",
    "FractionSyntheticEventsMeasure": "Mean fraction of synthetic events",
    "EmptyOutputProbabilityMeasure": "Empty output probability",
    "LengthUnchangedProbabilityMeasure": "Probability of unchanged length",
    "GlobalFractionRetainedEventsMeasure": "Global fraction of retained events",
    "GlobalFractionSyntheticEventsMeasure": "Global fraction of synthetic events",
}

PROTOCOL_COLORS = {
    "LenPro": "tab:blue",
    "RandomizedResponse": "tab:orange",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_measurements(directory):
    """Load one observation per measure, protocol, epsilon, and repetition."""
    rows = []

    for file_path in sorted(Path(directory).glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            run = json.load(file)

        experiment = run.get("experiment", {})
        if "repetition" not in experiment:
            raise ValueError(
                f"{file_path.name} has no repetition metadata. "
                "Use a directory containing only the repeated experiments."
            )

        generator = run["stream_generator"]
        distribution = generator["length_generator"]["distribution"]

        for result in run["results"]:
            protocol = result["protocol"]

            metadata = {
                "source_file": file_path.name,
                "repetition": experiment["repetition"],
                "protocol": protocol["name"],
                "epsilon": protocol["epsilon"],
                "delta_length": protocol["delta_length"],
                "data_set_size": run["data_set_size"],
                "fixed_horizon": generator.get("fixed_horizon"),
                "stream_generator_config": json.dumps(
                    generator,
                    sort_keys=True,
                ),
                "distribution": distribution["name"],
                "distribution_config": json.dumps(
                    distribution, 
                    sort_keys=True
                ),
            }

            # Stored within-run confidence intervals are intentionally omitted.
            # New intervals will be calculated across independent repetitions.
            for measurement in result["aggregate"]:
                if measurement["measure"]["name"] != "MeanMeasure":
                    continue

                config = measurement["entry_level_measure"]
                rows.append({
                    **metadata,
                    "level": "aggregate",
                    "measure": config["name"],
                    "measure_config": json.dumps(config, sort_keys=True),
                    "value": measurement["value"],
                })

            for measurement in result["global"]:
                config = measurement["measure"]
                rows.append({
                    **metadata,
                    "level": "global",
                    "measure": config["name"],
                    "measure_config": json.dumps(config, sort_keys=True),
                    "value": measurement["value"],
                })

    if not rows:
        raise FileNotFoundError(f"No measurements found in {directory}")

    df = pd.DataFrame(rows)

    if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise ValueError("Some measurement values are missing or non-finite.")

    observation_key = [
        "distribution_config",
        "data_set_size",
        "fixed_horizon",
        "delta_length",
        "protocol",
        "epsilon",
        "repetition",
        "level",
        "measure_config",
    ]

    if df.duplicated(observation_key).any():
        raise ValueError(
            "Duplicate configuration/repetition observations found. "
            "Select one experiment batch or remove duplicate files."
        )

    return df


# ---------------------------------------------------------------------------
# Summarizing
# ---------------------------------------------------------------------------

def summarize_repetitions(df, confidence):
    """Mean and pointwise Student's t interval across repetitions."""
    group_columns = [
        "distribution",
        "distribution_config",
        "data_set_size",
        "fixed_horizon",
        "delta_length",
        "protocol",
        "epsilon",
        "level",
        "measure",
        "measure_config",
    ]

    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            repetitions=("value", "size"),
        )
    )

    summary["ci_low"] = np.nan
    summary["ci_high"] = np.nan

    valid = summary["repetitions"] >= 2
    n = summary.loc[valid, "repetitions"]

    standard_error = summary.loc[valid, "std"] / np.sqrt(n)
    critical_value = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = critical_value * standard_error

    summary.loc[valid, "ci_low"] = summary.loc[valid, "mean"] - margin
    summary.loc[valid, "ci_high"] = summary.loc[valid, "mean"] + margin

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(summary, measure, level, output_directory, confidence):
    selected = summary[
        (summary["measure"] == measure)
        & (summary["level"] == level)
    ].copy()

    if selected.empty:
        raise ValueError(f"No results for {level}/{measure}")

    if selected["measure_config"].nunique() != 1:
        raise ValueError(
            f"Different parameter settings exist for {measure}. "
            "Filter to one measure configuration first."
        )

    sizes = sorted(selected["data_set_size"].unique())
    horizons = sorted(selected["fixed_horizon"].unique())
    epsilons = sorted(selected["epsilon"].unique())
    label = MEASURE_LABELS.get(measure, measure)

    # Shared limits make the separate distribution figures comparable.
    bounds = np.concatenate([
        selected["mean"].to_numpy(),
        selected["ci_low"].dropna().to_numpy(),
        selected["ci_high"].dropna().to_numpy(),
    ])

    lower, upper = bounds.min(), bounds.max()
    padding = 0.05 * (upper - lower) if upper > lower else 0.05
    y_limits = (lower - padding, upper + padding)

    for distribution_name, distribution_df in selected.groupby("distribution"):
        fig, axes = plt.subplots(
            len(sizes),
            len(horizons),
            figsize=(5 * len(horizons), 3.6 * len(sizes)),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )

        legend_entries = {}

        for row, size in enumerate(sizes):
            for column, horizon in enumerate(horizons):
                ax = axes[row, column]
                panel = distribution_df[
                    (distribution_df["data_set_size"] == size)
                    & (distribution_df["fixed_horizon"] == horizon)
                ]

                if (
                    panel["distribution_config"].nunique() > 1
                    or panel["delta_length"].nunique() > 1
                ):
                    raise ValueError(
                        "A panel contains different distribution parameters "
                        "or delta_length values. Filter them first."
                    )

                for protocol, series in panel.groupby("protocol"):
                    series = series.sort_values("epsilon")

                    line, = ax.plot(
                        series["epsilon"].to_numpy(),
                        series["mean"].to_numpy(),
                        marker="o",
                        color=PROTOCOL_COLORS.get(protocol),
                        label=protocol,
                    )
                    legend_entries[protocol] = line

                    intervals = series.dropna(subset=["ci_low", "ci_high"])
                    if not intervals.empty:
                        ax.fill_between(
                            intervals["epsilon"].to_numpy(),
                            intervals["ci_low"].to_numpy(),
                            intervals["ci_high"].to_numpy(),
                            color=line.get_color(),
                            alpha=0.18,
                        )

                ax.set_title(f"N={size:,}, horizon={horizon}")
                ax.set_xticks(epsilons)
                ax.set_ylim(*y_limits)
                ax.grid(True, alpha=0.3)

                if panel.empty:
                    ax.text(
                        0.5, 0.5, "No data",
                        transform=ax.transAxes,
                        ha="center", va="center",
                    )

                if row == len(sizes) - 1:
                    ax.set_xlabel("Epsilon")
                if column == 0:
                    ax.set_ylabel(label)

        if legend_entries:
            axes[0, -1].legend(
                list(legend_entries.values()),
                list(legend_entries.keys()),
            )

        minimum = int(distribution_df["repetitions"].min())
        maximum = int(distribution_df["repetitions"].max())
        repetition_label = (
            str(minimum) if minimum == maximum else f"{minimum}–{maximum}"
        )

        fig.suptitle(
            f"{label} — {distribution_name}\n"
            f"{repetition_label} repetitions per point; "
            f"{confidence:.0%} confidence intervals across repetitions",
            fontsize=13,
        )

        output_path = output_directory / (
            f"{level}_{measure}_{distribution_name}.png"
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved {output_path}")



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

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements = load_measurements(
        RESULTS_DIRECTORY
    )

    if SELECTED_MEASURES is not None:
        measurements = measurements[
            measurements["measure"].isin(
                SELECTED_MEASURES
            )
        ].copy()

    if measurements.empty:
        raise ValueError(
            "No measurements match SELECTED_MEASURES."
        )

    summary = summarize_repetitions(
        measurements,
        confidence=CONFIDENCE_LEVEL,
    )

    summary.to_csv(
        OUTPUT_DIRECTORY / "comparison_summary.csv",
        index=False,
    )

    measure_types = (
        summary[["level", "measure"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for level, measure in measure_types:
        plot_comparison(
            summary=summary,
            measure=measure,
            level=level,
            output_directory=OUTPUT_DIRECTORY,
            confidence=CONFIDENCE_LEVEL,
        )

    plot_sampling_distributions(
        results_directory=RESULTS_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        amount_of_samples=DISTRIBUTION_SAMPLES,
        seed=DISTRIBUTION_SEED,
    )


if __name__ == "__main__":
    main()