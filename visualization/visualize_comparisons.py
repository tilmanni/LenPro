from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# insert run directory for run here. 
RUN_DIRECTORY = Path("20261031T180421014966Z")
RESULTS_DIRECTORY = Path("results") / RUN_DIRECTORY
OUTPUT_DIRECTORY = Path("plots/comparisons") / RUN_DIRECTORY
CONFIDENCE_LEVEL = 0.95

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
                "max_length": generator["fixed_length"],
                "distribution": distribution["name"],
                "distribution_config": json.dumps(
                    distribution, sort_keys=True
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
        "max_length",
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
        "max_length",
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
    lengths = sorted(selected["max_length"].unique())
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
            len(lengths),
            figsize=(5 * len(lengths), 3.6 * len(sizes)),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )

        legend_entries = {}

        for row, size in enumerate(sizes):
            for column, length in enumerate(lengths):
                ax = axes[row, column]
                panel = distribution_df[
                    (distribution_df["data_set_size"] == size)
                    & (distribution_df["max_length"] == length)
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

                ax.set_title(f"N={size:,}, max length={length}")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    measurements = load_measurements(RESULTS_DIRECTORY)

    if SELECTED_MEASURES is not None:
        measurements = measurements[
            measurements["measure"].isin(SELECTED_MEASURES)
        ].copy()

    if measurements.empty:
        raise ValueError("No measurements match SELECTED_MEASURES.")

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


if __name__ == "__main__":
    main()