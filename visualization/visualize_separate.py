import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


MEASURE_LABELS = {
    "RelativeErrorMeasure": "Relative error",
    "FractionRetainedEventsMeasure": "Fraction of retained events",
    "FractionSyntheticEventsMeasure": "Fraction of synthetic events",
    "EmptyOutputProbabilityMeasure": "Empty output probability",
    "LengthUnchangedProbabilityMeasure": "Probability of unchanged length",
    "GlobalFractionRetainedEventsMeasure": "Global fraction of retained events",
    "GlobalFractionSyntheticEventsMeasure": "Global fraction of synthetic events",
}


def describe(config):
    """Create a readable label from a to_dict() result."""
    name = config["name"]
    parameters = ", ".join(
        f"{key}={describe(value) if isinstance(value, dict) else value}"
        for key, value in config.items()
        if key != "name"
    )
    return f"{name}({parameters})" if parameters else name


def measure_key(config):
    """Identify a measure by both its name and its parameters."""
    return json.dumps(config, sort_keys=True)


def measure_label(config):
    name = MEASURE_LABELS.get(config["name"], config["name"])
    parameters = ", ".join(
        f"{key}={value}"
        for key, value in config.items()
        if key != "name"
    )
    return f"{name} ({parameters})" if parameters else name


def plot_result_file(file_path, output_directory="plots"):
    file_path = Path(file_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with file_path.open("r", encoding="utf-8") as file:
        run = json.load(file)

    # Each series contains observations from all epsilon values.
    mean_series = {}
    interval_series = {}
    global_series = {}

    for result in run["results"]:
        epsilon = result["protocol"]["epsilon"]

        for measurement in result["aggregate"]:
            aggregate_name = measurement["measure"]["name"]
            entry_config = measurement["entry_level_measure"]
            key = measure_key(entry_config)

            if aggregate_name == "MeanMeasure":
                series = mean_series.setdefault(
                    key, {"config": entry_config, "points": []}
                )
                series["points"].append((epsilon, measurement["value"]))

            elif aggregate_name == "ConfidenceIntervalMeasure":
                # Support either [low, high] or {"low": ..., "high": ...}.
                interval = measurement["value"]
                if isinstance(interval, dict):
                    low, high = interval["low"], interval["high"]
                else:
                    low, high = interval

                confidence = measurement["measure"]["confidence_level"]
                series = interval_series.setdefault(
                    (key, confidence), []
                )
                series.append((epsilon, low, high))

            else:
                raise ValueError(
                    f"Unsupported aggregate measure: {aggregate_name}"
                )

        for measurement in result["global"]:
            config = measurement["measure"]
            key = measure_key(config)
            series = global_series.setdefault(
                key, {"config": config, "points": []}
            )
            series["points"].append((epsilon, measurement["value"]))

    number_of_plots = len(mean_series) + len(global_series)
    if number_of_plots == 0:
        raise ValueError(f"No measures to plot in {file_path}")

    # This script expects one observation per measure and epsilon.
    for series in [*mean_series.values(), *global_series.values()]:
        epsilons = [point[0] for point in series["points"]]
        if len(epsilons) != len(set(epsilons)):
            raise ValueError(
                f"Duplicate epsilon values in {file_path}. "
                "Separate different protocols/configurations before plotting."
            )

    columns = min(3, number_of_plots)
    rows = math.ceil(number_of_plots / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.8 * rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    plot_index = 0

    # Aggregate means, with their confidence intervals.
    for key, series in mean_series.items():
        ax = axes[plot_index]
        plot_index += 1

        points = sorted(series["points"])
        epsilons, values = zip(*points)

        ax.plot(epsilons, values, marker="o", color="tab:blue", label="Mean")

        for (interval_key, confidence), intervals in interval_series.items():
            if interval_key != key:
                continue

            intervals = sorted(intervals)
            ci_epsilons, lower, upper = zip(*intervals)

            ax.fill_between(
                ci_epsilons,
                lower,
                upper,
                color="tab:blue",
                alpha=0.2,
                label=f"{confidence:.0%} confidence interval",
            )

        ax.set_title(f"Mean: {measure_label(series['config'])}")
        ax.set_ylabel("Value")
        ax.set_xlabel("Epsilon")
        ax.set_xticks(epsilons)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Global measures.
    for series in global_series.values():
        ax = axes[plot_index]
        plot_index += 1

        points = sorted(series["points"])
        epsilons, values = zip(*points)

        ax.plot(epsilons, values, marker="o", color="tab:orange")

        ax.set_title(measure_label(series["config"]))
        ax.set_ylabel("Value")
        ax.set_xlabel("Epsilon")
        ax.set_xticks(epsilons)
        ax.grid(True, alpha=0.3)

    # Remove unused subplot slots.
    for ax in axes[plot_index:]:
        fig.delaxes(ax)

    generator = run["stream_generator"]
    distribution = generator["length_generator"]["distribution"]

    # All protocol parameters except epsilon are fixed in each of your files.
    protocol_config = {
        key: value
        for key, value in run["results"][0]["protocol"].items()
        if key != "epsilon"
    }

    fig.suptitle(
        f"{describe(protocol_config)}\n"
        f"Dataset size={run['data_set_size']}, "
        f"fixed length={generator['fixed_length']}\n"
        f"Length distribution: {describe(distribution)}",
        fontsize=11,
    )

    output_path = output_directory / f"{file_path.stem}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_all_results(results_directory="results", output_directory="plots"):
    files = sorted(Path(results_directory).glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {results_directory}"
        )

    for file_path in files:
        output_path = plot_result_file(file_path, output_directory)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_all_results(
        results_directory="results",
        output_directory="plots",
    )