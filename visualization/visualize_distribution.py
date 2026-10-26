import numpy as np
import matplotlib.pyplot as plt
from lenpro.distributions import distributions


def describe(distribution: distributions.Distribution) -> str:
    """'UniformDistribution(low=10, high=20)' built from to_dict()."""
    d = distribution.to_dict()
    name = d.pop("name")
    params = ", ".join(f"{k}={v}" for k, v in d.items())
    return f"{name}({params})"


def visualize_distribution(
    distribution: distributions.Distribution,
    amount_of_samples: int,
    max_value: int | None = None,
    log_y: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    samples = np.array([distribution.sample() for _ in range(amount_of_samples)])

    # Heavy-tailed distributions (Zipf) occasionally produce huge values that
    # would squash the plot; optionally cut them off and say so in the title.
    cut_off = 0
    if max_value is not None:
        cut_off = int((samples > max_value).sum())
        samples = samples[samples <= max_value]

    values, counts = np.unique(samples, return_counts=True)

    if ax is None:
        _, ax = plt.subplots()
    ax.bar(values, counts / amount_of_samples, width=1.0, edgecolor="black")
    ax.set_xlabel("value")
    ax.set_ylabel("relative frequency")
    if log_y:
        ax.set_yscale("log")

    title = f"{describe(distribution)}\n{amount_of_samples} samples"
    if cut_off:
        title += f", {cut_off} samples > {max_value} not shown"
    ax.set_title(title)
    return ax  

