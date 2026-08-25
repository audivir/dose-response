"""Plotting functions for DRC package."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

Float64Array: TypeAlias = NDArray[np.float64]

EPSILON = 1e-15


def scatter(ax: Axes, x: Float64Array, y: Float64Array) -> None:
    """Draws a simple scatter plot.

    Args:
        ax: Axes of the current plot.
        x: X values.
        y: Corresponding Y values.
    """
    ax.scatter(x, y)


def plot(ax: Axes, x: Sequence[float], y: Sequence[float]) -> None:
    """Draws a simple black plot.

    Args:
        ax: Axes of the current plot.
        x: X values.
        y: Corresponding Y values.
    """
    ax.plot(x, y, color="black")


def errorbars(ax: Axes, x: Sequence[float], y: Sequence[float], err: Sequence[float]) -> None:
    """Plots bold black errorbars without a connecting line.

    Args:
        ax: Axes of the current plot.
        x: X values.
        y: Mean values.
        err: Standard error of the means.
    """
    ax.errorbar(x, y, err, color="black", marker="o", capsize=4, linewidth=0, elinewidth=2)


def axes(ax: Axes) -> None:
    """Inverts the X-axis, removes the right and upper spines, and bolds lines and ticks.

    Args:
        ax: Axes of the current plot.
    """
    ax.invert_xaxis()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2)


def yticks(ax: Axes, y: Float64Array) -> None:
    """Plots bold ticks on the Y-axis with a 0.5 step size.

    Args:
        ax: Axes of the current plot.
        y: Y values.
    """
    max_val = np.ceil(max(y) * 2).astype(int)
    min_val = np.floor(min(y) * 2).astype(int)
    y_ticks = [x / 2 for x in range(min_val, max_val + 1)][1:-1]
    ax.set_yticks(y_ticks, [str(y_tick) for y_tick in y_ticks], weight="bold")


def labels(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Adds bold labels to the plot and its axis.

    Args:
        ax: Axes of the current plot.
        title: Title of the plot.
        xlabel: Label on the X-axis.
        ylabel: Label on the Y-axis.
    """
    ax.set_title(title, weight="bold", size="18")
    ax.set_xlabel(xlabel, weight="bold", size="16")
    ax.set_ylabel(ylabel, weight="bold", size="16")


def xticks(ax: Axes, xvals: Float64Array) -> None:
    """Plots ticks on the Y axis with logarithmic scaling and labels on every log step.

    Args:
        ax: Axes of the current plot.
        xvals: Logarithmic X values.
    """
    min_val = 10 ** np.floor(-max(xvals))
    max_val = 10 ** np.ceil(-min(xvals))

    x_ticks = [f * min_val for f in (0.7, 0.8, 0.9, 1.0)]
    x_tick_labels = ["", "", "", f"{min_val:.2f}"]
    while True:
        for f in range(2, 10):
            x_ticks.append(f * min_val)
            x_tick_labels.append("")
        min_val = 10 * min_val
        x_ticks.append(min_val)
        label = f"{min_val:.2f}" if min_val < 1 else f"{min_val:.0f}"
        x_tick_labels.append(label)
        if (max_val - min_val) < EPSILON:
            break
    log_x_ticks = -np.log10(x_ticks)
    ax.set_xticks(log_x_ticks, x_tick_labels, weight="bold")
