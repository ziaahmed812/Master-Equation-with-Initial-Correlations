from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ._types import ReferenceCurves


def _axis_spec(observable: str, x_range: tuple[float, float]) -> dict[str, object]:
    if x_range == (0.0, 5.0):
        x_ticks = np.arange(0.0, 5.0 + 0.001, 1.0)
    elif x_range == (0.0, 4.0):
        x_ticks = np.arange(0.0, 4.0 + 0.001, 1.0)
    elif x_range == (0.0, 2.0):
        x_ticks = np.arange(0.0, 2.0 + 0.001, 0.5)
    else:
        raise ValueError(f"Unsupported x-range {x_range}")

    if observable == "jx":
        y_ticks = np.arange(-1.0, 1.0 + 0.001, 0.5)
        y_label = r"$j_x$"
    elif observable == "jx2":
        y_ticks = np.arange(0.0, 1.0 + 0.001, 0.2)
        y_label = r"$j_x^{(2)}$"
    else:
        raise ValueError(f"Unsupported observable {observable!r}")

    return {
        "x_label": r"$t$",
        "y_label": y_label,
        "x_ticks": x_ticks.tolist(),
        "y_ticks": y_ticks.tolist(),
        "x_range": list(x_range),
    }


def apply_paper_axes_style(ax, *, observable: str, x_range: tuple[float, float], y_range: tuple[float, float]) -> dict[str, object]:
    spec = _axis_spec(observable, x_range)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(spec["x_label"], fontsize=15)
    ax.set_ylabel(spec["y_label"], fontsize=15)
    ax.set_xticks(spec["x_ticks"])
    ax.set_yticks(spec["y_ticks"])
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    return spec


def plot_reference_curves(curves: ReferenceCurves, output_dir: str | Path, filename: str | None = None) -> dict[str, Path | dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_asset = filename or curves.preset.paper_asset
    stem = Path(paper_asset).stem

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(curves.correlated[:, 0], curves.correlated[:, 1], color="black", linewidth=2.5, linestyle="-")
    ax.plot(curves.uncorrelated[:, 0], curves.uncorrelated[:, 1], color="red", linewidth=2.5, linestyle="--")

    if curves.exact_correlated is not None:
        ax.plot(
            curves.exact_correlated[:, 0],
            curves.exact_correlated[:, 1],
            linestyle="None",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="blue",
            markeredgewidth=1.2,
            markersize=6,
            zorder=4,
        )
        ax.plot(
            curves.exact_correlated[:, 0],
            curves.exact_correlated[:, 1],
            linestyle="None",
            marker="o",
            color="blue",
            markersize=2.4,
            zorder=5,
        )
    if curves.exact_uncorrelated is not None:
        ax.plot(
            curves.exact_uncorrelated[:, 0],
            curves.exact_uncorrelated[:, 1],
            linestyle="None",
            marker="s",
            markerfacecolor="none",
            markeredgecolor="purple",
            markeredgewidth=1.2,
            markersize=6,
            zorder=4,
        )

    axis_style = apply_paper_axes_style(
        ax,
        observable=curves.preset.observable,
        x_range=curves.preset.x_range,
        y_range=curves.preset.y_range,
    )
    fig.tight_layout()

    eps_path = output_dir / f"{stem}.eps"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(eps_path, format="eps")
    fig.savefig(png_path, format="png", dpi=200)
    plt.close(fig)
    return {"eps": eps_path, "png": png_path, "axis_style": axis_style}
