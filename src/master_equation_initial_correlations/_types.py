from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Preset:
    id: str
    figure_id: str
    paper_figure_number: int
    family: str
    label: str
    paper_asset: str
    observable: str
    solver_id: str
    execution_profile: str
    supports_exact: bool
    supports_rerun: bool
    parameters: dict[str, Any]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    plot_divisor: float
    asset_dir: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceCurves:
    preset: Preset
    correlated: np.ndarray
    uncorrelated: np.ndarray
    jz_correlated: np.ndarray | None = None
    jz_uncorrelated: np.ndarray | None = None
    exact_correlated: np.ndarray | None = None
    exact_uncorrelated: np.ndarray | None = None


@dataclass(frozen=True)
class RerunResult:
    preset: Preset
    output_dir: Path
    correlated_error: float
    uncorrelated_error: float
    jz_correlated_error: float | None
    jz_uncorrelated_error: float | None
    exact_correlated_error: float | None = None
    exact_uncorrelated_error: float | None = None
    rendered_eps: Path | None = None
    rendered_png: Path | None = None
    summary_path: Path | None = None
