from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferenceExample:
    id: str
    public_id: str
    bath: str
    model: str
    spectral: str
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


Preset = ReferenceExample


@dataclass(frozen=True)
class ReferenceCurves:
    example: ReferenceExample
    correlated: np.ndarray
    uncorrelated: np.ndarray
    jz_correlated: np.ndarray | None = None
    jz_uncorrelated: np.ndarray | None = None
    exact_correlated: np.ndarray | None = None
    exact_uncorrelated: np.ndarray | None = None


@dataclass(frozen=True)
class RerunResult:
    example: ReferenceExample
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


@dataclass(frozen=True)
class SimulationParams:
    bath: str
    model: str
    N: int
    epsilon0: float = 4.0
    epsilon: float = 2.5
    delta0: float = 0.5
    delta: float = 0.5
    beta: float = 1.0
    coupling: float = 0.05
    omega_c: float = 5.0
    spectral: str = "ohmic"
    observable: str = "jx"
    s: float | None = None


@dataclass(frozen=True)
class SimulationResult:
    params: SimulationParams
    output_dir: Path
    source: str
    example: ReferenceExample | None = None
    correlated_path: Path | None = None
    uncorrelated_path: Path | None = None
    correlated_error: float | None = None
    uncorrelated_error: float | None = None
    exact_correlated_error: float | None = None
    exact_uncorrelated_error: float | None = None
    rendered_eps: Path | None = None
    rendered_png: Path | None = None
    summary_path: Path | None = None
