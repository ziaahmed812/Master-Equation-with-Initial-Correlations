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


@dataclass(frozen=True)
class SystemParams:
    """Collective-spin system parameters.

    The public API uses N; internally the collective spin quantum number is
    J = N / 2.
    """

    N: int
    epsilon0: float = 4.0
    epsilon: float = 4.0
    delta0: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class BathParams:
    """Bath parameters for the supported model families."""

    kind: str = "ohmic"
    beta: float = 1.0
    coupling: float = 0.05
    omega_c: float = 5.0
    s: float | None = None


@dataclass(frozen=True)
class RunConfig:
    """Run-time options shared by notebook, script, and CLI workflows."""

    output_dir: Path | str | None = None
    overwrite: bool = False
    verify: bool = True
    plot: bool = False
    verbose: bool = True
    save_density: bool = False
    t_max: float = 5.0
    dt: float = 0.2
    numerics: "NumericsConfig | None" = None


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical controls for generated kernels and preserved Fortran runs.

    The defaults reproduce the paper-scale Mathematica/Fortran workflow. Users
    working away from that regime should choose these grids deliberately rather
    than relying on hidden constants.
    """

    omega_nodes: int = 500
    lambda_nodes: int = 100
    instate_omega_nodes: int = 260
    instate_lambda_nodes: int = 40
    instate_zeta_nodes: int = 40
    omega_max_coefficients: float = 500.0
    omega_max_tau: float = 510.0
    omega_max_instate: float = 500.0
    omega_min_coefficients: float = 1.0e-14
    omega_min_tau: float = 1.0e-11
    omega_min_instate: float = 1.0e-16
    coefficient_points: int = 2001
    coefficient_t_min: float = 1.0e-14
    coefficient_t_max: float = 5.00000000000001
    tau_points: int = 2001
    tau_t_min: float = 1.0e-14
    tau_t_max: float = 5.00000000000001
    fortran_dtau: float = 0.005
    fortran_dt: float = 0.01
    fortran_cutoff: float = 15.0
    fortran_t_final: float = 5.0


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
    example: ReferenceExample | None
    output_dir: Path
    correlated_error: float | None
    uncorrelated_error: float | None
    jz_correlated_error: float | None
    jz_uncorrelated_error: float | None
    exact_correlated_error: float | None = None
    exact_uncorrelated_error: float | None = None
    rendered_eps: Path | None = None
    rendered_png: Path | None = None
    summary_path: Path | None = None
    verification_performed: bool = True
    output_files: dict[str, Path] | None = None
    input_files: dict[str, Path] | None = None
    source_files: dict[str, Path] | None = None
    log_files: dict[str, Path] | None = None


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
    observable: str | np.ndarray = "jx"
    s: float | None = None
    initial_state: np.ndarray | None = None


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
    verification_performed: bool = True
    output_files: dict[str, Path] | None = None
    input_files: dict[str, Path] | None = None
    source_files: dict[str, Path] | None = None
    log_files: dict[str, Path] | None = None


SolverResult = SimulationResult
