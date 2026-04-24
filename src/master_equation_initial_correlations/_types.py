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


@dataclass(frozen=True, kw_only=True)
class BathParams:
    """Bath parameters for the supported bath types.

    The spectral exponent is explicit: Ohmic spectra use ``s=1.0``,
    sub-Ohmic spectra use ``0 < s < 1``, and super-Ohmic spectra use
    ``s > 1``.
    """

    bath_type: str = "bosonic"
    kind: str = "ohmic"
    beta: float = 1.0
    coupling: float = 0.05
    omega_c: float = 5.0
    s: float | None = None


@dataclass(frozen=True, init=False)
class NumericsConfig:
    """Numerical controls for bath kernels and Fortran-backed runs.

    The important user-facing knobs are the frequency quadrature used in the
    bath integrals and the time spacing of the generated coefficient and
    correlation tables. The defaults target the published benchmark parameter
    sets; users working away from that regime should check convergence.
    """

    omega_max: float
    omega_nodes: int = 500
    lambda_nodes: int = 100
    initial_state_omega_nodes: int = 260
    initial_state_lambda_nodes: int = 40
    initial_state_zeta_nodes: int = 40
    coefficient_omega_max: float = 500.0
    correlation_omega_max: float = 510.0
    initial_state_omega_max: float = 500.0
    coefficient_omega_min: float = 1.0e-14
    correlation_omega_min: float = 1.0e-11
    initial_state_omega_min: float = 1.0e-16
    coefficient_time_step: float = 0.0025
    correlation_tau_step: float = 0.0025
    coefficient_t_min: float = 1.0e-14
    coefficient_t_max: float = 5.00000000000001
    correlation_tau_min: float = 1.0e-14
    correlation_tau_max: float = 5.00000000000001
    fortran_dtau: float = 0.005
    fortran_dt: float = 0.01
    fortran_cutoff: float = 15.0
    fortran_t_final: float = 5.0

    def __init__(
        self,
        *,
        omega_max: float | None = None,
        omega_nodes: int = 500,
        lambda_nodes: int = 100,
        initial_state_omega_nodes: int = 260,
        initial_state_lambda_nodes: int = 40,
        initial_state_zeta_nodes: int = 40,
        coefficient_omega_max: float | None = None,
        correlation_omega_max: float | None = None,
        initial_state_omega_max: float | None = None,
        coefficient_omega_min: float = 1.0e-14,
        correlation_omega_min: float = 1.0e-11,
        initial_state_omega_min: float = 1.0e-16,
        coefficient_time_step: float = 0.0025,
        correlation_tau_step: float = 0.0025,
        coefficient_t_min: float = 1.0e-14,
        coefficient_t_max: float = 5.00000000000001,
        correlation_tau_min: float = 1.0e-14,
        correlation_tau_max: float = 5.00000000000001,
        fortran_dtau: float = 0.005,
        fortran_dt: float = 0.01,
        fortran_cutoff: float = 15.0,
        fortran_t_final: float = 5.0,
        **legacy_names: Any,
    ) -> None:
        """Create numerical controls.

        Legacy keyword aliases are accepted so older local scripts continue to
        run, but public examples use the clearer physical names above.
        """

        if "instate_omega_nodes" in legacy_names:
            initial_state_omega_nodes = legacy_names.pop("instate_omega_nodes")
        if "instate_lambda_nodes" in legacy_names:
            initial_state_lambda_nodes = legacy_names.pop("instate_lambda_nodes")
        if "instate_zeta_nodes" in legacy_names:
            initial_state_zeta_nodes = legacy_names.pop("instate_zeta_nodes")
        if "omega_max_coefficients" in legacy_names:
            coefficient_omega_max = legacy_names.pop("omega_max_coefficients")
        if "omega_max_tau" in legacy_names:
            correlation_omega_max = legacy_names.pop("omega_max_tau")
        if "omega_max_instate" in legacy_names:
            initial_state_omega_max = legacy_names.pop("omega_max_instate")
        if "omega_min_coefficients" in legacy_names:
            coefficient_omega_min = legacy_names.pop("omega_min_coefficients")
        if "omega_min_tau" in legacy_names:
            correlation_omega_min = legacy_names.pop("omega_min_tau")
        if "omega_min_instate" in legacy_names:
            initial_state_omega_min = legacy_names.pop("omega_min_instate")
        if "tau_t_min" in legacy_names:
            correlation_tau_min = legacy_names.pop("tau_t_min")
        if "tau_t_max" in legacy_names:
            correlation_tau_max = legacy_names.pop("tau_t_max")
        if "coefficient_points" in legacy_names:
            points = int(legacy_names.pop("coefficient_points"))
            if points > 1:
                coefficient_time_step = float(coefficient_t_max) / float(points - 1)
        if "tau_points" in legacy_names:
            points = int(legacy_names.pop("tau_points"))
            if points > 1:
                correlation_tau_step = float(correlation_tau_max) / float(points - 1)
        if legacy_names:
            unknown = ", ".join(sorted(legacy_names))
            raise TypeError(f"unknown NumericsConfig option(s): {unknown}")

        public_omega_max = 500.0 if omega_max is None else float(omega_max)
        object.__setattr__(self, "omega_max", public_omega_max)
        object.__setattr__(self, "omega_nodes", omega_nodes)
        object.__setattr__(self, "lambda_nodes", lambda_nodes)
        object.__setattr__(self, "initial_state_omega_nodes", initial_state_omega_nodes)
        object.__setattr__(self, "initial_state_lambda_nodes", initial_state_lambda_nodes)
        object.__setattr__(self, "initial_state_zeta_nodes", initial_state_zeta_nodes)
        object.__setattr__(self, "coefficient_omega_max", public_omega_max if coefficient_omega_max is None else float(coefficient_omega_max))
        object.__setattr__(self, "correlation_omega_max", (public_omega_max if omega_max is not None else 510.0) if correlation_omega_max is None else float(correlation_omega_max))
        object.__setattr__(self, "initial_state_omega_max", public_omega_max if initial_state_omega_max is None else float(initial_state_omega_max))
        object.__setattr__(self, "coefficient_omega_min", float(coefficient_omega_min))
        object.__setattr__(self, "correlation_omega_min", float(correlation_omega_min))
        object.__setattr__(self, "initial_state_omega_min", float(initial_state_omega_min))
        object.__setattr__(self, "coefficient_time_step", float(coefficient_time_step))
        object.__setattr__(self, "correlation_tau_step", float(correlation_tau_step))
        object.__setattr__(self, "coefficient_t_min", float(coefficient_t_min))
        object.__setattr__(self, "coefficient_t_max", float(coefficient_t_max))
        object.__setattr__(self, "correlation_tau_min", float(correlation_tau_min))
        object.__setattr__(self, "correlation_tau_max", float(correlation_tau_max))
        object.__setattr__(self, "fortran_dtau", float(fortran_dtau))
        object.__setattr__(self, "fortran_dt", float(fortran_dt))
        object.__setattr__(self, "fortran_cutoff", float(fortran_cutoff))
        object.__setattr__(self, "fortran_t_final", float(fortran_t_final))

    @staticmethod
    def _points(t_max: float, step: float) -> int:
        return int(round(float(t_max) / float(step))) + 1

    @property
    def coefficient_points(self) -> int:
        return self._points(self.coefficient_t_max, self.coefficient_time_step)

    @property
    def tau_points(self) -> int:
        return self._points(self.correlation_tau_max, self.correlation_tau_step)

    @property
    def instate_omega_nodes(self) -> int:
        return self.initial_state_omega_nodes

    @property
    def instate_lambda_nodes(self) -> int:
        return self.initial_state_lambda_nodes

    @property
    def instate_zeta_nodes(self) -> int:
        return self.initial_state_zeta_nodes

    @property
    def omega_max_coefficients(self) -> float:
        return self.coefficient_omega_max

    @property
    def omega_max_tau(self) -> float:
        return self.correlation_omega_max

    @property
    def omega_max_instate(self) -> float:
        return self.initial_state_omega_max

    @property
    def omega_min_coefficients(self) -> float:
        return self.coefficient_omega_min

    @property
    def omega_min_tau(self) -> float:
        return self.correlation_omega_min

    @property
    def omega_min_instate(self) -> float:
        return self.initial_state_omega_min

    @property
    def tau_t_min(self) -> float:
        return self.correlation_tau_min

    @property
    def tau_t_max(self) -> float:
        return self.correlation_tau_max


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
    summary_path: Path | None = None
    verification_performed: bool = True
    output_files: dict[str, Path] | None = None
    input_files: dict[str, Path] | None = None
    source_files: dict[str, Path] | None = None
    log_files: dict[str, Path] | None = None
