from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from ._types import NumericsConfig, SimulationParams
from .pure_dephasing import spin_operators


DEFAULT_NUMERICS = NumericsConfig()
QuadratureConfig = NumericsConfig
COEFFICIENT_TIMES = np.linspace(DEFAULT_NUMERICS.coefficient_t_min, DEFAULT_NUMERICS.coefficient_t_max, DEFAULT_NUMERICS.coefficient_points)
TAU_TIMES = np.linspace(DEFAULT_NUMERICS.tau_t_min, DEFAULT_NUMERICS.tau_t_max, DEFAULT_NUMERICS.tau_points)


@dataclass(frozen=True)
class GeneratedInputs:
    coefficient_a: np.ndarray
    coefficient_b: np.ndarray
    coefficient_c: np.ndarray
    integral_tau: np.ndarray
    bath_correlation_tau: np.ndarray
    initial_state: np.ndarray
    initial_state_source: str


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _validate_exact_step(t_min: float, t_max: float, step: float, label: str) -> None:
    interval = t_max - t_min
    intervals = interval / step
    rounded = round(intervals)
    if rounded < 1:
        raise ValueError(f"{label} step must produce at least two grid points.")
    if not np.isclose(intervals, rounded, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(
            f"{label} step must exactly divide max-min; got interval={interval:g}, "
            f"step={step:g}, interval/step={intervals:g}."
        )


def validate_numerics(numerics: NumericsConfig | None) -> NumericsConfig:
    config = DEFAULT_NUMERICS if numerics is None else numerics
    for name in (
        "omega_nodes",
        "lambda_nodes",
        "initial_state_omega_nodes",
        "initial_state_lambda_nodes",
        "initial_state_zeta_nodes",
    ):
        _positive_int(getattr(config, name), name)
    for name in (
        "omega_max",
        "coefficient_omega_max",
        "correlation_omega_max",
        "initial_state_omega_max",
        "coefficient_omega_min",
        "correlation_omega_min",
        "initial_state_omega_min",
        "coefficient_time_step",
        "correlation_tau_step",
        "coefficient_t_min",
        "coefficient_t_max",
        "correlation_tau_min",
        "correlation_tau_max",
        "fortran_dtau",
        "fortran_dt",
        "fortran_cutoff",
        "fortran_t_final",
    ):
        _finite_float(getattr(config, name), name)
    for name in ("coefficient_time_step", "correlation_tau_step"):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    for lower, upper, label in (
        (config.omega_min_coefficients, config.omega_max_coefficients, "coefficient omega grid"),
        (config.omega_min_tau, config.omega_max_tau, "tau omega grid"),
        (config.omega_min_instate, config.omega_max_instate, "initial-state omega grid"),
        (config.coefficient_t_min, config.coefficient_t_max, "coefficient time grid"),
        (config.tau_t_min, config.tau_t_max, "bath-correlation tau grid"),
    ):
        if lower < 0 or upper <= lower:
            raise ValueError(f"{label} requires 0 <= min < max.")
    _validate_exact_step(config.coefficient_t_min, config.coefficient_t_max, config.coefficient_time_step, "coefficient time grid")
    _validate_exact_step(config.tau_t_min, config.tau_t_max, config.correlation_tau_step, "bath-correlation tau grid")
    if config.coefficient_points < 2:
        raise ValueError("coefficient_time_step must produce at least two coefficient grid points.")
    if config.tau_points < 2:
        raise ValueError("correlation_tau_step must produce at least two correlation grid points.")
    for name in ("fortran_dtau", "fortran_dt", "fortran_cutoff", "fortran_t_final"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.coefficient_t_max + 1.0e-12 < config.fortran_t_final:
        raise ValueError("coefficient_t_max must be at least fortran_t_final.")
    if config.correlation_tau_max + 1.0e-12 < config.fortran_t_final:
        raise ValueError("correlation_tau_max must be at least fortran_t_final.")
    return config


def coefficient_times(numerics: NumericsConfig | None = None) -> np.ndarray:
    config = validate_numerics(numerics)
    return np.linspace(config.coefficient_t_min, config.coefficient_t_max, config.coefficient_points)


def tau_times(numerics: NumericsConfig | None = None) -> np.ndarray:
    config = validate_numerics(numerics)
    return np.linspace(config.tau_t_min, config.tau_t_max, config.tau_points)


def coefficient_index_step(numerics: NumericsConfig | None = None) -> float:
    config = validate_numerics(numerics)
    return (config.coefficient_t_max - config.coefficient_t_min) / (config.coefficient_points - 1)


def tau_index_step(numerics: NumericsConfig | None = None) -> float:
    config = validate_numerics(numerics)
    return (config.tau_t_max - config.tau_t_min) / (config.tau_points - 1)


def numerics_summary(numerics: NumericsConfig | None = None) -> dict[str, float | int]:
    config = validate_numerics(numerics)
    return {
        "omega_max": config.omega_max,
        "omega_nodes": config.omega_nodes,
        "lambda_nodes": config.lambda_nodes,
        "initial_state_omega_nodes": config.initial_state_omega_nodes,
        "initial_state_lambda_nodes": config.initial_state_lambda_nodes,
        "initial_state_zeta_nodes": config.initial_state_zeta_nodes,
        "coefficient_omega_max": config.coefficient_omega_max,
        "correlation_omega_max": config.correlation_omega_max,
        "initial_state_omega_max": config.initial_state_omega_max,
        "coefficient_time_step": config.coefficient_time_step,
        "coefficient_t_min": config.coefficient_t_min,
        "coefficient_t_max": config.coefficient_t_max,
        "coefficient_points": config.coefficient_points,
        "coefficient_index_step": coefficient_index_step(config),
        "correlation_tau_step": config.correlation_tau_step,
        "correlation_tau_min": config.tau_t_min,
        "correlation_tau_max": config.correlation_tau_max,
        "correlation_tau_points": config.tau_points,
        "tau_index_step": tau_index_step(config),
        "fortran_dtau": config.fortran_dtau,
        "fortran_dt": config.fortran_dt,
        "fortran_cutoff": config.fortran_cutoff,
        "fortran_t_final": config.fortran_t_final,
    }


def _gauss_interval(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _omega_rule(n: int, omega_min: float, omega_max: float, *, spectral: str, s: float | None) -> tuple[np.ndarray, np.ndarray]:
    if spectral == "subohmic" and (s is not None and s < 1.0):
        x, wx = _gauss_interval(n, np.sqrt(omega_min), np.sqrt(omega_max))
        omega = x * x
        return omega, wx * 2.0 * x
    return _gauss_interval(n, omega_min, omega_max)


def _spectral_weight(omega: np.ndarray, params: SimulationParams) -> np.ndarray:
    s = 1.0 if params.spectral == "ohmic" else float(params.s)
    return (omega**s) / (params.omega_c ** (s - 1.0)) * np.exp(-omega / params.omega_c)


def _safe_coth(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.tanh(x)


def _bosonic_even_odd(omega: np.ndarray, lam: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = -np.expm1(-beta * omega)
    even = (np.exp(-(beta - lam) * omega) + np.exp(-lam * omega)) / denominator
    odd = (np.exp(-(beta - lam) * omega) - np.exp(-lam * omega)) / denominator
    return even, odd


def _spin_even_odd(omega: np.ndarray, lam: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.exp(-beta * omega)
    denominator = 1.0 + q
    even = (np.exp(-(beta - lam) * omega) + np.exp(-lam * omega)) / denominator
    odd = (np.exp(-(beta - lam) * omega) - np.exp(-lam * omega)) / denominator
    return even, odd


def _bosonic_bathcorr_x(omega: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    qden = -np.expm1(-beta * omega)
    return np.exp(-omega * x) + (
        np.exp(-(beta - x) * omega) + np.exp(-(beta + x) * omega)
    ) / qden


def _spin_bathcorr_x(omega: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    return _bosonic_bathcorr_x(omega, x, beta) * np.tanh(0.5 * beta * omega)


def _bath_even_odd(omega: np.ndarray, lam: np.ndarray, params: SimulationParams) -> tuple[np.ndarray, np.ndarray]:
    if params.bath == "spin":
        return _spin_even_odd(omega, lam, params.beta)
    return _bosonic_even_odd(omega, lam, params.beta)


def _bathcorr_x(omega: np.ndarray, x: np.ndarray, params: SimulationParams) -> np.ndarray:
    if params.bath == "spin":
        return _spin_bathcorr_x(omega, x, params.beta)
    return _bosonic_bathcorr_x(omega, x, params.beta)


def _legacy_subohmic_eta_tau(params: SimulationParams, numerics: NumericsConfig) -> np.ndarray:
    times = tau_times(numerics)
    tempi = np.arctan(params.omega_c * times)
    numerator = params.omega_c * np.sqrt(np.pi) * np.sin(1.5 * tempi)
    denominator = 2.0 * (1.0 + (times * params.omega_c) ** 2.0) ** 0.75
    return numerator / denominator


def generate_integral_tau(params: SimulationParams, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> np.ndarray:
    numerics = validate_numerics(quadrature)
    correlation = generate_bath_correlation_tau(params, numerics)
    if params.bath == "spin":
        return correlation[:, 1]
    return correlation[:, 0]


def generate_bath_correlation_tau(params: SimulationParams, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> np.ndarray:
    numerics = validate_numerics(quadrature)
    times = tau_times(numerics)
    omega, weights = _omega_rule(
        numerics.omega_nodes,
        numerics.omega_min_tau,
        numerics.omega_max_tau,
        spectral=params.spectral,
        s=params.s,
    )
    spectral = _spectral_weight(omega, params)
    cos_terms = np.cos(np.outer(times, omega))
    sin_terms = np.sin(np.outer(times, omega))
    if params.bath == "spin":
        nu_weight = weights * spectral
        eta_weight = weights * spectral * np.tanh(0.5 * params.beta * omega)
        return np.column_stack([cos_terms @ nu_weight, sin_terms @ eta_weight])
    nu_weight = weights * spectral * _safe_coth(0.5 * params.beta * omega)
    eta_weight = weights * spectral
    eta = sin_terms @ eta_weight
    if params.spectral == "subohmic" and params.s is not None and abs(params.s - 0.5) <= 1.0e-14:
        # Preserve the dissipative kernel used by the published sub-Ohmic
        # Fortran branch. The real/noise kernel remains generated from the
        # integral table above.
        eta = _legacy_subohmic_eta_tau(params, numerics)
    return np.column_stack([cos_terms @ nu_weight, eta])


def generate_coefficients(params: SimulationParams, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerics = validate_numerics(quadrature)
    times = coefficient_times(numerics)
    omega, omega_weights = _omega_rule(
        numerics.omega_nodes,
        numerics.omega_min_coefficients,
        numerics.omega_max_coefficients,
        spectral=params.spectral,
        s=params.s,
    )
    lam, lam_weights = _gauss_interval(numerics.lambda_nodes, 0.0, params.beta)

    omega_col = omega[:, None]
    lam_row = lam[None, :]
    even, odd = _bath_even_odd(omega_col, lam_row, params)

    epsilon = float(params.epsilon)
    delta = float(params.delta)
    xi = float(params.epsilon0)
    delta0 = float(params.delta0)
    lambda_final = float(np.hypot(epsilon, delta))
    delta_initial = float(np.hypot(xi, delta0))

    u1 = xi * delta0 * (1.0 - np.cosh(lam_row * delta_initial))
    u2 = np.sinh(lam_row * delta_initial)
    u3 = xi * xi + delta0 * delta0 * np.cosh(lam_row * delta_initial)

    i1_even = (u1 * even * lam_weights).sum(axis=1)
    i1_odd = (u1 * odd * lam_weights).sum(axis=1)
    i2_even = (u2 * even * lam_weights).sum(axis=1)
    i2_odd = (u2 * odd * lam_weights).sum(axis=1)
    i3_even = (u3 * even * lam_weights).sum(axis=1)
    i3_odd = (u3 * odd * lam_weights).sum(axis=1)

    spectral = _spectral_weight(omega, params)
    integration_weight = omega_weights * spectral
    coefficient_c_weight = integration_weight
    if params.spectral == "subohmic":
        # The sub-Ohmic Mathematica notebook multiplies only C.dat by a
        # second cutoff factor. Preserve that published-code convention.
        coefficient_c_weight = coefficient_c_weight * np.exp(-omega / params.omega_c)
    d2 = delta_initial * delta_initial
    lf2 = lambda_final * lambda_final

    coefficient_a = np.empty_like(times, dtype=complex)
    coefficient_b = np.empty_like(times, dtype=complex)
    coefficient_c = np.empty_like(times, dtype=complex)

    for index, t_value in enumerate(times):
        cos_wt = np.cos(omega * t_value)
        sin_wt = np.sin(omega * t_value)
        cos_lt = np.cos(lambda_final * t_value)
        sin_lt = np.sin(lambda_final * t_value)

        i1c = i1_even * cos_wt
        i1s = i1_odd * sin_wt
        i2c = i2_even * cos_wt
        i2s = i2_odd * sin_wt
        i3c = i3_even * cos_wt
        i3s = i3_odd * sin_wt

        a_real = (
            (epsilon * delta * (1.0 - cos_lt) / (lf2 * d2)) * i1c
            + (epsilon * delta * sin_lt / (lambda_final * delta_initial)) * i2s
            - ((delta * delta + epsilon * epsilon * cos_lt) / (lf2 * d2)) * i3c
        )
        a_imag = (
            (-epsilon * delta * (1.0 - cos_lt) / (lf2 * d2)) * i1s
            + (epsilon * delta * sin_lt / (lambda_final * delta_initial)) * i2c
            + ((delta * delta + epsilon * epsilon * cos_lt) / (lf2 * d2)) * i3s
        )
        b_real = (
            -(delta * sin_lt / (lambda_final * d2)) * i1c
            - (delta * cos_lt / delta_initial) * i2s
            - (epsilon * sin_lt / (lambda_final * d2)) * i3c
        )
        b_imag = (
            (delta * sin_lt / (lambda_final * d2)) * i1s
            - (delta * cos_lt / delta_initial) * i2c
            + (epsilon * sin_lt / (lambda_final * d2)) * i3s
        )
        c_real = (
            (1.0 / d2 - delta * delta * (1.0 - cos_lt) / (lf2 * d2)) * i1c
            - (delta * delta0 * sin_lt / (lambda_final * delta_initial)) * i2s
            - (epsilon * delta * (1.0 - cos_lt) / (lf2 * d2)) * i3c
        )
        c_imag = (
            (-1.0 / d2 + delta * delta * (1.0 - cos_lt) / (lf2 * d2)) * i1s
            - (delta * delta0 * sin_lt / (lambda_final * delta_initial)) * i2c
            + (epsilon * delta * (1.0 - cos_lt) / (lf2 * d2)) * i3s
        )

        coefficient_a[index] = np.sum(integration_weight * (a_real + 1j * a_imag))
        coefficient_b[index] = np.sum(integration_weight * (b_real + 1j * b_imag))
        coefficient_c[index] = np.sum(coefficient_c_weight * (c_real + 1j * c_imag))

    # Mathematica added a tiny imaginary seed before FortranForm; keep the
    # numerical convention so files have the same complex shape near t=0.
    seed = 1.0e-18j
    return coefficient_a + seed, coefficient_b + seed, coefficient_c + seed


def _spin_matrix_exponential(hamiltonian: np.ndarray, value: float) -> np.ndarray:
    return expm(value * hamiltonian)


def _instate_kernel_values(x_values: np.ndarray, params: SimulationParams, quadrature: QuadratureConfig) -> np.ndarray:
    numerics = validate_numerics(quadrature)
    omega, omega_weights = _omega_rule(
        numerics.instate_omega_nodes,
        numerics.omega_min_instate,
        numerics.omega_max_instate,
        spectral=params.spectral,
        s=params.s,
    )
    spectral = _spectral_weight(omega, params)
    result = []
    for x_value in x_values:
        bath = _bathcorr_x(omega, x_value, params)
        result.append(np.sum(omega_weights * spectral * bath))
    return np.asarray(result, dtype=float)


def validate_initial_state(initial_state: np.ndarray, params: SimulationParams) -> np.ndarray:
    J = params.N / 2.0
    dim = int(round(2 * J + 1))
    matrix = np.asarray(initial_state, dtype=complex)
    if matrix.shape != (dim, dim):
        raise ValueError(f"initial_state has shape {matrix.shape}; expected {(dim, dim)} for N={params.N}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("initial_state contains non-finite values.")
    if np.max(np.abs(matrix - matrix.conj().T)) > 1.0e-10:
        raise ValueError("initial_state must be Hermitian.")
    trace = np.trace(matrix)
    if abs(trace - 1.0) > 1.0e-10:
        raise ValueError("initial_state must have trace 1.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) < -1.0e-10:
        raise ValueError("initial_state must be positive semidefinite.")
    return matrix


def _validate_generated_initial_state(initial_state: np.ndarray, params: SimulationParams) -> np.ndarray:
    matrix = np.asarray(initial_state, dtype=complex)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("generated initial_state contains non-finite values.")
    hermiticity_defect = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermiticity_defect > 5.0e-5:
        raise ValueError(
            "generated initial_state is not Hermitian within numerical tolerance "
            f"(defect={hermiticity_defect:.3g}); increase initial-state quadrature controls "
            "or check the requested parameter regime."
        )
    if hermiticity_defect > 1.0e-8:
        warnings.warn(
            "generated initial_state required Hermitian symmetrization after quadrature "
            f"(defect={hermiticity_defect:.3g}); check convergence for this parameter regime.",
            RuntimeWarning,
            stacklevel=3,
        )
    matrix = 0.5 * (matrix + matrix.conj().T)
    trace = np.trace(matrix)
    if abs(trace.imag) > 1.0e-10 or trace.real <= 0.0:
        raise ValueError(
            "generated initial_state has an invalid trace; increase initial-state "
            "quadrature controls or check the requested parameter regime."
        )
    matrix = matrix / trace.real
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -1.0e-10:
        raise ValueError(
            "generated initial_state is not positive semidefinite within numerical tolerance "
            f"(minimum eigenvalue={float(np.min(eigenvalues)):.3g}); increase initial-state "
            "quadrature controls or check the requested parameter regime."
        )
    return matrix


def generate_initial_state(params: SimulationParams, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> np.ndarray:
    if params.initial_state is not None:
        return validate_initial_state(params.initial_state, params)
    quadrature = validate_numerics(quadrature)
    J = params.N / 2.0
    _, jx, jy, jz = spin_operators(J)
    identity = np.eye(jx.shape[0], dtype=complex)
    h0 = params.epsilon0 * jz + params.delta0 * jx
    rotation = expm(1j * np.pi * jy / 2.0)
    rotation_dagger = rotation.conj().T
    rho_s0_run = rotation @ expm(-params.beta * h0) @ rotation_dagger

    lam_nodes, lam_weights = _gauss_interval(quadrature.instate_lambda_nodes, 0.0, params.beta)
    zeta_base, zeta_weights_base = _gauss_interval(quadrature.instate_zeta_nodes, 0.0, 1.0)
    pairs: list[tuple[float, float, float]] = []
    x_values: list[float] = []
    for lam, lam_weight in zip(lam_nodes, lam_weights):
        zeta_nodes = lam * zeta_base
        zeta_weights = lam * zeta_weights_base
        for zeta, zeta_weight in zip(zeta_nodes, zeta_weights):
            pairs.append((lam, zeta, lam_weight * zeta_weight))
            x_values.append(lam - zeta)

    kernel_values = _instate_kernel_values(np.asarray(x_values, dtype=float), params, quadrature)
    fl_cache: dict[float, np.ndarray] = {}

    def fl(value: float) -> np.ndarray:
        if value not in fl_cache:
            propagator = _spin_matrix_exponential(h0, value)
            fl_cache[value] = propagator @ jz @ np.linalg.inv(propagator)
        return fl_cache[value]

    beta_matrix = np.zeros_like(jx, dtype=complex)
    for (lam, zeta, weight), kernel in zip(pairs, kernel_values):
        beta_matrix += (
            params.coupling
            * weight
            * kernel
            * (rotation @ fl(lam) @ fl(zeta) @ rotation_dagger)
        )

    rho_prime = beta_matrix + identity
    rho = rho_s0_run @ rho_prime
    final_state = rho / np.trace(rho)
    return _validate_generated_initial_state(final_state, params)


def generate_inputs(params: SimulationParams, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> GeneratedInputs:
    numerics = validate_numerics(quadrature)
    coefficient_a, coefficient_b, coefficient_c = generate_coefficients(params, numerics)
    bath_correlation_tau = generate_bath_correlation_tau(params, numerics)
    integral_tau = bath_correlation_tau[:, 1] if params.bath == "spin" else bath_correlation_tau[:, 0]
    initial_state = generate_initial_state(params, numerics)
    return GeneratedInputs(
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        coefficient_c=coefficient_c,
        integral_tau=integral_tau,
        bath_correlation_tau=bath_correlation_tau,
        initial_state=initial_state,
        initial_state_source="user_supplied" if params.initial_state is not None else "generated_correlated_equilibrium",
    )


def write_complex_fortran(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(f"({value.real:.16g},{value.imag:.16g})\n")
    return path


def write_real_fortran(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            complex_value = complex(value)
            if abs(complex_value.imag) > 1.0e-10:
                raise ValueError(f"{path.name} expected real values but received {complex_value!r}")
            handle.write(f"{float(complex_value.real):.16g}\n")
    return path


def write_real_table_fortran(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{path.name} expected a two-dimensional table.")
    with path.open("w") as handle:
        for row in array:
            real_row = np.real_if_close(row)
            if np.max(np.abs(np.asarray(real_row).imag)) > 1.0e-10:
                raise ValueError(f"{path.name} expected real values but received complex entries.")
            handle.write(" ".join(f"{float(value):.16g}" for value in np.asarray(real_row).real) + "\n")
    return path


def write_generated_input_files(params: SimulationParams, run_dir: Path, quadrature: QuadratureConfig = DEFAULT_NUMERICS) -> dict[str, Path]:
    generated = generate_inputs(params, quadrature)
    files = {
        "A.dat": write_complex_fortran(run_dir / "A.dat", generated.coefficient_a),
        "B.dat": write_complex_fortran(run_dir / "B.dat", generated.coefficient_b),
        "C.dat": write_complex_fortran(run_dir / "C.dat", generated.coefficient_c),
        "integraldatasimpson.dat": write_real_fortran(run_dir / "integraldatasimpson.dat", generated.integral_tau),
        "bathcorrelation.dat": write_real_table_fortran(run_dir / "bathcorrelation.dat", generated.bath_correlation_tau),
        "INSTATE.dat": write_complex_fortran(run_dir / "INSTATE.dat", generated.initial_state.reshape(-1)),
    }
    return files
