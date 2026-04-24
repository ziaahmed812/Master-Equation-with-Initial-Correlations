"""Exact pure-dephasing reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import loggamma, logsumexp


@dataclass(frozen=True)
class PureDephasingParams:
    J: float
    epsilon: float = 4.0
    xi: float = 4.0
    beta: float = 1.0
    G: float = 0.05
    omega_c: float = 5.0


def _is_half_integer(value: float, *, atol: float = 1.0e-12) -> bool:
    return abs(2.0 * value - round(2.0 * value)) <= atol


def validate_pure_dephasing_params(params: PureDephasingParams) -> None:
    if params.J <= 0 or not _is_half_integer(params.J):
        raise ValueError("J must be a positive integer or half-integer collective-spin quantum number.")
    if params.beta <= 0:
        raise ValueError("beta must be positive.")
    if params.G < 0:
        raise ValueError("G/coupling must be non-negative.")
    if params.omega_c <= 0:
        raise ValueError("omega_c must be positive.")


def _positive_times(times: np.ndarray | float) -> np.ndarray:
    t_arr = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(t_arr <= 0):
        raise ValueError("Exact pure-dephasing formulas require strictly positive times; t=0 is singular.")
    return t_arr


def spin_operators(J: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if J <= 0 or not _is_half_integer(J):
        raise ValueError("J must be a positive integer or half-integer collective-spin quantum number.")
    ms = np.arange(J, -J - 1, -1, dtype=float)
    dim = len(ms)
    index = {m: i for i, m in enumerate(ms)}

    jz = np.diag(ms.astype(complex))
    jp = np.zeros((dim, dim), dtype=complex)
    jm = np.zeros((dim, dim), dtype=complex)

    for m in ms:
        if m + 1 <= J:
            jp[index[m + 1], index[m]] = np.sqrt(J * (J + 1) - m * (m + 1))
        if m - 1 >= -J:
            jm[index[m - 1], index[m]] = np.sqrt(J * (J + 1) - m * (m - 1))

    jx = 0.5 * (jp + jm)
    jy = (jp - jm) / (2j)
    return ms, jx, jy, jz


def phi_ohmic(t: np.ndarray | float, *, G: float, omega_c: float) -> np.ndarray:
    return G * np.arctan(omega_c * np.asarray(t))


def delta_shift(t: np.ndarray | float, *, G: float, omega_c: float) -> np.ndarray:
    t_arr = _positive_times(t)
    C = G * omega_c
    return (phi_ohmic(t_arr, G=G, omega_c=omega_c) - C * t_arr) / t_arr


def gamma_ohmic(t: np.ndarray | float, *, beta: float, G: float, omega_c: float) -> np.ndarray:
    t_arr = _positive_times(t)
    a = 1.0 + 1.0 / (omega_c * beta)
    gamma_vac = (G / (2.0 * t_arr)) * np.log1p((omega_c * t_arr) ** 2)
    gamma_th = ((2.0 * G) / t_arr) * (loggamma(a).real - loggamma(a + 1j * t_arr / beta).real)
    return gamma_vac + gamma_th


def rotated_thermal_state(params: PureDephasingParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_pure_dephasing_params(params)
    ms, _, jy, _ = spin_operators(params.J)
    rotation = expm(1j * np.pi * jy / 2.0)
    C = params.G * params.omega_c
    log_weights = -params.beta * params.xi * ms + params.beta * (ms**2) * C
    weights = np.exp(log_weights - logsumexp(log_weights))
    rho_s0 = rotation @ np.diag(weights) @ rotation.conj().T / weights.sum()
    return ms, rotation, rho_s0


def exact_density_matrices(
    times: np.ndarray | float,
    params: PureDephasingParams,
) -> tuple[np.ndarray, np.ndarray]:
    validate_pure_dephasing_params(params)
    t_arr = _positive_times(times)
    ms, rotation, rho_s0 = rotated_thermal_state(params)
    C = params.G * params.omega_c
    log_weights = -params.beta * params.xi * ms + params.beta * (ms**2) * C
    weights = np.exp(log_weights - logsumexp(log_weights))
    M, N = np.meshgrid(ms, ms, indexing="ij")

    correlated = []
    uncorrelated = []
    for t_val in t_arr:
        phase = phi_ohmic(t_val, G=params.G, omega_c=params.omega_c)
        delta = delta_shift(t_val, G=params.G, omega_c=params.omega_c)
        gamma = gamma_ohmic(t_val, beta=params.beta, G=params.G, omega_c=params.omega_c)

        AA = (
            np.exp(-1j * params.epsilon * (M - N) * t_val)
            * np.exp(-1j * delta * (M**2 - N**2) * t_val)
            * np.exp(-gamma * (M - N) ** 2 * t_val)
        )

        numerator = np.zeros_like(rho_s0, dtype=complex)
        denominator = np.zeros_like(rho_s0, dtype=complex)
        for col, l in enumerate(ms):
            outer = np.outer(rotation[:, col], rotation[:, col].conj()) * weights[col]
            denominator += outer
            numerator += outer * np.exp(-1j * 2.0 * (N - M) * l * phase)

        BB = np.divide(numerator, denominator, out=np.ones_like(numerator), where=np.abs(denominator) > 1.0e-15)
        correlated.append(rho_s0 * AA * BB)
        uncorrelated.append(rho_s0 * AA)

    return np.asarray(correlated), np.asarray(uncorrelated)


def exact_curves(
    params: PureDephasingParams,
    *,
    correlated_times: np.ndarray | None = None,
    uncorrelated_times: np.ndarray | None = None,
    observable: str | np.ndarray = "jx",
) -> tuple[np.ndarray, np.ndarray]:
    if correlated_times is None:
        correlated_times = np.arange(1.0e-10, 5.0, 0.2)
    if uncorrelated_times is None:
        uncorrelated_times = np.arange(0.1, 5.0, 0.2)

    validate_pure_dephasing_params(params)
    from .observables import expectation_from_density_matrices

    rho_c, _ = exact_density_matrices(correlated_times, params)
    _, rho_u = exact_density_matrices(uncorrelated_times, params)

    corr_values = expectation_from_density_matrices(rho_c, observable, params.J)
    unc_values = expectation_from_density_matrices(rho_u, observable, params.J)
    if np.iscomplexobj(corr_values) or np.iscomplexobj(unc_values):
        raise ValueError("exact pure-dephasing output currently requires a real observable expectation.")
    corr = np.asarray(corr_values, dtype=float)
    unc = np.asarray(unc_values, dtype=float)

    return (
        np.column_stack([np.asarray(correlated_times, dtype=float), corr]),
        np.column_stack([np.asarray(uncorrelated_times, dtype=float), unc]),
    )
