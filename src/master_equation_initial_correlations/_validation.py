from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ._types import BathParams, SimulationParams, SystemParams
from .observables import ObservableParseError, normalize_observable_expression, parse_observable


def clean_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def close(a: float, b: float, *, atol: float = 1.0e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


def positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer_value = int(value)
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer_value <= 0 or float_value != float(integer_value):
        raise ValueError(f"{name} must be a positive integer.")
    return integer_value


def normalize_bath_type(bath_type: str | None) -> str:
    value = clean_token(bath_type or "bosonic")
    aliases = {
        "boson": "bosonic",
        "bosonic-bath": "bosonic",
        "spin-bath": "spin",
        "spin-environment": "spin",
    }
    value = aliases.get(value, value)
    if value not in {"bosonic", "spin"}:
        raise ValueError("bath_type must be either 'bosonic' or 'spin'.")
    return value


def normalize_spectral(kind: str) -> str:
    value = clean_token(kind)
    aliases = {
        "sub-ohmic": "subohmic",
        "super-ohmic": "superohmic",
    }
    value = aliases.get(value, value)
    if value not in {"ohmic", "subohmic", "superohmic"}:
        raise ValueError("spectral kind must be 'ohmic', 'subohmic', or 'superohmic'.")
    return value


def normalize_model(model: str | None, *, bath_type: str, exact: bool = False) -> str:
    if model is None or clean_token(model) == "auto":
        if exact:
            return "pure-dephasing"
        return "spin-environment" if bath_type == "spin" else "spin-boson"
    value = clean_token(model)
    if value not in {"pure-dephasing", "spin-boson", "spin-environment"}:
        raise ValueError("model must be 'pure-dephasing', 'spin-boson', 'spin-environment', or 'auto'.")
    return value


def normalize_correlations(correlations: str) -> str:
    value = clean_token(correlations)
    aliases = {
        "wc": "with_correlations",
        "with": "with_correlations",
        "with-correlations": "with_correlations",
        "with-correlated": "with_correlations",
        "correlated": "with_correlations",
        "woc": "without_correlations",
        "without": "without_correlations",
        "without-correlations": "without_correlations",
        "uncorrelated": "without_correlations",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"with_correlations", "without_correlations"}:
        raise ValueError("correlations must be 'with'/'wc' or 'without'/'woc'.")
    return normalized


def validate_system_params(system: SystemParams) -> SystemParams:
    N = positive_integer(system.N, name="N")
    for name in ("epsilon0", "epsilon", "delta0", "delta"):
        value = float(getattr(system, name))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    return replace(
        system,
        N=N,
        epsilon0=float(system.epsilon0),
        epsilon=float(system.epsilon),
        delta0=float(system.delta0),
        delta=float(system.delta),
    )


def validate_bath_params(bath: BathParams, *, bath_type: str | None = None) -> BathParams:
    normalized_bath_type = normalize_bath_type(bath_type if bath_type is not None else bath.bath_type)
    spectral = normalize_spectral(bath.kind)
    beta = float(bath.beta)
    coupling = float(bath.coupling)
    omega_c = float(bath.omega_c)
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite.")
    if coupling < 0 or not np.isfinite(coupling):
        raise ValueError("coupling must be non-negative and finite.")
    if omega_c <= 0 or not np.isfinite(omega_c):
        raise ValueError("omega_c must be positive and finite.")
    s = None if bath.s is None else float(bath.s)
    if s is not None and not np.isfinite(s):
        raise ValueError("spectral exponent s must be finite.")
    if spectral == "ohmic":
        if s is not None and not close(s, 1.0):
            raise ValueError("Ohmic spectra use s=1.0.")
        s = 1.0
    elif spectral == "subohmic":
        if s is None:
            raise ValueError("sub-Ohmic spectra require s with 0 < s < 1.")
        if not (0.0 < s < 1.0):
            raise ValueError("sub-Ohmic spectra require 0 < s < 1.")
    elif spectral == "superohmic":
        if s is None:
            raise ValueError("super-Ohmic spectra require s > 1.")
        if s <= 1.0:
            raise ValueError("super-Ohmic spectra require s > 1.")
    return BathParams(
        bath_type=normalized_bath_type,
        kind=spectral,
        beta=beta,
        coupling=coupling,
        omega_c=omega_c,
        s=s,
    )


def validate_observable(observable: str | np.ndarray, *, N: int) -> str | np.ndarray:
    if isinstance(observable, np.ndarray):
        parsed = observable
    else:
        parsed = normalize_observable_expression(str(observable))
    try:
        parse_observable(parsed, N / 2.0)
    except ObservableParseError as exc:
        raise ValueError(str(exc)) from exc
    return parsed


def validate_model_compatibility(*, system: SystemParams, bath: BathParams, model: str, exact: bool = False) -> None:
    if bath.bath_type == "spin" and model != "spin-environment":
        raise ValueError("spin bath runs use model='spin-environment'.")
    if bath.bath_type == "bosonic" and model == "spin-environment":
        raise ValueError("spin-environment is a spin-bath model; use bath_type='spin'.")
    if exact and (bath.bath_type != "bosonic" or model != "pure-dephasing" or bath.kind != "ohmic"):
        raise ValueError("the exact analytical solver is implemented for bosonic Ohmic pure dephasing only.")
    if model == "pure-dephasing":
        if not close(system.delta0, 0.0) or not close(system.delta, 0.0):
            raise ValueError("pure-dephasing runs require delta0=0 and delta=0.")
    else:
        min_splitting = 1.0e-10
        if float(np.hypot(system.epsilon, system.delta)) <= min_splitting:
            raise ValueError("master-equation runs require a final system splitting larger than 1e-10.")
        if float(np.hypot(system.epsilon0, system.delta0)) <= min_splitting:
            raise ValueError("master-equation runs require an initial system splitting larger than 1e-10.")


def simulation_params_from_public(
    *,
    system: SystemParams,
    bath: BathParams,
    observable: str | np.ndarray,
    initial_state: np.ndarray | None = None,
    model: str | None = "auto",
    exact: bool = False,
) -> SimulationParams:
    normalized_system = validate_system_params(system)
    normalized_bath = validate_bath_params(bath)
    normalized_model = normalize_model(model, bath_type=normalized_bath.bath_type, exact=exact)
    validate_model_compatibility(system=normalized_system, bath=normalized_bath, model=normalized_model, exact=exact)
    normalized_observable = validate_observable(observable, N=normalized_system.N)
    return SimulationParams(
        bath=normalized_bath.bath_type,
        model=normalized_model,
        spectral=normalized_bath.kind,
        observable=normalized_observable,
        N=normalized_system.N,
        epsilon0=normalized_system.epsilon0,
        epsilon=normalized_system.epsilon,
        delta0=normalized_system.delta0,
        delta=normalized_system.delta,
        beta=normalized_bath.beta,
        coupling=normalized_bath.coupling,
        omega_c=normalized_bath.omega_c,
        s=normalized_bath.s,
        initial_state=initial_state,
    )


def normalize_simulation_params(params: SimulationParams) -> SimulationParams:
    system = SystemParams(
        N=params.N,
        epsilon0=params.epsilon0,
        epsilon=params.epsilon,
        delta0=params.delta0,
        delta=params.delta,
    )
    bath = BathParams(
        bath_type=params.bath,
        kind=params.spectral,
        beta=params.beta,
        coupling=params.coupling,
        omega_c=params.omega_c,
        s=params.s,
    )
    return simulation_params_from_public(
        system=system,
        bath=bath,
        observable=params.observable,
        initial_state=params.initial_state,
        model=params.model,
    )


def observable_is_hermitian(observable: str | np.ndarray, *, N: int) -> bool:
    parsed = parse_observable(validate_observable(observable, N=N), N / 2.0)
    return bool(np.allclose(parsed.dimensionless_matrix, parsed.dimensionless_matrix.conj().T, atol=1.0e-10))
