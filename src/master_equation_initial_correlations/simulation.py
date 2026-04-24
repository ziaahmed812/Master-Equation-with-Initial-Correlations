from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np

from ._resources import load_table, prepare_output_dir, write_json, write_table
from ._types import NumericsConfig, ReferenceExample, SimulationParams, SimulationResult
from .catalog import list_examples
from .fortran_runner import run_parameterized_fortran
from .generated_inputs import validate_initial_state, validate_numerics
from .observables import ObservableParseError, normalize_observable_expression, parse_observable
from .pure_dephasing import PureDephasingParams, exact_curves


EXACT_OUTPUTS = ("exact-correlated.dat", "exact-uncorrelated.dat", "simulation_summary.json")


def _clean_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _close(a: float, b: float, *, atol: float = 1.0e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


def _positive_integer(value: Any, *, name: str) -> int:
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


def _normalize_params(params: SimulationParams) -> SimulationParams:
    bath = _clean_token(params.bath)
    model = _clean_token(params.model)
    spectral = _clean_token(params.spectral)
    observable = params.observable
    if not isinstance(observable, np.ndarray):
        observable = normalize_observable_expression(str(observable))
    N = _positive_integer(params.N, name="N")
    if spectral == "sub-ohmic":
        spectral = "subohmic"
    if spectral == "super-ohmic":
        spectral = "superohmic"

    if bath not in {"bosonic", "spin"}:
        raise ValueError("bath must be either 'bosonic' or 'spin'")
    if model not in {"pure-dephasing", "spin-boson", "spin-environment"}:
        raise ValueError("model must be 'pure-dephasing', 'spin-boson', or 'spin-environment'")
    if bath == "spin" and model != "spin-environment":
        raise ValueError("spin bath runs use model='spin-environment'")
    if bath == "bosonic" and model == "spin-environment":
        raise ValueError("spin-environment is a spin bath model; use bath='spin'")
    if spectral not in {"ohmic", "subohmic", "superohmic"}:
        raise ValueError("spectral must be 'ohmic', 'subohmic', or 'superohmic'")
    try:
        parse_observable(observable, N / 2.0)
    except ObservableParseError as exc:
        raise ValueError(str(exc)) from exc
    if model == "pure-dephasing":
        if not _close(params.delta0, 0.0) or not _close(params.delta, 0.0):
            raise ValueError("pure-dephasing runs require delta0=0 and delta=0")
    if model != "pure-dephasing":
        if float(np.hypot(params.epsilon, params.delta)) <= 0.0:
            raise ValueError("non-pure-dephasing runs require a nonzero final system splitting")
        if float(np.hypot(params.epsilon0, params.delta0)) <= 0.0:
            raise ValueError("non-pure-dephasing runs require a nonzero initial system splitting")
    if params.beta <= 0:
        raise ValueError("beta must be positive")
    if params.coupling < 0:
        raise ValueError("coupling must be non-negative")
    if params.omega_c <= 0:
        raise ValueError("omega_c must be positive")
    if spectral == "ohmic" and params.s is not None and not _close(params.s, 1.0):
        raise ValueError("Ohmic spectra use s=1; omit s or set s=1.")
    if spectral == "subohmic":
        if params.s is None:
            raise ValueError("sub-Ohmic spectra require s with 0 < s < 1.")
        if not (0.0 < params.s < 1.0):
            raise ValueError("sub-Ohmic spectra require 0 < s < 1.")
    if spectral == "superohmic":
        if params.s is None:
            raise ValueError("super-Ohmic spectra require s > 1.")
        if params.s <= 1.0:
            raise ValueError("super-Ohmic spectra require s > 1.")

    return SimulationParams(
        bath=bath,
        model=model,
        N=N,
        epsilon0=params.epsilon0,
        epsilon=params.epsilon,
        delta0=params.delta0,
        delta=params.delta,
        beta=params.beta,
        coupling=params.coupling,
        omega_c=params.omega_c,
        spectral=spectral,
        observable=observable,
        s=1.0 if spectral == "ohmic" else params.s,
        initial_state=params.initial_state,
    )


def _example_matches(example: ReferenceExample, params: SimulationParams) -> bool:
    raw = example.parameters
    if params.initial_state is not None:
        return False
    if example.bath != params.bath or example.model != params.model or example.spectral != params.spectral:
        return False
    example_observable = normalize_observable_expression(example.observable)
    if isinstance(params.observable, np.ndarray) or example_observable != params.observable:
        return False
    checks = (
        (raw["N"], params.N),
        (raw["epsilon_0"], params.epsilon0),
        (raw["epsilon"], params.epsilon),
        (raw["Delta_0"], params.delta0),
        (raw["Delta"], params.delta),
        (raw["beta"], params.beta),
        (raw["G"], params.coupling),
        (raw["omega_c"], params.omega_c),
    )
    if not all(_close(a, b) for a, b in checks):
        return False
    if params.spectral == "subohmic" and not _close(raw.get("s"), params.s):
        return False
    return True


def find_reference_example(params: SimulationParams) -> ReferenceExample | None:
    normalized = _normalize_params(params)
    for example in list_examples():
        if _example_matches(example, normalized):
            return example
    return None


def _available_examples(params: SimulationParams) -> str:
    normalized = _normalize_params(params)
    matches = [
        example.public_id
        for example in list_examples()
        if example.bath == normalized.bath
        and example.model == normalized.model
        and example.spectral == normalized.spectral
        and normalize_observable_expression(example.observable) == normalized.observable
    ]
    return ", ".join(matches) if matches else "none"


def _header_lines(params: SimulationParams, *, source: str, columns: str) -> list[str]:
    return [
        "Generated by master-equation-with-initial-correlations",
        f"source: {source}",
        f"bath: {params.bath}",
        f"model: {params.model}",
        f"spectral: {params.spectral}",
        f"observable: {normalize_observable_expression(params.observable)}",
        f"N: {params.N}",
        f"J: {params.N / 2.0}",
        f"epsilon0: {params.epsilon0}",
        f"epsilon: {params.epsilon}",
        f"delta0: {params.delta0}",
        f"delta: {params.delta}",
        f"beta: {params.beta}",
        f"coupling: {params.coupling}",
        f"omega_c: {params.omega_c}",
        f"s: {params.s}",
        f"initial_state_source: {'user_supplied' if params.initial_state is not None else 'solver_default'}",
        f"columns: {columns}",
    ]


def _emit(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _serializable_params(params: SimulationParams) -> dict[str, object]:
    payload = asdict(params)
    payload["observable"] = normalize_observable_expression(params.observable)
    payload["initial_state"] = (
        None
        if params.initial_state is None
        else {"source": "user_supplied", "shape": list(np.asarray(params.initial_state).shape)}
    )
    return payload


def _observable_output_table(table: np.ndarray, observable: str, branch: str) -> tuple[np.ndarray, str]:
    array = np.asarray(table)
    if np.iscomplexobj(array) and np.max(np.abs(array[:, 1].imag)) > 1.0e-10:
        return (
            np.column_stack([array[:, 0].real, array[:, 1].real, array[:, 1].imag]),
            f"t {observable}_{branch}_real {observable}_{branch}_imag",
        )
    return np.asarray(array.real, dtype=float), f"t {observable}_{branch}"


def _warn_if_nonhermitian_observable(params: SimulationParams) -> None:
    spec = parse_observable(params.observable, params.N / 2.0)
    if np.max(np.abs(spec.dimensionless_matrix - spec.dimensionless_matrix.conj().T)) > 1.0e-10:
        warnings.warn(
            "observable is non-Hermitian; expectation values may be complex and output tables will keep real and imaginary columns when needed.",
            RuntimeWarning,
            stacklevel=3,
        )


def _run_exact_pure_dephasing(
    params: SimulationParams,
    output_dir: Path,
    *,
    t_max: float,
    dt: float,
    verify: bool,
    overwrite: bool,
    verbose: bool,
) -> SimulationResult:
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite, generated_names=EXACT_OUTPUTS)
    J = params.N / 2.0
    _emit(f"[meic] validating exact pure-dephasing parameters (N={params.N}, J={J})", verbose=verbose)
    pure_params = PureDephasingParams(
        J=J,
        epsilon=params.epsilon,
        xi=params.epsilon0,
        beta=params.beta,
        G=params.coupling,
        omega_c=params.omega_c,
    )
    correlated_times = np.arange(1.0e-10, t_max, dt)
    uncorrelated_times = np.arange(dt / 2.0, t_max, dt)
    _emit("[meic] evaluating exact correlated and uncorrelated curves", verbose=verbose)
    correlated, uncorrelated = exact_curves(
        pure_params,
        correlated_times=correlated_times,
        uncorrelated_times=uncorrelated_times,
        observable=params.observable,
    )
    corr_path = output_dir / "exact-correlated.dat"
    unc_path = output_dir / "exact-uncorrelated.dat"
    observable = normalize_observable_expression(params.observable)
    correlated_table, correlated_columns = _observable_output_table(correlated, observable, "correlated")
    uncorrelated_table, uncorrelated_columns = _observable_output_table(uncorrelated, observable, "uncorrelated")
    write_table(
        corr_path,
        correlated_table,
        header_lines=_header_lines(params, source="exact pure-dephasing Python solver", columns=correlated_columns),
    )
    write_table(
        unc_path,
        uncorrelated_table,
        header_lines=_header_lines(params, source="exact pure-dephasing Python solver", columns=uncorrelated_columns),
    )
    _emit(f"[meic] saved {corr_path}", verbose=verbose)
    _emit(f"[meic] saved {unc_path}", verbose=verbose)

    example = find_reference_example(params)
    exact_correlated_error = None
    exact_uncorrelated_error = None
    verification_performed = verify and example is not None
    if verification_performed:
        try:
            reference_correlated = load_table(f"{example.asset_dir}/tables/exact-correlated.dat")
            reference_uncorrelated = load_table(f"{example.asset_dir}/tables/exact-uncorrelated.dat")
            if reference_correlated.shape == correlated.shape:
                exact_correlated_error = float(np.max(np.abs(correlated - reference_correlated)))
            if reference_uncorrelated.shape == uncorrelated.shape:
                exact_uncorrelated_error = float(np.max(np.abs(uncorrelated - reference_uncorrelated)))
        except FileNotFoundError:
            pass

    summary_path = output_dir / "simulation_summary.json"
    write_json(
        summary_path,
        {
            "source": "exact pure-dephasing Python solver",
            "parameters": _serializable_params(params),
            "J": J,
            "reference_example": example.public_id if example else None,
            "exact_correlated_error": exact_correlated_error,
            "exact_uncorrelated_error": exact_uncorrelated_error,
            "correlated_path": str(corr_path),
            "uncorrelated_path": str(unc_path),
            "verification_performed": verification_performed,
        },
    )
    _emit(f"[meic] saved {summary_path}", verbose=verbose)
    return SimulationResult(
        params=params,
        output_dir=output_dir,
        source="exact pure-dephasing Python solver",
        example=example,
        correlated_path=corr_path,
        uncorrelated_path=unc_path,
        exact_correlated_error=exact_correlated_error,
        exact_uncorrelated_error=exact_uncorrelated_error,
        summary_path=summary_path,
        verification_performed=verification_performed,
        output_files={"exact_correlated": corr_path, "exact_uncorrelated": unc_path},
    )


def run_simulation(
    params: SimulationParams,
    output_dir: str | Path | None = None,
    *,
    plot: bool = False,
    verify: bool = True,
    t_max: float = 5.0,
    dt: float = 0.2,
    overwrite: bool = False,
    verbose: bool = True,
    save_density: bool = False,
    numerics: NumericsConfig | None = None,
) -> SimulationResult:
    params = _normalize_params(params)
    _warn_if_nonhermitian_observable(params)
    output_dir = Path.cwd() if output_dir is None else Path(output_dir)

    if params.bath == "bosonic" and params.model == "pure-dephasing":
        if params.initial_state is not None:
            raise ValueError("The exact pure-dephasing solver uses its analytical correlated/uncorrelated state construction and does not accept initial_state.")
        if params.spectral != "ohmic":
            raise ValueError("The closed-form exact pure-dephasing solver is defined for the Ohmic bosonic bath.")
        return _run_exact_pure_dephasing(
            params,
            output_dir,
            t_max=t_max,
            dt=dt,
            verify=verify,
            overwrite=overwrite,
            verbose=verbose,
        )

    numerics = validate_numerics(numerics)
    if params.initial_state is not None:
        validate_initial_state(params.initial_state, params)
    example = find_reference_example(params)
    if example is None:
        _emit("[meic] no bundled reference branch exactly matches these parameters; running without reference verification", verbose=verbose)
    else:
        _emit(f"[meic] matched bundled reference {example.public_id}; generated run will be numerically compared to it", verbose=verbose)

    _emit("[meic] generating A/B/C coefficients, bath integral table, and initial state from parameters", verbose=verbose)
    _emit("[meic] staging parameterized preserved Fortran solver", verbose=verbose)
    rerun = run_parameterized_fortran(
        params,
        output_dir,
        example=example,
        verify=verify,
        render=plot,
        overwrite=overwrite,
        verbose=verbose,
        save_density=save_density,
        numerics=numerics,
    )
    return SimulationResult(
        params=params,
        output_dir=rerun.output_dir,
        source="generated inputs + preserved Fortran solver",
        example=example,
        correlated_path=rerun.output_dir / "observable-correlated.dat",
        uncorrelated_path=rerun.output_dir / "observable-uncorrelated.dat",
        correlated_error=rerun.correlated_error,
        uncorrelated_error=rerun.uncorrelated_error,
        exact_correlated_error=rerun.exact_correlated_error,
        exact_uncorrelated_error=rerun.exact_uncorrelated_error,
        rendered_eps=rerun.rendered_eps,
        rendered_png=rerun.rendered_png,
        summary_path=rerun.summary_path,
        verification_performed=rerun.verification_performed,
        output_files=rerun.output_files,
        input_files=rerun.input_files,
        source_files=rerun.source_files,
        log_files=rerun.log_files,
    )
