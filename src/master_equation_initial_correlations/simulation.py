from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._resources import ensure_clean_dir, load_table, write_json
from ._types import ReferenceExample, SimulationParams, SimulationResult
from .catalog import list_examples
from .fortran_runner import rerun_example
from .pure_dephasing import PureDephasingParams, exact_curves


def _clean_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _close(a: float, b: float, *, atol: float = 1.0e-12) -> bool:
    return abs(float(a) - float(b)) <= atol


def _normalize_params(params: SimulationParams) -> SimulationParams:
    bath = _clean_token(params.bath)
    model = _clean_token(params.model)
    spectral = _clean_token(params.spectral)
    observable = _clean_token(params.observable)

    if bath not in {"bosonic", "spin"}:
        raise ValueError("bath must be either 'bosonic' or 'spin'")
    if model not in {"pure-dephasing", "spin-boson", "spin-environment"}:
        raise ValueError("model must be 'pure-dephasing', 'spin-boson', or 'spin-environment'")
    if bath == "spin" and model != "spin-environment":
        raise ValueError("spin bath runs use model='spin-environment'")
    if bath == "bosonic" and model == "spin-environment":
        raise ValueError("spin-environment is a spin bath model; use bath='spin'")
    if spectral not in {"ohmic", "subohmic"}:
        raise ValueError("spectral must be 'ohmic' or 'subohmic'")
    if observable not in {"jx", "jx2"}:
        raise ValueError("observable must be 'jx' or 'jx2'")
    if params.N <= 0:
        raise ValueError("N must be a positive integer for these collective-spin workflows")
    if spectral == "subohmic" and params.s is None:
        raise ValueError("sub-Ohmic runs require --s")

    return SimulationParams(
        bath=bath,
        model=model,
        N=params.N,
        epsilon0=params.epsilon0,
        epsilon=params.epsilon,
        delta0=params.delta0,
        delta=params.delta,
        beta=params.beta,
        coupling=params.coupling,
        omega_c=params.omega_c,
        spectral=spectral,
        observable=observable,
        s=params.s,
    )


def _example_matches(example: ReferenceExample, params: SimulationParams) -> bool:
    raw = example.parameters
    if example.bath != params.bath or example.model != params.model or example.spectral != params.spectral:
        return False
    if example.model == "spin-boson" and example.observable != params.observable:
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
        and (normalized.model != "spin-boson" or example.observable == normalized.observable)
    ]
    return ", ".join(matches) if matches else "none"


def _run_exact_pure_dephasing(params: SimulationParams, output_dir: Path, *, t_max: float, dt: float) -> SimulationResult:
    output_dir = ensure_clean_dir(output_dir)
    J = params.N / 2.0
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
    correlated, uncorrelated = exact_curves(
        pure_params,
        correlated_times=correlated_times,
        uncorrelated_times=uncorrelated_times,
    )
    corr_path = output_dir / "exact-correlated.dat"
    unc_path = output_dir / "exact-uncorrelated.dat"
    np.savetxt(corr_path, correlated, fmt="%.16e")
    np.savetxt(unc_path, uncorrelated, fmt="%.16e")

    example = find_reference_example(params)
    exact_correlated_error = None
    exact_uncorrelated_error = None
    if example is not None:
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
            "parameters": asdict(params),
            "J": J,
            "reference_example": example.public_id if example else None,
            "exact_correlated_error": exact_correlated_error,
            "exact_uncorrelated_error": exact_uncorrelated_error,
            "correlated_path": str(corr_path),
            "uncorrelated_path": str(unc_path),
        },
    )
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
    )


def run_simulation(
    params: SimulationParams,
    output_dir: str | Path,
    *,
    plot: bool = False,
    verify: bool = True,
    t_max: float = 5.0,
    dt: float = 0.2,
) -> SimulationResult:
    params = _normalize_params(params)
    output_dir = Path(output_dir)

    if params.bath == "bosonic" and params.model == "pure-dephasing":
        if params.spectral != "ohmic":
            raise ValueError("The exact pure-dephasing solver currently supports the Ohmic bosonic bath.")
        if params.observable != "jx":
            raise ValueError("The exact pure-dephasing solver currently returns the jx observable.")
        return _run_exact_pure_dephasing(params, output_dir, t_max=t_max, dt=dt)

    example = find_reference_example(params)
    if example is None:
        raise ValueError(
            "No packaged coefficient branch matches these parameters yet. "
            f"Available examples for this family: {_available_examples(params)}."
        )

    rerun = rerun_example(example.id, output_dir, verify=verify, render=plot)
    return SimulationResult(
        params=params,
        output_dir=rerun.output_dir,
        source="legacy Fortran solver",
        example=example,
        correlated_path=rerun.output_dir / "runs" / "correlated" / "EXPX-C.DAT",
        uncorrelated_path=rerun.output_dir / "runs" / "uncorrelated" / "EXPX-UNC.DAT",
        correlated_error=rerun.correlated_error,
        uncorrelated_error=rerun.uncorrelated_error,
        exact_correlated_error=rerun.exact_correlated_error,
        exact_uncorrelated_error=rerun.exact_uncorrelated_error,
        rendered_eps=rerun.rendered_eps,
        rendered_png=rerun.rendered_png,
        summary_path=rerun.summary_path,
    )
