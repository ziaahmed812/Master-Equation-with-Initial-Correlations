from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import tempfile
from typing import Any, ClassVar
import warnings

import numpy as np

from ._types import BathParams, NumericsConfig, SimulationParams, SystemParams
from .fortran_runner import run_parameterized_fortran_branch
from .generated_inputs import validate_numerics
from .observables import expectation_from_density_matrices, parse_observable
from .pure_dephasing import PureDephasingParams, exact_density_matrices, rotated_thermal_state
from .result import Result, observable_label
from ._validation import (
    normalize_correlations,
    observable_is_hermitian,
    simulation_params_from_public,
    validate_bath_params,
    validate_model_compatibility,
    validate_system_params,
)


Branch = str


def _validate_tlist(tlist: Any) -> tuple[np.ndarray, float, float]:
    times = np.asarray(tlist, dtype=float)
    if times.ndim != 1:
        raise ValueError("tlist must be a one-dimensional array of times.")
    if times.size < 2:
        raise ValueError("tlist must contain at least two time points.")
    if not np.all(np.isfinite(times)):
        raise ValueError("tlist must contain only finite values.")
    if abs(times[0]) > 1.0e-12:
        raise ValueError("tlist must start at 0.0 for the Fortran-backed master-equation solver.")
    steps = np.diff(times)
    if np.any(steps <= 0):
        raise ValueError("tlist must be strictly increasing.")
    dt = float(steps[0])
    if not np.allclose(steps, dt, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("tlist must be uniformly spaced in v1; use np.linspace or np.arange from 0.0.")
    return times, dt, float(times[-1])


def _numerics_from_tlist(numerics: NumericsConfig | None, *, dt: float, t_final: float) -> NumericsConfig:
    base = validate_numerics(numerics)
    # The Fortran loop writes while T < TFINAL, so use one internal
    # sentinel step beyond the requested public endpoint.
    internal_t_final = t_final + dt
    return validate_numerics(
        replace(
            base,
            coefficient_t_max=internal_t_final + 1.0e-14,
            correlation_tau_max=internal_t_final + 1.0e-14,
            fortran_dt=dt,
            fortran_t_final=internal_t_final,
        )
    )


def _normalize_e_ops(e_ops: Any) -> tuple[list[Any], list[str], dict[str, int]]:
    if e_ops is None:
        observables = ["jx"]
        labels = ["jx"]
    elif isinstance(e_ops, dict):
        observables = list(e_ops.values())
        labels = [str(key) for key in e_ops]
    elif isinstance(e_ops, (str, np.ndarray)):
        observables = [e_ops]
        labels = [observable_label(e_ops, "observable_0")]
    else:
        observables = list(e_ops)
        labels = [observable_label(observable, f"observable_{index}") for index, observable in enumerate(observables)]

    seen: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique_labels.append(label if count == 0 else f"{label}_{count + 1}")
    return observables, unique_labels, {label: index for index, label in enumerate(unique_labels)}


def _read_expectation_table(path: Path, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=float, comments="#")
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[0] != times.size or not np.allclose(table[:, 0], times, rtol=1.0e-8, atol=1.0e-10):
        raise RuntimeError(
            f"solver returned times that do not match tlist for {path}; "
            "choose a uniform tlist compatible with the Fortran time step."
        )
    if table.shape[1] == 2:
        return table[:, 0], table[:, 1]
    if table.shape[1] == 3:
        values = table[:, 1] + 1j * table[:, 2]
        if np.max(np.abs(values.imag)) <= 1.0e-10:
            return table[:, 0], values.real
        return table[:, 0], values
    raise RuntimeError(f"unexpected observable table shape {table.shape} in {path}.")


def _simulation_params(
    *,
    bath_type: str,
    model: str,
    system: SystemParams,
    bath: BathParams,
    observable: str | np.ndarray,
    initial_state: np.ndarray | None,
) -> SimulationParams:
    return simulation_params_from_public(
        model=model,
        system=system,
        bath=replace(bath, bath_type=bath_type),
        observable=observable,
        initial_state=initial_state,
    )


def _warn_for_non_hermitian_observable(observable: str | np.ndarray, *, system: SystemParams, label: str) -> None:
    if not observable_is_hermitian(observable, N=system.N):
        warnings.warn(
            f"observable {label!r} is not Hermitian; expectation values may be complex.",
            RuntimeWarning,
            stacklevel=3,
        )


def _pure_dephasing_params(system: SystemParams, bath: BathParams) -> PureDephasingParams:
    params = simulation_params_from_public(
        system=system,
        bath=bath,
        observable="jx",
        model="pure-dephasing",
        exact=True,
    )
    return PureDephasingParams(
        J=params.N / 2.0,
        epsilon=params.epsilon,
        xi=params.epsilon0,
        beta=params.beta,
        G=params.coupling,
        omega_c=params.omega_c,
    )


def _exact_states_for_branch(times: np.ndarray, params: PureDephasingParams, branch: Branch) -> np.ndarray:
    _, _, rho0 = rotated_thermal_state(params)
    states = np.empty((times.size, *rho0.shape), dtype=complex)
    zero_mask = np.isclose(times, 0.0, atol=1.0e-14)
    states[zero_mask] = rho0
    positive_mask = ~zero_mask
    if np.any(positive_mask):
        correlated, uncorrelated = exact_density_matrices(times[positive_mask], params)
        states[positive_mask] = correlated if branch == "with_correlations" else uncorrelated
    return states


@dataclass(frozen=True)
class _BasePureDephasingBranchSolver:
    system: SystemParams
    bath: BathParams = field(default_factory=lambda: BathParams(bath_type="bosonic", kind="ohmic", s=1.0))
    observable: str | np.ndarray = "jx"
    branch: ClassVar[Branch] = "with_correlations"

    def run(self, tlist: Any, e_ops: Any = None, *, store_states: bool = False, verbose: bool = True) -> Result:
        times, _, _ = _validate_tlist(tlist)
        observables, labels, label_index = _normalize_e_ops(self.observable if e_ops is None else e_ops)
        system = validate_system_params(self.system)
        bath = validate_bath_params(self.bath)
        params = _pure_dephasing_params(system, bath)
        for observable, label in zip(observables, labels):
            _warn_for_non_hermitian_observable(observable, system=system, label=label)
        states = _exact_states_for_branch(times, params, self.branch)
        expect = [
            expectation_from_density_matrices(states, observable, params.J)
            for observable in observables
        ]
        e_data = {label: expect[index] for label, index in label_index.items()}
        return Result(
            times=times,
            expect=expect,
            e_data=e_data,
            states=states if store_states else None,
            params=params,
            branch=self.branch,
            solver=self.__class__.__name__,
            e_ops=labels,
            metadata={
                "system": system,
                "bath": bath,
                "observables": labels,
            },
        )


@dataclass(frozen=True)
class PureDephasingSolverWC(_BasePureDephasingBranchSolver):
    branch: ClassVar[Branch] = "with_correlations"


@dataclass(frozen=True)
class PureDephasingSolverWOC(_BasePureDephasingBranchSolver):
    branch: ClassVar[Branch] = "without_correlations"


@dataclass(frozen=True)
class _BaseFortranBranchSolver:
    system: SystemParams
    bath: BathParams = field(default_factory=lambda: BathParams(bath_type="bosonic", kind="ohmic", s=1.0))
    observable: str | np.ndarray = "jx"
    initial_state: np.ndarray | None = None
    numerics: NumericsConfig | None = None

    branch: ClassVar[Branch] = "with_correlations"
    bath_type: ClassVar[str] = "bosonic"
    model: ClassVar[str] = "spin-boson"

    def run(
        self,
        tlist: Any,
        e_ops: Any = None,
        *,
        numerics: NumericsConfig | None = None,
        save_density: bool = False,
        verbose: bool = True,
    ) -> Result:
        times, dt, t_final = _validate_tlist(tlist)
        effective_numerics = _numerics_from_tlist(numerics or self.numerics, dt=dt, t_final=t_final)
        observables, labels, label_index = _normalize_e_ops(self.observable if e_ops is None else e_ops)
        expect: list[np.ndarray] = []
        artifacts: dict[str, Path] = {}
        temporary_directories: list[tempfile.TemporaryDirectory[str]] = []
        raw_times: np.ndarray | None = None
        system = validate_system_params(self.system)
        bath = validate_bath_params(self.bath, bath_type=self.bath_type)
        validate_model_compatibility(system=system, bath=bath, model=self.model)

        for observable, label in zip(observables, labels):
            _warn_for_non_hermitian_observable(observable, system=system, label=label)
            tmp = tempfile.TemporaryDirectory(prefix="meic-")
            temporary_directories.append(tmp)
            output_dir = Path(tmp.name) / label
            params = _simulation_params(
                bath_type=self.bath_type,
                model=self.model,
                system=self.system,
                bath=self.bath,
                observable=observable,
                initial_state=self.initial_state,
            )
            backend_branch = "correlated" if self.branch == "with_correlations" else "uncorrelated"
            result = run_parameterized_fortran_branch(
                params,
                output_dir,
                branch=backend_branch,
                overwrite=True,
                verbose=verbose,
                save_density=save_density,
                numerics=effective_numerics,
            )
            table_path = result.output_files.get("correlated" if backend_branch == "correlated" else "uncorrelated") if result.output_files else None
            if table_path is None:
                raise RuntimeError(f"{self.__class__.__name__} did not produce an observable table.")
            table_times, values = _read_expectation_table(table_path, times)
            if raw_times is None:
                raw_times = table_times
            elif not np.allclose(raw_times, table_times, rtol=0.0, atol=1.0e-14):
                raise RuntimeError(f"{self.__class__.__name__} returned inconsistent time grids across observables.")
            expect.append(values)
            artifacts[label] = result.output_dir

        e_data = {label: expect[index] for label, index in label_index.items()}
        params = _simulation_params(
            bath_type=self.bath_type,
            model=self.model,
            system=self.system,
            bath=self.bath,
            observable=observables[0],
            initial_state=self.initial_state,
        )
        return Result(
            times=raw_times if raw_times is not None else times,
            expect=expect,
            e_data=e_data,
            states=None,
            params=params,
            branch=self.branch,
            solver=self.__class__.__name__,
            e_ops=labels,
            numerics=effective_numerics,
            artifact_dirs=artifacts,
            _temporary_directories=temporary_directories,
            metadata={
                "system": system,
                "bath": bath,
                "observables": labels,
                "observable_parameters": [
                    _simulation_params(
                        bath_type=self.bath_type,
                        model=self.model,
                        system=self.system,
                        bath=self.bath,
                        observable=observable,
                        initial_state=self.initial_state,
                    )
                    for observable in observables
                ],
            },
        )


@dataclass(frozen=True)
class BosonicBathSolverWC(_BaseFortranBranchSolver):
    branch: ClassVar[Branch] = "with_correlations"
    bath_type: ClassVar[str] = "bosonic"
    model: ClassVar[str] = "spin-boson"


@dataclass(frozen=True)
class BosonicBathSolverWOC(_BaseFortranBranchSolver):
    branch: ClassVar[Branch] = "without_correlations"
    bath_type: ClassVar[str] = "bosonic"
    model: ClassVar[str] = "spin-boson"


@dataclass(frozen=True)
class SpinBathSolverWC(_BaseFortranBranchSolver):
    branch: ClassVar[Branch] = "with_correlations"
    bath_type: ClassVar[str] = "spin"
    model: ClassVar[str] = "spin-environment"


@dataclass(frozen=True)
class SpinBathSolverWOC(_BaseFortranBranchSolver):
    branch: ClassVar[Branch] = "without_correlations"
    bath_type: ClassVar[str] = "spin"
    model: ClassVar[str] = "spin-environment"


def solve(
    system: SystemParams,
    bath: BathParams,
    *,
    tlist: Any,
    e_ops: Any = None,
    correlations: str = "with",
    model: str = "auto",
    numerics: NumericsConfig | None = None,
    initial_state: np.ndarray | None = None,
    save_density: bool = False,
    verbose: bool = True,
) -> Result:
    """Run the master-equation solver from explicit physical inputs.

    The public form is ``solve(system, bath, tlist=..., e_ops=...)``.
    """

    if not isinstance(bath, BathParams):
        raise TypeError("solve(system, bath, ...) requires bath to be a BathParams instance.")
    normalized_bath = validate_bath_params(bath)
    normalized_correlations = normalize_correlations(correlations)
    normalized_model = "spin-environment" if normalized_bath.bath_type == "spin" else "spin-boson"
    model_token = "auto" if model is None else model.strip().lower().replace("_", "-")
    if model_token != "auto":
        requested_model = model_token
        if requested_model != normalized_model:
            raise ValueError(
                f"meic.solve uses model={normalized_model!r} for bath_type={normalized_bath.bath_type!r}; "
                "use meic.exact.solve(...) for the analytical pure-dephasing solver."
            )
    solver_cls: type[_BaseFortranBranchSolver]
    if normalized_bath.bath_type == "spin":
        solver_cls = SpinBathSolverWC if normalized_correlations == "with_correlations" else SpinBathSolverWOC
    else:
        solver_cls = BosonicBathSolverWC if normalized_correlations == "with_correlations" else BosonicBathSolverWOC
    solver = solver_cls(
        system=system,
        bath=normalized_bath,
        observable="jx",
        initial_state=initial_state,
        numerics=numerics,
    )
    return solver.run(tlist, e_ops=e_ops, numerics=numerics, save_density=save_density, verbose=verbose)


def _operator(system_or_n: SystemParams | int, expression: str) -> np.ndarray:
    if isinstance(system_or_n, SystemParams):
        J = system_or_n.N / 2.0
    else:
        J = int(system_or_n) / 2.0
    return parse_observable(expression, J).dimensionless_matrix


def jx(system_or_n: SystemParams | int) -> np.ndarray:
    return _operator(system_or_n, "jx")


def jy(system_or_n: SystemParams | int) -> np.ndarray:
    return _operator(system_or_n, "jy")


def jz(system_or_n: SystemParams | int) -> np.ndarray:
    return _operator(system_or_n, "jz")


def jx2(system_or_n: SystemParams | int) -> np.ndarray:
    return _operator(system_or_n, "jx^2")
