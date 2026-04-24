from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._types import BathParams, RunConfig, SimulationParams, SolverResult, SystemParams
from .simulation import run_simulation


def _run_config(config: RunConfig | None) -> RunConfig:
    return config if config is not None else RunConfig()


@dataclass(frozen=True)
class PureDephasingSolver:
    """Exact bosonic-bath pure-dephasing solver."""

    system: SystemParams
    bath: BathParams = field(default_factory=lambda: BathParams(kind="ohmic"))
    observable: str | np.ndarray = "jx"
    initial_state: np.ndarray | None = None

    def run(self, config: RunConfig | None = None, output_dir: str | Path | None = None, **overrides) -> SolverResult:
        cfg = _run_config(config)
        destination = output_dir if output_dir is not None else cfg.output_dir
        params = SimulationParams(
            bath="bosonic",
            model="pure-dephasing",
            spectral=self.bath.kind,
            observable=self.observable,
            N=self.system.N,
            epsilon0=self.system.epsilon0,
            epsilon=self.system.epsilon,
            delta0=self.system.delta0,
            delta=self.system.delta,
            beta=self.bath.beta,
            coupling=self.bath.coupling,
            omega_c=self.bath.omega_c,
            s=self.bath.s,
            initial_state=self.initial_state,
        )
        return run_simulation(
            params,
            destination,
            plot=overrides.get("plot", cfg.plot),
            verify=overrides.get("verify", cfg.verify),
            t_max=overrides.get("t_max", cfg.t_max),
            dt=overrides.get("dt", cfg.dt),
            overwrite=overrides.get("overwrite", cfg.overwrite),
            verbose=overrides.get("verbose", cfg.verbose),
            save_density=overrides.get("save_density", cfg.save_density),
            numerics=overrides.get("numerics", cfg.numerics),
        )


@dataclass(frozen=True)
class BosonicBathSolver:
    """Bosonic-bath spin-boson solver."""

    system: SystemParams
    bath: BathParams = field(default_factory=lambda: BathParams(kind="ohmic"))
    observable: str | np.ndarray = "jx"
    initial_state: np.ndarray | None = None

    def run(self, config: RunConfig | None = None, output_dir: str | Path | None = None, **overrides) -> SolverResult:
        cfg = _run_config(config)
        destination = output_dir if output_dir is not None else cfg.output_dir
        params = SimulationParams(
            bath="bosonic",
            model="spin-boson",
            spectral=self.bath.kind,
            observable=self.observable,
            N=self.system.N,
            epsilon0=self.system.epsilon0,
            epsilon=self.system.epsilon,
            delta0=self.system.delta0,
            delta=self.system.delta,
            beta=self.bath.beta,
            coupling=self.bath.coupling,
            omega_c=self.bath.omega_c,
            s=self.bath.s,
            initial_state=self.initial_state,
        )
        return run_simulation(
            params,
            destination,
            plot=overrides.get("plot", cfg.plot),
            verify=overrides.get("verify", cfg.verify),
            t_max=overrides.get("t_max", cfg.t_max),
            dt=overrides.get("dt", cfg.dt),
            overwrite=overrides.get("overwrite", cfg.overwrite),
            verbose=overrides.get("verbose", cfg.verbose),
            save_density=overrides.get("save_density", cfg.save_density),
            numerics=overrides.get("numerics", cfg.numerics),
        )


@dataclass(frozen=True)
class SpinBathSolver:
    """Spin-bath / spin-environment solver."""

    system: SystemParams
    bath: BathParams = field(default_factory=lambda: BathParams(kind="ohmic"))
    observable: str | np.ndarray = "jx"
    initial_state: np.ndarray | None = None

    def run(self, config: RunConfig | None = None, output_dir: str | Path | None = None, **overrides) -> SolverResult:
        cfg = _run_config(config)
        destination = output_dir if output_dir is not None else cfg.output_dir
        params = SimulationParams(
            bath="spin",
            model="spin-environment",
            spectral=self.bath.kind,
            observable=self.observable,
            N=self.system.N,
            epsilon0=self.system.epsilon0,
            epsilon=self.system.epsilon,
            delta0=self.system.delta0,
            delta=self.system.delta,
            beta=self.bath.beta,
            coupling=self.bath.coupling,
            omega_c=self.bath.omega_c,
            s=self.bath.s,
            initial_state=self.initial_state,
        )
        return run_simulation(
            params,
            destination,
            plot=overrides.get("plot", cfg.plot),
            verify=overrides.get("verify", cfg.verify),
            t_max=overrides.get("t_max", cfg.t_max),
            dt=overrides.get("dt", cfg.dt),
            overwrite=overrides.get("overwrite", cfg.overwrite),
            verbose=overrides.get("verbose", cfg.verbose),
            save_density=overrides.get("save_density", cfg.save_density),
            numerics=overrides.get("numerics", cfg.numerics),
        )
