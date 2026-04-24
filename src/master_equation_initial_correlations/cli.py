from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ._types import BathParams, NumericsConfig, SystemParams
from .blackbox import solve
from .exact import solve as exact_solve
from .fortran_runner import FortranExecutionError, doctor


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(json.dumps(doctor(), indent=2))
    return 0


def _default_epsilon(args: argparse.Namespace) -> float:
    if args.epsilon is not None:
        return args.epsilon
    return 4.0 if args.model == "pure-dephasing" else 2.5


def _default_delta(args: argparse.Namespace, name: str) -> float:
    value = getattr(args, name)
    if value is not None:
        return value
    return 0.0 if args.model == "pure-dephasing" else 0.5


def _numerics(args: argparse.Namespace) -> NumericsConfig:
    return NumericsConfig(
        omega_nodes=args.omega_nodes,
        omega_max=args.omega_max,
        lambda_nodes=args.lambda_nodes,
        initial_state_omega_nodes=args.initial_state_omega_nodes,
        initial_state_lambda_nodes=args.initial_state_lambda_nodes,
        initial_state_zeta_nodes=args.initial_state_zeta_nodes,
        initial_state_omega_max=args.initial_state_omega_max,
        correlation_omega_max=args.correlation_omega_max,
        coefficient_time_step=args.coefficient_time_step,
        correlation_tau_step=args.correlation_tau_step,
        fortran_dtau=args.fortran_dtau,
        fortran_cutoff=args.fortran_cutoff,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    system = SystemParams(
        N=args.N,
        epsilon0=args.epsilon0,
        epsilon=_default_epsilon(args),
        delta0=_default_delta(args, "delta0"),
        delta=_default_delta(args, "delta"),
    )
    bath = BathParams(
        bath_type=args.bath_type,
        kind=args.spectral,
        beta=args.beta,
        coupling=args.coupling,
        omega_c=args.omega_c,
        s=args.s,
    )
    tlist = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt)
    if args.model == "pure-dephasing":
        if args.artifacts:
            raise ValueError("the exact analytical pure-dephasing solver does not produce Fortran artifacts.")
        result = exact_solve(
            system,
            bath,
            tlist=tlist,
            e_ops=[args.observable],
            correlations=args.branch,
            verbose=not args.quiet,
        )
    else:
        result = solve(
            system,
            bath,
            tlist=tlist,
            e_ops=[args.observable],
            correlations=args.branch,
            model=args.model,
            numerics=_numerics(args),
            keep_artifacts=args.artifacts,
            verbose=not args.quiet,
        )
    try:
        destination = result.save(args.out, include_artifacts=args.artifacts, overwrite=args.overwrite)
    finally:
        result.close()
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "solver": result.solver,
                "branch": result.branch,
                "observable": args.observable,
                "time_points": int(result.times.size),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meic",
        description="Master-equation workflows with initial system-environment correlations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a solver and explicitly save the result.")
    run_parser.add_argument("--bath-type", choices=("bosonic", "spin"), default="bosonic")
    run_parser.add_argument("--model", choices=("auto", "pure-dephasing", "spin-boson", "spin-environment"), default="auto")
    run_parser.add_argument("--branch", choices=("wc", "woc"), default="wc", help="wc means with correlations; woc means without correlations.")
    run_parser.add_argument("--spectral", choices=("ohmic", "subohmic", "superohmic"), default="ohmic")
    run_parser.add_argument("--observable", default="jx", help="Observable expression, for example jx, jy, jz, jx^2, or jx+jy.")
    run_parser.add_argument("--N", type=int, required=True)
    run_parser.add_argument("--epsilon0", type=float, default=4.0)
    run_parser.add_argument("--epsilon", type=float)
    run_parser.add_argument("--delta0", type=float)
    run_parser.add_argument("--delta", type=float)
    run_parser.add_argument("--beta", type=float, default=1.0)
    run_parser.add_argument("--coupling", type=float, default=0.05)
    run_parser.add_argument("--omega-c", dest="omega_c", type=float, default=5.0)
    run_parser.add_argument("--s", type=float, help="Spectral exponent: use 1.0 for Ohmic, 0<s<1 for sub-Ohmic, and s>1 for super-Ohmic.")
    run_parser.add_argument("--tmax", type=float, default=5.0)
    run_parser.add_argument("--dt", type=float, default=0.1)
    run_parser.add_argument("--out", type=Path, default=Path("."), help="Output directory; defaults to the current working directory.")
    run_parser.add_argument("--artifacts", action="store_true", help="Also save generated .dat inputs, Fortran sources, and logs.")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--omega-max", type=float, help="Shared frequency cutoff for coefficient, correlation, and initial-state integrals.")
    run_parser.add_argument("--omega-nodes", type=int, default=500, help="Frequency quadrature nodes.")
    run_parser.add_argument("--lambda-nodes", type=int, default=100)
    run_parser.add_argument("--initial-state-omega-nodes", type=int, default=260)
    run_parser.add_argument("--initial-state-lambda-nodes", type=int, default=40)
    run_parser.add_argument("--initial-state-zeta-nodes", type=int, default=40)
    run_parser.add_argument("--initial-state-omega-max", type=float)
    run_parser.add_argument("--correlation-omega-max", type=float)
    run_parser.add_argument("--coefficient-time-step", type=float, default=0.0025, help="Time spacing for generated A/B/C coefficient tables.")
    run_parser.add_argument("--correlation-tau-step", type=float, default=0.0025, help="Tau spacing for generated bath-correlation tables.")
    run_parser.add_argument("--fortran-dtau", type=float, default=0.005)
    run_parser.add_argument("--fortran-cutoff", type=float, default=15.0)
    run_parser.set_defaults(func=_cmd_run)

    doctor_parser = sub.add_parser("doctor", help="Check the optional Fortran compiler/linker setup.")
    doctor_parser.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileExistsError, RuntimeError, FortranExecutionError) as exc:
        print(f"meic: error: {exc}", file=sys.stderr)
        return 2
