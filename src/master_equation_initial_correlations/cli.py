from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import list_examples
from .fortran_runner import doctor
from .reference import export_example_assets
from .simulation import SimulationParams, run_simulation


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cmd_examples(args: argparse.Namespace) -> int:
    examples = list_examples()
    if args.json:
        payload = [
            {
                "id": example.public_id,
                "bath": example.bath,
                "model": example.model,
                "spectral": example.spectral,
                "observable": example.observable,
                "parameters": example.parameters,
                "asset": example.paper_asset,
            }
            for example in examples
        ]
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0

    for example in examples:
        params = example.parameters
        print(
            f"{example.public_id}: {example.bath} bath, {example.model}, "
            f"{example.spectral}, N={params['N']}, observable={example.observable}"
        )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    destination = export_example_assets(args.example, args.out)
    print(destination)
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


def _cmd_run(args: argparse.Namespace) -> int:
    params = SimulationParams(
        bath=args.bath,
        model=args.model,
        spectral=args.spectral,
        observable=args.observable,
        N=args.N,
        epsilon0=args.epsilon0,
        epsilon=_default_epsilon(args),
        delta0=_default_delta(args, "delta0"),
        delta=_default_delta(args, "delta"),
        beta=args.beta,
        coupling=args.coupling,
        omega_c=args.omega_c,
        s=args.s,
    )
    result = run_simulation(
        params,
        args.out,
        plot=args.plot,
        verify=not args.no_verify,
        t_max=args.t_max,
        dt=args.dt,
    )
    print(
        json.dumps(
            {
                "source": result.source,
                "example": result.example.public_id if result.example else None,
                "output_dir": result.output_dir,
                "correlated_path": result.correlated_path,
                "uncorrelated_path": result.uncorrelated_path,
                "correlated_error": result.correlated_error,
                "uncorrelated_error": result.uncorrelated_error,
                "exact_correlated_error": result.exact_correlated_error,
                "exact_uncorrelated_error": result.exact_uncorrelated_error,
                "summary_path": result.summary_path,
            },
            indent=2,
            default=_json_default,
        )
    )
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(json.dumps(doctor(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meic",
        description="Master-equation workflows with initial system-environment correlations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    examples_parser = sub.add_parser("examples", help="List bundled reference examples.")
    examples_parser.add_argument("--json", action="store_true", help="Write the example catalog as JSON.")
    examples_parser.set_defaults(func=_cmd_examples)

    export_parser = sub.add_parser("export", help="Export bundled data and figure assets for one example.")
    export_parser.add_argument("example", help="Example id shown by `meic examples`.")
    export_parser.add_argument("--out", required=True, help="Destination directory.")
    export_parser.set_defaults(func=_cmd_export)

    run_parser = sub.add_parser("run", help="Run a supported bosonic-bath or spin-bath workflow.")
    run_parser.add_argument("--bath", choices=("bosonic", "spin"), required=True)
    run_parser.add_argument("--model", choices=("pure-dephasing", "spin-boson", "spin-environment"), required=True)
    run_parser.add_argument("--spectral", choices=("ohmic", "subohmic"), default="ohmic")
    run_parser.add_argument("--observable", choices=("jx", "jx2"), default="jx")
    run_parser.add_argument("--N", type=int, required=True)
    run_parser.add_argument("--epsilon0", type=float, default=4.0)
    run_parser.add_argument("--epsilon", type=float)
    run_parser.add_argument("--delta0", type=float)
    run_parser.add_argument("--delta", type=float)
    run_parser.add_argument("--beta", type=float, default=1.0)
    run_parser.add_argument("--coupling", type=float, default=0.05)
    run_parser.add_argument("--omega-c", dest="omega_c", type=float, default=5.0)
    run_parser.add_argument("--s", type=float, help="Sub-Ohmic spectral exponent.")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--plot", action="store_true", help="Also render EPS/PNG output when plotting data are available.")
    run_parser.add_argument("--no-verify", action="store_true", help="Skip comparison against bundled reference tables.")
    run_parser.add_argument("--t-max", type=float, default=5.0, help="Final time for exact pure-dephasing output.")
    run_parser.add_argument("--dt", type=float, default=0.2, help="Time spacing for exact pure-dephasing output.")
    run_parser.set_defaults(func=_cmd_run)

    doctor_parser = sub.add_parser("doctor", help="Show optional Fortran compiler/linker configuration.")
    doctor_parser.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
