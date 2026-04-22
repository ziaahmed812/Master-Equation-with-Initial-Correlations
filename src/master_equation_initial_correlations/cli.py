from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .catalog import get_preset, list_presets
from .fortran_runner import doctor, rerun_preset
from .pure_dephasing import PureDephasingParams, exact_curves
from .reference import export_figure_assets


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cmd_list(args: argparse.Namespace) -> int:
    for preset in list_presets(include_advanced=args.include_advanced, include_heavy=args.include_heavy):
        print(f"{preset.id}: {preset.label} [{preset.execution_profile}]")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    preset = get_preset(args.preset)
    payload = {
        "id": preset.id,
        "figure_id": preset.figure_id,
        "paper_figure_number": preset.paper_figure_number,
        "family": preset.family,
        "label": preset.label,
        "paper_asset": preset.paper_asset,
        "execution_profile": preset.execution_profile,
        "parameters": preset.parameters,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    destination = export_figure_assets(args.preset, args.out)
    print(destination)
    return 0


def _cmd_exact(args: argparse.Namespace) -> int:
    if args.preset:
        preset = get_preset(args.preset)
        params = PureDephasingParams(
            J=float(preset.parameters["J"]),
            epsilon=float(preset.parameters["epsilon"]),
            xi=float(preset.parameters["epsilon_0"]),
            beta=float(preset.parameters["beta"]),
            G=float(preset.parameters["G"]),
            omega_c=float(preset.parameters["omega_c"]),
        )
    else:
        params = PureDephasingParams(
            J=args.J,
            epsilon=args.epsilon,
            xi=args.xi,
            beta=args.beta,
            G=args.G,
            omega_c=args.omega_c,
        )
    correlated, uncorrelated = exact_curves(params)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "exact-correlated.dat", correlated, fmt="%.16e")
        np.savetxt(out / "exact-uncorrelated.dat", uncorrelated, fmt="%.16e")
    print(json.dumps({"correlated_points": int(correlated.shape[0]), "uncorrelated_points": int(uncorrelated.shape[0])}, indent=2))
    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    result = rerun_preset(
        args.preset,
        args.out,
        verify=not args.no_verify,
        render=args.render,
        allow_heavy=args.allow_heavy,
    )
    print(json.dumps({
        "preset_id": result.preset.id,
        "correlated_error": result.correlated_error,
        "uncorrelated_error": result.uncorrelated_error,
        "exact_correlated_error": result.exact_correlated_error,
        "exact_uncorrelated_error": result.exact_uncorrelated_error,
        "summary_path": result.summary_path,
    }, indent=2, default=_json_default))
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(json.dumps(doctor(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meic")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--include-advanced", action="store_true")
    list_parser.add_argument("--include-heavy", action="store_true")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = sub.add_parser("show")
    show_parser.add_argument("preset")
    show_parser.add_argument("--json", action="store_true")
    show_parser.set_defaults(func=_cmd_show)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("preset")
    export_parser.add_argument("--out", required=True)
    export_parser.set_defaults(func=_cmd_export)

    exact_parser = sub.add_parser("exact")
    exact_sub = exact_parser.add_subparsers(dest="exact_command", required=True)
    pd_parser = exact_sub.add_parser("pure-dephasing")
    pd_parser.add_argument("--preset")
    pd_parser.add_argument("--J", type=float)
    pd_parser.add_argument("--epsilon", type=float, default=4.0)
    pd_parser.add_argument("--xi", type=float, default=4.0)
    pd_parser.add_argument("--beta", type=float, default=1.0)
    pd_parser.add_argument("--G", type=float, default=0.05)
    pd_parser.add_argument("--omega-c", dest="omega_c", type=float, default=5.0)
    pd_parser.add_argument("--out")
    pd_parser.set_defaults(func=_cmd_exact)

    rerun_parser = sub.add_parser("rerun")
    rerun_parser.add_argument("preset")
    rerun_parser.add_argument("--out", required=True)
    rerun_parser.add_argument("--render", action="store_true")
    rerun_parser.add_argument("--allow-heavy", action="store_true")
    rerun_parser.add_argument("--no-verify", action="store_true")
    rerun_parser.set_defaults(func=_cmd_rerun)

    doctor_parser = sub.add_parser("doctor")
    doctor_parser.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "exact_command", None) == "pure-dephasing" and not args.preset and args.J is None:
        parser.error("pure-dephasing exact mode requires either --preset or --J")
    return args.func(args)
