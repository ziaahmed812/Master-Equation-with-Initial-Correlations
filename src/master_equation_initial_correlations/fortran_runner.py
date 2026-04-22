from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np

from ._resources import copy_resource_tree, ensure_clean_dir, load_table, write_json
from ._types import ReferenceCurves, RerunResult
from .catalog import get_preset
from .plotting import plot_reference_curves
from .pure_dephasing import PureDephasingParams, exact_curves


@dataclass(frozen=True)
class FortranBuildConfig:
    compiler: str = "gfortran"
    flags: tuple[str, ...] = ("-O2", "-std=legacy", "-ffixed-line-length-none")
    link_args: tuple[str, ...] = ("-llapack", "-lblas")


def _default_link_args() -> tuple[str, ...]:
    explicit = (
        "/lib/x86_64-linux-gnu/liblapack.so.3",
        "/lib/x86_64-linux-gnu/libblas.so.3",
    )
    if all(Path(arg).exists() for arg in explicit):
        return explicit
    return ("-llapack", "-lblas")


def _build_config() -> FortranBuildConfig:
    compiler = os.environ.get("MEIC_FORTRAN_COMPILER", "gfortran")
    flags = tuple(shlex.split(os.environ.get("MEIC_FORTRAN_FLAGS", "-O2 -std=legacy -ffixed-line-length-none")))
    link_args = tuple(shlex.split(os.environ["MEIC_FORTRAN_LINK_ARGS"])) if "MEIC_FORTRAN_LINK_ARGS" in os.environ else _default_link_args()
    return FortranBuildConfig(compiler=compiler, flags=flags, link_args=link_args)


def doctor() -> dict[str, object]:
    config = _build_config()
    compiler_path = shutil.which(config.compiler)
    return {
        "compiler": config.compiler,
        "compiler_found": compiler_path is not None,
        "compiler_path": compiler_path,
        "flags": list(config.flags),
        "link_args": list(config.link_args),
    }


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _max_abs(generated: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(generated - reference)))


def _stage_branch(preset_id: str, branch: str, output_dir: Path) -> Path:
    preset = get_preset(preset_id)
    run_dir = output_dir / "runs" / branch
    ensure_clean_dir(run_dir)
    copy_resource_tree(f"{preset.asset_dir}/inputs", run_dir)
    solver_dir = run_dir / "solver"
    copy_resource_tree(f"solvers/{preset.solver_id}", solver_dir)
    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    shutil.copy2(solver_dir / source_name, run_dir / source_name)
    shutil.rmtree(solver_dir)
    return run_dir


def _compile_and_run(run_dir: Path, branch: str, config: FortranBuildConfig) -> dict[str, object]:
    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    binary_name = f"{branch}.out"
    compile_cmd = [config.compiler, *config.flags, source_name, "-o", binary_name, *config.link_args]
    compile_proc = _run(compile_cmd, run_dir)
    (run_dir / "compile.stdout.log").write_text(compile_proc.stdout)
    (run_dir / "compile.stderr.log").write_text(compile_proc.stderr)
    run_proc = _run([str(run_dir / binary_name)], run_dir)
    (run_dir / "run.stdout.log").write_text(run_proc.stdout)
    (run_dir / "run.stderr.log").write_text(run_proc.stderr)
    return {"compile_command": compile_cmd}


def rerun_preset(
    preset_id: str,
    output_dir: str | Path,
    *,
    verify: bool = True,
    render: bool = False,
    allow_heavy: bool = False,
) -> RerunResult:
    preset = get_preset(preset_id)
    if preset.execution_profile == "heavy" and not allow_heavy:
        raise RuntimeError(f"{preset.id} is marked as a heavy preset. Pass allow_heavy=True to rerun it.")

    config = _build_config()
    compiler_path = shutil.which(config.compiler)
    if compiler_path is None:
        raise RuntimeError(f"Fortran compiler {config.compiler!r} was not found. Run `meic doctor` for details.")

    output_dir = Path(output_dir)
    ensure_clean_dir(output_dir)

    correlated_dir = _stage_branch(preset.id, "correlated", output_dir)
    uncorrelated_dir = _stage_branch(preset.id, "uncorrelated", output_dir)
    correlated_meta = _compile_and_run(correlated_dir, "correlated", config)
    uncorrelated_meta = _compile_and_run(uncorrelated_dir, "uncorrelated", config)

    correlated = np.loadtxt(correlated_dir / "EXPX-C.DAT", dtype=float)
    uncorrelated = np.loadtxt(uncorrelated_dir / "EXPX-UNC.DAT", dtype=float)
    reference_correlated = load_table(f"{preset.asset_dir}/tables/EXPX-C.DAT")
    reference_uncorrelated = load_table(f"{preset.asset_dir}/tables/EXPX-UNC.DAT")

    correlated_error = _max_abs(correlated, reference_correlated) if verify else 0.0
    uncorrelated_error = _max_abs(uncorrelated, reference_uncorrelated) if verify else 0.0

    jz_correlated_error = None
    jz_uncorrelated_error = None
    if (correlated_dir / "EXPZ-C.DAT").exists():
        jz_correlated = np.loadtxt(correlated_dir / "EXPZ-C.DAT", dtype=float)
        jz_reference = load_table(f"{preset.asset_dir}/tables/EXPZ-C.DAT")
        jz_correlated_error = _max_abs(jz_correlated, jz_reference) if verify else 0.0
    if (uncorrelated_dir / "EXPZ-UNC.DAT").exists():
        jz_uncorrelated = np.loadtxt(uncorrelated_dir / "EXPZ-UNC.DAT", dtype=float)
        jz_reference = load_table(f"{preset.asset_dir}/tables/EXPZ-UNC.DAT")
        jz_uncorrelated_error = _max_abs(jz_uncorrelated, jz_reference) if verify else 0.0

    exact_correlated_error = None
    exact_uncorrelated_error = None
    exact_correlated = None
    exact_uncorrelated = None
    if preset.supports_exact:
        params = PureDephasingParams(
            J=float(preset.parameters["J"]),
            epsilon=float(preset.parameters["epsilon"]),
            xi=float(preset.parameters["epsilon_0"]),
            beta=float(preset.parameters["beta"]),
            G=float(preset.parameters["G"]),
            omega_c=float(preset.parameters["omega_c"]),
        )
        exact_correlated, exact_uncorrelated = exact_curves(params)
        if verify:
            ref_exact_correlated = load_table(f"{preset.asset_dir}/tables/exact-correlated.dat")
            ref_exact_uncorrelated = load_table(f"{preset.asset_dir}/tables/exact-uncorrelated.dat")
            exact_correlated_error = _max_abs(exact_correlated, ref_exact_correlated)
            exact_uncorrelated_error = _max_abs(exact_uncorrelated, ref_exact_uncorrelated)

    rendered_eps = None
    rendered_png = None
    if render:
        plot_correlated = np.array(correlated, copy=True)
        plot_uncorrelated = np.array(uncorrelated, copy=True)
        if preset.plot_divisor != 1.0:
            plot_correlated[:, 1] = plot_correlated[:, 1] / preset.plot_divisor
            plot_uncorrelated[:, 1] = plot_uncorrelated[:, 1] / preset.plot_divisor
        curves = ReferenceCurves(
            preset=preset,
            correlated=plot_correlated,
            uncorrelated=plot_uncorrelated,
            exact_correlated=exact_correlated,
            exact_uncorrelated=exact_uncorrelated,
        )
        outputs = plot_reference_curves(curves, output_dir / "plots")
        rendered_eps = outputs["eps"]
        rendered_png = outputs["png"]

    summary = {
        "preset_id": preset.id,
        "compiler": compiler_path,
        "build": asdict(config),
        "fortran": {
            "correlated": {**correlated_meta, "max_abs_error": correlated_error},
            "uncorrelated": {**uncorrelated_meta, "max_abs_error": uncorrelated_error},
            "jz_correlated_error": jz_correlated_error,
            "jz_uncorrelated_error": jz_uncorrelated_error,
        },
        "exact": {
            "correlated_max_abs_error": exact_correlated_error,
            "uncorrelated_max_abs_error": exact_uncorrelated_error,
        },
        "rendered_eps": str(rendered_eps) if rendered_eps else None,
        "rendered_png": str(rendered_png) if rendered_png else None,
    }
    summary_path = output_dir / "regeneration_summary.json"
    write_json(summary_path, summary)
    return RerunResult(
        preset=preset,
        output_dir=output_dir,
        correlated_error=correlated_error,
        uncorrelated_error=uncorrelated_error,
        jz_correlated_error=jz_correlated_error,
        jz_uncorrelated_error=jz_uncorrelated_error,
        exact_correlated_error=exact_correlated_error,
        exact_uncorrelated_error=exact_uncorrelated_error,
        rendered_eps=rendered_eps,
        rendered_png=rendered_png,
        summary_path=summary_path,
    )
