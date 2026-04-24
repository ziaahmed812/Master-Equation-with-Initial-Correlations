from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ._resources import asset, copy_resource_tree, ensure_clean_dir, load_table, prepare_output_dir, write_json, write_table
from ._types import NumericsConfig, ReferenceExample, RerunResult, SimulationParams
from .catalog import get_example
from .generated_inputs import coefficient_index_step, numerics_summary, tau_index_step, validate_numerics, write_complex_fortran, write_generated_input_files
from .observables import normalize_observable_expression, parse_observable
from .pure_dephasing import PureDephasingParams, exact_curves
from ._validation import normalize_simulation_params


@dataclass(frozen=True)
class FortranBuildConfig:
    compiler: str = "gfortran"
    flags: tuple[str, ...] = ("-O2", "-std=legacy", "-ffixed-line-length-none")
    link_args: tuple[str, ...] = ("-llapack", "-lblas")


class FortranExecutionError(RuntimeError):
    """A compile, link, or run step failed; logs were written when possible."""


RERUN_OUTPUTS = (
    "runs",
    "sources",
    "logs",
    "regeneration_summary.json",
    "EXPX-C.DAT",
    "EXPX-UNC.DAT",
    "EXPZ-C.DAT",
    "EXPZ-UNC.DAT",
    "OBSERVABLE-C.DAT",
    "OBSERVABLE-UNC.DAT",
    "observable-correlated.dat",
    "observable-uncorrelated.dat",
    "density-correlated.dat",
    "density-uncorrelated.dat",
    "A.dat",
    "B.dat",
    "C.dat",
    "integraldatasimpson.dat",
    "bathcorrelation.dat",
    "OBSERVABLE.dat",
    "INSTATE-J=0.5.dat",
    "INSTATE-J=1.dat",
    "INSTATE-J=2.dat",
    "INSTATE-J=5.dat",
    "INSTATE.dat",
    "params.in",
    "dimensions.inc",
)


@dataclass(frozen=True)
class _GeneratedBranchRun:
    branch: str
    run_dir: Path
    output_files: dict[str, Path]
    source_files: dict[str, Path]
    log_files: dict[str, Path]
    input_files: dict[str, Path]
    metadata: dict[str, object]


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
    smoke = {"compile": False, "run": False, "error": None}
    if compiler_path is not None:
        with tempfile.TemporaryDirectory(prefix="meic-doctor-") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "lapack_smoke.f90"
            source.write_text(
                "\n".join(
                    [
                        "program lapack_smoke",
                        "  implicit none",
                        "  double precision :: a(1,1), b(1,1)",
                        "  integer :: ipiv(1), info",
                        "  a(1,1) = 1.0d0",
                        "  b(1,1) = 2.0d0",
                        "  call dgesv(1, 1, a, 1, ipiv, b, 1, info)",
                        "  if (info .ne. 0) stop 2",
                        "end program lapack_smoke",
                    ]
                )
            )
            binary = tmp_path / "lapack_smoke.out"
            cmd = [config.compiler, *config.flags, str(source), "-o", str(binary), *config.link_args]
            try:
                subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)
                smoke["compile"] = True
                subprocess.run([str(binary)], cwd=tmp_path, check=True, capture_output=True, text=True)
                smoke["run"] = True
            except (OSError, subprocess.CalledProcessError) as exc:
                smoke["error"] = str(exc)
    return {
        "compiler": config.compiler,
        "compiler_found": compiler_path is not None,
        "compiler_path": compiler_path,
        "flags": list(config.flags),
        "link_args": list(config.link_args),
        "lapack_blas_smoke": smoke,
    }


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _emit(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _max_abs(generated: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(generated - reference)))


def _stage_branch(example_id: str, branch: str, output_dir: Path) -> Path:
    example = get_example(example_id)
    run_dir = output_dir / "runs" / branch
    ensure_clean_dir(run_dir)
    copy_resource_tree(f"{example.asset_dir}/inputs", run_dir)
    solver_dir = run_dir / "solver"
    copy_resource_tree(f"solvers/{example.solver_id}", solver_dir)
    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    shutil.copy2(solver_dir / source_name, run_dir / source_name)
    shutil.rmtree(solver_dir)
    return run_dir


def _fortran_float(value: float) -> str:
    text = f"{float(value):.16g}".replace("e", "D").replace("E", "D")
    return text if "D" in text else f"{text}D0"


def _template_solver_id(params: SimulationParams) -> str:
    if params.bath == "spin":
        return "spin_environment/J2"
    if params.spectral == "subohmic":
        return "subohmic/J2"
    return "beyond_pd/J2"


def _parameter_reader_block() -> str:
    return "\n".join(
        [
            "",
            "\tOPEN (99,FILE = 'params.in',STATUS = 'OLD')",
            "\tREAD(99,*) DELTA",
            "\tREAD(99,*) ENERGYGAP",
            "\tREAD(99,*) G",
            "\tREAD(99,*) TEMPERATURE",
            "\tREAD(99,*) OMEGAC",
            "\tREAD(99,*) STEPSIZE_TAU",
            "\tREAD(99,*) STEPSIZE_T",
            "\tREAD(99,*) TCUTOFF",
            "\tREAD(99,*) TFINAL",
            "\tREAD(99,*) SAVE_DENSITY",
            "\tCLOSE (99)",
            "",
        ]
    )


def _expectation_complex_block() -> str:
    return "\n".join(
        [
            "",
            "\tSUBROUTINE EXPECTATION_COMPLEX(X,A,ANS)",
            "\tIMPLICIT NONE",
            "\tCOMPLEX*16, DIMENSION(DIMMAT,1),INTENT(IN) :: X",
            "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS),INTENT(IN)::A",
            "\tCOMPLEX*16, INTENT(OUT) :: ANS",
            "\tCOMPLEX*16, DIMENSION(DIMSYS, DIMSYS)::RHO",
            "\tCOMPLEX*16, DIMENSION(DIMSYS, DIMSYS) :: TEMP",
            "\tINTEGER :: J",
            "\tCOMPLEX*16::SUMM",
            "\tSUMM = (0.0D0,0.0D0)",
            "\tCALL DENSITYMATRIX(X,RHO)",
            "\tTEMP = MATMUL(A,RHO)",
            "\tDO J = 1, DIMSYS",
            "\tSUMM = SUMM + TEMP(J,J)",
            "\tEND DO",
            "\tANS = SUMM",
            "\tEND SUBROUTINE EXPECTATION_COMPLEX",
            "",
        ]
    )


def _densitymatrix_public_block() -> str:
    return "\n".join(
        [
            "",
            "\tSUBROUTINE DENSITYMATRIX_PUBLIC(X,MATOUT)",
            "\tIMPLICIT NONE",
            "\tCOMPLEX*16,DIMENSION(DIMMAT, 1),INTENT(IN)::X",
            "\tCOMPLEX*16,DIMENSION(DIMSYS, DIMSYS),INTENT(OUT)::MATOUT",
            "\tCOMPLEX*16,DIMENSION(DIMSYS, DIMSYS)::RHO",
            "\tCOMPLEX*16,DIMENSION(DIMSYS, DIMSYS)::R",
            "\tCOMPLEX*16,DIMENSION(DIMSYS, DIMSYS)::RT",
            "\tCALL UROT(R)",
            "\tCALL DENSITYMATRIX(X,RHO)",
            "\tRT = DCONJG(TRANSPOSE(R))",
            "\tMATOUT = MATMUL(RT,MATMUL(RHO,R))",
            "\tEND SUBROUTINE DENSITYMATRIX_PUBLIC",
            "",
        ]
    )


def _generic_correlator_block() -> str:
    return "\n".join(
        [
            "\tCOMPLEX*16 FUNCTION CORRELATOR(TAU) \t\t",
            "\tIMPLICIT NONE",
            "\tREAL(KIND = DBL), INTENT(IN)::TAU",
            "\tREAL(KIND = DBL) :: NU, ETA",
            "\tINTEGER :: J",
            "\tJ = NINT((TAU - TAU_INDEX_MIN)/TAU_INDEX_STEP)",
            "\tIF (J < 0) J = 0",
            "\tIF (J > NTAU - 1) J = NTAU - 1",
            "\tNU = G * NUARRAY(J + 1)",
            "\tETA = G * ETAARRAY(J + 1)",
            "\tCORRELATOR = NU - IMU * ETA",
            "\tEND FUNCTION CORRELATOR",
        ]
    )


def _density_output_name(branch: str) -> str:
    return "density-correlated.dat" if branch == "correlated" else "density-uncorrelated.dat"


def _observable_output_name(branch: str) -> str:
    return "OBSERVABLE-C.DAT" if branch == "correlated" else "OBSERVABLE-UNC.DAT"


def _serializable_params(params: SimulationParams) -> dict[str, object]:
    payload = asdict(params)
    payload["observable"] = normalize_observable_expression(params.observable)
    payload["initial_state"] = (
        None
        if params.initial_state is None
        else {"source": "user_supplied", "shape": list(np.asarray(params.initial_state).shape)}
    )
    return payload


def _parameterize_source(source: str, *, branch: str) -> str:
    source, dimension_replacements = re.subn(
        r"\s*REAL\(KIND = DBL\), PARAMETER :: JSYS = .*?!CHANGE HERE\s*\n"
        r"\s*INTEGER, PARAMETER :: DIMSYS = NINT\(2\*JSYS \+ 1\)\s*\n"
        r"\s*INTEGER, PARAMETER :: DIMMAT = DIMSYS \* DIMSYS\s*\n",
        "\n\tINCLUDE 'dimensions.inc'\n",
        source,
        count=1,
    )
    if dimension_replacements != 1:
        raise RuntimeError("Could not parameterize Fortran spin dimension block.")

    source = source.replace(
        "\tREAL(KIND = DBL), DIMENSION(2001)::NUARRAY",
        "\tREAL(KIND = DBL), DIMENSION(NTAU)::NUARRAY\n\tREAL(KIND = DBL), DIMENSION(NTAU)::ETAARRAY",
        1,
    )
    source = source.replace(
        "\tCOMPLEX*16, DIMENSION(2001) :: ADATA, BDATA, CDATA",
        "\tCOMPLEX*16, DIMENSION(NCOEFF) :: ADATA, BDATA, CDATA",
        1,
    )
    source = source.replace(
        "\tREAL*16, DIMENSION(DIMMAT) :: INSTATE_TEMP",
        "\tCOMPLEX*16, DIMENSION(DIMMAT) :: INSTATE_TEMP",
        1,
    )
    source = source.replace(
        "\tREAL(KIND = DBL) :: TCUTOFF",
        "\tREAL(KIND = DBL) :: TCUTOFF, TFINAL\n\n\tINTEGER :: SAVE_DENSITY",
        1,
    )
    source = source.replace(
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::Jx_EB,Jy_EB, Jz_EB",
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::Jx_EB,Jy_EB, Jz_EB\n"
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::OBSERVABLE, OBSERVABLE_EB",
        1,
    )

    source, parameter_replacements = re.subn(
        r"\s*DELTA\s*=.*?\n"
        r"\s*ENERGYGAP\s*=.*?\n"
        r"\s*G\s*=.*?\n"
        r"\s*TEMPERATURE\s*=.*?\n"
        r"\s*OMEGAC\s*=.*?\n"
        r"\s*STEPSIZE_TAU\s*=.*?\n"
        r"\s*STEPSIZE_T\s*=.*?\n"
        r"\s*TCUTOFF\s*=.*?\n",
        _parameter_reader_block(),
        source,
        count=1,
    )
    if parameter_replacements != 1:
        raise RuntimeError("Could not parameterize Fortran run-constant block.")
    source = source.replace(
        "\tJ = NINT(T/0.0025D0)",
        "\tJ = NINT((T - COEFF_INDEX_MIN)/COEFF_INDEX_STEP)\n\tIF (J < 0) J = 0\n\tIF (J > NCOEFF - 1) J = NCOEFF - 1",
        1,
    )

    source, instate_replacements = re.subn(
        r"OPEN\s*\(8,\s*FILE\s*=\s*'INSTATE-J=[^']+\.dat'\)\s*!CHANGE HERE",
        "OPEN (8,FILE = 'INSTATE.dat')",
        source,
        count=1,
    )
    if instate_replacements != 1:
        raise RuntimeError("Could not parameterize Fortran initial-state input filename.")

    source = source.replace(
        "\tEND SUBROUTINE EXPECTATION\n",
        f"\tEND SUBROUTINE EXPECTATION\n{_expectation_complex_block()}\n",
        1,
    )
    source = source.replace(
        "\tEND SUBROUTINE DENSITYMATRIX_EB\n",
        f"\tEND SUBROUTINE DENSITYMATRIX_EB\n{_densitymatrix_public_block()}\n",
        1,
    )
    source, correlator_replacements = re.subn(
        r"\tCOMPLEX\*16 FUNCTION CORRELATOR\(TAU\).*?\n\tEND FUNCTION CORRELATOR",
        _generic_correlator_block(),
        source,
        count=1,
        flags=re.DOTALL,
    )
    if correlator_replacements != 1:
        raise RuntimeError("Could not parameterize Fortran bath-correlation block.")

    source = source.replace(
        "OPEN (2,FILE = 'integraldatasimpson.dat')",
        "OPEN (2,FILE = 'bathcorrelation.dat')",
        1,
    )
    source = source.replace(
        "\tOPEN (7,FILE = 'C.dat')",
        "\tOPEN (7,FILE = 'C.dat')\n\n"
        f"\tOPEN (11,FILE = '{_observable_output_name(branch)}')\n"
        f"\tOPEN (12,FILE = '{_density_output_name(branch)}')\n\n"
        "\tOPEN (10,FILE = 'OBSERVABLE.dat')",
        1,
    )
    source = source.replace(
        "\tREAL(KIND = DBL) :: Z_EXPECTATION, X_EXPECTATION",
        "\tREAL(KIND = DBL) :: Z_EXPECTATION, X_EXPECTATION\n"
        "\tCOMPLEX*16 :: OBS_EXPECTATION\n"
        "\tINTEGER :: OBS_INDEX, OBS_I, OBS_J",
        1,
    )
    source = source.replace(
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::RHO_EB",
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::RHO_EB\n"
        "\tCOMPLEX*16, DIMENSION(DIMSYS,DIMSYS)::RHO_PUBLIC",
        1,
    )
    source = source.replace(
        "\tWRITE(3,*) T, X_EXPECTATION/JSYS",
        "\tWRITE(3,*) T, X_EXPECTATION/JSYS\n"
        "\tCALL EXPECTATION_COMPLEX(X,OBSERVABLE_EB, OBS_EXPECTATION)\n"
        "\tWRITE(11,*) T, DREAL(OBS_EXPECTATION)/JSYS, DIMAG(OBS_EXPECTATION)/JSYS\n"
        "\tIF (SAVE_DENSITY .EQ. 1) CALL DENSITYMATRIX_PUBLIC(X,RHO_PUBLIC)\n"
        "\tIF (SAVE_DENSITY .EQ. 1) WRITE(12,*) T, ((DREAL(RHO_PUBLIC(OBS_I,OBS_J)), DIMAG(RHO_PUBLIC(OBS_I,OBS_J)), OBS_J = 1, DIMSYS), OBS_I = 1, DIMSYS)",
        1,
    )
    source, bath_read_replacements = re.subn(
        r"\tDO J = 1, 2001\s*\n\tREAD \(2,\*\) NUARRAY\(J\)\s*\n\tEND DO",
        "\tDO J = 1, NTAU\n\tREAD (2,*) NUARRAY(J), ETAARRAY(J)\n\tEND DO",
        source,
        count=1,
    )
    if bath_read_replacements != 1:
        raise RuntimeError("Could not parameterize Fortran bath-correlation table reader.")
    source = source.replace(
        "\tDO J = 1, 2001\n\tREAD (4,*) ADATA(J)",
        "\tDO J = 1, NCOEFF\n\tREAD (4,*) ADATA(J)",
        1,
    )
    source = source.replace(
        "\tDO J = 1, 2001\n\tREAD (5,*) BDATA(J)",
        "\tDO J = 1, NCOEFF\n\tREAD (5,*) BDATA(J)",
        1,
    )
    source = source.replace(
        "\tDO J = 1, 2001\n\tREAD (7,*) CDATA(J)",
        "\tDO J = 1, NCOEFF\n\tREAD (7,*) CDATA(J)",
        1,
    )
    source = source.replace("CALL RUNGEKUTTA(5.0D0,ANS)", "CALL RUNGEKUTTA(TFINAL,ANS)", 1)
    source = source.replace(
        "\tCALL INSTATETOX_EB(INSTATE_BASISSTATES)\n\n\n\tCALL MATRIX_EB(Jz,Jz_EB)",
        "\tCALL INSTATETOX_EB(INSTATE_BASISSTATES)\n\n"
        "\tDO KK=1,DIMSYS\n"
        "\tDO K1=1,DIMSYS\n"
        "\tREAD(10,*) OBSERVABLE(KK,K1)\n"
        "\tEND DO\n"
        "\tEND DO\n"
        "\tCLOSE (10)\n\n"
        "\tCALL MATRIX_EB(Jz,Jz_EB)",
        1,
    )
    source = source.replace(
        "\tCALL MATRIX_EB(Jx,Jx_EB)\n\tCALL MATRIX_EB(INSTATE,INSTATE_EB)",
        "\tCALL MATRIX_EB(Jx,Jx_EB)\n\tCALL MATRIX_EB(OBSERVABLE,OBSERVABLE_EB)\n\tCALL MATRIX_EB(INSTATE,INSTATE_EB)",
        1,
    )
    return source


def _write_dimensions(path: Path, params: SimulationParams, numerics: NumericsConfig) -> Path:
    j_value = params.N / 2.0
    numerics = validate_numerics(numerics)
    path.write_text(
        "\n".join(
            [
                f"\tREAL(KIND = DBL), PARAMETER :: JSYS = {_fortran_float(j_value)}",
                "\tINTEGER, PARAMETER :: DIMSYS = NINT(2*JSYS + 1)",
                "\tINTEGER, PARAMETER :: DIMMAT = DIMSYS * DIMSYS",
                f"\tINTEGER, PARAMETER :: NCOEFF = {numerics.coefficient_points}",
                f"\tINTEGER, PARAMETER :: NTAU = {numerics.tau_points}",
                f"\tREAL(KIND = DBL), PARAMETER :: COEFF_INDEX_MIN = {_fortran_float(numerics.coefficient_t_min)}",
                f"\tREAL(KIND = DBL), PARAMETER :: COEFF_INDEX_STEP = {_fortran_float(coefficient_index_step(numerics))}",
                f"\tREAL(KIND = DBL), PARAMETER :: TAU_INDEX_MIN = {_fortran_float(numerics.tau_t_min)}",
                f"\tREAL(KIND = DBL), PARAMETER :: TAU_INDEX_STEP = {_fortran_float(tau_index_step(numerics))}",
                "",
            ]
        )
    )
    return path


def _write_params_in(path: Path, params: SimulationParams, numerics: NumericsConfig, *, save_density: bool) -> Path:
    numerics = validate_numerics(numerics)
    values = (
        params.delta,
        params.epsilon,
        params.coupling,
        params.beta,
        params.omega_c,
        numerics.fortran_dtau,
        numerics.fortran_dt,
        numerics.fortran_cutoff,
        numerics.fortran_t_final,
    )
    path.write_text("".join(f"{_fortran_float(value)}\n" for value in values) + f"{1 if save_density else 0}\n")
    return path


def _stage_generated_branch(params: SimulationParams, branch: str, output_dir: Path, numerics: NumericsConfig, *, save_density: bool) -> Path:
    run_dir = output_dir / "runs" / branch
    ensure_clean_dir(run_dir)
    solver_id = _template_solver_id(params)
    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    source = asset(f"solvers/{solver_id}/{source_name}").read_text()
    (run_dir / source_name).write_text(_parameterize_source(source, branch=branch))
    _write_dimensions(run_dir / "dimensions.inc", params, numerics)
    _write_params_in(run_dir / "params.in", params, numerics, save_density=save_density)
    return run_dir


def _copy_generated_inputs_to_branch(input_files: dict[str, Path], run_dir: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for name, source_path in input_files.items():
        destination = run_dir / name
        shutil.copy2(source_path, destination)
        copied[name] = destination
    return copied


def _write_observable_input(params: SimulationParams, run_dir: Path) -> Path:
    spec = parse_observable(params.observable, params.N / 2.0)
    return write_complex_fortran(run_dir / "OBSERVABLE.dat", spec.fortran_matrix.reshape(-1))


def _compile_and_run(run_dir: Path, branch: str, config: FortranBuildConfig) -> dict[str, object]:
    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    binary_name = f"{branch}.out"
    compile_cmd = [config.compiler, *config.flags, source_name, "-o", binary_name, *config.link_args]
    try:
        compile_proc = _run(compile_cmd, run_dir)
    except subprocess.CalledProcessError as exc:
        (run_dir / "compile.stdout.log").write_text(exc.stdout or "")
        (run_dir / "compile.stderr.log").write_text(exc.stderr or "")
        raise FortranExecutionError(
            f"Fortran compile failed for {branch}. Command: {' '.join(compile_cmd)}. "
            f"See {run_dir / 'compile.stderr.log'}."
        ) from exc
    (run_dir / "compile.stdout.log").write_text(compile_proc.stdout)
    (run_dir / "compile.stderr.log").write_text(compile_proc.stderr)
    run_cmd = [f"./{binary_name}"]
    try:
        run_proc = _run(run_cmd, run_dir)
    except subprocess.CalledProcessError as exc:
        (run_dir / "run.stdout.log").write_text(exc.stdout or "")
        (run_dir / "run.stderr.log").write_text(exc.stderr or "")
        raise FortranExecutionError(
            f"Fortran run failed for {branch}. Command: {' '.join(run_cmd)}. "
            f"See {run_dir / 'run.stderr.log'}."
        ) from exc
    (run_dir / "run.stdout.log").write_text(run_proc.stdout)
    (run_dir / "run.stderr.log").write_text(run_proc.stderr)
    return {"compile_command": compile_cmd}


def _table_header(example, *, branch: str, columns: str) -> list[str]:
    params = example.parameters
    return [
        "Generated by master-equation-with-initial-correlations",
        f"example: {example.public_id}",
        f"family: {example.family}",
        f"branch: {branch}",
        f"bath: {example.bath}",
        f"model: {example.model}",
        f"spectral: {example.spectral}",
        f"observable: {example.observable}",
        f"N: {params['N']}",
        f"J: {params['J']}",
        f"epsilon0: {params['epsilon_0']}",
        f"epsilon: {params['epsilon']}",
        f"delta0: {params['Delta_0']}",
        f"delta: {params['Delta']}",
        f"beta: {params['beta']}",
        f"coupling: {params['G']}",
        f"omega_c: {params['omega_c']}",
        "normalization: lowercase spin operators are dimensionless: jx=J_x/J, jy=J_y/J, jz=J_z/J",
        f"columns: {columns}",
    ]


def _parameter_header(params: SimulationParams, *, branch: str, columns: str, numerics: NumericsConfig | None = None) -> list[str]:
    config = validate_numerics(numerics)
    initial_state_source = "user_supplied" if params.initial_state is not None else "generated_correlated_equilibrium"
    lines = [
        "Generated by master-equation-with-initial-correlations",
        "input_mode: generated from user parameters",
        f"branch: {branch}",
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
        f"initial_state_source: {initial_state_source}",
        "normalization: lowercase spin operators are dimensionless: jx=J_x/J, jy=J_y/J, jz=J_z/J",
        f"omega_nodes: {config.omega_nodes}",
        f"omega_max: {config.omega_max}",
        f"coefficient_omega_max: {config.coefficient_omega_max}",
        f"correlation_omega_max: {config.correlation_omega_max}",
        f"coefficient_time_step: {config.coefficient_time_step}",
        f"coefficient_t_min: {config.coefficient_t_min}",
        f"coefficient_t_max: {config.coefficient_t_max}",
        f"coefficient_points: {config.coefficient_points}",
        f"coefficient_index_step: {coefficient_index_step(config)}",
        f"correlation_tau_step: {config.correlation_tau_step}",
        f"correlation_tau_min: {config.correlation_tau_min}",
        f"correlation_tau_max: {config.correlation_tau_max}",
        f"correlation_tau_points: {config.tau_points}",
        f"tau_index_step: {tau_index_step(config)}",
        f"fortran_dtau: {config.fortran_dtau}",
        f"fortran_dt: {config.fortran_dt}",
        f"fortran_cutoff: {config.fortran_cutoff}",
        f"fortran_t_final: {config.fortran_t_final}",
        f"columns: {columns}",
    ]
    if params.spectral == "subohmic" and params.s is not None and abs(params.s - 0.5) <= 1.0e-14:
        lines.append("subohmic_s0p5_note: dissipative eta(tau) uses the legacy analytic paper-parity branch")
    if params.bath == "spin":
        lines.append("integraldatasimpson_convention: spin bath stores eta(tau), the dissipative kernel")
    else:
        lines.append("integraldatasimpson_convention: bosonic bath stores nu(tau), the noise kernel")
    return lines


def _write_public_table(example, src: Path, dst: Path, *, branch: str, columns: str) -> Path:
    table = np.loadtxt(src, dtype=float)
    if example.plot_divisor != 1.0 and "EXPX" in src.name:
        table = np.array(table, copy=True)
        table[:, 1] = table[:, 1] / example.plot_divisor
    return write_table(dst, table, header_lines=_table_header(example, branch=branch, columns=columns))


def _write_generated_public_table(params: SimulationParams, src: Path, dst: Path, *, branch: str, columns: str, numerics: NumericsConfig) -> Path:
    table = np.loadtxt(src, dtype=float)
    return write_table(dst, table, header_lines=_parameter_header(params, branch=branch, columns=columns, numerics=numerics))


def _observable_table_for_public(src: Path) -> tuple[np.ndarray, str]:
    table = np.loadtxt(src, dtype=float)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] != 3:
        raise ValueError(f"{src} expected columns t, observable_real, observable_imag.")
    if np.max(np.abs(table[:, 2])) <= 1.0e-10:
        return table[:, :2], "t observable"
    return table, "t observable_real observable_imag"


def _write_observable_public_table(params: SimulationParams, src: Path, dst: Path, *, branch: str, numerics: NumericsConfig) -> Path:
    table, columns = _observable_table_for_public(src)
    return write_table(dst, table, header_lines=_parameter_header(params, branch=branch, columns=columns, numerics=numerics))


def _reference_observable_table(example: ReferenceExample, *, branch: str) -> np.ndarray:
    observable = normalize_observable_expression(example.observable)
    if observable == "jz":
        filename = "EXPZ-C.DAT" if branch == "correlated" else "EXPZ-UNC.DAT"
        return load_table(f"{example.asset_dir}/tables/{filename}")
    filename = "EXPX-C.DAT" if branch == "correlated" else "EXPX-UNC.DAT"
    table = load_table(f"{example.asset_dir}/tables/{filename}")
    if observable == "jx^2" and example.plot_divisor != 1.0:
        table = np.array(table, copy=True)
        table[:, 1] = table[:, 1] / example.plot_divisor
    return table


def _copy_public_text_with_header(example, src_rel: str, dst: Path, *, columns: str) -> Path:
    header = "\n".join(f"# {line}" for line in _table_header(example, branch="input", columns=columns))
    body = asset(src_rel).read_text().rstrip()
    dst.write_text(f"{header}\n{body}\n")
    return dst


def _copy_generated_text_with_header(params: SimulationParams, src: Path, dst: Path, *, columns: str, numerics: NumericsConfig) -> Path:
    header = "\n".join(f"# {line}" for line in _parameter_header(params, branch="input", columns=columns, numerics=numerics))
    body = src.read_text().rstrip()
    dst.write_text(f"{header}\n{body}\n")
    return dst


def _input_file_columns(params: SimulationParams, name: str) -> str:
    if name == "A.dat":
        return "complex A(t) coefficient on the generated coefficient time grid"
    if name == "B.dat":
        return "complex B(t) coefficient on the generated coefficient time grid"
    if name == "C.dat":
        return "complex C(t) coefficient on the generated coefficient time grid"
    if name == "bathcorrelation.dat":
        return "nu(tau) eta(tau) on the generated bath-correlation tau grid"
    if name == "integraldatasimpson.dat":
        if params.bath == "spin":
            return "legacy one-column eta(tau) table for spin-bath compatibility"
        return "legacy one-column nu(tau) table for bosonic-bath compatibility"
    if name == "INSTATE.dat":
        return "flattened initial density matrix; complex Fortran input ordered row-by-row"
    if name == "OBSERVABLE.dat":
        return "flattened observable matrix supplied to Fortran; expectation is divided by J internally"
    return name


def _copy_public_sources_and_logs(correlated_dir: Path, uncorrelated_dir: Path, output_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    sources_dir = output_dir / "sources"
    logs_dir = output_dir / "logs"
    sources_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_files = {
        "correlated_solver": sources_dir / "correlated_solver.f",
        "uncorrelated_solver": sources_dir / "uncorrelated_solver.f",
    }
    shutil.copy2(correlated_dir / "test6wc.f", source_files["correlated_solver"])
    shutil.copy2(uncorrelated_dir / "test6woc.f", source_files["uncorrelated_solver"])

    log_files: dict[str, Path] = {}
    for label, run_dir in (("correlated", correlated_dir), ("uncorrelated", uncorrelated_dir)):
        for name in ("compile.stdout.log", "compile.stderr.log", "run.stdout.log", "run.stderr.log"):
            src = run_dir / name
            if src.exists():
                key = f"{label}_{name.removesuffix('.log').replace('.', '_')}"
                dst = logs_dir / f"{label}_{name}"
                shutil.copy2(src, dst)
                log_files[key] = dst
    return source_files, log_files


def _copy_public_branch_sources_and_logs(run_dir: Path, output_dir: Path, branch: str) -> tuple[dict[str, Path], dict[str, Path]]:
    sources_dir = output_dir / "sources"
    logs_dir = output_dir / "logs"
    sources_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_name = "test6wc.f" if branch == "correlated" else "test6woc.f"
    source_files = {
        f"{branch}_solver": sources_dir / f"{branch}_solver.f",
        "dimensions_include": sources_dir / "dimensions.inc",
    }
    shutil.copy2(run_dir / source_name, source_files[f"{branch}_solver"])
    shutil.copy2(run_dir / "dimensions.inc", source_files["dimensions_include"])

    log_files: dict[str, Path] = {}
    for name in ("compile.stdout.log", "compile.stderr.log", "run.stdout.log", "run.stderr.log"):
        src = run_dir / name
        if src.exists():
            key = f"{branch}_{name.removesuffix('.log').replace('.', '_')}"
            dst = logs_dir / f"{branch}_{name}"
            shutil.copy2(src, dst)
            log_files[key] = dst
    return source_files, log_files


def _write_branch_public_outputs(
    params: SimulationParams,
    run_dir: Path,
    output_dir: Path,
    *,
    branch: str,
    numerics: NumericsConfig,
    save_density: bool,
) -> dict[str, Path]:
    observable_src = run_dir / _observable_output_name(branch)
    public_name = "observable-correlated.dat" if branch == "correlated" else "observable-uncorrelated.dat"
    public_key = "correlated" if branch == "correlated" else "uncorrelated"
    output_files = {
        public_key: _write_observable_public_table(
            params,
            observable_src,
            output_dir / public_name,
            branch=branch,
            numerics=numerics,
        )
    }

    observable_expression = normalize_observable_expression(params.observable)
    if observable_expression == "jx":
        legacy_name = "EXPX-C.DAT" if branch == "correlated" else "EXPX-UNC.DAT"
        if (run_dir / legacy_name).exists():
            output_files[f"legacy_jx_{public_key}"] = _write_generated_public_table(
                params,
                run_dir / legacy_name,
                output_dir / legacy_name,
                branch=branch,
                columns=f"t legacy_jx_{public_key}",
                numerics=numerics,
            )
    if observable_expression == "jz":
        legacy_name = "EXPZ-C.DAT" if branch == "correlated" else "EXPZ-UNC.DAT"
        if (run_dir / legacy_name).exists():
            output_files[f"jz_{public_key}"] = _write_generated_public_table(
                params,
                run_dir / legacy_name,
                output_dir / legacy_name,
                branch=branch,
                columns=f"t jz_{public_key}",
                numerics=numerics,
            )
    if save_density:
        density_name = _density_output_name(branch)
        output_files[f"density_{public_key}"] = _copy_generated_text_with_header(
            params,
            run_dir / density_name,
            output_dir / density_name,
            columns="t followed by row-major real/imag pairs for the reduced system density matrix",
            numerics=numerics,
        )
    return output_files


def _copy_public_inputs(params: SimulationParams, run_dir: Path, output_dir: Path, *, numerics: NumericsConfig) -> dict[str, Path]:
    input_files: dict[str, Path] = {}
    for input_name in ("A.dat", "B.dat", "C.dat", "integraldatasimpson.dat", "bathcorrelation.dat", "INSTATE.dat", "OBSERVABLE.dat"):
        input_files[input_name] = _copy_generated_text_with_header(
            params,
            run_dir / input_name,
            output_dir / input_name,
            columns=_input_file_columns(params, input_name),
            numerics=numerics,
        )
    input_files["params.in"] = _copy_generated_text_with_header(
        params,
        run_dir / "params.in",
        output_dir / "params.in",
        columns="Fortran run constants: delta, epsilon, coupling, beta, omega_c, dtau, dt, cutoff, t_final, save_density",
        numerics=numerics,
    )
    return input_files


def _run_generated_branch(
    params: SimulationParams,
    output_dir: Path,
    *,
    branch: str,
    numerics: NumericsConfig,
    save_density: bool,
    config: FortranBuildConfig,
    verbose: bool,
    input_files: dict[str, Path] | None = None,
) -> _GeneratedBranchRun:
    run_dir = _stage_generated_branch(params, branch, output_dir, numerics, save_density=save_density)
    _emit(f"[meic] raw {branch} run directory: {run_dir}", verbose=verbose)
    if input_files is None:
        branch_inputs = write_generated_input_files(params, run_dir, numerics)
        branch_inputs["OBSERVABLE.dat"] = _write_observable_input(params, run_dir)
    else:
        branch_inputs = _copy_generated_inputs_to_branch(input_files, run_dir)
    _emit(f"[meic] compiling/running {branch} Fortran branch", verbose=verbose)
    branch_meta = _compile_and_run(run_dir, branch, config)
    output_files = _write_branch_public_outputs(
        params,
        run_dir,
        output_dir,
        branch=branch,
        numerics=numerics,
        save_density=save_density,
    )
    source_files, log_files = _copy_public_branch_sources_and_logs(run_dir, output_dir, branch)
    return _GeneratedBranchRun(
        branch=branch,
        run_dir=run_dir,
        output_files=output_files,
        source_files=source_files,
        log_files=log_files,
        input_files=branch_inputs,
        metadata=branch_meta,
    )


def run_parameterized_fortran_branch(
    params: SimulationParams,
    output_dir: str | Path,
    *,
    branch: str,
    overwrite: bool = False,
    verbose: bool = True,
    save_density: bool = False,
    numerics: NumericsConfig | None = None,
) -> RerunResult:
    params = normalize_simulation_params(params)
    if branch not in {"correlated", "uncorrelated"}:
        raise ValueError("branch must be 'correlated' or 'uncorrelated'.")
    numerics = validate_numerics(numerics)
    config = _build_config()
    compiler_path = shutil.which(config.compiler)
    if compiler_path is None:
        raise RuntimeError(f"Fortran compiler {config.compiler!r} was not found. Run `meic doctor` for details.")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite, generated_names=RERUN_OUTPUTS)

    _emit("[meic] generated coefficient, correlation, initial-state, and observable input files", verbose=verbose)
    generated = _run_generated_branch(
        params,
        output_dir,
        branch=branch,
        numerics=numerics,
        save_density=save_density,
        config=config,
        verbose=verbose,
    )
    input_files = _copy_public_inputs(params, generated.run_dir, output_dir, numerics=numerics)

    summary = {
        "input_mode": "generated",
        "parameters": _serializable_params(params),
        "numerics": numerics_summary(numerics),
        "reference_example": None,
        "compiler": compiler_path,
        "build": asdict(config),
        "fortran": {
            "template_solver": _template_solver_id(params),
            "save_density": save_density,
            branch: generated.metadata,
            "verification_performed": False,
        },
        "public_outputs": {key: str(path) for key, path in generated.output_files.items()},
        "public_inputs": {key: str(path) for key, path in input_files.items()},
        "public_sources": {key: str(path) for key, path in generated.source_files.items()},
        "logs": {key: str(path) for key, path in generated.log_files.items()},
    }
    summary_path = output_dir / "regeneration_summary.json"
    write_json(summary_path, summary)
    _emit(f"[meic] saved branch outputs in {output_dir}", verbose=verbose)
    return RerunResult(
        example=None,
        output_dir=output_dir,
        correlated_error=None,
        uncorrelated_error=None,
        jz_correlated_error=None,
        jz_uncorrelated_error=None,
        summary_path=summary_path,
        verification_performed=False,
        output_files=generated.output_files,
        input_files=input_files,
        source_files=generated.source_files,
        log_files=generated.log_files,
    )


def run_parameterized_fortran(
    params: SimulationParams,
    output_dir: str | Path,
    *,
    example: ReferenceExample | None = None,
    verify: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    save_density: bool = False,
    numerics: NumericsConfig | None = None,
) -> RerunResult:
    params = normalize_simulation_params(params)
    numerics = validate_numerics(numerics)
    config = _build_config()
    compiler_path = shutil.which(config.compiler)
    if compiler_path is None:
        raise RuntimeError(f"Fortran compiler {config.compiler!r} was not found. Run `meic doctor` for details.")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite, generated_names=RERUN_OUTPUTS)

    _emit("[meic] generated coefficient, correlation, initial-state, and observable input files", verbose=verbose)
    correlated_run = _run_generated_branch(
        params,
        output_dir,
        branch="correlated",
        numerics=numerics,
        save_density=save_density,
        config=config,
        verbose=verbose,
    )
    uncorrelated_run = _run_generated_branch(
        params,
        output_dir,
        branch="uncorrelated",
        numerics=numerics,
        save_density=save_density,
        config=config,
        verbose=verbose,
        input_files=correlated_run.input_files,
    )
    correlated_dir = correlated_run.run_dir
    uncorrelated_dir = uncorrelated_run.run_dir

    correlated, _ = _observable_table_for_public(correlated_dir / "OBSERVABLE-C.DAT")
    uncorrelated, _ = _observable_table_for_public(uncorrelated_dir / "OBSERVABLE-UNC.DAT")
    verification_performed = bool(verify and example is not None)
    correlated_error = None
    uncorrelated_error = None
    if verification_performed:
        reference_correlated = _reference_observable_table(example, branch="correlated")
        reference_uncorrelated = _reference_observable_table(example, branch="uncorrelated")
        correlated_error = _max_abs(correlated, reference_correlated)
        uncorrelated_error = _max_abs(uncorrelated, reference_uncorrelated)

    jz_correlated_error = None
    jz_uncorrelated_error = None
    if (correlated_dir / "EXPZ-C.DAT").exists() and verification_performed:
        jz_correlated = np.loadtxt(correlated_dir / "EXPZ-C.DAT", dtype=float)
        jz_reference = load_table(f"{example.asset_dir}/tables/EXPZ-C.DAT")
        jz_correlated_error = _max_abs(jz_correlated, jz_reference)
    if (uncorrelated_dir / "EXPZ-UNC.DAT").exists() and verification_performed:
        jz_uncorrelated = np.loadtxt(uncorrelated_dir / "EXPZ-UNC.DAT", dtype=float)
        jz_reference = load_table(f"{example.asset_dir}/tables/EXPZ-UNC.DAT")
        jz_uncorrelated_error = _max_abs(jz_uncorrelated, jz_reference)

    output_files = {**correlated_run.output_files, **uncorrelated_run.output_files}
    input_files = _copy_public_inputs(params, correlated_dir, output_dir, numerics=numerics)
    source_files = {**correlated_run.source_files, **uncorrelated_run.source_files}
    log_files = {**correlated_run.log_files, **uncorrelated_run.log_files}

    summary = {
        "input_mode": "generated",
        "parameters": _serializable_params(params),
        "numerics": numerics_summary(numerics),
        "reference_example": example.public_id if example else None,
        "compiler": compiler_path,
        "build": asdict(config),
        "fortran": {
            "template_solver": _template_solver_id(params),
            "save_density": save_density,
            "correlated": {**correlated_run.metadata, "max_abs_error": correlated_error},
            "uncorrelated": {**uncorrelated_run.metadata, "max_abs_error": uncorrelated_error},
            "jz_correlated_error": jz_correlated_error,
            "jz_uncorrelated_error": jz_uncorrelated_error,
            "verification_performed": verification_performed,
        },
        "public_outputs": {key: str(path) for key, path in output_files.items()},
        "public_inputs": {key: str(path) for key, path in input_files.items()},
        "public_sources": {key: str(path) for key, path in source_files.items()},
        "logs": {key: str(path) for key, path in log_files.items()},
    }
    summary_path = output_dir / "regeneration_summary.json"
    write_json(summary_path, summary)
    _emit(f"[meic] saved public outputs in {output_dir}", verbose=verbose)
    _emit(f"[meic] saved summary {summary_path}", verbose=verbose)
    return RerunResult(
        example=example,
        output_dir=output_dir,
        correlated_error=correlated_error,
        uncorrelated_error=uncorrelated_error,
        jz_correlated_error=jz_correlated_error,
        jz_uncorrelated_error=jz_uncorrelated_error,
        summary_path=summary_path,
        verification_performed=verification_performed,
        output_files=output_files,
        input_files=input_files,
        source_files=source_files,
        log_files=log_files,
    )


def rerun_example(
    example_id: str,
    output_dir: str | Path,
    *,
    verify: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> RerunResult:
    example = get_example(example_id)

    config = _build_config()
    compiler_path = shutil.which(config.compiler)
    if compiler_path is None:
        raise RuntimeError(f"Fortran compiler {config.compiler!r} was not found. Run `meic doctor` for details.")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite, generated_names=RERUN_OUTPUTS)

    correlated_dir = _stage_branch(example.id, "correlated", output_dir)
    uncorrelated_dir = _stage_branch(example.id, "uncorrelated", output_dir)
    correlated_meta = _compile_and_run(correlated_dir, "correlated", config)
    uncorrelated_meta = _compile_and_run(uncorrelated_dir, "uncorrelated", config)

    correlated = np.loadtxt(correlated_dir / "EXPX-C.DAT", dtype=float)
    uncorrelated = np.loadtxt(uncorrelated_dir / "EXPX-UNC.DAT", dtype=float)
    reference_correlated = load_table(f"{example.asset_dir}/tables/EXPX-C.DAT")
    reference_uncorrelated = load_table(f"{example.asset_dir}/tables/EXPX-UNC.DAT")

    correlated_error = _max_abs(correlated, reference_correlated) if verify else None
    uncorrelated_error = _max_abs(uncorrelated, reference_uncorrelated) if verify else None

    jz_correlated_error = None
    jz_uncorrelated_error = None
    if (correlated_dir / "EXPZ-C.DAT").exists():
        jz_correlated = np.loadtxt(correlated_dir / "EXPZ-C.DAT", dtype=float)
        jz_reference = load_table(f"{example.asset_dir}/tables/EXPZ-C.DAT")
        jz_correlated_error = _max_abs(jz_correlated, jz_reference) if verify else None
    if (uncorrelated_dir / "EXPZ-UNC.DAT").exists():
        jz_uncorrelated = np.loadtxt(uncorrelated_dir / "EXPZ-UNC.DAT", dtype=float)
        jz_reference = load_table(f"{example.asset_dir}/tables/EXPZ-UNC.DAT")
        jz_uncorrelated_error = _max_abs(jz_uncorrelated, jz_reference) if verify else None

    exact_correlated_error = None
    exact_uncorrelated_error = None
    exact_correlated = None
    exact_uncorrelated = None
    if example.supports_exact:
        params = PureDephasingParams(
            J=float(example.parameters["J"]),
            epsilon=float(example.parameters["epsilon"]),
            xi=float(example.parameters["epsilon_0"]),
            beta=float(example.parameters["beta"]),
            G=float(example.parameters["G"]),
            omega_c=float(example.parameters["omega_c"]),
        )
        exact_correlated, exact_uncorrelated = exact_curves(params)
        if verify:
            ref_exact_correlated = load_table(f"{example.asset_dir}/tables/exact-correlated.dat")
            ref_exact_uncorrelated = load_table(f"{example.asset_dir}/tables/exact-uncorrelated.dat")
            exact_correlated_error = _max_abs(exact_correlated, ref_exact_correlated)
            exact_uncorrelated_error = _max_abs(exact_uncorrelated, ref_exact_uncorrelated)

    output_files = {
        "correlated": _write_public_table(
            example,
            correlated_dir / "EXPX-C.DAT",
            output_dir / "EXPX-C.DAT",
            branch="correlated",
            columns="t observable_correlated",
        ),
        "uncorrelated": _write_public_table(
            example,
            uncorrelated_dir / "EXPX-UNC.DAT",
            output_dir / "EXPX-UNC.DAT",
            branch="uncorrelated",
            columns="t observable_uncorrelated",
        ),
    }
    if (correlated_dir / "EXPZ-C.DAT").exists():
        output_files["jz_correlated"] = _write_public_table(
            example,
            correlated_dir / "EXPZ-C.DAT",
            output_dir / "EXPZ-C.DAT",
            branch="correlated",
            columns="t jz_correlated",
        )
    if (uncorrelated_dir / "EXPZ-UNC.DAT").exists():
        output_files["jz_uncorrelated"] = _write_public_table(
            example,
            uncorrelated_dir / "EXPZ-UNC.DAT",
            output_dir / "EXPZ-UNC.DAT",
            branch="uncorrelated",
            columns="t jz_uncorrelated",
        )

    input_files: dict[str, Path] = {}
    for input_name in ("A.dat", "B.dat", "C.dat", "integraldatasimpson.dat"):
        input_files[input_name] = _copy_public_text_with_header(
            example,
            f"{example.asset_dir}/inputs/{input_name}",
            output_dir / input_name,
            columns=input_name,
        )
    instate = next(name for name in ("INSTATE-J=0.5.dat", "INSTATE-J=1.dat", "INSTATE-J=2.dat", "INSTATE-J=5.dat") if (correlated_dir / name).exists())
    instate_header = "\n".join(f"# {line}" for line in _table_header(example, branch="input", columns=f"{instate} flattened density matrix"))
    instate_body = (correlated_dir / instate).read_text().rstrip()
    input_files[instate] = output_dir / instate
    input_files[instate].write_text(f"{instate_header}\n{instate_body}\n")
    source_files, log_files = _copy_public_sources_and_logs(correlated_dir, uncorrelated_dir, output_dir)

    summary = {
        "example_id": example.public_id,
        "compiler": compiler_path,
        "build": asdict(config),
        "fortran": {
            "correlated": {**correlated_meta, "max_abs_error": correlated_error},
            "uncorrelated": {**uncorrelated_meta, "max_abs_error": uncorrelated_error},
            "jz_correlated_error": jz_correlated_error,
            "jz_uncorrelated_error": jz_uncorrelated_error,
            "verification_performed": verify,
        },
        "exact": {
            "correlated_max_abs_error": exact_correlated_error,
            "uncorrelated_max_abs_error": exact_uncorrelated_error,
        },
        "public_outputs": {key: str(path) for key, path in output_files.items()},
        "public_inputs": {key: str(path) for key, path in input_files.items()},
        "public_sources": {key: str(path) for key, path in source_files.items()},
        "logs": {key: str(path) for key, path in log_files.items()},
    }
    summary_path = output_dir / "regeneration_summary.json"
    write_json(summary_path, summary)
    return RerunResult(
        example=example,
        output_dir=output_dir,
        correlated_error=correlated_error,
        uncorrelated_error=uncorrelated_error,
        jz_correlated_error=jz_correlated_error,
        jz_uncorrelated_error=jz_uncorrelated_error,
        exact_correlated_error=exact_correlated_error,
        exact_uncorrelated_error=exact_uncorrelated_error,
        summary_path=summary_path,
        verification_performed=verify,
        output_files=output_files,
        input_files=input_files,
        source_files=source_files,
        log_files=log_files,
    )
