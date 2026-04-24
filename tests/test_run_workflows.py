from pathlib import Path

import pytest

from master_equation_initial_correlations import SimulationParams, doctor, run_simulation


def _compiler_ready() -> bool:
    return bool(doctor()["compiler_found"])


def test_run_pure_dephasing_free_form_n1(tmp_path: Path) -> None:
    result = run_simulation(
        SimulationParams(
            bath="bosonic",
            model="pure-dephasing",
            spectral="ohmic",
            N=1,
            epsilon0=4.0,
            epsilon=4.0,
            delta0=0.0,
            delta=0.0,
        ),
        tmp_path / "pure-n1",
    )
    assert result.exact_correlated_error is not None
    assert result.exact_correlated_error <= 1.0e-12


def test_run_pure_dephasing_free_form_n4(tmp_path: Path) -> None:
    result = run_simulation(
        SimulationParams(
            bath="bosonic",
            model="pure-dephasing",
            spectral="ohmic",
            N=4,
            epsilon0=4.0,
            epsilon=4.0,
            delta0=0.0,
            delta=0.0,
        ),
        tmp_path / "pure-n4",
    )
    assert result.exact_correlated_error is not None
    assert result.exact_correlated_error <= 1.0e-12


@pytest.mark.solver_rerun
def test_run_bosonic_spin_boson_n2(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = run_simulation(
        SimulationParams(bath="bosonic", model="spin-boson", spectral="ohmic", observable="jx", N=2),
        tmp_path / "spin-boson-n2",
        verify=True,
    )
    assert result.correlated_error is not None
    assert result.correlated_error <= 1.0e-12
    assert result.uncorrelated_error is not None
    assert result.uncorrelated_error <= 1.0e-12


@pytest.mark.solver_rerun
def test_run_bosonic_spin_boson_n4(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = run_simulation(
        SimulationParams(bath="bosonic", model="spin-boson", spectral="ohmic", observable="jx", N=4),
        tmp_path / "spin-boson-n4",
        verify=True,
    )
    assert result.correlated_error is not None
    assert result.correlated_error <= 1.0e-12


@pytest.mark.solver_rerun
def test_run_bosonic_jx2_n4(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = run_simulation(
        SimulationParams(
            bath="bosonic",
            model="spin-boson",
            spectral="ohmic",
            observable="jx2",
            N=4,
            epsilon=3.5,
        ),
        tmp_path / "jx2-n4",
        verify=True,
    )
    assert result.correlated_error is not None
    assert result.correlated_error <= 1.0e-12


@pytest.mark.solver_rerun
def test_run_bosonic_subohmic_n4(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = run_simulation(
        SimulationParams(bath="bosonic", model="spin-boson", spectral="subohmic", observable="jx", N=4, s=0.5),
        tmp_path / "subohmic-n4",
        verify=True,
    )
    assert result.correlated_error is not None
    assert result.correlated_error <= 1.0e-12


@pytest.mark.solver_rerun
def test_run_spin_bath_n4(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = run_simulation(
        SimulationParams(bath="spin", model="spin-environment", spectral="ohmic", observable="jx", N=4),
        tmp_path / "spin-bath-n4",
        verify=True,
    )
    assert result.correlated_error is not None
    assert result.correlated_error <= 1.0e-12
