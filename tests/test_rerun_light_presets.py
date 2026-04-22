from pathlib import Path

import pytest

from master_equation_initial_correlations import doctor, rerun_preset


def _compiler_ready() -> bool:
    return bool(doctor()["compiler_found"])


@pytest.mark.light_rerun
def test_rerun_pure_dephasing_j0p5(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = rerun_preset("pure_dephasing_ohmic_j0p5", tmp_path / "f09", verify=True, render=False)
    assert result.correlated_error <= 1.0e-12
    assert result.uncorrelated_error <= 1.0e-12
    assert result.exact_correlated_error is not None
    assert result.exact_correlated_error <= 1.0e-12


@pytest.mark.light_rerun
def test_rerun_beyond_pd_j1(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = rerun_preset("beyond_pure_dephasing_ohmic_j1", tmp_path / "f02", verify=True, render=False)
    assert result.correlated_error <= 1.0e-12
    assert result.uncorrelated_error <= 1.0e-12


@pytest.mark.light_rerun
def test_rerun_pure_dephasing_j2(tmp_path: Path) -> None:
    if not _compiler_ready():
        pytest.skip("gfortran not available")
    result = rerun_preset("pure_dephasing_ohmic_j2", tmp_path / "f10", verify=True, render=False)
    assert result.correlated_error <= 1.0e-12
    assert result.uncorrelated_error <= 1.0e-12
    assert result.exact_correlated_error is not None
    assert result.exact_correlated_error <= 1.0e-12
