import json
from pathlib import Path

import pytest

from master_equation_initial_correlations import doctor
from master_equation_initial_correlations.cli import main


def test_cli_doctor_reports_json(capsys) -> None:
    code = main(["doctor"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert "compiler" in payload


def test_cli_run_exact_pure_dephasing_saves_result(tmp_path: Path, capsys) -> None:
    code = main([
        "run",
        "--model", "pure-dephasing",
        "--branch", "wc",
        "--N", "1",
        "--epsilon0", "4",
        "--epsilon", "4",
        "--delta0", "0",
        "--delta", "0",
        "--tmax", "0.2",
        "--dt", "0.1",
        "--out", str(tmp_path),
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["branch"] == "with_correlations"
    assert (tmp_path / "expect-jx.dat").exists()


def test_cli_run_defaults_to_current_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = main([
        "run",
        "--model", "pure-dephasing",
        "--branch", "wc",
        "--N", "1",
        "--epsilon0", "4",
        "--epsilon", "4",
        "--delta0", "0",
        "--delta", "0",
        "--tmax", "0.1",
        "--dt", "0.1",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["output_dir"] == "."
    assert (tmp_path / "expect-jx.dat").exists()


def test_cli_rejects_incompatible_bath_model(capsys) -> None:
    code = main([
        "run",
        "--bath-type", "spin",
        "--model", "spin-boson",
        "--N", "2",
        "--tmax", "0.1",
        "--dt", "0.1",
        "--out", "unused",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "spin-environment" in captured.err


@pytest.mark.solver_rerun
def test_cli_auto_model_routes_spin_bath(tmp_path: Path, capsys) -> None:
    if not doctor()["compiler_found"]:
        pytest.skip("gfortran not available")
    code = main([
        "run",
        "--bath-type", "spin",
        "--N", "2",
        "--tmax", "0.01",
        "--dt", "0.01",
        "--omega-nodes", "8",
        "--lambda-nodes", "4",
        "--initial-state-omega-nodes", "4",
        "--initial-state-lambda-nodes", "3",
        "--initial-state-zeta-nodes", "3",
        "--out", str(tmp_path),
        "--quiet",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["solver"] == "SpinBathSolverWC"
    assert (tmp_path / "expect-jx.dat").exists()
