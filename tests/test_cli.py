import json
from pathlib import Path

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
