import json
from pathlib import Path

from master_equation_initial_correlations.cli import main


def test_cli_examples_uses_physical_language(capsys) -> None:
    code = main(["examples"])
    captured = capsys.readouterr()
    assert code == 0
    assert "bosonic bath" in captured.out
    assert "spin bath" in captured.out
    assert "pure-dephasing-ohmic-N1" in captured.out


def test_cli_examples_json(capsys) -> None:
    code = main(["examples", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload[0]["id"] == "pure-dephasing-ohmic-N1"


def test_cli_export(tmp_path: Path, capsys) -> None:
    code = main(["export", "pure-dephasing-ohmic-N4", "--out", str(tmp_path)])
    captured = capsys.readouterr()
    exported_dir = Path(captured.out.strip())
    assert code == 0
    assert exported_dir.exists()
    assert (exported_dir / "rendered" / "Puredephasing-N=4-unitary.eps").exists()


def test_cli_export_unknown_example_is_clean_error(capsys) -> None:
    code = main(["export", "not-a-real-example", "--out", "unused"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown reference example" in captured.err


def test_cli_run_exact_pure_dephasing(tmp_path: Path, capsys) -> None:
    code = main([
        "run",
        "--bath", "bosonic",
        "--model", "pure-dephasing",
        "--spectral", "ohmic",
        "--N", "1",
        "--epsilon0", "4",
        "--epsilon", "4",
        "--delta0", "0",
        "--delta", "0",
        "--out", str(tmp_path),
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["source"] == "exact pure-dephasing Python solver"
    assert (tmp_path / "exact-correlated.dat").exists()
