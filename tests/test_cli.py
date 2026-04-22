import json
from pathlib import Path

from master_equation_initial_correlations.cli import main


def test_cli_list_default_hides_advanced_and_heavy(capsys) -> None:
    code = main(["list"])
    captured = capsys.readouterr()
    assert code == 0
    assert "pure_dephasing_ohmic_j0p5" in captured.out
    assert "beta_compare_ohmic_beta0p5_j5" not in captured.out
    assert "pure_dephasing_ohmic_j5" not in captured.out


def test_cli_list_can_include_advanced_and_heavy(capsys) -> None:
    code = main(["list", "--include-advanced", "--include-heavy"])
    captured = capsys.readouterr()
    assert code == 0
    assert "beta_compare_ohmic_beta0p5_j5" in captured.out
    assert "pure_dephasing_ohmic_j5" in captured.out


def test_cli_show_json(capsys) -> None:
    code = main(["show", "pure_dephasing_ohmic_j0p5", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["id"] == "pure_dephasing_ohmic_j0p5"


def test_cli_export(tmp_path: Path, capsys) -> None:
    code = main(["export", "pure_dephasing_ohmic_j2", "--out", str(tmp_path)])
    captured = capsys.readouterr()
    exported_dir = Path(captured.out.strip())
    assert code == 0
    assert exported_dir.exists()
    assert (exported_dir / "rendered" / "Puredephasing-N=4-unitary.eps").exists()

