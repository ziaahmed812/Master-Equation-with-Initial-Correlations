from master_equation_initial_correlations import get_preset, list_presets


def test_list_presets_excludes_heavy_by_default() -> None:
    presets = list_presets()
    ids = {preset.id for preset in presets}
    assert "pure_dephasing_ohmic_j0p5" in ids
    assert "pure_dephasing_ohmic_j5" not in ids
    assert "beta_compare_ohmic_beta0p5_j5" not in ids


def test_list_presets_can_include_advanced_without_heavy() -> None:
    presets = list_presets(include_advanced=True)
    ids = {preset.id for preset in presets}
    assert "beta_compare_ohmic_beta0p5_j5" in ids
    assert "spin_environment_ohmic_j5" in ids
    assert "pure_dephasing_ohmic_j5" not in ids


def test_get_preset_supports_aliases() -> None:
    preset = get_preset("Puredephasing-N=1-unitary")
    assert preset.id == "pure_dephasing_ohmic_j0p5"
    assert preset.paper_figure_number == 1
