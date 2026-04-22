import numpy as np

from master_equation_initial_correlations import load_reference_curves


def test_load_reference_curves_pure_dephasing_has_exact() -> None:
    curves = load_reference_curves("pure_dephasing_ohmic_j0p5")
    assert curves.correlated.shape[1] == 2
    assert curves.uncorrelated.shape[1] == 2
    assert curves.exact_correlated is not None
    assert curves.exact_uncorrelated is not None


def test_load_reference_curves_jx2_is_plot_normalized() -> None:
    curves = load_reference_curves("jx2_ohmic_j2")
    assert np.max(curves.correlated[:, 1]) <= 1.1
    assert np.max(curves.uncorrelated[:, 1]) <= 1.1
