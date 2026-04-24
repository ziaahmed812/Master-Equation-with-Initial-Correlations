import numpy as np

from master_equation_initial_correlations import PureDephasingParams, exact_curves, load_reference_curves


def test_exact_curves_match_packaged_reference() -> None:
    params = PureDephasingParams(J=0.5, epsilon=4.0, xi=4.0, beta=1.0, G=0.05, omega_c=5.0)
    correlated, uncorrelated = exact_curves(params)
    curves = load_reference_curves("pure-dephasing-ohmic-N1")

    assert curves.exact_correlated is not None
    assert curves.exact_uncorrelated is not None
    np.testing.assert_allclose(correlated, curves.exact_correlated, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(uncorrelated, curves.exact_uncorrelated, atol=1.0e-12, rtol=0.0)
