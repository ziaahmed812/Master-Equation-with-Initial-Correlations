import numpy as np

import master_equation_initial_correlations as meic
from master_equation_initial_correlations.pure_dephasing import PureDephasingParams, exact_curves
from master_equation_initial_correlations.reference import load_reference_curves


def test_exact_curves_match_packaged_reference() -> None:
    params = PureDephasingParams(J=0.5, epsilon=4.0, xi=4.0, beta=1.0, G=0.05, omega_c=5.0)
    correlated, uncorrelated = exact_curves(params)
    curves = load_reference_curves("pure-dephasing-ohmic-N1")

    assert curves.exact_correlated is not None
    assert curves.exact_uncorrelated is not None
    np.testing.assert_allclose(correlated, curves.exact_correlated, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(uncorrelated, curves.exact_uncorrelated, atol=1.0e-12, rtol=0.0)


def test_exact_public_api_matches_reference_n4() -> None:
    curves = load_reference_curves("pure-dephasing-ohmic-N4")
    assert curves.exact_correlated is not None
    assert curves.exact_uncorrelated is not None
    correlated_tlist = curves.exact_correlated[:, 0].copy()
    correlated_tlist[0] = 0.0
    uncorrelated_tlist = np.arange(0.0, 5.0, 0.1)
    system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
    bath = meic.BathParams(family="bosonic", kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)

    wc = meic.exact.solve(system, bath, tlist=correlated_tlist, e_ops=["jx"], correlations="with")
    woc = meic.exact.solve(system, bath, tlist=uncorrelated_tlist, e_ops=["jx"], correlations="without")

    np.testing.assert_allclose(wc.e_data["jx"], curves.exact_correlated[:, 1], atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(woc.e_data["jx"][1::2], curves.exact_uncorrelated[:, 1], atol=1.0e-10, rtol=0.0)
