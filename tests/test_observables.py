import numpy as np
import pytest

from master_equation_initial_correlations import ObservableParseError, parse_observable
from master_equation_initial_correlations._types import SimulationParams
from master_equation_initial_correlations.simulation import run_simulation


def test_jx2_alias_matches_explicit_square() -> None:
    alias = parse_observable("jx2", J=2.0)
    parenthesized = parse_observable("jx(2)", J=2.0)
    explicit = parse_observable("jx^2", J=2.0)
    assert np.max(np.abs(alias.dimensionless_matrix - explicit.dimensionless_matrix)) <= 1.0e-14
    assert np.max(np.abs(parenthesized.dimensionless_matrix - explicit.dimensionless_matrix)) <= 1.0e-14


def test_safe_observable_expression_supports_operator_sums() -> None:
    spec = parse_observable("jx/2 + jy - 2*jz", J=2.0)
    assert spec.dimensionless_matrix.shape == (5, 5)
    assert np.max(np.abs(spec.dimensionless_matrix - spec.dimensionless_matrix.conj().T)) <= 1.0e-14


def test_observable_parser_rejects_unsafe_syntax() -> None:
    with pytest.raises(ObservableParseError):
        parse_observable("__import__('os').system('echo bad')", J=1.0)


def test_custom_observable_matrix_shape_is_validated() -> None:
    with pytest.raises(ObservableParseError, match="expected"):
        parse_observable(np.eye(3), J=2.0)


def test_nonhermitian_observable_warns_user(tmp_path) -> None:
    with pytest.warns(RuntimeWarning, match="non-Hermitian"):
        run_simulation(
            SimulationParams(
                bath="bosonic",
                model="pure-dephasing",
                spectral="ohmic",
                observable="jx*jy",
                N=1,
                epsilon=4.0,
                delta0=0.0,
                delta=0.0,
            ),
            tmp_path,
            verify=False,
            verbose=False,
        )
