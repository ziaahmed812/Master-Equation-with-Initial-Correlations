from pathlib import Path

import numpy as np
import pytest

from master_equation_initial_correlations._resources import asset
from master_equation_initial_correlations.generated_inputs import (
    QuadratureConfig,
    _validate_generated_initial_state,
    coefficient_index_step,
    coefficient_times,
    generate_bath_correlation_tau,
    generate_coefficients,
    generate_inputs,
    generate_initial_state,
    generate_integral_tau,
    tau_index_step,
    tau_times,
    validate_numerics,
    write_generated_input_files,
)
from master_equation_initial_correlations.fortran_runner import _write_dimensions
from master_equation_initial_correlations.simulation import SimulationParams


def _read_fortran_complex(path: object, rows: int) -> np.ndarray:
    values: list[complex] = []
    text = path.read_text() if hasattr(path, "read_text") else Path(path).read_text()
    for line in text.splitlines()[:rows]:
        text = line.strip().strip("()")
        if not text:
            continue
        real, imag = text.split(",")
        values.append(float(real) + 1j * float(imag))
    return np.asarray(values)


def test_generated_ohmic_inputs_match_reference_branch() -> None:
    params = SimulationParams(
        bath="bosonic",
        model="spin-boson",
        spectral="ohmic",
        observable="jx",
        N=4,
    )
    coefficient_a, coefficient_b, coefficient_c = generate_coefficients(params)
    for name, generated in (("A.dat", coefficient_a), ("B.dat", coefficient_b), ("C.dat", coefficient_c)):
        reference = _read_fortran_complex(asset(f"figures/beyond_pure_dephasing_ohmic_j2/inputs/{name}"), rows=12)
        assert np.max(np.abs(generated[:12] - reference)) <= 5.0e-9

    integral = generate_integral_tau(params)
    reference_integral = np.loadtxt(asset("figures/beyond_pure_dephasing_ohmic_j2/inputs/integraldatasimpson.dat"), max_rows=12)
    assert np.max(np.abs(integral[:12] - reference_integral)) <= 1.0e-10

    initial_state = generate_initial_state(params)
    reference_state = np.loadtxt(asset("figures/beyond_pure_dephasing_ohmic_j2/inputs/INSTATE-J=2.dat"))
    assert np.max(np.abs(initial_state.reshape(-1).real - reference_state)) <= 2.0e-6
    assert np.max(np.abs(initial_state.imag)) <= 1.0e-14


def test_generated_subohmic_coefficients_preserve_legacy_c_cutoff() -> None:
    params = SimulationParams(
        bath="bosonic",
        model="spin-boson",
        spectral="subohmic",
        observable="jx",
        N=4,
        s=0.5,
    )
    coefficient_a, coefficient_b, coefficient_c = generate_coefficients(params)
    references = {
        "A.dat": coefficient_a,
        "B.dat": coefficient_b,
        "C.dat": coefficient_c,
    }
    for name, generated in references.items():
        reference = _read_fortran_complex(asset(f"figures/subohmic_s0p5_j2/inputs/{name}"), rows=12)
        assert np.max(np.abs(generated[:12] - reference)) <= 2.0e-5

    integral = generate_integral_tau(params)
    reference_integral = np.loadtxt(asset("figures/subohmic_s0p5_j2/inputs/integraldatasimpson.dat"), max_rows=12)
    assert abs(integral[0] - reference_integral[0]) <= 1.0e-8


def test_generated_superohmic_correlation_table_has_nu_and_eta_columns() -> None:
    params = SimulationParams(
        bath="bosonic",
        model="spin-boson",
        spectral="superohmic",
        observable="jx+jy",
        N=2,
        s=3.0,
    )
    table = generate_bath_correlation_tau(params)
    assert table.shape == (2001, 2)
    assert np.all(np.isfinite(table[:20]))


def test_generated_initial_state_fails_loudly_when_quadrature_is_unphysical() -> None:
    params = SimulationParams(
        bath="bosonic",
        model="spin-boson",
        spectral="superohmic",
        observable="jx",
        N=2,
        s=3.0,
    )
    numerics = QuadratureConfig(
        omega_nodes=2,
        omega_max=5.0,
        lambda_nodes=2,
        initial_state_omega_nodes=2,
        initial_state_lambda_nodes=2,
        initial_state_zeta_nodes=2,
    )
    with pytest.raises(ValueError, match="generated initial_state"):
        generate_initial_state(params, numerics)


def test_generated_initial_state_warns_before_symmetrizing_marginal_hermiticity_defect() -> None:
    params = SimulationParams(bath="bosonic", model="spin-boson", spectral="ohmic", observable="jx", N=1)
    matrix = np.asarray([[0.5, 2.0e-8], [0.0, 0.5]], dtype=complex)
    with pytest.warns(RuntimeWarning, match="Hermitian symmetrization"):
        validated = _validate_generated_initial_state(matrix, params)
    np.testing.assert_allclose(validated, validated.conj().T)


def test_user_supplied_initial_state_is_written_as_complex_fortran_input(tmp_path: Path) -> None:
    params = SimulationParams(
        bath="bosonic",
        model="spin-boson",
        spectral="ohmic",
        observable="jx",
        N=2,
        initial_state=np.eye(3, dtype=complex) / 3.0,
    )
    numerics = QuadratureConfig(
        omega_nodes=8,
        omega_max=20.0,
        lambda_nodes=4,
        initial_state_omega_nodes=4,
        initial_state_lambda_nodes=3,
        initial_state_zeta_nodes=3,
        coefficient_time_step=1.25,
        correlation_tau_step=1.25,
    )
    generated = generate_inputs(params, numerics)
    assert generated.initial_state_source == "user_supplied"
    files = write_generated_input_files(params, tmp_path, numerics)
    assert files["INSTATE.dat"].read_text().splitlines()[0].startswith("(")


def test_custom_tau_grid_is_explicitly_configurable() -> None:
    numerics = QuadratureConfig(correlation_tau_step=0.2, correlation_tau_max=2.0, fortran_t_final=2.0)
    times = tau_times(numerics)
    assert len(times) == 11
    assert times[-1] == 2.0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_numerics_reject_nonfinite_controls(bad_value: float) -> None:
    with pytest.raises(ValueError, match="omega_max must be finite"):
        validate_numerics(QuadratureConfig(omega_max=bad_value))


def test_numerics_rejects_nondividing_time_steps() -> None:
    with pytest.raises(ValueError, match="coefficient time grid step must exactly divide"):
        validate_numerics(
            QuadratureConfig(
                coefficient_t_min=0.0,
                coefficient_t_max=1.0,
                coefficient_time_step=0.3,
                fortran_t_final=1.0,
            )
        )
    with pytest.raises(ValueError, match="bath-correlation tau grid step must exactly divide"):
        validate_numerics(
            QuadratureConfig(
                correlation_tau_min=0.0,
                correlation_tau_max=1.0,
                correlation_tau_step=0.3,
                fortran_t_final=1.0,
            )
        )


def test_nonzero_time_grid_minima_are_reflected_in_index_steps_and_fortran_constants(tmp_path: Path) -> None:
    numerics = QuadratureConfig(
        coefficient_t_min=1.0,
        coefficient_t_max=2.0,
        coefficient_time_step=0.25,
        correlation_tau_min=0.5,
        correlation_tau_max=1.5,
        correlation_tau_step=0.25,
        fortran_t_final=1.5,
    )
    assert coefficient_times(numerics).tolist() == pytest.approx([1.0, 1.25, 1.5, 1.75, 2.0])
    assert tau_times(numerics).tolist() == pytest.approx([0.5, 0.75, 1.0, 1.25, 1.5])
    assert coefficient_index_step(numerics) == pytest.approx(0.25)
    assert tau_index_step(numerics) == pytest.approx(0.25)

    path = _write_dimensions(tmp_path / "dimensions.inc", SimulationParams(bath="bosonic", model="spin-boson", spectral="ohmic", N=2), numerics)
    text = path.read_text()
    assert "COEFF_INDEX_MIN" in text
    assert "TAU_INDEX_MIN" in text
