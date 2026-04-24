from pathlib import Path

from master_equation_initial_correlations import (
    BathParams,
    BosonicBathSolver,
    NumericsConfig,
    PureDephasingSolver,
    RunConfig,
    SystemParams,
)


def main() -> None:
    root = Path("smoke-output")
    exact = PureDephasingSolver(
        system=SystemParams(N=1, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0),
        bath=BathParams(kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0),
        observable="jx",
    )
    exact.run(RunConfig(output_dir=root / "exact-pure-dephasing", overwrite=True, verbose=False))

    numerics = NumericsConfig(
        omega_nodes=24,
        lambda_nodes=8,
        instate_omega_nodes=12,
        instate_lambda_nodes=6,
        instate_zeta_nodes=6,
        omega_max_coefficients=60.0,
        omega_max_tau=60.0,
        omega_max_instate=60.0,
        coefficient_points=41,
        tau_points=41,
        fortran_t_final=0.05,
    )
    bosonic = BosonicBathSolver(
        system=SystemParams(N=2, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5),
        bath=BathParams(kind="superohmic", s=3.0, beta=1.0, coupling=0.05, omega_c=5.0),
        observable="jy",
    )
    bosonic.run(RunConfig(output_dir=root / "bosonic-superohmic", overwrite=True, verbose=False, verify=False, numerics=numerics))
    print(root)


if __name__ == "__main__":
    main()
