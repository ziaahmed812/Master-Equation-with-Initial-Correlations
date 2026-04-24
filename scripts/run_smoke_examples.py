import numpy as np
import master_equation_initial_correlations as meic


def main() -> None:
    exact_system = meic.SystemParams(N=1, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
    exact_bath = meic.BathParams(family="bosonic", kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)
    exact_tlist = np.linspace(0.0, 0.5, 6)
    exact_result = meic.exact.solve(exact_system, exact_bath, tlist=exact_tlist, e_ops=["jx"])

    numerics = meic.NumericsConfig(
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
    )
    system = meic.SystemParams(N=2, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
    bath = meic.BathParams(family="bosonic", kind="superohmic", s=3.0, beta=1.0, coupling=0.05, omega_c=5.0)
    tlist = np.linspace(0.0, 0.05, 6)
    bosonic_result = meic.solve(system, bath, tlist=tlist, e_ops=["jy"], numerics=numerics, verbose=False)

    print(exact_result.e_data["jx"][:2])
    print(bosonic_result.e_data["jy"][:2])


if __name__ == "__main__":
    main()
