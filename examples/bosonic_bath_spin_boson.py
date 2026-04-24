import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=2, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = meic.BathParams(family="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
numerics = meic.NumericsConfig(
    omega_nodes=48,
    lambda_nodes=12,
    instate_omega_nodes=24,
    instate_lambda_nodes=8,
    instate_zeta_nodes=8,
    omega_max_coefficients=80.0,
    omega_max_tau=80.0,
    omega_max_instate=80.0,
    coefficient_points=81,
    tau_points=81,
)

tlist = np.linspace(0.0, 0.2, 21)
e_ops = ["jx", "jz"]

wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with", numerics=numerics)
woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without", numerics=numerics)

print(wc_result.e_data["jx"][:5])
print(woc_result.e_data["jx"][:5])
