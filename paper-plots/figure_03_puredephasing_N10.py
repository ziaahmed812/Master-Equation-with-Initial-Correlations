import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=10, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(bath_type="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 4.0, 41)
e_ops = ["jx"]
paper_numerics = meic.NumericsConfig(
    coefficient_omega_max=500.0,
    correlation_omega_max=510.0,
    initial_state_omega_max=500.0,
    omega_nodes=500,
    lambda_nodes=100,
    initial_state_omega_nodes=260,
    initial_state_lambda_nodes=40,
    initial_state_zeta_nodes=40,
    coefficient_time_step=0.0025,
    correlation_tau_step=0.0025,
)

exact_wc_result = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
exact_woc_result = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")
me_wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with", numerics=paper_numerics)
me_woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without", numerics=paper_numerics)

print(me_wc_result.e_data["jx"][:5])
print(me_woc_result.e_data["jx"][:5])
