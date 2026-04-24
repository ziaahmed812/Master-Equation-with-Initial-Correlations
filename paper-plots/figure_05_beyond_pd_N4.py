import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = meic.BathParams(bath_type="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 5.0, 51)
e_ops = ["jx"]
paper_numerics = meic.NumericsConfig(
    omega_max=500.0,
    omega_nodes=500,
    lambda_nodes=100,
    initial_state_omega_nodes=260,
    initial_state_lambda_nodes=40,
    initial_state_zeta_nodes=40,
    coefficient_time_step=0.0025,
    correlation_tau_step=0.0025,
)

me_wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with", numerics=paper_numerics)
me_woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without", numerics=paper_numerics)

print(me_wc_result.e_data["jx"][:5])
print(me_woc_result.e_data["jx"][:5])
