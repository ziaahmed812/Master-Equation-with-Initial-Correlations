import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=2, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = meic.BathParams(bath_type="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
numerics = meic.NumericsConfig(
    omega_nodes=48,
    omega_max=80.0,
    lambda_nodes=12,
    initial_state_omega_nodes=24,
    initial_state_lambda_nodes=8,
    initial_state_zeta_nodes=8,
    coefficient_time_step=0.0025,
    correlation_tau_step=0.0025,
)

tlist = np.linspace(0.0, 0.2, 21)
e_ops = ["jx", "jz"]

# Default initial state: reduced system state from joint system-bath equilibrium.
wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with", numerics=numerics)
woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without", numerics=numerics)

print(wc_result.e_data["jx"][:5])
print(woc_result.e_data["jx"][:5])
