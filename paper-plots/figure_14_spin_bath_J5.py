import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=10, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = meic.BathParams(bath_type="spin", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 5.0, 51)
e_ops = ["jx"]

spin_wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
spin_woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")

print(spin_wc_result.e_data["jx"][:5])
print(spin_woc_result.e_data["jx"][:5])
