import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=10, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = meic.BathParams(family="bosonic", kind="ohmic", beta=0.5, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 5.0, 51)
e_ops = ["jx"]

me_wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
me_woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")

print(me_wc_result.e_data["jx"][:5])
print(me_woc_result.e_data["jx"][:5])
