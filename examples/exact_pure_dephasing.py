import numpy as np
import master_equation_initial_correlations as meic


system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(family="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)

tlist = np.linspace(0.0, 5.0, 101)
e_ops = ["jx"]

wc_result = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
woc_result = meic.exact.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")

print(wc_result.times[:5])
print(wc_result.e_data["jx"][:5])
print(woc_result.e_data["jx"][:5])
