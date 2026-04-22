import numpy as np

from master_equation_initial_correlations import PureDephasingParams, exact_curves


params = PureDephasingParams(J=0.5)
correlated, uncorrelated = exact_curves(params, correlated_times=np.arange(1.0e-10, 1.0, 0.2))
print(correlated[:3])
print(uncorrelated[:3])
