from master_equation_initial_correlations import BathParams, PureDephasingSolver, RunConfig, SystemParams


system = SystemParams(
    N=4,
    epsilon0=4.0,
    epsilon=4.0,
    delta0=0.0,
    delta=0.0,
)
bath = BathParams(
    kind="ohmic",
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

solver = PureDephasingSolver(system=system, bath=bath, observable="jx")
result = solver.run(RunConfig(output_dir="output-pure-dephasing-N4"))

print(f"Saved correlated curve: {result.output_files['exact_correlated']}")
print(f"Saved uncorrelated curve: {result.output_files['exact_uncorrelated']}")
