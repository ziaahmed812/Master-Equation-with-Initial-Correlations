from master_equation_initial_correlations import BathParams, RunConfig, SpinBathSolver, SystemParams


system = SystemParams(
    N=4,
    epsilon0=4.0,
    epsilon=2.5,
    delta0=0.5,
    delta=0.5,
)
bath = BathParams(
    kind="ohmic",
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

solver = SpinBathSolver(system=system, bath=bath, observable="jx^2")
result = solver.run(RunConfig(output_dir="output-spin-bath-jx2-N4"))

print(f"Saved correlated observable: {result.output_files['correlated']}")
print(f"Saved uncorrelated observable: {result.output_files['uncorrelated']}")
