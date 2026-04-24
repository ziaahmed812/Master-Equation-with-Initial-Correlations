from master_equation_initial_correlations import BathParams, BosonicBathSolver, RunConfig, SystemParams


system = SystemParams(
    N=2,
    epsilon0=4.0,
    epsilon=2.5,
    delta0=0.5,
    delta=0.5,
)
bath = BathParams(
    kind="superohmic",
    s=3.0,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

solver = BosonicBathSolver(system=system, bath=bath, observable="jx+jy")
result = solver.run(RunConfig(output_dir="output-superohmic-bosonic-N2"))

print(f"Saved correlated observable: {result.output_files['correlated']}")
print(f"Saved uncorrelated observable: {result.output_files['uncorrelated']}")
