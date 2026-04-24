from master_equation_initial_correlations import BathParams, BosonicBathSolver, RunConfig, SystemParams


system = SystemParams(
    N=4,
    epsilon0=4.0,
    epsilon=2.5,
    delta0=0.5,
    delta=0.5,
)
bath = BathParams(
    kind="subohmic",
    s=0.5,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

solver = BosonicBathSolver(system=system, bath=bath, observable="jx")
result = solver.run(RunConfig(output_dir="output-subohmic-bosonic-N4"))

print(f"Saved correlated curve: {result.output_files['correlated']}")
print(f"Saved uncorrelated curve: {result.output_files['uncorrelated']}")
