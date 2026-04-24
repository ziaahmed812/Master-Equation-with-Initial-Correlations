# API Reference

This short reference is meant for notebook and script users who want to define
parameters explicitly and run one of the supported solver families.

## Parameter Objects

`SystemParams` defines the collective spin system. The public spin-size
parameter is `N`, with `J = N / 2` internally.

`BathParams` defines the bath family parameters:

- `kind="ohmic"` uses spectral exponent `s=1`.
- `kind="subohmic"` requires `0 < s < 1`.
- `kind="superohmic"` requires `s > 1`.

`NumericsConfig` controls quadrature and preserved-Fortran time grids. The
defaults reproduce the paper-scale workflow. For new regimes, explicitly refine
`omega_nodes`, omega cutoffs, `coefficient_points`, `tau_points`,
`fortran_dtau`, `fortran_dt`, and `fortran_t_final`, then check convergence.

`RunConfig` controls output behavior. If `output_dir` is omitted, outputs are
written to the current working directory. Existing package-generated files are
not replaced unless `overwrite=True`.

## Solver Classes

`PureDephasingSolver` evaluates the exact Ohmic bosonic pure-dephasing model in
Python. It requires `delta0=0` and `delta=0`.

`BosonicBathSolver` generates coefficient, bath-correlation, initial-state, and
observable inputs, then runs the preserved Fortran master-equation solver for a
bosonic bath.

`SpinBathSolver` uses the same generated-input and preserved-Fortran workflow
for a spin bath.

## Initial States

For the Fortran-backed solvers, the default initial state is the
correlated-equilibrium construction associated with the paper. Users may pass a
NumPy density matrix through `initial_state=`. The package validates the shape,
trace, Hermiticity, and positive semidefiniteness before writing `INSTATE.dat`.

The exact pure-dephasing solver does not accept a custom initial state because
its correlated and uncorrelated states are built into the analytical solution.

## Observables

String observables are built from dimensionless `jx`, `jy`, `jz`, and `id`
using a safe expression grammar. Examples: `jx`, `jy`, `jz`, `jx^2`, `jx+jy`,
and `0.5*jx + 2*jz`.

Advanced users may pass a NumPy operator matrix. Non-Hermitian observables are
allowed, but the package warns because expectation values can be complex.
