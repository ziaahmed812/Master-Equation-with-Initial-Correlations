# Master-Equation-with-Initial-Correlations

This package implements master-equation workflows for collective-spin open
systems with initial system-environment correlations. It is associated with
the paper *A master equation incorporating the system-environment correlations
present in the joint equilibrium state*.

Paper: https://arxiv.org/abs/2012.14853

The package supports two bath families:

- a bosonic bath of harmonic oscillators
- a spin bath

For both bath families, the same public interface accepts Ohmic, sub-Ohmic, and
super-Ohmic spectral densities. The user supplies the physical parameters and,
when needed, numerical grid controls; the package generates the coefficient
tables, bath-correlation tables, initial state, observable operator, and Fortran
run inputs needed by the solver.

The bundled reference data validate the paper parameter sets. Other parameter
sets are executable workflows with explicit numerical settings, but they are not
automatically benchmarked unless they match a bundled reference example.

## Install

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

The exact pure-dephasing solver is Python-only. The master-equation workflows
beyond exact pure dephasing use preserved Fortran solvers and require
`gfortran` with BLAS/LAPACK:

```bash
meic doctor
```

## Basic Python Workflow

The intended use is notebook- and script-friendly: define the parameters at the
top, choose a bath solver, choose an observable, and run.

```python
from master_equation_initial_correlations import (
    BathParams,
    BosonicBathSolver,
    NumericsConfig,
    RunConfig,
    SystemParams,
)

system = SystemParams(
    N=4,          # collective spin is J = N/2
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

solver = BosonicBathSolver(system=system, bath=bath, observable="jx")
result = solver.run(RunConfig(output_dir="output-bosonic-ohmic-N4"))

print(result.output_files["correlated"])
print(result.summary_path)
```

If `output_dir` is omitted, files are written in the current working directory.
Existing package-generated files are not overwritten unless `overwrite=True` is
passed.

The paper defaults are collected in `NumericsConfig`. You can keep them, or make
the quadrature and time grids explicit:

```python
numerics = NumericsConfig(
    omega_nodes=800,
    omega_max_coefficients=800.0,
    omega_max_tau=820.0,
    coefficient_points=4001,
    tau_points=4001,
    fortran_dt=0.005,
    fortran_dtau=0.0025,
    fortran_t_final=5.0,
)

result = solver.run(RunConfig(output_dir="output-refined", numerics=numerics))
```

## Parameters

`N` is the public spin-size parameter. Internally the collective spin quantum
number is

```text
J = N / 2
```

For example, `N=4` means `J=2`, and `N=10` means `J=5`.

Spectral-density choices:

- `kind="ohmic"` uses `s=1`; omit `s` or set `s=1`.
- `kind="subohmic"` requires `0 < s < 1`.
- `kind="superohmic"` requires `s > 1`.

Observables are safe algebraic expressions built from dimensionless collective
spin operators:

```text
jx = J_x / J
jy = J_y / J
jz = J_z / J
```

Examples include `jx`, `jy`, `jz`, `jx^2`, `jx+jy`, `jx/2`, and
`0.5*jx + 2*jz`. The expression parser is deliberately small and does not
execute Python code. Advanced users may pass a NumPy operator matrix through
the Python API. Non-Hermitian observables are allowed, but the package warns
because the expectation values can be complex; in that case output tables keep
real and imaginary columns.

For the second moment, the public expression `jx^2` is dimensionless. It
corresponds to the paper's normalized quantity `4 <J_x^2> / N^2`. Older
reference metadata may call this a plot divisor; physically it is the
observable normalization.

## Solver Families

`PureDephasingSolver` evaluates the exact Ohmic bosonic-bath pure-dephasing
model directly in Python. Pure dephasing requires `delta0=0` and `delta=0`.

`BosonicBathSolver` generates the master-equation inputs for a collective spin
coupled to a bosonic bath, then runs the preserved Fortran solver.

`SpinBathSolver` uses the same generated-input plus preserved-Fortran execution
model for a collective spin coupled to a spin bath.

For generated-input Fortran workflows, the default initial state is the
correlated equilibrium construction used in the paper. Notebook or script users
may pass a density matrix through `initial_state=` on `BosonicBathSolver`,
`SpinBathSolver`, or `SimulationParams`. The matrix is validated for shape,
trace, Hermiticity, and positive semidefiniteness, then written to `INSTATE.dat`.
The exact pure-dephasing Python solver uses its analytical correlated and
uncorrelated state construction and does not accept a custom `initial_state`.

When a parameter set matches a bundled paper reference example, the package
compares the regenerated numerical output against the curated reference table.
For new parameter sets, the solver still runs and records that no bundled
reference comparison was available.

## More Examples

Bosonic bath, second moment:

```python
system = SystemParams(N=4, epsilon0=4.0, epsilon=3.5, delta0=0.5, delta=0.5)
bath = BathParams(kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)
result = BosonicBathSolver(system, bath, observable="jx^2").run(
    RunConfig(output_dir="output-bosonic-jx2-N4")
)
```

Sub-Ohmic bosonic bath:

```python
system = SystemParams(N=4, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = BathParams(kind="subohmic", s=0.5, beta=1.0, coupling=0.05, omega_c=5.0)
result = BosonicBathSolver(system, bath, observable="jx").run(
    RunConfig(output_dir="output-bosonic-subohmic-N4")
)
```

Super-Ohmic bosonic bath with an observable sum:

```python
system = SystemParams(N=2, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = BathParams(kind="superohmic", s=3.0, beta=1.0, coupling=0.05, omega_c=5.0)
result = BosonicBathSolver(system, bath, observable="jx+jy").run(
    RunConfig(output_dir="output-bosonic-superohmic-sum-N2")
)
```

Spin bath with the second moment:

```python
from master_equation_initial_correlations import SpinBathSolver

system = SystemParams(N=4, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
bath = BathParams(kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)
result = SpinBathSolver(system, bath, observable="jx^2").run(
    RunConfig(output_dir="output-spin-bath-jx2-N4")
)
```

Runnable scripts live in `examples/`.

## Command Line

List bundled reference examples:

```bash
meic examples
```

Run exact pure dephasing:

```bash
meic run --bath bosonic --model pure-dephasing --N 4 \
  --epsilon0 4 --epsilon 4 --delta0 0 --delta 0 \
  --beta 1 --coupling 0.05 --omega-c 5 \
  --observable jx --out output-pure-dephasing-N4
```

Run a bosonic-bath master-equation workflow:

```bash
meic run --bath bosonic --model spin-boson --N 4 \
  --epsilon0 4 --epsilon 2.5 --delta0 0.5 --delta 0.5 \
  --beta 1 --coupling 0.05 --omega-c 5 \
  --spectral subohmic --s 0.5 --observable 'jx^2' \
  --out output-bosonic-subohmic-jx2-N4
```

Run a spin-bath workflow:

```bash
meic run --bath spin --model spin-environment --N 4 \
  --epsilon0 4 --epsilon 2.5 --delta0 0.5 --delta 0.5 \
  --beta 1 --coupling 0.05 --omega-c 5 \
  --spectral ohmic --observable 'jx+jy' \
  --out output-spin-bath-sum-N4
```

Use `--overwrite` only when you intentionally want to replace
package-generated files in an existing output directory. Use `--save-density`
when you also want the internal Fortran density-vector trajectory.

CLI users can also expose the numerical controls, for example:

```bash
meic run --bath bosonic --model spin-boson --N 2 \
  --spectral superohmic --s 3 --observable jy \
  --omega-nodes 800 --omega-max-coefficients 800 \
  --tau-points 4001 --tau-t-max 5 \
  --fortran-dt 0.005 --fortran-dtau 0.0025 \
  --out output-refined-superohmic
```

## Output Files

For exact pure dephasing, the main outputs are:

- `exact-correlated.dat`: time and correlated observable expectation.
- `exact-uncorrelated.dat`: time and uncorrelated observable expectation.
- `simulation_summary.json`: parameters, output paths, and verification metadata.

For generated-input plus preserved-Fortran workflows, the main public outputs
are:

- `observable-correlated.dat`: time and correlated observable expectation.
- `observable-uncorrelated.dat`: time and uncorrelated observable expectation.
- `A.dat`, `B.dat`, `C.dat`: complex coefficient tables on the generated coefficient time grid.
- `bathcorrelation.dat`: two-column `nu(tau) eta(tau)` bath-correlation table on the generated tau grid.
- `integraldatasimpson.dat`: legacy one-column table retained for Fortran provenance; for a bosonic bath it stores `nu(tau)`, while for a spin bath it stores `eta(tau)`.
- `INSTATE.dat`: flattened initial system density matrix, generated from the correlated-equilibrium construction unless the user supplied `initial_state`.
- `OBSERVABLE.dat`: flattened observable matrix supplied to the Fortran solver.
- `params.in`: scalar run constants read by the Fortran solver, including internal time steps and final time.
- `sources/`: parameterized Fortran sources and generated `dimensions.inc`.
- `logs/`: compile and run logs.
- `runs/`: isolated raw Fortran run directories.
- `regeneration_summary.json`: parameters, compiler, output paths, and verification metadata.

If the requested observable is exactly `jx`, compatibility aliases
`EXPX-C.DAT` and `EXPX-UNC.DAT` are also written. If the requested observable
is exactly `jz`, compatibility aliases `EXPZ-C.DAT` and `EXPZ-UNC.DAT` are
written.

All top-level `.dat` files include comment headers beginning with `#`, so
`numpy.loadtxt` can read them directly. Raw files inside `runs/` remain
headerless because the preserved Fortran readers expect plain numeric input.

For sub-Ohmic bosonic runs with `s=0.5`, the dissipative kernel `eta(tau)` uses
the analytic legacy branch from the paper workflow, while nearby `s` values use
quadrature. This is deliberate paper-parity behavior and is recorded in output
headers.

## Numerical Scope

The default quadrature cutoffs, node counts, tau grid, and Fortran time steps
are the validated paper-scale settings. If you move to large `omega_c`, large
spectral exponent, low temperature, sharp spectral structure, or longer final
times, increase the relevant `NumericsConfig` values and check convergence.
The package records those numerical choices in every generated header and in
`regeneration_summary.json`.

## Reference Assets

Curated paper assets and reference tables are available through the API:

```python
import master_equation_initial_correlations as meic

for example in meic.list_examples():
    print(example.public_id, example.bath, example.model, example.observable)

curves = meic.load_reference_curves("pure-dephasing-ohmic-N1")
```

Export one bundled reference branch:

```bash
meic export pure-dephasing-ohmic-N4 --out exported-assets
```

## Citation

If you use this package in research, please cite the paper. A BibTeX entry is
provided in `CITATION.bib`.

## License

This package is released under the MIT License. See `LICENSE` and `NOTICE` for
notes about bundled research assets and preserved Fortran solver sources.
