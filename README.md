# Master-Equation-with-Initial-Correlations

Scientific Python tools for the workflows associated with the paper
*A master equation incorporating the system-environment correlations present in
the joint equilibrium state*.

Paper: https://arxiv.org/abs/2012.14853

The public API is intentionally notebook-friendly: define the system, define
the bath, choose a time grid and observables, call a solver, and work with
arrays in memory. Plotting is not part of the solver API; use matplotlib,
seaborn, notebooks, or any plotting tool you prefer.

## Install

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

The exact pure-dephasing solver is Python-only. The master-equation solver uses
a Fortran numerical backend and needs `gfortran` plus BLAS/LAPACK:

```bash
meic doctor
```

## Master-Equation Solver

```python
import numpy as np
import master_equation_initial_correlations as meic

system = meic.SystemParams(
    N=4,          # collective spin J = N/2
    epsilon0=4.0,
    epsilon=2.5,
    delta0=0.5,
    delta=0.5,
)

bath = meic.BathParams(
    bath_type="bosonic",
    kind="ohmic",
    s=1.0,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)

tlist = np.linspace(0.0, 5.0, 501)
e_ops = ["jx", "jz", "jx^2"]

wc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="with")
woc_result = meic.solve(system, bath, tlist=tlist, e_ops=e_ops, correlations="without")

wc_result.times
wc_result.expect[0]
wc_result.e_data["jx"]
```

Use `bath_type="spin"` for a spin bath:

```python
spin_bath = meic.BathParams(bath_type="spin", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
spin_result = meic.solve(system, spin_bath, tlist=tlist, e_ops=["jx"], correlations="with")
```

## Exact Pure Dephasing

The analytical pure-dephasing solver has its own namespace so it is never
confused with the master-equation workflow:

```python
system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(bath_type="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 5.0, 501)

exact_wc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="with")
exact_woc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="without")
```

This exact solver is valid for bosonic Ohmic pure dephasing with
`delta0 = 0` and `delta = 0`.

## Bath And Spectrum

`BathParams.bath_type` chooses the bath:

- `bath_type="bosonic"` for the bosonic bath master-equation workflow.
- `bath_type="spin"` for the spin-bath master-equation workflow.

`BathParams.kind` chooses the spectrum:

- `kind="ohmic"` uses the spectral exponent `s=1.0`.
- `kind="subohmic"` requires `0 < s < 1`.
- `kind="superohmic"` requires `s > 1`.

The package validates these choices before generating coefficient tables or
running Fortran, so invalid spectra fail with physical error messages rather
than low-level numerical errors.

## Observables

Observables can be strings or matrices:

```python
e_ops = ["jx", "jy", "jz", "jx^2", "jx+jy", "jx*jy"]
e_ops = [meic.jx(system), meic.jz(system)]
```

The lowercase operators are dimensionless:

```text
jx = J_x / J
jy = J_y / J
jz = J_z / J
```

The expression `jx^2` uses the paper normalization `4 <J_x^2> / N^2`.
Non-Hermitian observables are allowed, but the solver warns because complex
expectation values may appear.

For Fortran-backed master-equation runs, each observable currently triggers a
separate numerical run. This keeps each observable evaluation explicit, but
multi-observable calls can take longer than single-observable calls.

## Results And Export

Solver calls are RAM-first:

```python
result.times
result.expect
result.e_data
result.states
```

No output folder is written unless you ask for one:

```python
result.save("my-run")
result.save("my-run-with-artifacts", include_artifacts=True)
result.close()
```

`include_artifacts=True` exports generated coefficient, bath-correlation,
initial-state, observable, Fortran-source, and log files when they are still
available. If artifacts have been cleaned up or were never produced, the export
fails loudly rather than silently writing an incomplete folder.

## Numerics

You can omit `numerics`; the package then uses the benchmark grid settings.
Only change these controls when you want to check convergence or run outside
the benchmark regime.

The main controls are the bath-integral grids:

```python
numerics = meic.NumericsConfig(
    omega_max=800.0,              # frequency cutoff in the omega integrals
    omega_nodes=800,              # quadrature nodes for omega
    coefficient_time_step=0.00125, # spacing for A.dat, B.dat, C.dat
    correlation_tau_step=0.00125,  # spacing for bath-correlation tau tables
)

result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], numerics=numerics)
```

For ordinary `meic.solve(...)` calls, `tlist` sets the requested physical time
window. `NumericsConfig` controls how finely the coefficient and bath
correlation tables are generated inside that window. Advanced users can also
set separate cutoffs such as `correlation_omega_max`,
`initial_state_omega_max`, or initial-state quadrature nodes.

Fortran-backed solvers currently require a one-dimensional, strictly
increasing, uniformly spaced `tlist` starting at `0.0`. The first returned
Fortran time is a tiny positive value (`1e-11`), and `result.times` records the
solver output without relabeling it.

## Examples And Paper Parameters

Runnable examples live in `examples/`. They are plain scientific Python
scripts: imports, parameters, `tlist`, observables, solver call, inspect
`result.expect`.

Paper-parameter scripts live in `paper-plots/`. They compute arrays for the
published parameter sets using the same public API. They do not generate plots
or ship rendered image assets.

## Citation

If this package helps your work, please cite the paper using `CITATION.bib`.
