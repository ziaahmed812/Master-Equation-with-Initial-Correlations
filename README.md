# Master-Equation-with-Initial-Correlations

`master_equation_initial_correlations` is a scientific Python package for
open-quantum-system dynamics with initial system-environment correlations.

It implements the master-equation workflows associated with the paper
*A master equation incorporating the system-environment correlations present in
the joint equilibrium state*:

https://arxiv.org/abs/2012.14853

The package is designed for notebook-style use. Define the physical system,
define the bath, choose times and observables, call a solver, and work with
NumPy arrays in memory. Plotting and file export are left to the user.

## Install

From the repository root:

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

The exact pure-dephasing solver is Python-only. The master-equation solver uses
a Fortran backend, so it needs `gfortran` and BLAS/LAPACK:

```bash
meic doctor
```

## Quick Start

```python
import numpy as np
import master_equation_initial_correlations as meic

system = meic.SystemParams(
    N=2,
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

tlist = np.linspace(0.0, 0.2, 21)
e_ops = ["jx"]

# Small grids keep this first run quick. For production calculations, increase
# these controls and check convergence of the observables you care about.
numerics = meic.NumericsConfig(
    omega_max=80.0,
    omega_nodes=64,
    lambda_nodes=16,
    initial_state_omega_nodes=40,
    initial_state_lambda_nodes=12,
    initial_state_zeta_nodes=12,
)

with_correlations = meic.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=e_ops,
    correlations="with",
    numerics=numerics,
)

without_correlations = meic.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=e_ops,
    correlations="without",
    numerics=numerics,
)

print(with_correlations.times[:5])
print(with_correlations.e_data["jx"][:5])
print(without_correlations.e_data["jx"][:5])
```

Nothing is saved to disk in this workflow. `result.times`, `result.expect`,
and `result.e_data` are ordinary in-memory arrays and dictionaries.

## Physical Model

The master-equation solver follows the preparation protocol used in the paper:

1. The system and environment are first described by a joint thermal
   equilibrium state.
2. A unitary operation prepares the desired reduced system state.
3. The reduced dynamics are evolved either with or without the second-order
   terms produced by the initial system-environment correlations.

This is a time-local Redfield-type master-equation workflow, correct to second
order in the system-environment coupling. The paper parameter sets are the
validated reference regimes. Other parameter choices are executable, but should
be treated like any numerical open-system calculation: check convergence with
respect to cutoff, quadrature, and time-table settings.

The public master-equation solver intentionally does not accept a custom
reduced initial density matrix yet. A reduced density matrix alone does not
define the joint system-environment correlations needed by the correlated
branch, so the public API keeps the initial state tied to the equilibrium
preparation map.

## Baths And Spectra

Use `BathParams.bath_type` to choose the environment:

- `bath_type="bosonic"` uses the bosonic-bath master-equation workflow.
- `bath_type="spin"` uses the spin-bath master-equation workflow.

Use `BathParams.kind` and `BathParams.s` to choose the spectral class:

- `kind="ohmic"` uses `s=1.0`.
- `kind="subohmic"` requires `0 < s < 1`.
- `kind="superohmic"` requires `s > 1`.

The implemented spectral weight is

```text
J_s(omega) proportional to
coupling * omega^s * omega_c^(1-s) * exp(-omega / omega_c).
```

Arbitrary user-defined spectral functions are not part of the public API yet.
For `kind="subohmic", s=0.5`, the solver preserves the analytic convention
used by the paper workflow for the dissipative kernel.

## Observables

Observables may be strings:

```python
e_ops = ["jx", "jy", "jz", "jx^2", "jx+jy", "0.5*jx + 2*jz"]
```

or explicit matrices:

```python
e_ops = [meic.jx(system), meic.jz(system)]
```

The built-in collective-spin observables are dimensionless:

```text
jx = J_x / J
jy = J_y / J
jz = J_z / J
```

The second moment `"jx^2"` follows the paper normalization
`4 <J_x^2> / N^2`. If you pass your own matrix observable, use the same
dimensionless convention when you want to compare with the built-in strings.

Non-Hermitian observables are allowed. The solver warns because complex
expectation values may then be physically expected.

## Exact Pure Dephasing

The analytical pure-dephasing solver lives under `meic.exact` so it is not
confused with the Fortran-backed master-equation solver:

```python
system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(
    bath_type="bosonic",
    kind="ohmic",
    s=1.0,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)
tlist = np.linspace(0.0, 5.0, 501)

exact_wc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="with")
exact_woc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="without")
```

This solver is valid for bosonic Ohmic pure dephasing with
`delta0 = 0` and `delta = 0`.

## Results

Every solver returns a `Result`:

```python
result.times      # returned time grid
result.expect     # list of expectation arrays, in e_ops order
result.e_data     # dictionary keyed by observable label
result.states     # density matrices when save_density=True
result.as_dict()  # copy-friendly dictionary of arrays and metadata
```

To use pandas, install it in your environment and call:

```python
df = result.to_dataframe()
```

To save arrays and metadata:

```python
result.save("my-run")
```

The saved directory contains expectation tables and `result_metadata.json`.
If density matrices were requested, it also contains `states.npz`.

## Numerical Controls

You can usually start without passing `numerics`; the package then uses the
benchmark settings. For convergence checks or exploratory parameters, pass a
`NumericsConfig`:

```python
numerics = meic.NumericsConfig(
    omega_max=800.0,
    omega_nodes=800,
    coefficient_time_step=0.00125,
    correlation_tau_step=0.00125,
)

result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], numerics=numerics)
```

These controls set the frequency quadrature cutoff and nodes, plus the spacing
of the coefficient and bath-correlation time tables used by the backend. The
time-table spacings are exact: if a requested step does not divide the
configured interval, the package raises `ValueError` instead of silently
changing the grid.

For `meic.solve(...)`, `tlist` must be one-dimensional, strictly increasing,
uniformly spaced, and start at `0.0`. The first returned master-equation time
is the backend's tiny positive initial value (`1e-11`), not a relabelled zero.
The analytical `meic.exact.solve(...)` solver accepts arbitrary finite,
nonnegative, strictly increasing time samples.

## Advanced Export

Most users never need backend files. If you want to inspect the generated
coefficient tables, bath-correlation tables, Fortran sources, and logs, request
artifact retention explicitly:

```python
result = meic.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=["jx"],
    keep_artifacts=True,
)
result.save("my-run-with-artifacts", include_artifacts=True)
```

This copies the retained backend files into the saved output directory.

## Examples

Runnable examples live in `examples/`:

```bash
python examples/exact_pure_dephasing.py
python examples/bosonic_bath_spin_boson.py
python examples/spin_bath.py
```

They are written like notebook cells: parameters at the top, `tlist`, `e_ops`,
solver call, then array inspection.

The folder `paper-plots/` contains one script per paper parameter set. These
scripts use the same public API and are meant as transparent parameter recipes,
not as a separate plotting interface.

## Citation

If this package helps your work, please cite the paper using `CITATION.bib`.
