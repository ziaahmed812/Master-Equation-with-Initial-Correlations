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
preserved Fortran kernels under the hood and needs `gfortran` plus BLAS/LAPACK:

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
    family="bosonic",
    kind="ohmic",
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

Use `family="spin"` for a spin bath:

```python
spin_bath = meic.BathParams(family="spin", kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)
spin_result = meic.solve(system, spin_bath, tlist=tlist, e_ops=["jx"], correlations="with")
```

## Exact Pure Dephasing

The analytical pure-dephasing solver has its own namespace so it is never
confused with the master-equation workflow:

```python
system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=4.0, delta0=0.0, delta=0.0)
bath = meic.BathParams(family="bosonic", kind="ohmic", beta=1.0, coupling=0.05, omega_c=5.0)
tlist = np.linspace(0.0, 5.0, 501)

exact_wc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="with")
exact_woc = meic.exact.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="without")
```

This exact solver is valid for bosonic Ohmic pure dephasing with
`delta0 = 0` and `delta = 0`.

## Bath And Spectrum

`BathParams.family` chooses the bath:

- `family="bosonic"` for the bosonic bath master-equation workflow.
- `family="spin"` for the spin-bath master-equation workflow.

`BathParams.kind` chooses the spectrum:

- `kind="ohmic"` uses `s=1`; omit `s` or set `s=1.0`.
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

For Fortran-backed master-equation runs, each observable currently triggers its
own preserved-kernel run. This is deliberately explicit and correct, but
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

The defaults reproduce the paper-scale workflows. Away from those parameter
sets, make numerical controls explicit and check convergence:

```python
numerics = meic.NumericsConfig(
    omega_nodes=800,
    omega_max_coefficients=800.0,
    omega_max_tau=820.0,
    coefficient_points=4001,
    tau_points=4001,
)

result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], numerics=numerics)
```

Fortran-backed solvers currently require a one-dimensional, strictly
increasing, uniformly spaced `tlist` starting at `0.0`. The preserved Fortran
tables start at a tiny positive time internally; `result.times` records the
actual returned solver times.

## Examples And Paper Parameters

Runnable examples live in `examples/`. They are plain scientific Python
scripts: imports, parameters, `tlist`, observables, solver call, inspect
`result.expect`.

Paper-parameter scripts live in `paper-plots/`. They compute arrays for the
published parameter sets using the same public API. They do not generate plots
or ship rendered image assets.

## Citation

If this package helps your work, please cite the paper using `CITATION.bib`.
