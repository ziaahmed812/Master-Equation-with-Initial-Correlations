# API Reference

The public API is centered on two solver calls:

```python
meic.solve(...)        # master-equation workflow
meic.exact.solve(...)  # analytical pure-dephasing workflow
```

Both return an in-memory `Result`.

## Parameters

```python
system = meic.SystemParams(N=4, epsilon0=4.0, epsilon=2.5, delta0=0.5, delta=0.5)
```

`N` is the public spin-size parameter; internally `J = N / 2`.

```python
bath = meic.BathParams(bath_type="bosonic", kind="ohmic", s=1.0, beta=1.0, coupling=0.05, omega_c=5.0)
bath = meic.BathParams(bath_type="bosonic", kind="subohmic", s=0.5, beta=1.0, coupling=0.05, omega_c=5.0)
bath = meic.BathParams(bath_type="spin", kind="superohmic", s=3.0, beta=1.0, coupling=0.05, omega_c=5.0)
```

`bath_type` is either `"bosonic"` or `"spin"`. `kind` is `"ohmic"`,
`"subohmic"`, or `"superohmic"`. Ohmic spectra use `s=1.0`; sub-Ohmic spectra
require `0 < s < 1`; super-Ohmic spectra require `s > 1`.

The implemented spectral weight is proportional to
`coupling * omega**s * omega_c**(1-s) * exp(-omega / omega_c)`. Arbitrary
user-defined spectral-density callables are not part of this public API yet.
For `kind="subohmic", s=0.5`, the package preserves the paper-code convention:
the dissipative kernel uses the legacy analytic branch and `C.dat` keeps the
extra cutoff used in the published Mathematica/Fortran workflow.

`NumericsConfig` controls the bath-integral grids used to generate coefficient
and correlation tables:

```python
numerics = meic.NumericsConfig(
    omega_max=800.0,
    omega_nodes=800,
    coefficient_time_step=0.00125,
    correlation_tau_step=0.00125,
)
```

`omega_max` and `omega_nodes` control the frequency quadrature. The coefficient
time step controls the `A.dat`, `B.dat`, and `C.dat` tables. The correlation
tau step controls the bath-correlation table used by the Fortran backend. In
normal `meic.solve(...)` calls, the maximum time comes from `tlist`.

## Master-Equation Solver

```python
result = meic.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=["jx", "jz"],
    correlations="with",
    numerics=numerics,
)
```

`correlations` accepts `"with"`, `"wc"`, `"without"`, or `"woc"`. The
`bath_type` determines the physical backend: bosonic bath uses the spin-boson
master-equation workflow, and spin bath uses the spin-environment workflow.

Fortran-backed master-equation runs require a one-dimensional, strictly
increasing, uniformly spaced `tlist` starting at `0.0`.

## Exact Solver

```python
result = meic.exact.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=["jx"],
    correlations="without",
)
```

The exact solver is analytical, Python-only, and valid for bosonic Ohmic pure
dephasing with `delta0=0` and `delta=0`. It accepts arbitrary finite,
nonnegative, strictly increasing time samples.

## Result

```python
result.times      # NumPy array of returned solver times
result.expect     # list of expectation arrays in e_ops order
result.e_data     # dict keyed by observable label
result.states     # density matrices when requested and available
result.as_dict()  # copy-friendly dictionary of arrays and metadata
```

Default behavior is RAM-only. Export is explicit:

```python
result.save("my-run")
```

Normal solver calls clean temporary backend staging folders before returning.
For provenance export, request artifact retention explicitly:

```python
result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], keep_artifacts=True)
result.save("my-run-with-artifacts", include_artifacts=True)
result.close()
```

`include_artifacts=True` copies generated input tables, Fortran sources/logs,
and provenance files. It raises an error if artifacts were not retained or are
no longer available. For Fortran-backed runs, `save_density=True` fills
`result.states` with reconstructed reduced density matrices in the public
collective-spin basis.

## Observables

String observables are safe algebraic expressions built from dimensionless
collective operators:

```python
e_ops = ["jx", "jy", "jz", "jx^2", "jx+jy", "0.5*jx + 2*jz", "jx*jy"]
```

Matrix observables are also accepted:

```python
e_ops = [meic.jx(system), meic.jz(system)]
```

`jx^2` follows the paper normalization `4 <J_x^2> / N^2`. Non-Hermitian
observables are allowed and produce a warning because complex expectation
values may be expected. Custom matrix observables must follow the same
dimensionless convention as the string observables: pass `J_x / J`, not the
physical matrix `J_x`, when you want `"jx"` normalization.

For Fortran-backed master-equation runs, each observable currently triggers a
separate numerical run. Multi-observable calls are therefore more expensive
than single-observable calls.

## Initial States

For master-equation workflows, the default initial state is the reduced system
density matrix generated from the correlated joint equilibrium state of the
system and bath. This is the initial-state construction used in the paper:
`SystemParams`, `BathParams`, and the initial-state quadrature controls in
`NumericsConfig` determine the generated reduced density matrix.

```python
result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"])
```

The call above uses the generated correlated-equilibrium reduced system state.
The `correlations` argument selects the dynamical branch:

- `"with"` keeps the initial system-bath correlation terms.
- `"without"` runs the corresponding uncorrelated comparison branch.

The public master-equation API does not accept a custom `initial_state` yet. A
custom reduced density matrix alone would not define the joint system-bath
correlations in the `correlations="with"` branch, so the public solver keeps
the preparation protocol tied to the paper's correlated-equilibrium
construction.

The exact pure-dephasing solver uses its analytical correlated or uncorrelated
construction and does not accept a custom initial state.

Ordinary notebook use should start with `meic.solve(...)` for master-equation
runs or `meic.exact.solve(...)` for analytical pure dephasing.
