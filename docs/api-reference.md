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

Fortran-backed runs require a one-dimensional, strictly increasing, uniformly
spaced `tlist` starting at `0.0`.

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
dephasing with `delta0=0` and `delta=0`.

## Result

```python
result.times      # NumPy array of returned solver times
result.expect     # list of expectation arrays in e_ops order
result.e_data     # dict keyed by observable label
result.states     # density matrices when requested and available
```

Default behavior is RAM-only. Export is explicit:

```python
result.save("my-run")
result.save("my-run-with-artifacts", include_artifacts=True)
result.close()
```

`include_artifacts=True` copies generated input tables, Fortran sources/logs,
and provenance files. It raises an error if those artifacts are unavailable.
`result.close()` removes temporary Fortran staging folders held alive by the
result object.

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
values may be expected.

For Fortran-backed master-equation runs, each observable currently triggers a
separate numerical run. Multi-observable calls are therefore more expensive
than single-observable calls.

## Initial States

For Fortran-backed workflows, the default initial state is the correlated
equilibrium construction associated with the paper. Advanced users may pass a
validated density matrix through `initial_state=` in `meic.solve(...)`.

The exact pure-dephasing solver uses its analytical correlated or uncorrelated
construction and does not accept a custom initial state.

Ordinary notebook use should start with `meic.solve(...)` for master-equation
runs or `meic.exact.solve(...)` for analytical pure dephasing.
