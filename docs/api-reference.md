# API Reference

This page summarizes the public objects most users need in notebooks and
scripts.

```python
import master_equation_initial_correlations as meic
```

## Solvers

### `meic.solve(...)`

Runs the master-equation workflow.

```python
result = meic.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=["jx"],
    correlations="with",
)
```

Common arguments:

- `system`: a `SystemParams` object.
- `bath`: a `BathParams` object.
- `tlist`: a one-dimensional, uniformly spaced NumPy array starting at `0.0`.
- `e_ops`: one or more observables, either strings or matrices.
- `correlations`: `"with"` or `"without"`.
- `numerics`: optional `NumericsConfig` for convergence studies.
- `save_density`: set `True` to include reduced density matrices in
  `result.states`.

The master-equation solver keeps results in memory by default and writes no
output directory unless you call `result.save(...)`.

### `meic.exact.solve(...)`

Runs the analytical pure-dephasing solver.

```python
result = meic.exact.solve(
    system,
    bath,
    tlist=tlist,
    e_ops=["jx"],
    correlations="without",
)
```

This solver is valid for bosonic Ohmic pure dephasing with `delta0=0` and
`delta=0`. Its `tlist` only needs to be finite, nonnegative, strictly
increasing, and one-dimensional.

## System Parameters

```python
system = meic.SystemParams(
    N=4,
    epsilon0=4.0,
    epsilon=2.5,
    delta0=0.5,
    delta=0.5,
)
```

`N` sets the collective spin size through `J = N / 2`. The parameters
`epsilon0` and `delta0` describe the preparation Hamiltonian, while `epsilon`
and `delta` describe the Hamiltonian used during subsequent dynamics.

For pure dephasing, use `delta0=0.0` and `delta=0.0`.

## Bath Parameters

```python
bath = meic.BathParams(
    bath_type="bosonic",
    kind="ohmic",
    s=1.0,
    beta=1.0,
    coupling=0.05,
    omega_c=5.0,
)
```

`bath_type` chooses the environment:

- `"bosonic"`: bosonic bath.
- `"spin"`: spin bath.

`kind` chooses the spectral class:

- `"ohmic"` with `s=1.0`.
- `"subohmic"` with `0 < s < 1`.
- `"superohmic"` with `s > 1`.

The implemented spectral weight is proportional to

```text
coupling * omega^s * omega_c^(1-s) * exp(-omega / omega_c).
```

The public API does not yet accept arbitrary user-defined spectral-density
functions.

## Initial State

For master-equation runs, the initial state is generated from the joint
thermal equilibrium state of the system and environment, followed by the
system preparation step used in the paper.

```python
result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], correlations="with")
```

The `correlations` argument selects the dynamical branch:

- `"with"` includes the initial system-environment correlation terms.
- `"without"` runs the corresponding uncorrelated comparison.

A custom reduced density matrix is not part of the public master-equation API
yet, because it would not by itself define the joint correlations required by
the correlated branch.

## Observables

String observables:

```python
e_ops = ["jx", "jy", "jz", "jx^2", "jx+jy", "0.5*jx + 2*jz"]
```

Matrix observables:

```python
e_ops = [meic.jx(system), meic.jz(system)]
```

The built-in operators are dimensionless:

```text
jx = J_x / J
jy = J_y / J
jz = J_z / J
```

The string `"jx^2"` returns the paper-normalized second moment
`4 <J_x^2> / N^2`. If you pass a custom matrix and want the same convention,
pass the dimensionless operator rather than the unscaled physical spin matrix.

For master-equation runs, each observable is evaluated by its own backend run.
This is simple and explicit, but multi-observable calls can be slower.

## Result Objects

```python
result.times
result.expect
result.e_data
result.states
```

- `times` is the returned time grid.
- `expect` is a list of expectation arrays in the same order as `e_ops`.
- `e_data` maps observable labels to arrays.
- `states` contains density matrices when `save_density=True`.

Convenience helpers:

```python
data = result.as_dict()
df = result.to_dataframe()
```

`to_dataframe()` requires pandas to be installed by the user.

## Saving Results

Results stay in memory unless you explicitly save them:

```python
result.save("my-run")
```

The output directory contains text expectation tables and
`result_metadata.json`. If `result.states` is available, `states.npz` is also
written.

By default, the solver does not keep backend working files. To export those
advanced files as well:

```python
result = meic.solve(system, bath, tlist=tlist, e_ops=["jx"], keep_artifacts=True)
result.save("my-run-with-artifacts", include_artifacts=True)
```

Use this only when you want to inspect coefficient tables, generated inputs,
Fortran sources, or logs. Normal notebook workflows do not need it.

## Numerical Controls

```python
numerics = meic.NumericsConfig(
    omega_max=800.0,
    omega_nodes=800,
    coefficient_time_step=0.00125,
    correlation_tau_step=0.00125,
)
```

`omega_max` and `omega_nodes` control the frequency quadrature. The time-step
settings control the generated coefficient and bath-correlation tables used by
the backend. The requested table steps must divide their intervals exactly;
otherwise the package raises `ValueError`.

For ordinary `meic.solve(...)` calls, the physical time window comes from
`tlist`. Increase quadrature cutoffs, node counts, and table resolution when
checking convergence away from the paper benchmark regimes.
