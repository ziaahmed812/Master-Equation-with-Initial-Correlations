# Examples

These examples are meant to read like short notebook cells. Each script defines
the physical parameters, builds a time grid, chooses observables, calls a
solver, and prints a few values from the returned arrays.

```bash
python examples/exact_pure_dephasing.py
python examples/bosonic_bath_spin_boson.py
python examples/spin_bath.py
```

The master-equation examples use small grids so they finish quickly. They are
good for learning the API, not for declaring a calculation converged. For
production runs, increase the numerical controls and repeat the calculation
until the observables of interest are stable.

Available examples:

- `exact_pure_dephasing.py`: analytical pure dephasing, Python-only.
- `bosonic_bath_spin_boson.py`: bosonic bath, Ohmic spectrum.
- `bosonic_bath_second_moment.py`: normalized `"jx^2"` observable.
- `subohmic_bosonic_bath.py`: bosonic bath, sub-Ohmic spectrum.
- `superohmic_bosonic_bath.py`: bosonic bath, super-Ohmic spectrum.
- `spin_bath.py`: spin bath, Ohmic spectrum.
- `plot_paper_figure_02_pure_dephasing_N4.py`: optional matplotlib plot built
  from solver results.

The solvers do not require matplotlib. The plotting example imports matplotlib
only because that script draws a figure.
