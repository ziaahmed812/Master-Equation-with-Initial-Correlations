# Examples

These scripts are written like notebook cells. They define parameters, define
`tlist`, choose observables, call `meic.solve(...)` or `meic.exact.solve(...)`,
and inspect in-memory arrays.

The Fortran-backed examples require `gfortran` plus BLAS/LAPACK:

```bash
meic doctor
```

Examples:

- `exact_pure_dephasing.py`: Python-only exact pure-dephasing workflow.
- `bosonic_bath_spin_boson.py`: bosonic bath with an Ohmic spectrum.
- `bosonic_bath_second_moment.py`: bosonic bath with normalized `jx^2`.
- `subohmic_bosonic_bath.py`: bosonic bath with a sub-Ohmic spectrum.
- `superohmic_bosonic_bath.py`: bosonic bath with a super-Ohmic spectrum.
- `spin_bath.py`: spin bath with an Ohmic spectrum.

Nothing is saved unless the script explicitly calls `result.save(...)`.
