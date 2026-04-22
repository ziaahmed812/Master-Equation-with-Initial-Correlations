# Master-Equation-with-Initial-Correlations

`Master-Equation-with-Initial-Correlations` is a reusable implementation framework for the master-equation workflows associated with the paper *A master equation incorporating the system-environment correlations present in the joint equilibrium state*. It packages validated figure assets, reusable Python interfaces, and optional legacy-Fortran reruns for the supported model families studied in the paper.

The package is designed to be useful in two ways:

- as a Python-first reference library for loading bundled results, plotting them, and working with the exact pure-dephasing benchmark
- as an optional reproduction layer for the legacy Fortran-backed workflows when you want to rerun the original solver paths

## What The Package Provides

- a public preset catalog covering all 14 published figure configurations
- packaged figure assets in `EPS`, `PNG`, and numeric table form
- a free-form Python implementation of the exact pure-dephasing benchmark
- paper-style plotting helpers for `j_x` and `j_x^{(2)}`
- optional rerun support for curated legacy Fortran solvers
- a small command-line interface for listing presets, exporting assets, computing exact pure-dephasing curves, and rerunning supported presets

## Validated Lightweight Workflows

The default public validation path is centered on lightweight presets that are fast to install, inspect, and rerun:

- `pure_dephasing_ohmic_j0p5` (`N=1`)
- `pure_dephasing_ohmic_j2` (`N=4`)
- `beyond_pure_dephasing_ohmic_j1` (`N=2`)

Other presets are still packaged and available, but some are marked `advanced` or `heavy` because they are not part of the default lightweight validation lane.

## Installation

For Python-only use:

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

For optional legacy reruns, install a Fortran compiler and BLAS/LAPACK on your system as well. The base package does not compile anything at install time.

If you plan to use the legacy rerun path, start with:

```bash
meic doctor
```

to confirm that the package can see your compiler and link configuration.

## Quick Start

### Load a bundled reference preset

```python
import master_equation_initial_correlations as meic

preset = meic.get_preset("pure_dephasing_ohmic_j0p5")
curves = meic.load_reference_curves(preset.id)

print(preset.label)
print(curves.correlated[:3])
```

### Compute exact pure-dephasing curves in Python

```python
import numpy as np
from master_equation_initial_correlations import PureDephasingParams, exact_curves

params = PureDephasingParams(J=0.5, epsilon=4.0, xi=4.0, beta=1.0, G=0.05, omega_c=5.0)
times = np.arange(1.0e-10, 5.0, 0.2)
correlated, uncorrelated = exact_curves(params, correlated_times=times)
```

### Use the CLI

```bash
meic list
meic list --include-advanced --include-heavy
meic show pure_dephasing_ohmic_j2
meic export pure_dephasing_ohmic_j2 --out exported-assets
meic exact pure-dephasing --preset pure_dephasing_ohmic_j0p5 --out exact-output
meic doctor
```

## Supported Model Families

- `pure_dephasing_ohmic`
- `beyond_pure_dephasing_ohmic`
- `beta_compare_ohmic`
- `jx2_ohmic`
- `spin_environment_ohmic`
- `subohmic_s0p5`

In v1, only the pure-dephasing family has a fully parameterized Python solver API. The other families are exposed through curated presets and optional rerun helpers.

By default, the public preset listing shows the lightweight validated presets first. Use `--include-advanced` and `--include-heavy` in the CLI, or the corresponding Python API flags, if you want the broader preset catalog.

## Python-Only Use vs Legacy Reruns

The package works immediately without Fortran if you want to:

- inspect the bundled published figures
- load the packaged reference tables
- compute exact pure-dephasing curves
- export preset assets
- make plots from the packaged data

You only need Fortran, BLAS, and LAPACK if you want to rerun the legacy solver-backed presets.

The default public path is intentionally light. Presets marked `advanced` or `heavy` are packaged and available, but they are not part of the default lightweight validation lane.

## Citation

If you use this package in research, please cite the paper:

- Ali Raza Mirza
- Muhammad Zia
- Adam Zaman Chaudhry

*A master equation incorporating the system-environment correlations present in the joint equilibrium state*, arXiv:2012.14853  
https://arxiv.org/abs/2012.14853

The repository includes:

- `CITATION.bib` for BibTeX users
- `CITATION.cff` for GitHub/software citation metadata

## License

The new Python package code is released under the MIT License. See `LICENSE` and `NOTICE` for details and provenance notes about bundled legacy materials.
