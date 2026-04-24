# Master-Equation-with-Initial-Correlations

`Master-Equation-with-Initial-Correlations` is a research software package for the master-equation workflows in the paper *A master equation incorporating the system-environment correlations present in the joint equilibrium state*.

The package focuses on two physical settings from the paper:

- a collective spin system coupled to a bosonic bath of harmonic oscillators
- a collective spin system coupled to a spin bath

It provides bundled reference examples, paper-style plotting helpers, an exact pure-dephasing benchmark written in Python, and optional reruns of the preserved Fortran solver paths.

Paper: https://arxiv.org/abs/2012.14853

## Install

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

Python-only workflows work immediately. Rerunning the legacy solver paths also requires `gfortran` plus BLAS/LAPACK:

```bash
meic doctor
```

## Command Line

List the bundled reference examples:

```bash
meic examples
```

Export one bundled example:

```bash
meic export pure-dephasing-ohmic-N4 --out exported-example
```

Run the exact bosonic-bath pure-dephasing benchmark with your own parameters:

```bash
meic run --bath bosonic --model pure-dephasing --spectral ohmic \
  --N 4 --epsilon0 4 --epsilon 4 --delta0 0 --delta 0 \
  --beta 1 --coupling 0.05 --omega-c 5 --out pure-dephasing-output
```

Rerun a bosonic-bath spin-boson workflow for a packaged coefficient branch:

```bash
meic run --bath bosonic --model spin-boson --spectral ohmic \
  --N 4 --epsilon0 4 --epsilon 2.5 --delta0 0.5 --delta 0.5 \
  --beta 1 --coupling 0.05 --omega-c 5 --observable jx \
  --out spin-boson-output
```

Rerun the spin-bath workflow:

```bash
meic run --bath spin --model spin-environment --spectral ohmic \
  --N 4 --epsilon0 4 --epsilon 2.5 --delta0 0.5 --delta 0.5 \
  --beta 1 --coupling 0.05 --omega-c 5 --out spin-bath-output
```

## Python API

```python
import master_equation_initial_correlations as meic

for example in meic.list_examples():
    print(example.public_id, example.bath, example.model)

curves = meic.load_reference_curves("pure-dephasing-ohmic-N1")
print(curves.correlated[:3])
```

Exact pure dephasing can be evaluated directly:

```python
import numpy as np
from master_equation_initial_correlations import PureDephasingParams, exact_curves

params = PureDephasingParams(J=0.5, epsilon=4.0, xi=4.0, beta=1.0, G=0.05, omega_c=5.0)
times = np.arange(1.0e-10, 5.0, 0.2)
correlated, uncorrelated = exact_curves(params, correlated_times=times)
```

The higher-dimensional master-equation workflows use preserved Fortran solvers and precomputed coefficient inputs. The command line accepts physical parameters, selects the matching packaged coefficient branch, and gives a clear error if that branch is not packaged yet. The exact pure-dephasing benchmark is the free-form Python solver in this release.

## Model Families

- Bosonic bath, Ohmic pure-dephasing benchmark
- Bosonic bath, Ohmic spin-boson model beyond pure dephasing
- Bosonic bath, Ohmic second-moment observable `j_x^(2)`
- Bosonic bath, sub-Ohmic spin-boson model with `s = 0.5`
- Spin bath, Ohmic spin-environment model

For the collective-spin examples, pass `N`; the package uses `J = N / 2`.

## Citation

If you use this package in research, please cite the paper. A BibTeX entry is provided in `CITATION.bib`.

```bibtex
@article{mirza2020masterequation,
  title = {A master equation incorporating the system-environment correlations present in the joint equilibrium state},
  author = {Mirza, Ali Raza and Zia, Muhammad and Chaudhry, Adam Zaman},
  year = {2020},
  eprint = {2012.14853},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url = {https://arxiv.org/abs/2012.14853}
}
```

## License

The new Python package code is released under the MIT License. See `LICENSE` and `NOTICE` for details and provenance notes about bundled legacy materials.
