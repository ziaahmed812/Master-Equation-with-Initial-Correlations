from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ._resources import copy_resource_tree, load_table
from ._types import ReferenceCurves
from .catalog import get_example


def _scaled_curve(table: np.ndarray, divisor: float) -> np.ndarray:
    scaled = np.array(table, copy=True)
    scaled[:, 1] = scaled[:, 1] / divisor
    return scaled


def load_reference_curves(example_id: str, include_exact: str = "auto") -> ReferenceCurves:
    example = get_example(example_id)
    base = example.asset_dir

    correlated = load_table(f"{base}/tables/EXPX-C.DAT")
    uncorrelated = load_table(f"{base}/tables/EXPX-UNC.DAT")
    jz_correlated = None
    jz_uncorrelated = None

    if example.plot_divisor != 1.0:
        correlated = _scaled_curve(correlated, example.plot_divisor)
        uncorrelated = _scaled_curve(uncorrelated, example.plot_divisor)

    expz_c_path = f"{base}/tables/EXPZ-C.DAT"
    expz_u_path = f"{base}/tables/EXPZ-UNC.DAT"
    try:
        jz_correlated = load_table(expz_c_path)
        jz_uncorrelated = load_table(expz_u_path)
    except FileNotFoundError:
        pass

    want_exact = include_exact == "always" or (include_exact == "auto" and example.supports_exact)
    exact_correlated = None
    exact_uncorrelated = None
    if want_exact:
        exact_correlated = load_table(f"{base}/tables/exact-correlated.dat")
        exact_uncorrelated = load_table(f"{base}/tables/exact-uncorrelated.dat")

    return ReferenceCurves(
        example=example,
        correlated=correlated,
        uncorrelated=uncorrelated,
        jz_correlated=jz_correlated,
        jz_uncorrelated=jz_uncorrelated,
        exact_correlated=exact_correlated,
        exact_uncorrelated=exact_uncorrelated,
    )


def export_example_assets(example_id: str, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    example = get_example(example_id)
    destination = Path(output_dir) / example.public_id
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing export directory {destination}.")
        shutil.rmtree(destination)
    copy_resource_tree(example.asset_dir, destination)
    return destination
