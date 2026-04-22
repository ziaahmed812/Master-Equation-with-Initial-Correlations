from __future__ import annotations

import json
import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

import numpy as np


_ASSETS = files("master_equation_initial_correlations").joinpath("_assets")


def asset(path: str):
    target = _ASSETS
    for part in Path(path).parts:
        target = target.joinpath(part)
    return target


def read_json(path: str) -> dict:
    return json.loads(asset(path).read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_table(path: str) -> np.ndarray:
    with as_file(asset(path)) as resolved_path:
        return np.loadtxt(resolved_path, dtype=float)


def copy_resource_file(src: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(asset(src).read_bytes())
    return dst


def _iter_tree(rel_path: str) -> Iterator[tuple[str, object]]:
    root = asset(rel_path)
    for child in root.iterdir():
        yield child.name, child


def copy_resource_tree(src_dir: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name, child in _iter_tree(src_dir):
        dst = dst_dir / name
        if child.is_dir():
            copy_resource_tree(f"{src_dir}/{name}", dst)
        else:
            dst.write_bytes(child.read_bytes())
    return dst_dir


def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
