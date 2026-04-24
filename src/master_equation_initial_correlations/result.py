from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ._resources import write_table
from .observables import normalize_observable_expression


def _safe_name(label: str) -> str:
    cleaned = []
    for char in label:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        elif char == "^":
            continue
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    return name or "observable"


def _expectation_table(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(values)
    if np.iscomplexobj(values) and np.max(np.abs(values.imag)) > 1.0e-10:
        return np.column_stack([times, values.real, values.imag]), "t expectation_real expectation_imag"
    return np.column_stack([times, values.real]), "t expectation"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


@dataclass
class Result:
    """In-memory solver result, intentionally close to QuTiP's result shape."""

    times: np.ndarray
    expect: list[np.ndarray]
    e_data: dict[str, np.ndarray]
    states: np.ndarray | None
    params: Any
    branch: str
    solver: str
    e_ops: list[str]
    numerics: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_dirs: dict[str, Path] = field(default_factory=dict)
    _temporary_directories: list[Any] = field(default_factory=list, repr=False)

    def __enter__(self) -> "Result":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def as_dict(self) -> dict[str, Any]:
        """Return a copy-friendly dictionary of the in-memory solver result."""

        return {
            "times": self.times.copy(),
            "expect": {label: np.asarray(values).copy() for label, values in zip(self.e_ops, self.expect)},
            "states": None if self.states is None else self.states.copy(),
            "branch": self.branch,
            "solver": self.solver,
            "observables": list(self.e_ops),
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> Any:
        """Return expectation values as a pandas DataFrame if pandas is installed."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Result.to_dataframe() requires pandas; install pandas to use this helper.") from exc
        data: dict[str, np.ndarray] = {"t": self.times}
        for label, values in zip(self.e_ops, self.expect):
            data[label] = np.asarray(values)
        return pd.DataFrame(data)

    def save(self, output_dir: str | Path, *, include_artifacts: bool = False, overwrite: bool = False) -> Path:
        """Save expectation data, metadata, and optional generated artifacts."""

        destination = Path(output_dir)
        if include_artifacts:
            if not self.artifact_dirs:
                raise FileNotFoundError(
                    "include_artifacts=True was requested, but this Result has no retained artifacts. "
                    "For Fortran-backed runs, call meic.solve(..., keep_artifacts=True) and save before close()."
                )
            missing = [f"{label}: {source}" for label, source in self.artifact_dirs.items() if not source.exists()]
            if missing:
                joined = "; ".join(missing)
                raise FileNotFoundError(
                    f"include_artifacts=True was requested, but artifact directories are unavailable: {joined}. "
                    "For Fortran-backed runs, rerun with keep_artifacts=True and save before close()."
                )
        generated_targets: list[Path] = []
        if destination.exists():
            generated_targets.extend(sorted(destination.glob("expect-*.dat")))
            metadata_path = destination / "result_metadata.json"
            if metadata_path.exists():
                generated_targets.append(metadata_path)
            artifact_root = destination / "artifacts"
            if artifact_root.exists():
                generated_targets.append(artifact_root)
        conflicts = [path.name for path in generated_targets]
        if conflicts and not overwrite:
            joined = ", ".join(conflicts)
            raise FileExistsError(f"Refusing to overwrite existing result file(s) in {destination}: {joined}.")
        destination.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for target in generated_targets:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        for label, values in zip(self.e_ops, self.expect):
            table, columns = _expectation_table(self.times, values)
            initial_state = self.metadata.get("initial_state", {})
            header = [
                "Generated by master-equation-with-initial-correlations",
                "storage: exported from an in-memory Result object",
                f"solver: {self.solver}",
                f"branch: {self.branch}",
                f"observable: {label}",
                f"initial_state_source: {initial_state.get('source', 'not_recorded')}",
                "normalization: lowercase spin operators are dimensionless; jx^2 is 4 <J_x^2> / N^2",
                f"columns: {columns}",
            ]
            write_table(destination / f"expect-{_safe_name(label)}.dat", table, header_lines=header)

        observable_parameters = self.metadata.get("observable_parameters")
        if observable_parameters is None:
            parameters_by_observable = {self.e_ops[0]: _json_ready(self.params)} if self.e_ops else {}
        else:
            parameters_by_observable = {
                label: _json_ready(params)
                for label, params in zip(self.e_ops, observable_parameters)
            }
        parameters = _json_ready(self.params)
        if len(self.e_ops) > 1:
            parameters = {
                "note": "multiple observables were calculated; see parameters_by_observable for observable-specific provenance",
                "first_observable": parameters,
            }

        metadata = {
            "solver": self.solver,
            "branch": self.branch,
            "observables": self.e_ops,
            "times": {
                "start": float(self.times[0]),
                "stop": float(self.times[-1]),
                "count": int(self.times.size),
                "dt": float(self.times[1] - self.times[0]) if self.times.size > 1 else None,
            },
            "parameters": parameters,
            "parameters_by_observable": parameters_by_observable,
            "metadata": _json_ready(self.metadata),
            "artifacts_available": {label: str(path) for label, path in self.artifact_dirs.items()},
        }
        (destination / "result_metadata.json").write_text(json.dumps(metadata, indent=2))

        if include_artifacts:
            artifact_root = destination / "artifacts"
            if artifact_root.exists() and overwrite:
                shutil.rmtree(artifact_root)
            artifact_root.mkdir(parents=True, exist_ok=True)
            for label, source in self.artifact_dirs.items():
                target = artifact_root / _safe_name(label)
                if target.exists() and overwrite:
                    shutil.rmtree(target)
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite existing artifact directory {target}.")
                shutil.copytree(source, target)

        return destination

    def close(self) -> None:
        """Remove temporary solver artifacts kept alive by this result."""

        for temporary_directory in self._temporary_directories:
            temporary_directory.cleanup()
        self._temporary_directories.clear()


def observable_label(observable: str | np.ndarray, fallback: str) -> str:
    if isinstance(observable, str):
        return normalize_observable_expression(observable)
    return fallback
