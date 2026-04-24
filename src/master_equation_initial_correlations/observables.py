from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pure_dephasing import spin_operators


@dataclass(frozen=True)
class ObservableSpec:
    """A parsed observable and the matrix convention used by the solvers.

    The public expression is dimensionless: ``jx`` means ``J_x / J`` and
    ``jx^2`` means ``J_x^2 / J^2``.  The preserved Fortran programs divide
    expectations by ``J``, so they are supplied with ``J * matrix``.
    """

    expression: str
    dimensionless_matrix: np.ndarray
    fortran_matrix: np.ndarray


class ObservableParseError(ValueError):
    """Raised when an observable expression is not part of the safe grammar."""


def normalize_observable_expression(observable: Any) -> str:
    if isinstance(observable, np.ndarray):
        return "custom_matrix"
    if not isinstance(observable, str):
        raise ObservableParseError("observable must be a string expression or a NumPy matrix.")
    expression = observable.strip().lower().replace(" ", "")
    aliases = {
        "jx2": "jx^2",
        "jx(2)": "jx^2",
        "jx^(2)": "jx^2",
        "jy2": "jy^2",
        "jy(2)": "jy^2",
        "jy^(2)": "jy^2",
        "jz2": "jz^2",
        "jz(2)": "jz^2",
        "jz^(2)": "jz^2",
        "j_x": "jx",
        "j_y": "jy",
        "j_z": "jz",
    }
    return aliases.get(expression, expression)


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float, complex, np.number))


def _matrix_power(matrix: np.ndarray, exponent: object) -> np.ndarray:
    if not _is_scalar(exponent):
        raise ObservableParseError("matrix powers require a scalar integer exponent.")
    exponent_value = complex(exponent)
    if abs(exponent_value.imag) > 1.0e-14:
        raise ObservableParseError("matrix powers require a real integer exponent.")
    integer_exponent = int(round(exponent_value.real))
    if integer_exponent < 0 or abs(exponent_value.real - integer_exponent) > 1.0e-14:
        raise ObservableParseError("matrix powers require a non-negative integer exponent.")
    return np.linalg.matrix_power(matrix, integer_exponent)


def _mul(left: object, right: object) -> object:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left @ right
    if isinstance(left, np.ndarray) and _is_scalar(right):
        return left * right
    if _is_scalar(left) and isinstance(right, np.ndarray):
        return left * right
    if _is_scalar(left) and _is_scalar(right):
        return left * right
    raise ObservableParseError("unsupported multiplication in observable expression.")


def _div(left: object, right: object) -> object:
    if not _is_scalar(right):
        raise ObservableParseError("division is only supported by scalar denominators.")
    if abs(complex(right)) <= 1.0e-15:
        raise ObservableParseError("observable expression divides by zero.")
    if isinstance(left, np.ndarray) or _is_scalar(left):
        return left / right
    raise ObservableParseError("unsupported division in observable expression.")


def _add(left: object, right: object, sign: int = 1) -> object:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ObservableParseError("cannot add a scalar directly to an operator; use the identity operator `id`.")
        return left + sign * right
    if _is_scalar(left) and _is_scalar(right):
        return left + sign * right
    raise ObservableParseError("unsupported addition in observable expression.")


def _evaluate_node(node: ast.AST, names: dict[str, np.ndarray]) -> object:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ObservableParseError("observable constants must be numeric.")
    if isinstance(node, ast.Name):
        if node.id not in names:
            allowed = ", ".join(sorted(names))
            raise ObservableParseError(f"unknown observable symbol {node.id!r}; allowed symbols are {allowed}.")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _evaluate_node(node.operand, names)
        return -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _evaluate_node(node.operand, names)
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, names)
        right = _evaluate_node(node.right, names)
        if isinstance(node.op, ast.Add):
            return _add(left, right, sign=1)
        if isinstance(node.op, ast.Sub):
            return _add(left, right, sign=-1)
        if isinstance(node.op, ast.Mult):
            return _mul(left, right)
        if isinstance(node.op, ast.Div):
            return _div(left, right)
        if isinstance(node.op, ast.Pow):
            if isinstance(left, np.ndarray):
                return _matrix_power(left, right)
            if _is_scalar(left) and _is_scalar(right):
                return left**right
        raise ObservableParseError("unsupported operator in observable expression.")
    raise ObservableParseError("unsupported syntax in observable expression.")


def parse_observable(observable: str | np.ndarray, J: float) -> ObservableSpec:
    if J <= 0:
        raise ObservableParseError("J must be positive when constructing observables.")
    _, raw_jx, raw_jy, raw_jz = spin_operators(J)
    identity = np.eye(raw_jx.shape[0], dtype=complex)

    if isinstance(observable, np.ndarray):
        matrix = np.asarray(observable, dtype=complex)
        if matrix.shape != raw_jx.shape:
            raise ObservableParseError(
                f"custom observable matrix has shape {matrix.shape}, expected {raw_jx.shape} for J={J}."
            )
        return ObservableSpec(
            expression="custom_matrix",
            dimensionless_matrix=matrix,
            fortran_matrix=J * matrix,
        )

    expression = normalize_observable_expression(observable)
    python_expression = expression.replace("^", "**")
    names = {
        "id": identity,
        "jx": raw_jx / J,
        "jy": raw_jy / J,
        "jz": raw_jz / J,
    }
    try:
        tree = ast.parse(python_expression, mode="eval")
    except SyntaxError as exc:
        raise ObservableParseError(f"invalid observable expression {observable!r}.") from exc
    value = _evaluate_node(tree, names)
    if _is_scalar(value):
        value = complex(value) * identity
    if not isinstance(value, np.ndarray):
        raise ObservableParseError("observable expression did not produce an operator matrix.")
    return ObservableSpec(
        expression=expression,
        dimensionless_matrix=np.asarray(value, dtype=complex),
        fortran_matrix=J * np.asarray(value, dtype=complex),
    )


def expectation_from_density_matrices(
    densities: np.ndarray,
    observable: str | np.ndarray,
    J: float,
    *,
    real_if_close: bool = True,
) -> np.ndarray:
    spec = parse_observable(observable, J)
    values = np.asarray([np.trace(spec.dimensionless_matrix @ rho) for rho in densities], dtype=complex)
    if real_if_close and np.max(np.abs(values.imag)) <= 1.0e-10:
        return values.real
    return values
