from __future__ import annotations

from typing import Any

from ._types import BathParams, SystemParams
from ._validation import normalize_correlations
from .blackbox import PureDephasingSolverWC, PureDephasingSolverWOC
from .result import Result


def solve(
    system: SystemParams,
    bath: BathParams,
    *,
    tlist: Any,
    e_ops: Any = None,
    correlations: str = "with",
    store_states: bool = False,
    verbose: bool = True,
) -> Result:
    """Run the exact analytical pure-dephasing solver.

    This solver is Python-only and is valid for the bosonic Ohmic
    pure-dephasing limit with ``delta0=0`` and ``delta=0``.
    It uses the analytical correlated or uncorrelated state construction and
    does not accept a custom ``initial_state``. Unlike the Fortran-backed
    master-equation solver, this analytical path accepts arbitrary finite,
    nonnegative, strictly increasing time samples.
    """

    branch = normalize_correlations(correlations)
    solver_cls = PureDephasingSolverWC if branch == "with_correlations" else PureDephasingSolverWOC
    return solver_cls(system=system, bath=bath).run(
        tlist,
        e_ops=e_ops,
        store_states=store_states,
        verbose=verbose,
    )
