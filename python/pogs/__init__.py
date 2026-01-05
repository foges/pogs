"""
POGS - Proximal Operator Graph Solver

High-performance convex optimization using ADMM.

Example usage:
    from pogs import solve_lasso, solve_logistic

    # Lasso regression
    result = solve_lasso(A, b, lambd=0.1)

    # Logistic regression
    result = solve_logistic(A, y, lambd=0.01)

    # With CVXPY (auto-detects graph-form patterns)
    from pogs import pogs_solve
    pogs_solve(problem)
"""

from __future__ import annotations


__version__ = "0.4.2"

# Graph-form solvers (main API)
# CVXPY integration (cvxpy checked at runtime, not import time)
from pogs.cvxpy import pogs_solve
from pogs.graph import (
    solve_elastic_net,
    solve_huber,
    solve_lasso,
    solve_logistic,
    solve_nonneg_ls,
    solve_ridge,
    solve_svm,
)


__all__ = [
    "__version__",
    # CVXPY integration
    "pogs_solve",
    # Graph-form solvers
    "solve_elastic_net",
    "solve_huber",
    "solve_lasso",
    "solve_logistic",
    "solve_nonneg_ls",
    "solve_ridge",
    "solve_svm",
]
