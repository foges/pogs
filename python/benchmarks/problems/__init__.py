"""
Problem generators for POGS benchmarks.

Each module contains a generate() function that creates a CVXPY problem
with standardized attributes:
- problem.name: Human-readable problem name
- problem.size_metrics: Dict of problem dimensions
"""

from problems import lasso, logistic, lp, portfolio, qp, sdp, socp


__all__ = ["lasso", "logistic", "lp", "portfolio", "qp", "sdp", "socp"]
