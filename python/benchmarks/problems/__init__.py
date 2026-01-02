"""
Problem generators for POGS benchmarks.

Each module contains a generate() function that creates a CVXPY problem
with standardized attributes:
- problem.name: Human-readable problem name
- problem.size_metrics: Dict of problem dimensions
"""

from . import lasso
from . import logistic
from . import portfolio
from . import lp
from . import qp
from . import socp
from . import sdp

__all__ = ['lasso', 'logistic', 'portfolio', 'lp', 'qp', 'socp', 'sdp']
