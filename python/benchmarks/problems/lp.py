"""
Linear Program problem generator.

Problem: minimize c'*x
         subject to A*x = b, x >= 0
"""

import cvxpy as cp
import numpy as np


def generate(m=100, n=200, density=1.0, seed=None):
    """
    Generate a linear program in standard form.

    Args:
        m: Number of equality constraints
        n: Number of variables
        density: Sparsity of A (1.0 = dense)
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate problem data
    if density < 1.0:
        nnz = int(m * n * density)
        rows = np.random.randint(0, m, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.randn(nnz)
        from scipy.sparse import coo_matrix

        A = coo_matrix((data, (rows, cols)), shape=(m, n)).toarray()
    else:
        A = np.random.randn(m, n)

    # Generate feasible point to ensure problem is bounded
    x_feas = np.random.rand(n) + 0.1  # Ensure x > 0
    b = A @ x_feas

    # Generate cost vector
    c = np.random.randn(n)

    # Formulate problem
    x = cp.Variable(n)
    objective = cp.Minimize(c.T @ x)
    constraints = [A @ x == b, x >= 0]
    problem = cp.Problem(objective, constraints)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"LP (m={m}, n={n}, density={density:.2f})"
    problem._custom_size_metrics = {"m": m, "n": n, "density": density}

    return problem


def generate_small(seed=None):
    """Small LP (m=100, n=200)."""
    return generate(m=100, n=200, seed=seed)


def generate_medium(seed=None):
    """Medium LP (m=500, n=1000)."""
    return generate(m=500, n=1000, seed=seed)


def generate_large(seed=None):
    """Large LP (m=2000, n=5000)."""
    return generate(m=2000, n=5000, seed=seed)


def generate_sparse(seed=None):
    """Sparse LP (m=1000, n=2000, 5% density)."""
    return generate(m=1000, n=2000, density=0.05, seed=seed)
