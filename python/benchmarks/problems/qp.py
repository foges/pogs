"""
Quadratic Program problem generator.

Problem: minimize 0.5*x'*Q*x + c'*x
         subject to A*x <= b
"""

import cvxpy as cp
import numpy as np


def generate(n=100, m=50, seed=None):
    """
    Generate a quadratic program.

    Args:
        n: Number of variables
        m: Number of inequality constraints
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate positive definite Q
    A_mat = np.random.randn(n, n)
    Q = A_mat.T @ A_mat / n + 0.1 * np.eye(n)

    # Generate linear term
    c = np.random.randn(n)

    # Generate inequality constraints
    A = np.random.randn(m, n)

    # Generate feasible point
    x_feas = np.random.randn(n)
    b = A @ x_feas + np.random.rand(m) + 0.1  # Ensure Ax < b

    # Formulate problem
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
    constraints = [A @ x <= b]
    problem = cp.Problem(objective, constraints)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"QP (n={n}, m={m})"
    problem._custom_size_metrics = {
        "n": n,
        "m": m
    }

    return problem


def generate_small(seed=None):
    """Small QP (n=50, m=25)."""
    return generate(n=50, m=25, seed=seed)


def generate_medium(seed=None):
    """Medium QP (n=500, m=250)."""
    return generate(n=500, m=250, seed=seed)


def generate_large(seed=None):
    """Large QP (n=2000, m=1000)."""
    return generate(n=2000, m=1000, seed=seed)
