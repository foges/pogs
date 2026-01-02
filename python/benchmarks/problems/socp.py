"""
Second-Order Cone Program problem generator.

Problem: minimize c'*x
         subject to ||A_i*x + b_i|| <= c_i'*x + d_i
"""

import cvxpy as cp
import numpy as np


def generate_robust_ls(m=100, n=50, seed=None):
    """
    Generate a robust least squares problem (SOCP formulation).

    Problem: minimize ||A*x - b||_2 + lambda*||x||_1

    Args:
        m: Number of measurements
        n: Number of variables
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate problem data
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[np.random.rand(n) < 0.8] = 0  # Sparse
    b = A @ x_true + 0.1 * np.random.randn(m)

    # Formulate as SOCP
    x = cp.Variable(n)
    t = cp.Variable()

    objective = cp.Minimize(t + 0.1 * cp.norm(x, 1))
    constraints = [
        cp.norm(A @ x - b) <= t
    ]
    problem = cp.Problem(objective, constraints)

    # Add metadata
    problem.name = f"SOCP-RobustLS (m={m}, n={n})"
    problem.size_metrics = {
        "m": m,
        "n": n
    }

    return problem


def generate_portfolio_robust(n_assets=50, seed=None):
    """
    Generate a robust portfolio optimization problem (SOCP).

    Args:
        n_assets: Number of assets
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Expected returns
    mu = np.random.rand(n_assets) * 0.2 + 0.05

    # Uncertainty in returns
    Sigma_sqrt = np.random.randn(n_assets, n_assets) / np.sqrt(n_assets)

    # Risk parameter
    gamma = 2.0

    # Formulate problem
    w = cp.Variable(n_assets)

    # Robust objective: minimize worst-case risk - expected return
    objective = cp.Minimize(
        gamma * cp.norm(Sigma_sqrt.T @ w) - mu.T @ w
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    problem = cp.Problem(objective, constraints)

    # Add metadata
    problem.name = f"SOCP-Portfolio (n={n_assets})"
    problem.size_metrics = {
        "n_assets": n_assets
    }

    return problem


def generate_small(seed=None):
    """Small SOCP (m=100, n=50)."""
    return generate_robust_ls(m=100, n=50, seed=seed)


def generate_medium(seed=None):
    """Medium SOCP (m=500, n=250)."""
    return generate_robust_ls(m=500, n=250, seed=seed)
