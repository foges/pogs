"""
Portfolio Optimization problem generator.

Problem: minimize w'*Sigma*w - gamma*mu'*w
         subject to sum(w) = 1, w >= 0
"""

import cvxpy as cp
import numpy as np


def generate(n_assets=50, gamma=1.0, seed=None):
    """
    Generate a Markowitz portfolio optimization problem.

    Args:
        n_assets: Number of assets
        gamma: Risk aversion parameter (higher = less risk)
        seed: Random seed for reproducibility

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate expected returns
    mu = np.random.rand(n_assets) * 0.2 + 0.05  # Returns between 5% and 25%

    # Generate covariance matrix (positive definite)
    A = np.random.randn(n_assets, n_assets)
    Sigma = A.T @ A / n_assets + 0.01 * np.eye(n_assets)  # Regularize

    # Formulate problem
    w = cp.Variable(n_assets)

    # Risk - expected return tradeoff
    risk = cp.quad_form(w, Sigma)
    expected_return = mu.T @ w

    objective = cp.Minimize(risk - gamma * expected_return)
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= 0           # Long only
    ]
    problem = cp.Problem(objective, constraints)

    # Add metadata
    problem.name = f"Portfolio (n={n_assets}, gamma={gamma:.1f})"
    problem.size_metrics = {
        "n_assets": n_assets,
        "gamma": gamma
    }

    return problem


def generate_small(seed=None):
    """Small portfolio (50 assets)."""
    return generate(n_assets=50, seed=seed)


def generate_medium(seed=None):
    """Medium portfolio (500 assets)."""
    return generate(n_assets=500, seed=seed)


def generate_large(seed=None):
    """Large portfolio (2000 assets)."""
    return generate(n_assets=2000, seed=seed)
