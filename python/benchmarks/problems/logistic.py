"""
Logistic Regression problem generator.

Problem: minimize sum(logistic(-y_i * (w'*x_i + b))) + lambda * ||w||^2
"""

import cvxpy as cp
import numpy as np


def generate(n_samples=100, n_features=20, seed=None):
    """
    Generate a logistic regression problem.

    Args:
        n_samples: Number of training examples
        n_features: Number of features
        seed: Random seed for reproducibility

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate true weights (sparse)
    w_true = np.random.randn(n_features)
    w_true[np.random.rand(n_features) < 0.7] = 0  # 70% sparse
    b_true = np.random.randn()

    # Generate labels with logistic noise
    logits = X @ w_true + b_true
    probs = 1 / (1 + np.exp(-logits))
    y = 2 * (np.random.rand(n_samples) < probs) - 1  # Labels in {-1, 1}

    # Formulate problem
    w = cp.Variable(n_features)
    b = cp.Variable()

    # Logistic loss + L2 regularization
    lambda_val = 0.1
    losses = cp.sum(cp.logistic(-cp.multiply(y, X @ w + b)))
    regularization = lambda_val * cp.sum_squares(w)

    objective = cp.Minimize(losses + regularization)
    problem = cp.Problem(objective)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"Logistic (n={n_samples}, d={n_features})"
    problem._custom_size_metrics = {"n_samples": n_samples, "n_features": n_features}

    return problem


def generate_small(seed=None):
    """Small logistic regression (n=100, d=20)."""
    return generate(n_samples=100, n_features=20, seed=seed)


def generate_medium(seed=None):
    """Medium logistic regression (n=1000, d=100)."""
    return generate(n_samples=1000, n_features=100, seed=seed)


def generate_large(seed=None):
    """Large logistic regression (n=10000, d=500)."""
    return generate(n_samples=10000, n_features=500, seed=seed)
