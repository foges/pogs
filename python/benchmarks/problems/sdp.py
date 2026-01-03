"""
Semidefinite Program problem generator.

Problem: minimize c'*x
         subject to F(x) >= 0 (positive semidefinite)
"""

import cvxpy as cp
import numpy as np


def generate_maxcut(n=20, seed=None):
    """
    Generate a max-cut SDP relaxation.

    Args:
        n: Number of nodes in graph
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random graph (adjacency matrix)
    # Symmetric matrix with random edges
    W = np.random.rand(n, n)
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # No self-loops

    # SDP formulation of max-cut relaxation
    X = cp.Variable((n, n), symmetric=True)

    # Objective: maximize trace(W*X) = minimize -trace(W*X)
    objective = cp.Minimize(-cp.trace(W @ X))

    # Constraints
    constraints = [
        X >> 0,  # Positive semidefinite
        cp.diag(X) == 1  # Diagonal entries are 1
    ]

    problem = cp.Problem(objective, constraints)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"SDP-MaxCut (n={n})"
    problem._custom_size_metrics = {
        "n": n,
        "matrix_size": n
    }

    return problem


def generate_matrix_completion(n=30, m=20, seed=None):
    """
    Generate a matrix completion SDP.

    Args:
        n: Matrix size (n x n)
        m: Number of observed entries
        seed: Random seed

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate low-rank matrix to recover
    U = np.random.randn(n, 3)  # Rank-3 matrix
    M_true = U @ U.T

    # Sample m entries
    obs_indices = []
    obs_values = []
    for _ in range(m):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        obs_indices.append((i, j))
        obs_values.append(M_true[i, j])

    # Formulate SDP
    M = cp.Variable((n, n), symmetric=True)

    # Minimize nuclear norm (trace of M since M is PSD)
    objective = cp.Minimize(cp.trace(M))

    # Constraints: match observed entries, M is PSD
    constraints = [M >> 0]
    for (i, j), val in zip(obs_indices, obs_values):
        constraints.append(M[i, j] == val)

    problem = cp.Problem(objective, constraints)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"SDP-MatrixCompletion (n={n}, m={m})"
    problem._custom_size_metrics = {
        "n": n,
        "m": m
    }

    return problem


def generate_small(seed=None):
    """Small SDP (10x10 matrix)."""
    return generate_maxcut(n=10, seed=seed)


def generate_medium(seed=None):
    """Medium SDP (30x30 matrix)."""
    return generate_maxcut(n=30, seed=seed)


def generate_large(seed=None):
    """Large SDP (50x50 matrix)."""
    return generate_maxcut(n=50, seed=seed)
