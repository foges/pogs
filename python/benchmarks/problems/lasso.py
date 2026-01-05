"""
Lasso Regression problem generator.

Problem: minimize 0.5 * ||Ax - b||^2 + lambda * ||x||_1
"""

import cvxpy as cp
import numpy as np


def generate(m=100, n=50, density=1.0, condition_number=1.0, seed=None):
    """
    Generate a Lasso regression problem.

    Args:
        m: Number of samples
        n: Number of features
        density: Sparsity of A (1.0 = dense, 0.01 = 1% nonzero)
        condition_number: Condition number of A (1.0 = well-conditioned)
        seed: Random seed for reproducibility

    Returns:
        CVXPY Problem with name and size_metrics attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate matrix A
    if density < 1.0:
        # Sparse matrix
        nnz = int(m * n * density)
        rows = np.random.randint(0, m, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.randn(nnz)
        from scipy.sparse import coo_matrix

        A = coo_matrix((data, (rows, cols)), shape=(m, n)).toarray()
    else:
        # Dense matrix
        A = np.random.randn(m, n)

    # Apply ill-conditioning if requested
    if condition_number > 1.0:
        for j in range(n):
            scale = condition_number ** (-j / (n - 1))
            A[:, j] *= scale

    # Generate sparse true solution
    x_true = np.random.randn(n)
    x_true[np.random.rand(n) < 0.9] = 0  # 90% sparse

    # Generate observations with noise
    b = A @ x_true + 0.1 * np.random.randn(m)

    # Set regularization parameter
    lambda_max = np.linalg.norm(A.T @ b, np.inf)
    lambda_val = 0.1 * lambda_max

    # Formulate problem
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1))
    problem = cp.Problem(objective)

    # Add metadata (use custom attribute since size_metrics is read-only in newer CVXPY)
    problem.name = f"Lasso (m={m}, n={n}, density={density:.2f}, kappa={condition_number:.0f})"
    problem._custom_size_metrics = {
        "m": m,
        "n": n,
        "density": density,
        "condition_number": condition_number,
    }

    return problem


def generate_small(seed=None):
    """Small Lasso problem (m=100, n=50)."""
    return generate(m=100, n=50, seed=seed)


def generate_medium(seed=None):
    """Medium Lasso problem (m=1000, n=500)."""
    return generate(m=1000, n=500, seed=seed)


def generate_large(seed=None):
    """Large Lasso problem (m=10000, n=5000)."""
    return generate(m=10000, n=5000, seed=seed)


def generate_sparse(seed=None):
    """Sparse Lasso problem (m=10000, n=5000, 1% density)."""
    return generate(m=10000, n=5000, density=0.01, seed=seed)


def generate_ill_conditioned(seed=None):
    """Ill-conditioned Lasso (m=1000, n=500, kappa=1e4)."""
    return generate(m=1000, n=500, condition_number=1e4, seed=seed)
