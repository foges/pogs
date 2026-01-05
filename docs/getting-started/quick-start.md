# Quick Start

Solve your first optimization problem in under a minute.

---

## Lasso Regression

The most common use case—sparse linear regression:

```python
from pogs import solve_lasso
import numpy as np

# Generate problem data
np.random.seed(0)
m, n = 500, 300
A = np.random.randn(m, n)
x_true = np.zeros(n)
x_true[:10] = np.random.randn(10)  # Sparse ground truth
b = A @ x_true + 0.1 * np.random.randn(m)

# Solve: minimize ½||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)

print(f"Status: {'Solved' if result['status'] == 0 else 'Failed'}")
print(f"Iterations: {result['iterations']}")
print(f"Nonzeros: {np.sum(np.abs(result['x']) > 1e-4)}")
```

Output:
```
Status: Solved
Iterations: 60
Nonzeros: 10
```

---

## All Solvers

POGS provides optimized solvers for common ML problems:

### Regression

```python
from pogs import solve_lasso, solve_ridge, solve_elastic_net, solve_huber

# Lasso: ½||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)

# Ridge: ½||Ax - b||² + λ||x||²
result = solve_ridge(A, b, lambd=0.1)

# Elastic Net: ½||Ax - b||² + λ₁||x||₁ + λ₂||x||²
result = solve_elastic_net(A, b, lambda1=0.1, lambda2=0.05)

# Huber: Σ huber(Aᵢx - bᵢ) + λ||x||₁
result = solve_huber(A, b, lambd=0.1, delta=1.0)
```

### Classification

```python
from pogs import solve_logistic, solve_svm

# Labels in {-1, +1}
y = np.sign(A @ np.random.randn(n))

# Logistic: Σ log(1 + exp(-yᵢaᵢᵀx)) + λ||x||₁
result = solve_logistic(A, y, lambd=0.01)

# SVM: Σ max(0, 1 - yᵢaᵢᵀx) + λ||x||²
result = solve_svm(A, y, lambd=0.01)
```

### Constrained

```python
from pogs import solve_nonneg_ls

# Non-negative Least Squares: ½||Ax - b||² s.t. x ≥ 0
result = solve_nonneg_ls(A, b)
```

---

## Result Dictionary

All solvers return a dictionary:

| Key | Type | Description |
|:----|:-----|:------------|
| `x` | ndarray | Solution vector |
| `status` | int | 0=converged, 1=max_iter, 2=error |
| `iterations` | int | Number of ADMM iterations |
| `optval` | float | Optimal objective value |

```python
result = solve_lasso(A, b, lambd=0.1)

x = result['x']           # Solution
status = result['status'] # 0 = success
iters = result['iterations']
```

---

## Tuning Parameters

All solvers accept the same parameters:

```python
result = solve_lasso(
    A, b,
    lambd=0.1,       # Regularization strength
    abs_tol=1e-4,    # Absolute tolerance (default)
    rel_tol=1e-4,    # Relative tolerance (default)
    max_iter=2500,   # Maximum iterations (default)
    verbose=0,       # 0=quiet, 1=summary, 2=per-iteration
)
```

### Higher Accuracy

```python
result = solve_lasso(A, b, lambd=0.1, abs_tol=1e-6, rel_tol=1e-6)
```

### Faster (Lower Accuracy)

```python
result = solve_lasso(A, b, lambd=0.1, abs_tol=1e-3, rel_tol=1e-3)
```

### Verbose Output

```python
result = solve_lasso(A, b, lambd=0.1, verbose=2)
```

---

## CVXPY Integration

For more complex problems, use `pogs_solve()` with CVXPY:

```python
import cvxpy as cp
from pogs import pogs_solve

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

pogs_solve(prob)  # Auto-detects Lasso
print(x.value)
```

See [CVXPY Integration](../user-guide/cvxpy-integration.md) for details.

---

## Troubleshooting

### "Max iterations reached"

The problem may be poorly scaled or difficult:

```python
# Try more iterations
result = solve_lasso(A, b, lambd=0.1, max_iter=5000)

# Or loosen tolerance
result = solve_lasso(A, b, lambd=0.1, rel_tol=1e-3)
```

### Slow convergence

Normalize your data:

```python
# Normalize columns of A
A = A / np.linalg.norm(A, axis=0, keepdims=True)
result = solve_lasso(A, b, lambd=0.1)
```

---

## Next Steps

- [Lasso Example](../examples/lasso.md) - Detailed Lasso walkthrough
- [Logistic Example](../examples/logistic.md) - Classification example
- [API Reference](../api/solver.md) - Full documentation
