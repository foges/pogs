# Quick Start

Solve your first optimization problem in under 2 minutes.

---

## Lasso Regression

The most common use case - sparse linear regression:

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

# Solve
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

## All Supported Problems

POGS provides optimized solvers for common ML problems:

### Regression

```python
from pogs import solve_lasso, solve_ridge, solve_elastic_net, solve_huber

# Lasso: ||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)

# Ridge: ||Ax - b||² + λ||x||²
result = solve_ridge(A, b, lambd=0.1)

# Elastic Net: ||Ax - b||² + λ₁||x||₁ + λ₂||x||²
result = solve_elastic_net(A, b, lambda1=0.1, lambda2=0.05)

# Huber: ρ(Ax - b) + λ||x||₁ (robust regression)
result = solve_huber(A, b, lambd=0.1, delta=1.0)
```

### Classification

```python
from pogs import solve_logistic, solve_svm

# Logistic Regression: Σ log(1 + exp(-yᵢaᵢ'x)) + λ||x||₁
y = np.sign(A @ np.random.randn(n))  # Binary labels {-1, +1}
result = solve_logistic(A, y, lambd=0.01)

# SVM: Σ max(0, 1 - yᵢaᵢ'x) + λ||x||²
result = solve_svm(A, y, lambd=0.01)
```

### Constrained Problems

```python
from pogs import solve_nonneg_ls

# Non-negative Least Squares: ||Ax - b||² s.t. x ≥ 0
result = solve_nonneg_ls(A, b)
```

---

## Result Dictionary

All solvers return a dictionary with:

| Key | Description |
|-----|-------------|
| `x` | Solution vector |
| `status` | 0=success, other=failure |
| `iterations` | Number of iterations |
| `optval` | Optimal objective value |

```python
result = solve_lasso(A, b, lambd=0.1)

x = result['x']           # Solution
print(f"Solution norm: {np.linalg.norm(x):.4f}")
print(f"Iterations: {result['iterations']}")
```

---

## Tuning Parameters

### Tolerance

```python
# More accurate (slower)
result = solve_lasso(A, b, lambd=0.1, rel_tol=1e-6, abs_tol=1e-6)

# Faster (less accurate)
result = solve_lasso(A, b, lambd=0.1, rel_tol=1e-3, abs_tol=1e-3)
```

### Max Iterations

```python
# For difficult problems
result = solve_lasso(A, b, lambd=0.1, max_iter=5000)
```

### Verbose Output

```python
# See iteration progress
result = solve_lasso(A, b, lambd=0.1, verbose=2)
```

---

## Using with CVXPY

For more general problems, use POGS as a CVXPY backend:

```python
import cvxpy as cp
import numpy as np

# Define problem
x = cp.Variable(300)
A = np.random.randn(500, 300)
b = np.random.randn(500)

objective = cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1))
prob = cp.Problem(objective)

# Solve with POGS
prob.solve(solver='POGS')

print(f"Status: {prob.status}")
print(f"Optimal value: {prob.value:.4f}")
```

---

## Performance Comparison

POGS is optimized for these ML problems:

| Problem | Size | POGS | OSQP | SCS |
|---------|------|------|------|-----|
| Lasso | 500x300 | **51ms** | 399ms | 206ms |
| Ridge | 500x300 | **8ms** | 89ms | 64ms |
| Logistic | 500x300 | **34ms** | 312ms | 198ms |

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

Check your data scaling:

```python
# Normalize columns of A
A = A / np.linalg.norm(A, axis=0, keepdims=True)
result = solve_lasso(A, b, lambd=0.1)
```

---

## Next Steps

- [Examples](../examples/lasso.md) - Detailed examples with explanations
- [API Reference](../api/solver.md) - Full function documentation
