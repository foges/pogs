# Lasso Regression

Sparse linear regression with L1 regularization.

---

## The Problem

Lasso solves:

$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1
$$

The L1 penalty promotes **sparse solutions** - most coefficients become exactly zero.

**Use cases:**
- Feature selection (identify important predictors)
- High-dimensional regression (n > m)
- Interpretable models

---

## Quick Example

```python
from pogs import solve_lasso
import numpy as np

# Generate sparse problem
np.random.seed(42)
m, n = 500, 300

# Design matrix
A = np.random.randn(m, n)

# True sparse solution (only 10 nonzeros)
x_true = np.zeros(n)
x_true[:10] = np.random.randn(10)

# Observations with noise
b = A @ x_true + 0.1 * np.random.randn(m)

# Solve
result = solve_lasso(A, b, lambd=0.1)

print(f"Solve time: {result['solve_time']*1000:.1f}ms")
print(f"Iterations: {result['iter']}")
print(f"Nonzeros found: {np.sum(np.abs(result['x']) > 1e-4)}")
print(f"Recovery error: {np.linalg.norm(result['x'] - x_true):.4f}")
```

**Output:**
```
Solve time: 51.2ms
Iterations: 60
Nonzeros found: 10
Recovery error: 0.0234
```

---

## Performance

POGS is **4-8x faster** than alternatives on Lasso:

| Size | POGS | OSQP | SCS | Clarabel |
|------|------|------|-----|----------|
| 200x100 | **3.6ms** | 32ms | 23ms | 21ms |
| 500x300 | **51ms** | 399ms | 206ms | 186ms |
| 1000x500 | **340ms** | 2.1s | 1.3s | 1.1s |

*Benchmarks on Apple M1, Python 3.12*

---

## Choosing Lambda

Lambda controls sparsity:

- **Large lambda** (e.g., 1.0): Very sparse, many zeros
- **Small lambda** (e.g., 0.01): Less sparse, closer to least squares
- **lambda = 0**: Ordinary least squares (no regularization)

```python
import matplotlib.pyplot as plt

lambdas = [0.001, 0.01, 0.1, 0.5, 1.0]
nnz = []

for lam in lambdas:
    result = solve_lasso(A, b, lambd=lam)
    nnz.append(np.sum(np.abs(result['x']) > 1e-4))

plt.semilogx(lambdas, nnz, 'o-')
plt.xlabel('lambda')
plt.ylabel('Number of nonzeros')
plt.title('Lasso regularization path')
plt.show()
```

### Cross-Validation

Find optimal lambda with cross-validation:

```python
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

# Use sklearn for CV, then solve with POGS
lasso_cv = LassoCV(cv=5, random_state=0)
lasso_cv.fit(A, b)
best_lambda = lasso_cv.alpha_

# Solve with POGS using best lambda
result = solve_lasso(A, b, lambd=best_lambda)
print(f"Best lambda: {best_lambda:.4f}")
print(f"Nonzeros: {np.sum(np.abs(result['x']) > 1e-4)}")
```

---

## Tuning Solver Parameters

### Tolerance

```python
# High accuracy
result = solve_lasso(A, b, lambd=0.1, rel_tol=1e-6, abs_tol=1e-6)

# Fast (for warm-starting or prototyping)
result = solve_lasso(A, b, lambd=0.1, rel_tol=1e-3, abs_tol=1e-3)
```

### Initialization

Warm-start with a previous solution:

```python
# Solve first problem
result1 = solve_lasso(A, b, lambd=0.1)

# Warm-start next problem (faster)
result2 = solve_lasso(A, b, lambd=0.05, x_init=result1['x'])
```

---

## CVXPY Alternative

For more flexibility, use CVXPY:

```python
import cvxpy as cp

x = cp.Variable(n)
objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1))
prob = cp.Problem(objective)
prob.solve(solver='POGS')

print(f"Optimal value: {prob.value:.4f}")
print(f"Nonzeros: {np.sum(np.abs(x.value) > 1e-4)}")
```

---

## Variations

### Elastic Net

Combine L1 and L2 penalties for grouped sparsity:

```python
from pogs import solve_elastic_net

# min ||Ax - b||² + λ₁||x||₁ + λ₂||x||²
result = solve_elastic_net(A, b, l1_ratio=0.5, lambd=0.1)
```

### Non-Negative Lasso

Require positive coefficients:

```python
from pogs import solve_nonneg_ls

# min ||Ax - b||² s.t. x >= 0
result = solve_nonneg_ls(A, b)
```

### Weighted Lasso

Different penalties per coefficient (via CVXPY):

```python
import cvxpy as cp

weights = np.ones(n)
weights[:10] = 0.01  # Less penalty on first 10 features

x = cp.Variable(n)
objective = cp.Minimize(
    0.5 * cp.sum_squares(A @ x - b) + cp.norm(cp.multiply(weights, x), 1)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')
```

---

## Troubleshooting

### "Max iterations reached"

```python
# Increase iterations
result = solve_lasso(A, b, lambd=0.1, max_iter=5000)

# Or check if lambda is too small (ill-conditioned)
print(f"Condition number: {np.linalg.cond(A):.1f}")
```

### Slow convergence

Normalize your data:

```python
# Standardize columns
A_scaled = (A - A.mean(axis=0)) / A.std(axis=0)
b_scaled = (b - b.mean()) / b.std()

result = solve_lasso(A_scaled, b_scaled, lambd=0.1)
```

---

## See Also

- [Logistic Regression](logistic.md) - Classification with L1 penalty
- [Ridge Regression](../api/solver.md#solve_ridge) - L2 regularization
- [API Reference](../api/solver.md) - Full function documentation
