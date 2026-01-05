# Python API Reference

Complete reference for POGS Python functions.

---

## Quick Reference

| Function | Problem |
|:---------|:--------|
| [`solve_lasso`](#solve_lasso) | Sparse regression (L1) |
| [`solve_ridge`](#solve_ridge) | Ridge regression (L2) |
| [`solve_elastic_net`](#solve_elastic_net) | L1 + L2 regularization |
| [`solve_logistic`](#solve_logistic) | Logistic regression |
| [`solve_svm`](#solve_svm) | Support vector machine |
| [`solve_huber`](#solve_huber) | Robust regression |
| [`solve_nonneg_ls`](#solve_nonneg_ls) | Non-negative least squares |
| [`pogs_solve`](#pogs_solve) | CVXPY integration |

---

## solve_lasso

Solve L1-regularized least squares (Lasso):

$$\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1$$

```python
from pogs import solve_lasso

result = solve_lasso(A, b, lambd,
                     abs_tol=1e-4,
                     rel_tol=1e-4,
                     max_iter=2500,
                     verbose=0,
                     rho=1.0)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array (m, n) | Data matrix |
| `b` | array (m,) | Target vector |
| `lambd` | float | L1 regularization strength |
| `abs_tol` | float | Absolute tolerance (default: 1e-4) |
| `rel_tol` | float | Relative tolerance (default: 1e-4) |
| `max_iter` | int | Maximum iterations (default: 2500) |
| `verbose` | int | 0=quiet, 1=summary, 2=progress |
| `rho` | float | ADMM penalty parameter (default: 1.0) |

**Returns:** [Result dictionary](#result-dictionary)

**Example:**
```python
import numpy as np
from pogs import solve_lasso

A = np.random.randn(500, 300)
b = np.random.randn(500)

result = solve_lasso(A, b, lambd=0.1)
print(f"Nonzeros: {np.sum(np.abs(result['x']) > 1e-4)}")
```

---

## solve_ridge

Solve L2-regularized least squares (Ridge):

$$\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|x\|_2^2$$

```python
from pogs import solve_ridge

result = solve_ridge(A, b, lambd,
                     abs_tol=1e-4,
                     rel_tol=1e-4,
                     max_iter=2500,
                     verbose=0,
                     rho=1.0)
```

**Parameters:** Same as `solve_lasso`

**Example:**
```python
result = solve_ridge(A, b, lambd=0.1)
print(f"Solution norm: {np.linalg.norm(result['x']):.4f}")
```

---

## solve_elastic_net

Solve Elastic Net (L1 + L2 regularization):

$$\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda_1\|x\|_1 + \frac{\lambda_2}{2}\|x\|_2^2$$

```python
from pogs import solve_elastic_net

result = solve_elastic_net(A, b, lambda1, lambda2,
                           abs_tol=1e-4,
                           rel_tol=1e-4,
                           max_iter=2500,
                           verbose=0,
                           rho=1.0)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array (m, n) | Data matrix |
| `b` | array (m,) | Target vector |
| `lambda1` | float | L1 regularization strength |
| `lambda2` | float | L2 regularization strength |

**Example:**
```python
result = solve_elastic_net(A, b, lambda1=0.1, lambda2=0.05)
```

---

## solve_logistic

Solve L1-regularized logistic regression:

$$\text{minimize} \quad \sum_i \log(1 + e^{-y_i (a_i^T x)}) + \lambda\|x\|_1$$

```python
from pogs import solve_logistic

result = solve_logistic(A, b, lambd=0.0,
                        abs_tol=1e-4,
                        rel_tol=1e-4,
                        max_iter=2500,
                        verbose=0,
                        rho=1.0)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array (m, n) | Feature matrix |
| `b` | array (m,) | Labels in **{-1, +1}** |
| `lambd` | float | L1 regularization (default: 0.0) |

**Example:**
```python
y = np.sign(A @ np.random.randn(n))  # Labels in {-1, +1}
result = solve_logistic(A, y, lambd=0.01)

# Predict
pred = np.sign(A @ result['x'])
accuracy = np.mean(pred == y)
```

---

## solve_svm

Solve L2-regularized SVM (hinge loss):

$$\text{minimize} \quad \sum_i \max(0, 1 - y_i (a_i^T x)) + \frac{\lambda}{2}\|x\|_2^2$$

```python
from pogs import solve_svm

result = solve_svm(A, b, lambd=1.0,
                   abs_tol=1e-4,
                   rel_tol=1e-4,
                   max_iter=2500,
                   verbose=0,
                   rho=1.0)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array (m, n) | Feature matrix |
| `b` | array (m,) | Labels in **{-1, +1}** |
| `lambd` | float | L2 regularization (default: 1.0) |

**Example:**
```python
y = np.sign(A @ np.random.randn(n))
result = solve_svm(A, y, lambd=0.1)
```

---

## solve_huber

Solve robust regression with Huber loss:

$$\text{minimize} \quad \sum_i \text{huber}_\delta(a_i^T x - b_i) + \lambda\|x\|_1$$

where $\text{huber}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \le \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$

```python
from pogs import solve_huber

result = solve_huber(A, b, delta=1.0, lambd=0.0,
                     abs_tol=1e-4,
                     rel_tol=1e-4,
                     max_iter=2500,
                     verbose=0,
                     rho=1.0)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array (m, n) | Data matrix |
| `b` | array (m,) | Target vector |
| `delta` | float | Huber threshold (default: 1.0) |
| `lambd` | float | L1 regularization (default: 0.0) |

**Example:**
```python
# Add outliers to data
b_noisy = b.copy()
b_noisy[:10] += 100  # Outliers

result = solve_huber(A, b_noisy, delta=1.0)
```

---

## solve_nonneg_ls

Solve non-negative least squares:

$$\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 \quad \text{subject to} \quad x \ge 0$$

```python
from pogs import solve_nonneg_ls

result = solve_nonneg_ls(A, b,
                         abs_tol=1e-4,
                         rel_tol=1e-4,
                         max_iter=2500,
                         verbose=0,
                         rho=1.0)
```

**Example:**
```python
result = solve_nonneg_ls(A, b)
print(f"Min value: {result['x'].min():.6f}")  # Should be >= 0
```

---

## Result Dictionary

All solvers return a dictionary with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `x` | array (n,) | Solution vector |
| `y` | array (m,) | y = Ax |
| `l` | array (m,) | Dual variable |
| `optval` | float | Optimal objective value |
| `iterations` | int | Number of iterations |
| `status` | int | 0=success, other=failure |

**Example:**
```python
result = solve_lasso(A, b, lambd=0.1)

x = result['x']           # Solution
obj = result['optval']    # Objective value
iters = result['iterations']

if result['status'] == 0:
    print(f"Solved in {iters} iterations")
else:
    print("Solver failed")
```

---

## Common Parameters

### Tolerance

Controls solution accuracy:

```python
# High accuracy (slower)
result = solve_lasso(A, b, lambd=0.1, abs_tol=1e-6, rel_tol=1e-6)

# Lower accuracy (faster)
result = solve_lasso(A, b, lambd=0.1, abs_tol=1e-3, rel_tol=1e-3)
```

### Maximum Iterations

```python
# For difficult problems
result = solve_lasso(A, b, lambd=0.1, max_iter=5000)
```

### Verbosity

```python
result = solve_lasso(A, b, lambd=0.1, verbose=2)
# 0 = quiet
# 1 = summary only
# 2 = iteration progress
```

---

## Sparse Matrices

All solvers support scipy sparse matrices:

```python
import scipy.sparse as sp

A_sparse = sp.random(1000, 500, density=0.1, format='csr')
b = np.random.randn(1000)

result = solve_lasso(A_sparse, b, lambd=0.1)
```

---

## pogs_solve

Solve CVXPY problems with automatic pattern detection:

```python
from pogs import pogs_solve

optval = pogs_solve(problem, verbose=False, **solver_opts)
```

**Parameters:**

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `problem` | cvxpy.Problem | The CVXPY problem to solve |
| `verbose` | bool | Print solver output (default: False) |
| `abs_tol` | float | Absolute tolerance (default: 1e-4) |
| `rel_tol` | float | Relative tolerance (default: 1e-4) |
| `max_iter` | int | Maximum iterations (default: 2500) |
| `rho` | float | ADMM penalty parameter (default: 1.0) |

**Returns:** float - optimal objective value

**Supported Patterns:**

| Pattern | CVXPY Expression |
|:--------|:-----------------|
| Lasso | `sum_squares(A @ x - b) + λ * norm(x, 1)` |
| Ridge | `sum_squares(A @ x - b) + λ * sum_squares(x)` |
| NNLS | `sum_squares(A @ x - b)` with `x >= 0` |

For unsupported patterns, falls back to CVXPY's default solver.

**Example:**

```python
import cvxpy as cp
import numpy as np
from pogs import pogs_solve

A = np.random.randn(100, 50)
b = np.random.randn(100)

x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

pogs_solve(prob, verbose=True)
# Output: POGS: Detected lasso pattern, using fast graph-form solver

print(x.value)  # Solution is stored in the variable
```

**Registering as a Method:**

```python
cp.Problem.register_solve("POGS", pogs_solve)
prob.solve(method="POGS")
```

---

## See Also

- [Quick Start](../getting-started/quick-start.md) - Getting started guide
- [CVXPY Integration](../user-guide/cvxpy-integration.md) - Detailed CVXPY usage
- [Lasso Example](../examples/lasso.md) - Detailed Lasso example
- [Logistic Example](../examples/logistic.md) - Classification example
