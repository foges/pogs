# POGS

**Fast convex optimization for machine learning**

[![PyPI](https://img.shields.io/pypi/v/pogs)](https://pypi.org/project/pogs/)
[![CI](https://github.com/foges/pogs/actions/workflows/ci.yml/badge.svg)](https://github.com/foges/pogs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

POGS is **4-14x faster** than general-purpose solvers on common ML optimization problems.

| Problem | POGS | OSQP | SCS | Clarabel |
|---------|------|------|-----|----------|
| Lasso (500x300) | **51ms** | 399ms | 206ms | 186ms |
| Ridge (500x300) | **8ms** | 89ms | 64ms | 51ms |
| Logistic (500x300) | **34ms** | 312ms | 198ms | 167ms |

*Benchmarks on Apple M1, Python 3.12*

## Installation

```bash
pip install pogs
```

Works on **macOS**, **Linux**, and **Windows**.

## Quick Start

```python
from pogs import solve_lasso
import numpy as np

A = np.random.randn(500, 300)
b = np.random.randn(500)

result = solve_lasso(A, b, lambd=0.1)
print(f"Solved in {result['iterations']} iterations")
```

## Supported Problems

```python
from pogs import (
    solve_lasso,       # L1-regularized least squares
    solve_ridge,       # L2-regularized least squares
    solve_elastic_net, # L1 + L2 regularization
    solve_logistic,    # Logistic regression
    solve_svm,         # Support vector machine
    solve_huber,       # Robust regression
    solve_nonneg_ls,   # Non-negative least squares
)
```

### Lasso Regression

```python
# minimize ||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)
x = result['x']
```

### Logistic Regression

```python
# minimize Σ log(1 + exp(-yᵢaᵢ'x)) + λ||x||₁
y = np.sign(A @ np.random.randn(n))  # Labels in {-1, +1}
result = solve_logistic(A, y, lambd=0.01)
```

### Ridge Regression

```python
# minimize ||Ax - b||² + λ||x||²
result = solve_ridge(A, b, lambd=0.1)
```

## Why POGS is Fast

POGS uses [ADMM](https://stanford.edu/~boyd/papers/admm_distr_stats.html) with problem-specific proximal operators. For ML problems like Lasso, Ridge, and Logistic Regression, these operators have closed-form solutions—no inner iterations needed.

General-purpose solvers (OSQP, SCS, Clarabel) reformulate everything as cone programs, adding overhead. POGS solves the original problem directly.

## Parameters

All solvers accept:

```python
result = solve_lasso(A, b, lambd=0.1,
    abs_tol=1e-4,   # Absolute tolerance
    rel_tol=1e-4,   # Relative tolerance
    max_iter=2500,  # Maximum iterations
    verbose=0,      # 0=quiet, 1=summary, 2=progress
)
```

## Result Dictionary

```python
result = solve_lasso(A, b, lambd=0.1)

result['x']           # Solution vector
result['status']      # 0 = success
result['iterations']  # Number of iterations
result['optval']      # Optimal objective value
```

## Using with CVXPY

Register POGS with CVXPY, then use `method='POGS'`:

```python
import cvxpy as cp
import numpy as np
from pogs import register

register()  # One-time registration

A = np.random.randn(100, 50)
b = np.random.randn(100)

x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
prob.solve(method='POGS')  # Auto-detects Lasso, uses fast solver

print(x.value)
```

Or call `pogs_solve()` directly without registration:

```python
from pogs import pogs_solve

pogs_solve(prob)  # Same result, no registration needed
```

## C++ Library

POGS is written in C++ with Python bindings. If you need the C++ library directly:

```bash
git clone https://github.com/foges/pogs.git
cd pogs
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Documentation

Full documentation: **[foges.github.io/pogs](https://foges.github.io/pogs/)**

## Citation

```bibtex
@article{fougner2018pogs,
  title={Parameter selection and preconditioning for a graph form solver},
  author={Fougner, Christopher and Boyd, Stephen},
  journal={Emerging Applications of Control and Systems Theory},
  year={2018},
  publisher={Springer}
}
```

## License

Apache 2.0
