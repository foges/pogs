<p align="center">
  <h1 align="center">POGS</h1>
  <p align="center"><strong>Blazing fast convex optimization for machine learning</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/pogs/"><img src="https://img.shields.io/pypi/v/pogs?color=blue" alt="PyPI"></a>
  <a href="https://github.com/foges/pogs/actions/workflows/ci.yml"><img src="https://github.com/foges/pogs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://foges.github.io/pogs/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#why-pogs-is-fast">Why It's Fast</a> •
  <a href="#documentation">Docs</a>
</p>

---

## Performance

POGS uses the **graph-form ADMM** algorithm with closed-form proximal operators, making it efficient for problems that fit this structure (Lasso, Ridge, Elastic Net, Logistic, SVM, Huber).

**Best for:**
- Dense, well-conditioned design matrices
- Signal processing, image reconstruction, compressed sensing
- Problems where graph-form structure can be exploited

**Consider alternatives (SCS, OSQP) for:**
- Sparse or ill-conditioned matrices
- Rank-deficient problems
- General QPs without special structure

Run `python benchmarks/libsvm_benchmark.py` to compare on real datasets.

---

## Installation

```bash
pip install pogs
```

**Requirements:** Python 3.9+ and NumPy. Works on **macOS**, **Linux**, and **Windows**.

---

## Quick Start

```python
from pogs import solve_lasso
import numpy as np

# Your data
A = np.random.randn(500, 300)
b = np.random.randn(500)

# Solve Lasso: minimize ½||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)

print(result['x'])          # Solution
print(result['iterations']) # Typically 50-100
```

---

## Solvers

POGS provides specialized solvers for common ML problems:

```python
from pogs import (
    solve_lasso,        # L1-regularized regression
    solve_ridge,        # L2-regularized regression
    solve_elastic_net,  # L1 + L2 regularization
    solve_logistic,     # Logistic regression
    solve_svm,          # Support vector machine
    solve_huber,        # Robust regression
    solve_nonneg_ls,    # Non-negative least squares
)
```

### Regression

```python
# Lasso: minimize ½||Ax - b||² + λ||x||₁
result = solve_lasso(A, b, lambd=0.1)

# Ridge: minimize ½||Ax - b||² + λ||x||²
result = solve_ridge(A, b, lambd=0.1)

# Elastic Net: minimize ½||Ax - b||² + λ₁||x||₁ + λ₂||x||²
result = solve_elastic_net(A, b, lambda1=0.1, lambda2=0.05)

# Huber: minimize Σ huber(Aᵢx - bᵢ) + λ||x||₁
result = solve_huber(A, b, lambd=0.1, delta=1.0)
```

### Classification

```python
# Labels must be in {-1, +1}
y = np.sign(A @ np.random.randn(A.shape[1]))

# Logistic: minimize Σ log(1 + exp(-yᵢ·aᵢᵀx)) + λ||x||₁
result = solve_logistic(A, y, lambd=0.01)

# SVM: minimize Σ max(0, 1 - yᵢ·aᵢᵀx) + λ||x||²
result = solve_svm(A, y, lambd=0.01)
```

### Constrained

```python
# Non-negative least squares: minimize ½||Ax - b||² s.t. x ≥ 0
result = solve_nonneg_ls(A, b)
```

---

## Parameters

All solvers accept the same tuning parameters:

```python
result = solve_lasso(
    A, b,
    lambd=0.1,       # Regularization strength
    abs_tol=1e-4,    # Absolute tolerance (default: 1e-4)
    rel_tol=1e-4,    # Relative tolerance (default: 1e-4)
    max_iter=2500,   # Maximum iterations (default: 2500)
    verbose=0,       # 0=quiet, 1=summary, 2=per-iteration
)
```

## Return Value

All solvers return a dictionary:

```python
result = solve_lasso(A, b, lambd=0.1)

result['x']           # numpy array - solution vector
result['status']      # int - 0=converged, 1=max_iter, 2=error
result['iterations']  # int - number of ADMM iterations
result['optval']      # float - optimal objective value
```

---

## CVXPY Integration

Use `pogs_solve()` to solve CVXPY problems. It auto-detects supported patterns and uses the fast solver:

```python
import cvxpy as cp
import numpy as np
from pogs import pogs_solve

A = np.random.randn(100, 50)
b = np.random.randn(100)

x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

pogs_solve(prob)  # Detects Lasso, uses solve_lasso internally
print(x.value)
```

Or register as a named method:

```python
cp.Problem.register_solve("POGS", pogs_solve)
prob.solve(method="POGS")
```

**Supported patterns:** Lasso, Ridge, Non-negative least squares. Other problems fall back to CVXPY's default solver.

---

## Why POGS is Fast

POGS uses [ADMM](https://stanford.edu/~boyd/papers/admm_distr_stats.html) (Alternating Direction Method of Multipliers) with **closed-form proximal operators**.

For Lasso, Ridge, and Logistic regression, the proximal operators have analytical solutions:

| Function | Proximal Operator |
|:---------|:------------------|
| ½‖·‖² | x/(1+ρ) |
| λ‖·‖₁ | soft_threshold(x, λ/ρ) |
| λ‖·‖² | x/(1+2λ/ρ) |

General-purpose solvers (OSQP, SCS, Clarabel) reformulate these as cone programs, requiring more iterations and overhead. POGS solves the original problem directly.

---

## C++ Library

POGS is implemented in C++20 with Python bindings via pybind11. To build from source:

```bash
git clone https://github.com/foges/pogs.git
cd pogs
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

---

## Documentation

Full documentation: **[foges.github.io/pogs](https://foges.github.io/pogs/)**

- [Installation Guide](https://foges.github.io/pogs/getting-started/installation/)
- [Quick Start](https://foges.github.io/pogs/getting-started/quick-start/)
- [API Reference](https://foges.github.io/pogs/api/solver/)
- [Examples](https://foges.github.io/pogs/examples/lasso/)

---

## Citation

```bibtex
@article{fougner2018pogs,
  title={Parameter selection and preconditioning for a graph form solver},
  author={Fougner, Christopher and Boyd, Stephen},
  journal={Emerging Applications of Control and Systems Theory},
  pages={41--61},
  year={2018},
  publisher={Springer}
}
```

---

## License

Apache 2.0
