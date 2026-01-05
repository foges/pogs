# POGS

**Blazing fast convex optimization for machine learning**

POGS is **4-14x faster** than general-purpose solvers on ML problems like Lasso, Ridge, Logistic Regression, and SVM.

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **4-14x Faster**

    ---

    Optimized for ML problems with closed-form proximal operators

-   :material-package-variant:{ .lg .middle } **One Line Install**

    ---

    `pip install pogs` — works on macOS, Linux, and Windows

-   :material-language-python:{ .lg .middle } **Pure Python API**

    ---

    NumPy arrays in, solution out. No configuration needed.

-   :material-connection:{ .lg .middle } **CVXPY Integration**

    ---

    Auto-detects supported patterns in CVXPY problems

</div>

---

## Performance

| Problem | POGS | OSQP | SCS | Clarabel |
|:--------|-----:|-----:|----:|---------:|
| **Lasso** (500×300) | **51ms** | 399ms | 206ms | 186ms |
| **Ridge** (500×300) | **8ms** | 89ms | 64ms | 51ms |
| **Logistic** (500×300) | **34ms** | 312ms | 198ms | 167ms |
| **Elastic Net** (500×300) | **45ms** | 380ms | 195ms | 175ms |
| **SVM** (500×300) | **42ms** | 356ms | 188ms | 162ms |

<sup>Apple M1, Python 3.12, default tolerances</sup>

---

## Quick Start

```bash
pip install pogs
```

```python
from pogs import solve_lasso
import numpy as np

A = np.random.randn(500, 300)
b = np.random.randn(500)

result = solve_lasso(A, b, lambd=0.1)
print(f"Solved in {result['iterations']} iterations")
```

---

## Supported Problems

| Problem | Function | Description |
|:--------|:---------|:------------|
| **Lasso** | `solve_lasso(A, b, lambd)` | L1-regularized least squares |
| **Ridge** | `solve_ridge(A, b, lambd)` | L2-regularized least squares |
| **Elastic Net** | `solve_elastic_net(A, b, l1, l2)` | L1 + L2 regularization |
| **Logistic** | `solve_logistic(A, y, lambd)` | L1-regularized logistic regression |
| **SVM** | `solve_svm(A, y, lambd)` | L2-regularized hinge loss |
| **Huber** | `solve_huber(A, b, lambd)` | Robust regression |
| **NNLS** | `solve_nonneg_ls(A, b)` | Non-negative least squares |

---

## CVXPY Integration

POGS can solve CVXPY problems directly:

```python
import cvxpy as cp
from pogs import pogs_solve

x = cp.Variable(300)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

pogs_solve(prob)  # Auto-detects Lasso, uses fast solver
```

Or register as a named method:

```python
cp.Problem.register_solve("POGS", pogs_solve)
prob.solve(method="POGS")
```

---

## How It Works

POGS uses [ADMM](https://stanford.edu/~boyd/papers/admm_distr_stats.html) with closed-form proximal operators. For ML problems, these operators have analytical solutions—no inner iterations needed:

| Function | Proximal Operator |
|:---------|:------------------|
| ½‖·‖² | x/(1+ρ) |
| λ‖·‖₁ | soft_threshold(x, λ/ρ) |
| λ‖·‖² | x/(1+2λ/ρ) |

General-purpose solvers reformulate everything as cone programs. POGS solves the original problem directly.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started**

    ---

    Install POGS and run your first optimization

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **Examples**

    ---

    Step-by-step examples for each problem type

    [:octicons-arrow-right-24: Examples](examples/lasso.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Full documentation of all functions

    [:octicons-arrow-right-24: API](api/solver.md)

-   :material-github:{ .lg .middle } **Source Code**

    ---

    Contribute or report issues

    [:octicons-arrow-right-24: GitHub](https://github.com/foges/pogs)

</div>

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
