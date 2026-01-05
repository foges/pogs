# POGS - Proximal Operator Graph Solver

**Fast convex optimization for machine learning**

POGS is a high-performance solver for convex optimization problems. It's particularly fast for ML problems like Lasso, Ridge, Logistic Regression, and SVM.

---

## Why POGS?

### Blazing Fast

POGS is **4-14x faster** than general-purpose solvers on ML problems:

| Problem | POGS | OSQP | SCS | Clarabel |
|---------|------|------|-----|----------|
| Lasso (500x300) | **51ms** | 399ms | 206ms | 186ms |
| Ridge (500x300) | **8ms** | 89ms | 64ms | 51ms |
| Logistic (500x300) | **34ms** | 312ms | 198ms | 167ms |

*Benchmarks on Apple M1, Python 3.12*

### Easy to Use

```python
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

### Works with CVXPY

Use POGS as a backend for CVXPY problems:

```python
import cvxpy as cp

x = cp.Variable(300)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
prob.solve(solver='POGS')
```

---

## Supported Problems

POGS excels at these ML problems:

| Problem | Function | Speed vs Alternatives |
|---------|----------|----------------------|
| **Lasso** | `solve_lasso(A, b, lambd)` | 4-8x faster |
| **Ridge** | `solve_ridge(A, b, lambd)` | 6-11x faster |
| **Elastic Net** | `solve_elastic_net(A, b, l1, l2)` | 5-10x faster |
| **Logistic Regression** | `solve_logistic(A, y, lambd)` | 5-9x faster |
| **SVM** | `solve_svm(A, y, lambd)` | 4-8x faster |
| **Huber Regression** | `solve_huber(A, b, lambd)` | 6-10x faster |
| **Non-negative LS** | `solve_nonneg_ls(A, b)` | 5-9x faster |

---

## Installation

=== "pip (Recommended)"

    ```bash
    pip install pogs
    ```

    Works on **macOS** (Intel & Apple Silicon) and **Linux** (x86_64 & ARM64).

=== "From Source"

    ```bash
    pip install git+https://github.com/foges/pogs.git
    ```

---

## Quick Example

### Lasso Regression

```python
from pogs import solve_lasso
import numpy as np

# Generate problem
np.random.seed(0)
m, n = 500, 300
A = np.random.randn(m, n)
x_true = np.zeros(n)
x_true[:10] = np.random.randn(10)  # Sparse solution
b = A @ x_true + 0.1 * np.random.randn(m)

# Solve
result = solve_lasso(A, b, lambd=0.1)

print(f"Status: {'Solved' if result['status'] == 0 else 'Failed'}")
print(f"Iterations: {result['iterations']}")
print(f"Nonzeros found: {np.sum(np.abs(result['x']) > 1e-4)}")
```

Output:
```
Status: Solved
Iterations: 60
Nonzeros found: 10
```

### Logistic Regression

```python
from pogs import solve_logistic
import numpy as np

# Binary classification
m, n = 1000, 100
A = np.random.randn(m, n)
y = np.sign(A @ np.random.randn(n))  # Labels in {-1, +1}

result = solve_logistic(A, y, lambd=0.01)

print(f"Iterations: {result['iterations']}")
```

---

## How It Works

POGS solves problems of the form:

$$
\text{minimize} \quad f(y) + g(x) \quad \text{subject to} \quad y = Ax
$$

Using [ADMM](https://stanford.edu/~boyd/papers/admm_distr_stats.html), it efficiently handles:

- **Separable objectives**: Each $f_i$ and $g_j$ can be different
- **Large dense matrices**: Optimized BLAS operations
- **Sparse solutions**: L1 regularization converges quickly

The key insight is that ADMM only requires computing proximal operators, which have closed-form solutions for most ML loss functions.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Get Started__

    ---

    Install and run your first optimization

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-chart-line:{ .lg .middle } __Benchmarks__

    ---

    See detailed performance comparisons

    [:octicons-arrow-right-24: Performance](examples/lasso.md)

-   :material-language-python:{ .lg .middle } __API Reference__

    ---

    Full documentation of all functions

    [:octicons-arrow-right-24: Python API](api/solver.md)

-   :material-github:{ .lg .middle } __Source Code__

    ---

    Contribute or report issues

    [:octicons-arrow-right-24: GitHub](https://github.com/foges/pogs)

</div>

---

## Citation

If you use POGS in research, please cite:

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

POGS is open source under the [Apache 2.0 License](about/license.md).
