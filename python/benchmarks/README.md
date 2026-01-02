# POGS Benchmark Suite

Comprehensive benchmarks comparing POGS against other CVXPY solvers (ECOS, SCS, OSQP, CVXOPT) on standard optimization problems.

## Quick Start

```bash
# Install dependencies
pip install cvxpy numpy scipy pandas matplotlib

# Run all benchmarks
cd python/benchmarks
python run_benchmarks.py

# Run specific problem class
python run_benchmarks.py --problem lasso
python run_benchmarks.py --problem portfolio

# Quick test (fewer trials, smaller problems)
python run_benchmarks.py --quick
```

## Problem Classes

| Problem | Solvers Tested | Sizes |
|---------|----------------|-------|
| **Lasso Regression** | POGS, ECOS, SCS, OSQP | Small, Medium, Large, Sparse |
| **Logistic Regression** | POGS, ECOS, SCS | Small, Medium, Large |
| **Portfolio Optimization** | POGS, ECOS, SCS, OSQP | Small, Medium, Large |
| **Linear Programs (LP)** | POGS, ECOS, SCS, OSQP | Small, Medium, Large |
| **Quadratic Programs (QP)** | POGS, ECOS, SCS, OSQP | Small, Medium, Large |
| **SOCP** | POGS, ECOS, SCS | Small, Medium |
| **SDP** | POGS, SCS, CVXOPT | Small, Medium |

## Metrics Tracked

For each problem/solver combination:

- **Solve time** (seconds)
- **Setup time** (problem transformation)
- **Total time** (setup + solve)
- **Iterations** (solver iterations)
- **Optimal value** (objective)
- **Status** (optimal, infeasible, etc.)
- **Success rate** (over multiple trials)

## Output

Results are saved to:
- `results/latest.json` - Detailed JSON results
- `results/summary.txt` - Human-readable summary
- `results/latest.html` - HTML report with charts (if matplotlib available)

## Expected Performance

Based on the [POGS paper](http://stanford.edu/~boyd/papers/pogs.html):

**POGS Strengths**:
- Large-scale problems (billions of coefficients)
- Dense problems (efficient BLAS usage)
- GPU acceleration (for very large problems)
- Modest accuracy requirements (3-4 digits)

**POGS Limitations**:
- High accuracy (8+ digits) - Interior-point methods may be faster
- Very sparse problems - SCS may be more efficient
- Warm starts - Not currently supported (OSQP supports this)

## Directory Structure

```
python/benchmarks/
├── README.md              # This file
├── run_benchmarks.py      # Main benchmark runner
├── benchmark_utils.py     # Utilities (timing, reporting)
├── problems/              # Problem generators
│   ├── __init__.py
│   ├── lasso.py          # Lasso regression
│   ├── logistic.py       # Logistic regression
│   ├── portfolio.py      # Portfolio optimization
│   ├── lp.py             # Linear programs
│   ├── qp.py             # Quadratic programs
│   ├── socp.py           # Second-order cone programs
│   └── sdp.py            # Semidefinite programs
└── results/              # Benchmark results (gitignored)
    ├── latest.json
    ├── summary.txt
    └── archive/
```

## Adding New Problems

1. Create problem generator in `problems/<name>.py`
2. Add to `run_benchmarks.py` problem list
3. Run benchmarks

Example problem generator:
```python
import cvxpy as cp
import numpy as np

def generate(m=100, n=50, seed=None):
    """Generate a Lasso regression problem."""
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[np.random.rand(n) < 0.9] = 0  # Sparse
    b = A @ x_true + 0.1 * np.random.randn(m)

    x = cp.Variable(n)
    lambda_val = 0.1 * np.linalg.norm(A.T @ b, np.inf)

    objective = cp.Minimize(
        0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1)
    )
    problem = cp.Problem(objective)
    problem.name = f"Lasso (m={m}, n={n})"
    problem.size_metrics = {"m": m, "n": n}

    return problem
```

## Interpreting Results

### Solve Time
- **Lower is better**
- POGS typically excels on large dense problems
- Interior-point methods (ECOS) may be faster for small problems

### Iterations
- First-order methods (POGS, SCS) typically need more iterations
- Interior-point methods (ECOS) converge in fewer iterations
- Iterations × time per iteration = solve time

### Success Rate
- Should be 100% for well-formed problems
- Lower rates indicate numerical issues or solver limitations

## Citation

If you use these benchmarks in research, please cite:

```bibtex
@article{fougner2015pogs,
  title={Parameter selection and preconditioning for a graph form solver},
  author={Fougner, Chris and Boyd, Stephen},
  journal={Optimization and Engineering},
  year={2015}
}
```

## Contributing

Contributions welcome! Please:
1. Add new problem classes with realistic problem sizes
2. Ensure problems are well-formed (bounded, feasible)
3. Document expected solver performance
4. Update this README

---

**Last Updated**: v0.4.0 (January 2, 2026)
