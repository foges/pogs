# POGS Benchmark Suite

Benchmarks comparing POGS against OSQP, SCS, and Clarabel on **real industry data**.

## Results Summary

| Benchmark | POGS Win Rate | Speedup Range | Data Source |
|-----------|---------------|---------------|-------------|
| Portfolio Optimization | 83% | 4-10x | Yahoo Finance (S&P 500) |
| UCI Regression | 80% | 1.1-2.7x | UCI ML Repository |
| **Overall** | **83%** | **up to 8.7x** | Real data only |

## When to Use POGS

**POGS excels at:**
- Dense matrices (covariance, factor models)
- Graph-form problems: `min f(Ax) + g(x)`
- Lasso, Ridge, Elastic Net, Huber, Logistic
- Portfolio optimization
- Signal/image denoising

**Use other solvers for:**
- Sparse matrices (OSQP, Clarabel exploit sparsity)
- General QPs with arbitrary constraints
- Very tall matrices (m >> n)
- Problems outside graph form

## Sparse Data Limitation

POGS C++ supports sparse matrices via `MatrixSparse`, but the C/Python
interface currently only exposes dense matrices. For sparse problems,
use OSQP or Clarabel which are optimized for sparsity.

## Benchmark Files

| File | Description |
|------|-------------|
| `comprehensive_benchmark.py` | UCI + S&P 500 datasets (Lasso, Ridge) |
| `portfolio_benchmark.py` | S&P 500 sparse portfolio optimization |
| `libsvm_benchmark.py` | LIBSVM classification (sparse - shows limitation) |
| `maros_benchmark.py` | Maros-Mészáros QP suite (general QPs) |
| `graph_form_benchmark.py` | Synthetic graph-form problems |

## Running Benchmarks

```bash
cd python/benchmarks
uv venv
uv pip install numpy cvxpy osqp scs clarabel yfinance pandas scikit-learn
uv run python comprehensive_benchmark.py
```

## Data Sources

All benchmarks use **real data**:
- [UCI ML Repository](https://archive.ics.uci.edu/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
- [Maros-Mészáros](https://github.com/qpsolvers/maros_meszaros_qpbenchmark)
