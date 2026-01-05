# CVXPY Integration

POGS can solve CVXPY problems directly using `pogs_solve()`. It auto-detects supported problem patterns and uses the fast graph-form solver.

---

## Basic Usage

```python
import cvxpy as cp
import numpy as np
from pogs import pogs_solve

# Problem data
A = np.random.randn(100, 50)
b = np.random.randn(100)

# Define CVXPY problem
x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

# Solve with POGS
pogs_solve(prob)

print(f"Status: {prob.status}")
print(f"Optimal value: {prob.value}")
print(f"Solution: {x.value}")
```

---

## How It Works

`pogs_solve()` inspects the CVXPY problem structure and detects if it matches a supported pattern:

| Pattern | CVXPY Expression | POGS Solver |
|:--------|:-----------------|:------------|
| **Lasso** | `sum_squares(A @ x - b) + λ * norm(x, 1)` | `solve_lasso` |
| **Ridge** | `sum_squares(A @ x - b) + λ * sum_squares(x)` | `solve_ridge` |
| **NNLS** | `sum_squares(A @ x - b)` with `x >= 0` | `solve_nonneg_ls` |

If a pattern is detected, POGS uses its fast graph-form solver. Otherwise, it falls back to CVXPY's default solver.

---

## Registering as a Solve Method

You can register `pogs_solve` as a named method:

```python
import cvxpy as cp
from pogs import pogs_solve

# Register once
cp.Problem.register_solve("POGS", pogs_solve)

# Now use like any other solver
prob.solve(method="POGS")
```

This lets you use the familiar `solve(method=...)` syntax.

---

## Examples

### Lasso

```python
import cvxpy as cp
import numpy as np
from pogs import pogs_solve

m, n = 500, 300
A = np.random.randn(m, n)
b = np.random.randn(m)

x = cp.Variable(n)
lambd = 0.1
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1)))

pogs_solve(prob, verbose=True)
# Output: POGS: Detected lasso pattern, using fast graph-form solver
```

### Ridge

```python
x = cp.Variable(n)
lambd = 0.1
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.sum_squares(x)))

pogs_solve(prob, verbose=True)
# Output: POGS: Detected ridge pattern, using fast graph-form solver
```

### Non-negative Least Squares

```python
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

pogs_solve(prob, verbose=True)
# Output: POGS: Detected nonneg_ls pattern, using fast graph-form solver
```

### Unsupported Problems

For problems that don't match a supported pattern:

```python
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(A @ x - b, 1)))  # L1 loss, not L2

pogs_solve(prob, verbose=True)
# Output: POGS: No graph-form pattern detected, using default solver
```

---

## Solver Options

Pass solver options to `pogs_solve()`:

```python
pogs_solve(
    prob,
    verbose=True,      # Print solver output
    abs_tol=1e-6,      # Absolute tolerance
    rel_tol=1e-6,      # Relative tolerance
    max_iter=5000,     # Maximum iterations
    rho=1.0,           # ADMM penalty parameter
)
```

---

## When to Use POGS vs Direct Solvers

### Use `pogs_solve()` when:

- You have existing CVXPY code
- You want automatic pattern detection
- You're not sure which solver function to use

### Use direct solvers (`solve_lasso`, etc.) when:

- You know the exact problem type
- You want maximum performance
- You're writing new code

Direct solvers are slightly faster because they skip pattern detection:

```python
# Direct (faster)
from pogs import solve_lasso
result = solve_lasso(A, b, lambd=0.1)
x = result['x']

# Via CVXPY (convenient)
from pogs import pogs_solve
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
pogs_solve(prob)
x_val = x.value
```

---

## Supported Patterns

POGS detects these specific CVXPY expression patterns:

### Lasso

```python
# minimize ½||Ax - b||² + λ||x||₁
cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1))
cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm1(x))
```

### Ridge

```python
# minimize ½||Ax - b||² + λ||x||²
cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.sum_squares(x))
```

### Non-negative Least Squares

```python
# minimize ½||Ax - b||² s.t. x >= 0
cp.Minimize(cp.sum_squares(A @ x - b)), constraints=[x >= 0]
```

---

## Limitations

- **Single variable**: Only problems with one optimization variable are supported
- **Minimization**: Must be a minimization problem
- **Specific patterns**: Only Lasso, Ridge, and NNLS patterns are detected
- **Dense matrices**: Works best with dense matrices A

For problems outside these patterns, POGS falls back to CVXPY's default solver, which may be slower but handles general convex problems.
