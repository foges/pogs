# CVXPY Integration

POGS can be used as a solver backend for CVXPY, allowing you to express optimization problems in a high-level modeling language.

---

## Overview

The CVXPY interface allows you to solve convex optimization problems using POGS as the solver backend. POGS supports the following cone types:

- **Zero cone** (`{x : x = 0}`): Equality constraints
- **Non-negative cone** (`{x : x ≥ 0}`): Inequality constraints
- **Second-order cone** (`{(p,x) : ||x||₂ ≤ p}`): Quadratic constraints
- **Semidefinite cone** (`{X : X ⪰ 0}`): PSD matrix constraints
- **Exponential cone**: Exponential constraints

---

## Installation

### Prerequisites

1. **Build POGS library:**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
   cmake --build build
   sudo cmake --install build
   ```

2. **Install Python dependencies:**
   ```bash
   pip install numpy cvxpy
   ```

### Verify Installation

```bash
cd python
python3 verify_cvxpy_interface.py
```

Expected output: `✓ All components verified!`

---

## Usage Examples

### Basic Linear Program

```python
import cvxpy as cp
import sys
sys.path.insert(0, '/path/to/pogs/python')

# Problem: minimize x[0] subject to x[0] + x[1] = 2, x >= 0
x = cp.Variable(2)
objective = cp.Minimize(x[0])
constraints = [
    x[0] + x[1] == 2,
    x >= 0
]
prob = cp.Problem(objective, constraints)

# Solve with POGS
result = prob.solve(solver='POGS', verbose=True)

print(f"Optimal value: {prob.value}")
print(f"Solution: x = {x.value}")
# Output: x = [0, 2], optimal value = 0
```

### Quadratic Program

```python
# Problem: minimize 0.5*||x||^2 + c^T*x subject to Ax <= b
x = cp.Variable(n)
objective = cp.Minimize(0.5 * cp.sum_squares(x) + c @ x)
constraints = [A @ x <= b]
prob = cp.Problem(objective, constraints)

result = prob.solve(solver='POGS')
```

### Second-Order Cone Program

```python
# Problem: minimize ||Ax - b||_2 + lambda * ||x||_2
x = cp.Variable(n)
objective = cp.Minimize(cp.norm(A @ x - b) + lam * cp.norm(x))
prob = cp.Problem(objective)

result = prob.solve(solver='POGS')
```

### Semidefinite Program

```python
# Problem: minimize trace(C @ X) subject to trace(A_i @ X) = b_i, X PSD
X = cp.Variable((n, n), PSD=True)
objective = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A[i] @ X) == b[i] for i in range(m)]
prob = cp.Problem(objective, constraints)

result = prob.solve(solver='POGS')
```

---

## Solver Options

You can customize solver behavior with additional options:

```python
prob.solve(
    solver='POGS',
    verbose=True,       # Print solver output
    abs_tol=1e-4,      # Absolute tolerance
    rel_tol=1e-3,      # Relative tolerance
    max_iter=10000,    # Maximum iterations
    rho=1.0            # Initial penalty parameter
)
```

---

## Performance Considerations

### When to Use POGS

POGS is well-suited for:

- **Medium-scale problems** (thousands of variables)
- **Dense or moderately sparse** problems
- **Problems with SDP constraints** (POGS supports SDP natively)
- **Custom cone constraints** (easy to extend)

### When to Use Other Solvers

Consider alternatives for:

- **Very large sparse problems**: Use SCS or OSQP
- **Mixed-integer programs**: Use ECOS_BB, GLPK_MI, or Gurobi
- **High-precision requirements**: Use CVXOPT or MOSEK

### Tuning

For better performance:

- **Adjust `rho`**: Start with 1.0, try 0.1-10.0 range
- **Tighten tolerances**: Use `abs_tol=1e-6, rel_tol=1e-6` for higher accuracy
- **Increase iterations**: Set `max_iter=20000` for difficult problems

---

## Troubleshooting

### "POGS library not found"

**Solution**: Build the POGS library first:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### "Solver did not converge"

**Solutions**:
1. Increase `max_iter`
2. Adjust `rho` parameter
3. Check problem formulation (is it convex?)
4. Scale the problem (normalize data)

### Problem is infeasible/unbounded

POGS returns status code != 0. Check:
- Problem constraints are feasible
- Problem is bounded
- Data is correctly formatted

---

## Examples

See the Python examples directory for more:

- `python/test_cone_simple.py` - Python example without CVXPY
- `python/test_cvxpy_interface.py` - Full CVXPY integration tests

---

## References

- CVXPY documentation: [https://www.cvxpy.org](https://www.cvxpy.org)
- Cone programming: [Convex Optimization by Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/)
