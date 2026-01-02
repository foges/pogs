# POGS CVXPY Interface

Complete CVXPY solver interface for POGS, allowing CVXPY to use POGS as a backend for conic optimization problems.

## Overview

The CVXPY interface allows you to solve convex optimization problems using POGS as the solver backend. POGS supports the following cone types:

- **Zero cone** (`{x : x = 0}`): Equality constraints
- **Non-negative cone** (`{x : x ≥ 0}`): Inequality constraints
- **Second-order cone** (`{(p,x) : ||x||₂ ≤ p}`): Quadratic constraints
- **Semidefinite cone** (`{X : X ⪰ 0}`): PSD matrix constraints
- **Exponential cone**: Exponential constraints

## Installation

### Prerequisites

1. **Build POGS library:**
   ```bash
   cd src
   make cpu
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

## Usage

### Basic Example: Linear Program

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

## API Reference

### `solve_cone_problem(c, A, b, dims, ...)`

Low-level interface for solving cone problems directly.

**Parameters:**
- `c` (array): Objective vector (n,)
- `A` (array): Constraint matrix (m, n)
- `b` (array): Constraint vector (m,)
- `dims` (dict): Cone dimensions
  - `'f'`: Number of free variables (zero cone)
  - `'l'`: Number of non-negative variables
  - `'q'`: List of SOC dimensions
  - `'s'`: List of SDP dimensions (matrix sizes)
  - `'ep'`: Number of primal exponential cones
  - `'ed'`: Number of dual exponential cones
- `rho` (float): Penalty parameter (default: 1.0)
- `abs_tol` (float): Absolute tolerance (default: 1e-4)
- `rel_tol` (float): Relative tolerance (default: 1e-3)
- `max_iter` (int): Maximum iterations (default: 10000)
- `verbose` (int): Verbosity level (default: 0)

**Returns:**
- `dict` with keys:
  - `'x'`: Primal solution
  - `'y'`: Slack variables
  - `'z'`: Dual variables
  - `'s'`: Slack (same as y)
  - `'status'`: 0 for success
  - `'num_iters'`: Iterations taken
  - `'optval'`: Optimal value

**Example:**
```python
from pogs_cvxpy import solve_cone_problem
import numpy as np

# Solve: min x[0] s.t. x[0] + x[1] = 2, x >= 0
c = np.array([1.0, 0.0])
A = np.array([[1.0, 1.0]])
b = np.array([2.0])
dims = {'f': 1, 'l': 0}  # One equality constraint

result = solve_cone_problem(c, A, b, dims, verbose=5)
print(f"Solution: {result['x']}")
```

### `POGS` Class

CVXPY solver class implementing the ConicSolver interface.

**Attributes:**
- `MIP_CAPABLE = False`: Does not support mixed-integer programs
- `SUPPORTED_CONSTRAINTS`: List of supported constraint types

**Methods:**
- `name()`: Returns `"POGS"`
- `import_solver()`: Checks POGS availability
- `solve_via_data(data, ...)`: Solves problem from CVXPY data
- `invert(solution, inverse_data)`: Converts solution to CVXPY format

## Implementation Details

### How It Works

1. **CVXPY → Cone Form**: CVXPY converts the problem to standard conic form:
   ```
   minimize    c^T * x
   subject to  b - A*x ∈ K
   ```

2. **C Code Generation**: The Python interface generates C code that:
   - Encodes the problem data (A, b, c, cones)
   - Calls `PogsConeD()` from the C interface
   - Outputs the solution in machine-readable format

3. **Compilation & Execution**: The generated C code is compiled and executed:
   ```bash
   gcc -o solver solver.c pogs.a -framework Accelerate
   ./solver
   ```

4. **Solution Parsing**: The output is parsed and returned to CVXPY

### Architecture

```
CVXPY Problem
     ↓
POGS(ConicSolver)
     ↓
solve_cone_problem()
     ↓
_generate_c_code()     ← Generates C code
     ↓
_compile_and_run()     ← Compiles & executes
     ↓
PogsConeD() [C]        ← C interface
     ↓
PogsIndirectCone [C++] ← POGS solver
     ↓
ProxConeSdpCpu() etc.  ← Cone projections
     ↓
Solution
```

## Testing

### Without Dependencies

Test the cone solver without CVXPY:
```bash
cd python
python3 test_cone_simple.py
```

### With CVXPY

Run full integration tests:
```bash
cd python
python3 test_cvxpy_interface.py
```

Tests include:
- Simple LP
- LP with inequalities
- Quadratic program
- Second-order cone problem
- Feasibility problem

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

## Troubleshooting

### "POGS library not found"

**Solution**: Build the POGS library first:
```bash
cd src
make clean && make cpu
```

### "Compilation failed"

**Causes**:
- Accelerate framework not found (macOS only)
- Missing C compiler

**Solution** (Linux):
Update `pogs_cvxpy.py` to use `-lopenblas` instead of `-framework Accelerate`

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

## Examples

See the `examples/` directory for complete working examples:
- `examples/cpp_cone/test_c_interface.c` - C interface example
- `python/test_cone_simple.py` - Python example without CVXPY
- `python/test_cvxpy_interface.py` - Full CVXPY integration tests

## Limitations

Current limitations:
1. **CPU only**: GPU version not yet implemented for SDP cones
2. **Dense matrices**: Sparse matrix support exists but not exposed in Python interface
3. **No warm starting**: Currently doesn't support warm start
4. **Compilation overhead**: Each solve requires C compilation (can be optimized with caching)

## Future Improvements

Potential enhancements:
1. **Direct ctypes binding**: Eliminate compilation overhead
2. **Sparse matrix support**: Expose sparse matrix interface
3. **Warm starting**: Support warm start from previous solution
4. **GPU support**: Implement SDP cone for GPU
5. **Solver caching**: Cache compiled solver for repeated solves

## Contributing

To extend the solver:

1. **Add new cone type**:
   - Implement projection in `src/include/prox_lib_cone.h`
   - Add to `Cone` enum in `src/interface_c/pogs_c.h`
   - Update Python mapping in `pogs_cvxpy.py`

2. **Optimize performance**:
   - Profile with `verbose=5`
   - Tune parameters (rho, tolerances)
   - Consider sparse matrix support

3. **Add tests**:
   - Add test case to `test_cvxpy_interface.py`
   - Verify solution accuracy
   - Check edge cases

## References

- POGS paper: [Block Splitting for Distributed Optimization](https://stanford.edu/~boyd/papers/block_splitting.html)
- CVXPY documentation: https://www.cvxpy.org
- Cone programming: https://web.stanford.edu/~boyd/cvxbook/

## License

POGS is distributed under the Apache 2.0 license. See LICENSE file for details.
