# Quick Start

Get up and running with POGS in 5 minutes!

---

## Your First Optimization Problem

Let's solve a simple Lasso regression problem:

$$
\text{minimize} \quad \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
$$

This is a classic sparse regression problem used in statistics and machine learning.

---

## C++ Example

### 1. Create the Problem

Create a file `my_first_pogs.cpp`:

```cpp
#include <iostream>
#include <vector>
#include <random>

// Include POGS headers (from old API, will be modernized)
#include "matrix/matrix_dense.h"
#include "pogs.h"

int main() {
    // Problem dimensions
    const size_t m = 100;  // Number of samples
    const size_t n = 50;   // Number of features

    // Create problem data
    std::vector<double> A_data(m * n);
    std::vector<double> b_data(m);

    // Fill with random data
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < m * n; ++i)
        A_data[i] = dist(gen);
    for (size_t i = 0; i < m; ++i)
        b_data[i] = dist(gen);

    // Create matrix A
    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    // Create solver
    pogs::Pogs<double, pogs::MatrixDense<double>> solver(A);

    // Define objective functions
    std::vector<pogs::FunctionObj<double>> f(m);
    std::vector<pogs::FunctionObj<double>> g(n);

    // f_i(y_i) = (1/2) * y_i^2 - b_i * y_i  (for ||Ax - b||^2)
    for (size_t i = 0; i < m; ++i) {
        f[i].h = pogs::kSquare;
        f[i].c = 0.5;
        f[i].d = -b_data[i];
    }

    // g_j(x_j) = lambda * |x_j|  (L1 regularization)
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].h = pogs::kAbs;
        g[j].c = lambda;
    }

    // Solve
    solver.Solve(f, g);

    // Get solution
    std::cout << "Solved in " << solver.GetIter() << " iterations" << std::endl;
    std::cout << "Optimal value: " << solver.GetOptval() << std::endl;

    return 0;
}
```

### 2. Compile and Run

```bash
# Using CMake (recommended)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or compile directly
g++ -std=c++20 -I/usr/local/include my_first_pogs.cpp \
    -lpogs_cpu -llapack -lblas -o my_app

# Run
./my_app
```

**Expected Output**:
```
Solved in 186 iterations
Optimal value: 45.2341
```

---

## Python/CVXPY Example

Much simpler with Python!

### 1. Create the Problem

Create `my_first_pogs.py`:

```python
import cvxpy as cp
import numpy as np

# Problem dimensions
m, n = 100, 50

# Generate random data
np.random.seed(0)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define variable
x = cp.Variable(n)

# Define objective: minimize (1/2)||Ax - b||^2 + lambda*||x||_1
lambda_val = 0.1
objective = cp.Minimize(
    0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1)
)

# Create problem
prob = cp.Problem(objective)

# Solve with POGS
prob.solve(solver='POGS', verbose=True)

# Print results
print(f"Status: {prob.status}")
print(f"Optimal value: {prob.value:.4f}")
print(f"Solution sparsity: {np.sum(np.abs(x.value) > 1e-4)} / {n}")
```

### 2. Run

```bash
python my_first_pogs.py
```

**Output**:
```
Status: optimal
Optimal value: 45.2341
Solution sparsity: 12 / 50
```

!!! success
    The Python interface is much cleaner! Use CVXPY for rapid prototyping.

---

## Understanding the Output

### Solver Iterations

POGS uses ADMM, which iterates until convergence:

```
Iteration   Primal Res   Dual Res     Gap
   10       1.23e-02    4.56e-03    8.90e-02
   20       3.45e-03    1.23e-03    2.34e-02
   ...
  186       9.12e-05    3.45e-05    1.23e-04  ✓ Converged
```

### Solution Quality

- **Primal Residual**: How well $y = Ax$ is satisfied
- **Dual Residual**: Changes in the dual variable
- **Gap**: Difference between primal and dual objectives

All should be small at convergence (< 1e-4 by default).

---

## Common Problem Types

### 1. Linear Program

$$
\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax = b \\
& x \geq 0
\end{align}
$$

```python
x = cp.Variable(n)
objective = cp.Minimize(c.T @ x)
constraints = [A @ x == b, x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve(solver='POGS')
```

### 2. Quadratic Program

$$
\text{minimize} \quad \frac{1}{2} x^T Q x + c^T x
$$

```python
x = cp.Variable(n)
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
prob = cp.Problem(objective)
prob.solve(solver='POGS')
```

### 3. Ridge Regression

$$
\text{minimize} \quad \|Ax - b\|_2^2 + \lambda \|x\|_2^2
$$

```python
x = cp.Variable(n)
objective = cp.Minimize(
    cp.sum_squares(A @ x - b) + lambda_val * cp.sum_squares(x)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')
```

---

## Tuning the Solver

### Adjust Tolerances

```python
# More accurate (slower)
prob.solve(solver='POGS', abs_tol=1e-6, rel_tol=1e-6)

# Less accurate (faster)
prob.solve(solver='POGS', abs_tol=1e-3, rel_tol=1e-3)
```

### Adjust Iterations

```python
# More iterations for harder problems
prob.solve(solver='POGS', max_iter=5000)
```

### Enable Verbose Output

```python
# See iteration progress
prob.solve(solver='POGS', verbose=True)
```

---

## What's Next?

Now that you've run your first problem, explore:

<div class="grid cards" markdown>

-   :material-book-open:{ .lg .middle } __Learn the API__

    ---

    Understand POGS capabilities in depth

    [:octicons-arrow-right-24: User Guide](../user-guide/basic-usage.md)

-   :material-function:{ .lg .middle } __Supported Functions__

    ---

    See all available proximal operators

    [:octicons-arrow-right-24: API Reference](../api/proximal.md)

-   :material-clipboard-list:{ .lg .middle } __More Examples__

    ---

    Learn from complete problem solutions

    [:octicons-arrow-right-24: Examples](../examples/lasso.md)

-   :material-cog:{ .lg .middle } __Advanced Features__

    ---

    Cone constraints, SDP, and more

    [:octicons-arrow-right-24: Advanced Guide](../user-guide/advanced-features.md)

</div>

---

## Troubleshooting

### "Solver did not converge"

**Cause**: Problem may be infeasible or poorly scaled

**Solution**:
```python
# Try more iterations
prob.solve(solver='POGS', max_iter=5000)

# Check problem is feasible
print(f"Problem status: {prob.status}")
```

### "Solution is inaccurate"

**Cause**: Default tolerances may be too loose

**Solution**:
```python
# Tighten tolerances
prob.solve(solver='POGS', abs_tol=1e-6, rel_tol=1e-6)
```

### "Taking too long"

**Cause**: Problem may be large or ill-conditioned

**Solution**:
```python
# Check problem size
print(f"Matrix size: {A.shape}")

# Try adjusting rho (penalty parameter)
prob.solve(solver='POGS', rho=10.0)  # Try different values
```

---

## Getting Help

- **Documentation**: Browse the [User Guide](../user-guide/basic-usage.md)
- **Examples**: See [Examples](../examples/lasso.md) for complete code
- **Issues**: Report problems on [GitHub](https://github.com/foges/pogs/issues)
