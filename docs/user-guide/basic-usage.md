# Basic Usage

This guide covers the fundamental usage patterns for POGS.

---

## Problem Formulation

POGS solves problems in **graph form**:

$$
\begin{align}
\text{minimize} \quad & f(y) + g(x) \\
\text{subject to} \quad & y = Ax
\end{align}
$$

where:
- $f:\mathbb{R}^m \to \mathbb{R}$ and $g:\mathbb{R}^n \to \mathbb{R}$ are convex, separable functions
- $A \in \mathbb{R}^{m \times n}$ is the data matrix
- $x \in \mathbb{R}^n$ are the optimization variables
- $y \in \mathbb{R}^m$ are auxiliary variables

**Separability** means:
$$
f(y) = \sum_{i=1}^m f_i(y_i), \quad g(x) = \sum_{j=1}^n g_j(x_j)
$$

---

## C++ Interface

### 1. Define the Problem Data

```cpp
#include <pogs/pogs.hpp>
#include <vector>

const size_t m = 100;  // Number of samples
const size_t n = 50;   // Number of features

// Create matrix A (dense example)
std::vector<double> A_data(m * n);
// ... fill A_data ...

auto A = std::make_unique<pogs::MatrixDense<double>>(m, n);
// Copy data to matrix
```

### 2. Define Objective Functions

```cpp
#include <pogs/types.hpp>

std::vector<pogs::FunctionObj<double>> f(m);
std::vector<pogs::FunctionObj<double>> g(n);

// Example: Least squares + L1 regularization
// f_i(y_i) = (1/2) * y_i^2 - b_i * y_i  (for ||Ax - b||^2)
for (size_t i = 0; i < m; ++i) {
    f[i].type = pogs::FunctionType::Square;
    f[i].c = 0.5;
    f[i].d = -b[i];
}

// g_j(x_j) = lambda * |x_j|  (L1 regularization)
double lambda = 0.1;
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::Abs;
    g[j].c = lambda;
}
```

### 3. Configure the Solver

```cpp
#include <pogs/config.hpp>

// Using C++20 designated initializers
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4,
    .rel_tol = 1e-3,
    .max_iter = 1000,
    .verbose = true,
    .adaptive_rho = true,
    .gap_stop = true
};
```

### 4. Solve the Problem

```cpp
auto solver = pogs::make_solver<double>(std::move(A));
solver.configure(config);

auto result = solver.solve(f, g);

if (result.status == pogs::Status::Success) {
    std::cout << "Converged in " << result.iterations << " iterations\n";
    std::cout << "Optimal value: " << result.primal_obj.value() << "\n";

    // Access solution
    for (size_t j = 0; j < n; ++j) {
        std::cout << "x[" << j << "] = " << result.x[j] << "\n";
    }
}
```

---

## Function Types

POGS supports many common functions:

### Identity and Zero

| Function | Mathematical Form | Use Case |
|----------|------------------|----------|
| `Zero` | $f(x) = 0$ | Unconstrained variables |
| `Identity` | $f(x) = x$ | Linear objectives |

### Norms and Regularization

| Function | Mathematical Form | Use Case |
|----------|------------------|----------|
| `Abs` | $f(x) = c\|x\|$ | L1 regularization (Lasso) |
| `Square` | $f(x) = (c/2)x^2$ | Least squares, L2 penalty |

### Indicators (Constraints)

| Function | Mathematical Form | Use Case |
|----------|------------------|----------|
| `IndBox01` | $I_{[0,1]}(x)$ | Box constraints |
| `IndEq0` | $I_{\{0\}}(x)$ | Equality to zero |
| `IndGe0` | $I_{[0,\infty)}(x)$ | Non-negativity |
| `IndLe0` | $I_{(-\infty,0]}(x)$ | Non-positivity |

### Nonlinear Functions

| Function | Mathematical Form | Use Case |
|----------|------------------|----------|
| `Logistic` | $f(x) = \log(1 + e^x)$ | Logistic regression |
| `Huber` | Huber loss | Robust regression |
| `NegLog` | $f(x) = -\log(x)$ | Barrier functions |
| `Exp` | $f(x) = e^x$ | Exponential objectives |

---

## Common Problem Types

### Lasso Regression

$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1
$$

```cpp
// f_i(y_i) = (1/2) * y_i^2 - b_i * y_i
for (size_t i = 0; i < m; ++i) {
    f[i].type = pogs::FunctionType::Square;
    f[i].c = 0.5;
    f[i].d = -b[i];
}

// g_j(x_j) = lambda * |x_j|
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::Abs;
    g[j].c = lambda;
}
```

### Ridge Regression

$$
\text{minimize} \quad \|Ax - b\|_2^2 + \lambda\|x\|_2^2
$$

```cpp
// f_i(y_i) = y_i^2 - 2*b_i*y_i
for (size_t i = 0; i < m; ++i) {
    f[i].type = pogs::FunctionType::Square;
    f[i].c = 1.0;
    f[i].d = -2.0 * b[i];
}

// g_j(x_j) = lambda * x_j^2
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::Square;
    g[j].c = lambda;
}
```

### Non-Negative Least Squares

$$
\begin{align}
\text{minimize} \quad & \|Ax - b\|_2^2 \\
\text{subject to} \quad & x \geq 0
\end{align}
$$

```cpp
// f_i(y_i) = y_i^2 - 2*b_i*y_i
for (size_t i = 0; i < m; ++i) {
    f[i].type = pogs::FunctionType::Square;
    f[i].c = 1.0;
    f[i].d = -2.0 * b[i];
}

// g_j(x_j) = I_{x >= 0}(x_j)
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::IndGe0;
}
```

---

## Solver Parameters

### Penalty Parameter (ρ)

- Controls the weight of the augmented Lagrangian term
- Default: `1.0`
- Try range: `0.1` to `10.0`
- Larger ρ: Faster convergence but potentially less accurate
- Smaller ρ: More accurate but slower convergence

### Tolerances

**Absolute tolerance** (`abs_tol`):
- Default: `1e-4`
- For high accuracy: `1e-6` or smaller

**Relative tolerance** (`rel_tol`):
- Default: `1e-3`
- For high accuracy: `1e-5` or smaller

### Convergence Criteria

The solver stops when:

$$
\begin{align}
\text{Primal residual:} \quad & \|r\|_2 \leq \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \max(\|Ax\|_2, \|y\|_2) \\
\text{Dual residual:} \quad & \|s\|_2 \leq \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \|A^T\lambda\|_2
\end{align}
$$

### Adaptive ρ

When `adaptive_rho = true`, ρ is automatically adjusted based on residuals:

- If primal residual >> dual residual: increase ρ
- If dual residual >> primal residual: decrease ρ

This can significantly improve convergence for difficult problems.

---

## Matrix Formats

### Dense Matrices

```cpp
auto A = std::make_unique<pogs::MatrixDense<double>>(m, n);
// Column-major or row-major storage
```

Best for:
- Small to medium problems (< 10,000 variables)
- Most matrix elements are non-zero

### Sparse Matrices

```cpp
auto A = std::make_unique<pogs::MatrixSparse<double>>(m, n, nnz);
// CSR or CSC format
```

Best for:
- Large problems (> 10,000 variables)
- Most matrix elements are zero
- Significant memory savings

---

## Checking Convergence

```cpp
auto result = solver.solve(f, g);

switch (result.status) {
    case pogs::Status::Success:
        std::cout << "Converged successfully\n";
        break;
    case pogs::Status::MaxIterations:
        std::cout << "Maximum iterations reached\n";
        break;
    case pogs::Status::NumericalError:
        std::cout << "Numerical error encountered\n";
        break;
    case pogs::Status::InfeasibleOrUnbounded:
        std::cout << "Problem is infeasible or unbounded\n";
        break;
}
```

---

## Next Steps

- [Advanced Features](advanced-features.md) - Warm starting, custom functions
- [Cone Problems](cone-problems.md) - LP, QP, SOCP, SDP formulations
- [Examples](../examples/lasso.md) - Complete working examples
