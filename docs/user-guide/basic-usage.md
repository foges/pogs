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

### 1. Include Headers and Create Matrix

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

const size_t m = 100;  // Number of samples
const size_t n = 50;   // Number of features

// Create matrix A (row-major, m x n)
std::vector<double> A_data(m * n);
// ... fill A_data ...

pogs::MatrixDense<double> A('r', m, n, A_data.data());
```

### 2. Create the Solver

```cpp
// PogsDirect - uses direct factorization (dense problems)
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

// Alternative: PogsCgls - uses iterative method (large/sparse problems)
// pogs::PogsCgls<double, pogs::MatrixDense<double>> solver(A);
```

### 3. Define Objective Functions

```cpp
// f_i(y_i) and g_j(x_j) are defined using FunctionObj
std::vector<FunctionObj<double>> f(m);
std::vector<FunctionObj<double>> g(n);

// Example: Lasso (least squares + L1 regularization)
// f_i(y_i) = (1/2) * y_i^2 - b_i * y_i  (for ||Ax - b||^2)
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;   // Base function: (1/2) x^2
    f[i].c = 1.0;       // Scaling
    f[i].d = -b[i];     // Linear term: creates ||y - b||^2
}

// g_j(x_j) = lambda * |x_j|  (L1 regularization)
double lambda = 0.1;
for (size_t j = 0; j < n; ++j) {
    g[j].h = kAbs;      // Base function: |x|
    g[j].c = lambda;    // Regularization weight
}
```

### 4. Configure the Solver

```cpp
solver.SetRho(1.0);           // ADMM penalty parameter
solver.SetAbsTol(1e-4);       // Absolute tolerance
solver.SetRelTol(1e-3);       // Relative tolerance
solver.SetMaxIter(1000);      // Maximum iterations
solver.SetVerbose(2);         // Show progress
solver.SetAdaptiveRho(true);  // Enable adaptive rho
```

### 5. Solve and Get Results

```cpp
PogsStatus status = solver.Solve(f, g);

if (status == POGS_SUCCESS) {
    printf("Converged in %u iterations\n", solver.GetFinalIter());
    printf("Optimal value: %f\n", solver.GetOptval());

    // Access solution
    const double* x = solver.GetX();
    for (size_t j = 0; j < n; ++j) {
        printf("x[%zu] = %f\n", j, x[j]);
    }
}
```

---

## Complete Example

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

int main() {
    const size_t m = 100;  // samples
    const size_t n = 50;   // features
    const double lambda = 0.1;

    // Generate random data
    std::vector<double> A_data(m * n);
    std::vector<double> b(m);
    for (size_t i = 0; i < m * n; ++i)
        A_data[i] = (double)rand() / RAND_MAX - 0.5;
    for (size_t i = 0; i < m; ++i)
        b[i] = (double)rand() / RAND_MAX - 0.5;

    // Create matrix and solver
    pogs::MatrixDense<double> A('r', m, n, A_data.data());
    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    // Configure
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetMaxIter(1000);
    solver.SetVerbose(2);

    // Define objective functions
    std::vector<FunctionObj<double>> f(m), g(n);
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].d = -b[i];
    }
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kAbs;
        g[j].c = lambda;
    }

    // Solve
    PogsStatus status = solver.Solve(f, g);

    if (status == POGS_SUCCESS) {
        printf("Converged! Optimal value: %f\n", solver.GetOptval());
    } else {
        printf("Failed with status: %d\n", status);
    }

    return 0;
}
```

---

## Function Types

POGS supports many common functions via the `Function` enum:

### Norms and Regularization

| Enum | Mathematical Form | Use Case |
|------|------------------|----------|
| `kAbs` | $\|x\|$ | L1 regularization (Lasso) |
| `kSquare` | $(1/2)x^2$ | Least squares, L2 penalty |
| `kHuber` | Huber loss | Robust regression |

### Indicator Functions (Constraints)

| Enum | Mathematical Form | Use Case |
|------|------------------|----------|
| `kIndBox01` | $I_{[0,1]}(x)$ | Box constraints |
| `kIndEq0` | $I_{\{0\}}(x)$ | Equality to zero |
| `kIndGe0` | $I_{[0,\infty)}(x)$ | Non-negativity |
| `kIndLe0` | $I_{(-\infty,0]}(x)$ | Non-positivity |

### Nonlinear Functions

| Enum | Mathematical Form | Use Case |
|------|------------------|----------|
| `kLogistic` | $\log(1 + e^x)$ | Logistic regression |
| `kNegLog` | $-\log(x)$ | Barrier functions |
| `kExp` | $e^x$ | Exponential objectives |
| `kIdentity` | $x$ | Linear terms |
| `kZero` | $0$ | Unconstrained |

---

## FunctionObj Parameters

Each `FunctionObj<T>` represents:

$$
c \cdot h(ax - b) + d \cdot x + e \cdot x^2
$$

| Field | Default | Description |
|-------|---------|-------------|
| `h` | `kZero` | Base function type |
| `a` | `1.0` | Input scaling |
| `b` | `0.0` | Input shift |
| `c` | `1.0` | Output scaling |
| `d` | `0.0` | Linear term coefficient |
| `e` | `0.0` | Quadratic term coefficient |

**Example:**
```cpp
FunctionObj<double> obj;
obj.h = kSquare;  // h(x) = (1/2)x^2
obj.c = 0.5;      // Scale by 0.5
obj.d = -b[i];    // Add linear term -b[i]*x
// Result: 0.5 * (1/2) * x^2 - b[i] * x = (1/4)x^2 - b[i]*x
```

---

## Common Problem Types

### Lasso Regression

$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1
$$

```cpp
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].d = -b[i];
}
for (size_t j = 0; j < n; ++j) {
    g[j].h = kAbs;
    g[j].c = lambda;
}
```

### Ridge Regression

$$
\text{minimize} \quad \|Ax - b\|_2^2 + \lambda\|x\|_2^2
$$

```cpp
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].d = -2.0 * b[i];
}
for (size_t j = 0; j < n; ++j) {
    g[j].h = kSquare;
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
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].d = -2.0 * b[i];
}
for (size_t j = 0; j < n; ++j) {
    g[j].h = kIndGe0;  // Non-negativity constraint
}
```

---

## Status Codes

| Status | Meaning |
|--------|---------|
| `POGS_SUCCESS` | Converged successfully |
| `POGS_MAX_ITER` | Maximum iterations reached |
| `POGS_NAN_FOUND` | Numerical error (NaN) |
| `POGS_INFEASIBLE` | Problem likely infeasible |
| `POGS_UNBOUNDED` | Problem likely unbounded |
| `POGS_ERROR` | Generic error |

---

## Solver Parameters

### Penalty Parameter (rho)

- Controls the weight of the augmented Lagrangian term
- Default: `1.0`
- Larger rho: Faster convergence but potentially less accurate
- Smaller rho: More accurate but slower convergence

### Tolerances

**Absolute tolerance** (`abs_tol`): Default `1e-4`

**Relative tolerance** (`rel_tol`): Default `1e-3`

The solver stops when primal and dual residuals satisfy:
$$
\|r\|_2 \leq \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \max(\|Ax\|_2, \|y\|_2)
$$

### Adaptive Rho

When `SetAdaptiveRho(true)`, rho is automatically adjusted:
- If primal residual >> dual residual: increase rho
- If dual residual >> primal residual: decrease rho

---

## Next Steps

- [Advanced Features](advanced-features.md) - Warm starting, Anderson acceleration
- [Cone Problems](cone-problems.md) - LP, QP, SOCP, SDP formulations
- [Examples](../examples/lasso.md) - Complete working examples
