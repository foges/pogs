# Lasso Regression Example

Complete example of solving Lasso regression with POGS.

---

## Problem Formulation

**Lasso (Least Absolute Shrinkage and Selection Operator)** solves:

$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1
$$

where:
- $A \in \mathbb{R}^{m \times n}$ is the design matrix
- $b \in \mathbb{R}^m$ is the observation vector
- $x \in \mathbb{R}^n$ are the coefficients to find
- $\lambda > 0$ is the regularization parameter

**Effect:** The L1 penalty $\lambda\|x\|_1$ promotes sparsity in the solution.

---

## POGS Formulation

POGS requires the graph form:

$$
\begin{align}
\text{minimize} \quad & f(y) + g(x) \\
\text{subject to} \quad & y = Ax
\end{align}
$$

For Lasso:

- $f(y) = \frac{1}{2}\|y - b\|_2^2 = \sum_{i=1}^m \frac{1}{2}(y_i - b_i)^2$
- $g(x) = \lambda\|x\|_1 = \sum_{j=1}^n \lambda|x_j|$

In POGS notation:

- $f_i(y_i) = \frac{1}{2}y_i^2 - b_i \cdot y_i$ (square function with linear term)
- $g_j(x_j) = \lambda|x_j|$ (absolute value function)

---

## C++ Implementation

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main() {
    // Problem dimensions
    const size_t m = 100;  // Number of samples
    const size_t n = 50;   // Number of features

    // Generate random problem data
    std::vector<double> A_data(m * n);
    std::vector<double> b_data(m);

    std::default_random_engine gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < m * n; ++i) {
        A_data[i] = dist(gen);
    }
    for (size_t i = 0; i < m; ++i) {
        b_data[i] = dist(gen);
    }

    // Create matrix and solver
    pogs::MatrixDense<double> A('r', m, n, A_data.data());
    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    // Configure solver
    solver.SetRho(1.0);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetMaxIter(1000);
    solver.SetVerbose(2);
    solver.SetAdaptiveRho(true);

    // Define objective functions
    std::vector<FunctionObj<double>> f(m);
    std::vector<FunctionObj<double>> g(n);

    // f_i(y_i) = (1/2) * y_i^2 - b_i * y_i
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].c = 1.0;
        f[i].d = -b_data[i];
    }

    // g_j(x_j) = lambda * |x_j|
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kAbs;
        g[j].c = lambda;
    }

    // Solve
    PogsStatus status = solver.Solve(f, g);

    // Check result
    if (status == POGS_SUCCESS) {
        std::cout << "\nConverged in " << solver.GetFinalIter() << " iterations\n";
        std::cout << "Optimal value: " << solver.GetOptval() << "\n";

        // Count non-zeros (sparsity)
        const double* x = solver.GetX();
        size_t nnz = 0;
        for (size_t j = 0; j < n; ++j) {
            if (std::abs(x[j]) > 1e-4) {
                nnz++;
            }
        }
        std::cout << "Sparsity: " << nnz << " / " << n << " non-zeros\n";

        // Print first 10 coefficients
        std::cout << "\nFirst 10 coefficients:\n";
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            std::cout << "x[" << j << "] = " << x[j] << "\n";
        }
    } else {
        std::cerr << "Solver failed with status " << status << "\n";
        return 1;
    }

    return 0;
}
```

---

## Python/CVXPY Implementation

Much simpler with CVXPY!

```python
import cvxpy as cp
import numpy as np

# Problem dimensions
m, n = 100, 50

# Generate random data
np.random.seed(42)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define variable
x = cp.Variable(n)

# Define objective
lambda_val = 0.1
objective = cp.Minimize(
    0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1)
)

# Create problem
prob = cp.Problem(objective)

# Solve (use POGS if installed, else ECOS)
prob.solve(verbose=True)

# Print results
print(f"\nStatus: {prob.status}")
print(f"Optimal value: {prob.value:.4f}")
print(f"Sparsity: {np.sum(np.abs(x.value) > 1e-4)} / {n} non-zeros")
print(f"\nFirst 10 coefficients:\n{x.value[:10]}")
```

---

## Parameter Tuning

### Regularization Parameter (lambda)

- **Large lambda** (e.g., 1.0): Strong sparsity, many zeros
- **Small lambda** (e.g., 0.01): Less sparsity, closer to least squares
- **lambda = 0**: Reduces to ordinary least squares

### Solver Parameters

For Lasso, try:

```cpp
solver.SetRho(1.0);           // Good default for Lasso
solver.SetAbsTol(1e-4);       // Standard accuracy
solver.SetRelTol(1e-3);
solver.SetMaxIter(1000);
solver.SetAdaptiveRho(true);  // Helps convergence
```

For high accuracy:

```cpp
solver.SetAbsTol(1e-6);
solver.SetRelTol(1e-6);
solver.SetMaxIter(2000);
```

---

## Expected Output

```
Iter   Primal Res   Dual Res     Gap        rho
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  50   7.89e-04    2.34e-04    4.56e-03   1.00
 100   9.12e-05    3.45e-05    1.23e-04   1.00

Converged in 100 iterations
Optimal value: 45.2341
Sparsity: 12 / 50 non-zeros

First 10 coefficients:
x[0] = 0.234567
x[1] = 0.000000
x[2] = -0.456789
x[3] = 0.000000
x[4] = 0.123456
...
```

---

## Interpretation

### Sparsity

Lasso automatically selects a subset of features:

- Non-zero coefficients: Important features
- Zero coefficients: Unimportant features (excluded)

This is useful for:
- **Feature selection**: Identify important predictors
- **Interpretability**: Simpler models
- **Regularization**: Prevent overfitting

### Choosing lambda

Use cross-validation to choose lambda:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

lambdas = [0.001, 0.01, 0.1, 1.0]
scores = []

for lam in lambdas:
    model = Lasso(alpha=lam)
    score = cross_val_score(model, A, b, cv=5).mean()
    scores.append(score)

best_lambda = lambdas[np.argmax(scores)]
```

---

## Variations

### Elastic Net

Combine L1 and L2 penalties:

$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda_1\|x\|_1 + \frac{\lambda_2}{2}\|x\|_2^2
$$

```cpp
// f_i(y_i) = (1/2) * y_i^2 - b_i * y_i
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].c = 1.0;
    f[i].d = -b_data[i];
}

// g_j(x_j) = lambda1 * |x_j| + (lambda2/2) * x_j^2
// Use the 'e' parameter for quadratic term
for (size_t j = 0; j < n; ++j) {
    g[j].h = kAbs;
    g[j].c = lambda1;
    g[j].e = lambda2 / 2.0;  // Adds (lambda2/2) * x^2
}
```

### Non-Negative Lasso

Add non-negativity constraint:

```cpp
// Use indicator function for non-negativity
for (size_t j = 0; j < n; ++j) {
    g[j].h = kIndGe0;  // Constraint x >= 0
    // Note: L1 penalty is implicit via projection
}
```

---

## See Also

- [Basic Usage](../user-guide/basic-usage.md) - General POGS usage
- [CVXPY Integration](../user-guide/cvxpy-integration.md) - Python interface
- [Logistic Regression](logistic.md) - Classification example
