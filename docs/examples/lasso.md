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
#include <pogs/pogs.hpp>
#include <pogs/types.hpp>
#include <pogs/config.hpp>
#include <iostream>
#include <vector>
#include <random>

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

    // Create matrix
    auto A = std::make_unique<pogs::MatrixDense<double>>(m, n);
    // Copy data to matrix... (implementation specific)

    // Define objective functions
    std::vector<pogs::FunctionObj<double>> f(m);
    std::vector<pogs::FunctionObj<double>> g(n);

    // f_i(y_i) = (1/2) * y_i^2 - b_i * y_i
    for (size_t i = 0; i < m; ++i) {
        f[i].type = pogs::FunctionType::Square;
        f[i].c = 0.5;
        f[i].d = -b_data[i];
    }

    // g_j(x_j) = lambda * |x_j|
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].type = pogs::FunctionType::Abs;
        g[j].c = lambda;
    }

    // Configure solver
    auto config = pogs::SolverConfig{
        .rho = 1.0,
        .abs_tol = 1e-4,
        .rel_tol = 1e-3,
        .max_iter = 1000,
        .verbose = true,
        .adaptive_rho = true
    };

    // Create solver
    auto solver = pogs::make_solver<double>(std::move(A));
    solver.configure(config);

    // Solve
    auto result = solver.solve(f, g);

    // Check result
    if (result.status == pogs::Status::Success) {
        std::cout << "\nConverged in " << result.iterations << " iterations\n";
        std::cout << "Optimal value: " << result.primal_obj.value() << "\n";

        // Count non-zeros (sparsity)
        size_t nnz = 0;
        for (size_t j = 0; j < n; ++j) {
            if (std::abs(result.x[j]) > 1e-4) {
                nnz++;
            }
        }
        std::cout << "Sparsity: " << nnz << " / " << n << " non-zeros\n";

        // Print first 10 coefficients
        std::cout << "\nFirst 10 coefficients:\n";
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            std::cout << "x[" << j << "] = " << result.x[j] << "\n";
        }
    } else {
        std::cerr << "Solver failed with status "
                  << static_cast<int>(result.status) << "\n";
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

# Solve with POGS
prob.solve(solver='POGS', verbose=True)

# Print results
print(f"\nStatus: {prob.status}")
print(f"Optimal value: {prob.value:.4f}")
print(f"Sparsity: {np.sum(np.abs(x.value) > 1e-4)} / {n} non-zeros")
print(f"\nFirst 10 coefficients:\n{x.value[:10]}")
```

---

## Parameter Tuning

### Regularization Parameter (λ)

- **Large λ** (e.g., 1.0): Strong sparsity, many zeros
- **Small λ** (e.g., 0.01): Less sparsity, closer to least squares
- **λ = 0**: Reduces to ordinary least squares

### Solver Parameters

For Lasso, try:

```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,           // Good default for Lasso
    .abs_tol = 1e-4,      // Standard accuracy
    .rel_tol = 1e-3,
    .max_iter = 1000,
    .adaptive_rho = true  // Helps convergence
};
```

For high accuracy:

```cpp
config.abs_tol = 1e-6;
config.rel_tol = 1e-6;
config.max_iter = 2000;
```

---

## Expected Output

```
Iter   Primal Res   Dual Res     Gap        ρ
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  50   7.89e-04    2.34e-04    4.56e-03   1.00
 100   9.12e-05    3.45e-05    1.23e-04   1.00  ✓ Converged

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

### Choosing λ

Use cross-validation to choose λ:

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
    f[i].type = pogs::FunctionType::Square;
    f[i].c = 0.5;
    f[i].d = -b_data[i];
}

// g_j(x_j) = lambda1 * |x_j| + (lambda2/2) * x_j^2
// Split into two terms or use combined function
```

### Non-Negative Lasso

Add non-negativity constraint:

```cpp
// Use indicator function instead of Abs
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::IndGe0;
    g[j].c = lambda;
}
```

---

## See Also

- [Basic Usage](../user-guide/basic-usage.md) - General POGS usage
- [CVXPY Integration](../user-guide/cvxpy-integration.md) - Python interface
- [Logistic Regression](logistic.md) - Classification example
