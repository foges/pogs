# Logistic Regression Example

Complete example of solving logistic regression with POGS.

---

## Problem Formulation

**Logistic Regression** for binary classification solves:

$$
\text{minimize} \quad \sum_{i=1}^m \log(1 + e^{-y_i (a_i^T x + b)}) + \frac{\lambda}{2}\|x\|_2^2
$$

where:
- $a_i \in \mathbb{R}^n$ are the feature vectors
- $y_i \in \{-1, +1\}$ are the binary labels
- $x \in \mathbb{R}^n$ are the coefficients
- $b \in \mathbb{R}$ is the intercept
- $\lambda > 0$ is the L2 regularization parameter

---

## POGS Formulation

For POGS, we reformulate as:

$$
\begin{align}
\text{minimize} \quad & \sum_{i=1}^m \log(1 + e^{z_i}) + \frac{\lambda}{2}\|x\|_2^2 \\
\text{subject to} \quad & z_i = -y_i(a_i^T x + b)
\end{align}
$$

In graph form with $A = \text{diag}(-y) \cdot [\mathbf{a}_1, \ldots, \mathbf{a}_m]^T$:

- $f_i(z_i) = \log(1 + e^{z_i})$ (logistic loss)
- $g_j(x_j) = \frac{\lambda}{2}x_j^2$ (L2 regularization)

---

## C++ Implementation

```cpp
#include <pogs/pogs.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main() {
    // Problem dimensions
    const size_t m = 200;  // Number of samples
    const size_t n = 50;   // Number of features

    // Generate random classification data
    std::vector<double> A_data(m * n);
    std::vector<int> y_data(m);

    std::default_random_engine gen(42);
    std::normal_distribution<double> feat_dist(0.0, 1.0);
    std::uniform_int_distribution<int> label_dist(0, 1);

    // Generate features
    for (size_t i = 0; i < m * n; ++i) {
        A_data[i] = feat_dist(gen);
    }

    // Generate labels {-1, +1}
    for (size_t i = 0; i < m; ++i) {
        y_data[i] = 2 * label_dist(gen) - 1;  // Convert {0,1} to {-1,+1}
    }

    // Scale A by -y (row-wise)
    for (size_t i = 0; i < m; ++i) {
        double scale = -static_cast<double>(y_data[i]);
        for (size_t j = 0; j < n; ++j) {
            A_data[i * n + j] *= scale;
        }
    }

    // Create matrix
    auto A = std::make_unique<pogs::MatrixDense<double>>(m, n);
    // Copy scaled data to matrix...

    // Define objective functions
    std::vector<pogs::FunctionObj<double>> f(m);
    std::vector<pogs::FunctionObj<double>> g(n);

    // f_i(z_i) = log(1 + exp(z_i))
    for (size_t i = 0; i < m; ++i) {
        f[i].type = pogs::FunctionType::Logistic;
        f[i].c = 1.0;
    }

    // g_j(x_j) = (lambda/2) * x_j^2
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].type = pogs::FunctionType::Square;
        g[j].c = lambda / 2.0;
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

        // Compute training accuracy
        size_t correct = 0;
        for (size_t i = 0; i < m; ++i) {
            double pred = 0.0;
            for (size_t j = 0; j < n; ++j) {
                pred += A_data[i * n + j] * result.x[j];
            }
            int pred_label = (pred > 0) ? 1 : -1;
            if (pred_label == y_data[i]) {
                correct++;
            }
        }
        double accuracy = static_cast<double>(correct) / m;
        std::cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";

        // Print first 10 coefficients
        std::cout << "\nFirst 10 coefficients:\n";
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            std::cout << "x[" << j << "] = " << result.x[j] << "\n";
        }
    } else {
        std::cerr << "Solver failed\n";
        return 1;
    }

    return 0;
}
```

---

## Python/CVXPY Implementation

```python
import cvxpy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate classification data
X, y = make_classification(
    n_samples=200,
    n_features=50,
    n_informative=20,
    n_redundant=10,
    random_state=42
)

# Convert labels to {-1, +1}
y = 2 * y - 1

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Problem dimensions
m, n = X.shape

# Define variables
x = cp.Variable(n)
b = cp.Variable()

# Define objective: sum of logistic losses + L2 regularization
lambda_val = 0.1
losses = cp.sum(cp.logistic(-cp.multiply(y, X @ x + b)))
regularization = (lambda_val / 2) * cp.sum_squares(x)
objective = cp.Minimize(losses + regularization)

# Create problem
prob = cp.Problem(objective)

# Solve with POGS
prob.solve(solver='POGS', verbose=True)

# Print results
print(f"\nStatus: {prob.status}")
print(f"Optimal value: {prob.value:.4f}")

# Compute accuracy
predictions = np.sign(X @ x.value + b.value)
accuracy = np.mean(predictions == y)
print(f"Training accuracy: {accuracy * 100:.2f}%")

print(f"\nFirst 10 coefficients:\n{x.value[:10]}")
print(f"Intercept: {b.value:.4f}")
```

---

## Parameter Tuning

### Regularization Parameter (λ)

- **Large λ** (e.g., 1.0): Strong regularization, simpler model
- **Small λ** (e.g., 0.01): Weak regularization, complex model
- **λ = 0**: No regularization (may overfit)

### Solver Parameters

For logistic regression, try:

```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,           // Good default
    .abs_tol = 1e-4,      // Standard accuracy
    .rel_tol = 1e-3,
    .max_iter = 2000,     // May need more iterations
    .adaptive_rho = true  // Helps convergence
};
```

---

## Expected Output

```
Iter   Primal Res   Dual Res     Gap        ρ
  10   2.34e-02    5.67e-03    1.23e-01   1.00
  20   6.78e-03    2.34e-03    4.56e-02   1.00
  50   1.23e-03    4.56e-04    8.90e-03   1.00
 150   8.90e-05    3.21e-05    1.23e-04   1.00  ✓ Converged

Converged in 150 iterations
Optimal value: 68.4521
Training accuracy: 92.5%

First 10 coefficients:
x[0] = 0.456789
x[1] = -0.234567
x[2] = 0.789012
...
```

---

## Model Interpretation

### Coefficients

- **Positive coefficient**: Feature increases probability of class +1
- **Negative coefficient**: Feature increases probability of class -1
- **Large magnitude**: Strong predictor
- **Near zero**: Weak predictor

### Probability Prediction

For a new sample $a$, the probability of class +1 is:

$$
P(y = +1 | a) = \frac{1}{1 + e^{-(a^T x + b)}}
$$

```cpp
double predict_probability(const std::vector<double>& features,
                          const std::vector<double>& coefficients,
                          double intercept) {
    double z = intercept;
    for (size_t j = 0; j < features.size(); ++j) {
        z += features[j] * coefficients[j];
    }
    return 1.0 / (1.0 + std::exp(-z));
}
```

---

## Variations

### L1-Regularized Logistic Regression

For feature selection, use L1 instead of L2:

```cpp
// g_j(x_j) = lambda * |x_j|
for (size_t j = 0; j < n; ++j) {
    g[j].type = pogs::FunctionType::Abs;
    g[j].c = lambda;
}
```

This produces sparse solutions (automatic feature selection).

### Multinomial Logistic Regression

For multi-class classification, reformulate as multiple binary problems or use softmax.

---

## Cross-Validation

Choose λ using cross-validation:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lambdas = [0.001, 0.01, 0.1, 1.0]
scores = []

for lam in lambdas:
    model = LogisticRegression(C=1.0/lam, penalty='l2')
    score = cross_val_score(model, X, y, cv=5).mean()
    scores.append(score)

best_lambda = lambdas[np.argmax(scores)]
print(f"Best lambda: {best_lambda}")
```

---

## See Also

- [Lasso Example](lasso.md) - Regression with L1 penalty
- [Basic Usage](../user-guide/basic-usage.md) - General POGS usage
- [CVXPY Integration](../user-guide/cvxpy-integration.md) - Python interface
