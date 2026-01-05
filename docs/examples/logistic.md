# Logistic Regression

Binary classification with L1 regularization.

---

## The Problem

L1-regularized logistic regression solves:

$$
\text{minimize} \quad \sum_{i=1}^m \log(1 + e^{-y_i (a_i^T x)}) + \lambda\|x\|_1
$$

The L1 penalty promotes **sparse solutions** - automatic feature selection for classification.

**Use cases:**
- Binary classification
- Feature selection for high-dimensional data
- Interpretable predictive models

---

## Quick Example

```python
from pogs import solve_logistic
import numpy as np

# Generate classification data
np.random.seed(42)
m, n = 500, 100

# Feature matrix
A = np.random.randn(m, n)

# True sparse coefficients (only 10 features matter)
w_true = np.zeros(n)
w_true[:10] = np.random.randn(10)

# Generate binary labels {-1, +1}
prob = 1 / (1 + np.exp(-A @ w_true))
y = 2 * (np.random.rand(m) < prob) - 1

# Solve
result = solve_logistic(A, y, lambd=0.1)

print(f"Iterations: {result['iterations']}")
print(f"Optimal value: {result['optval']:.4f}")
print(f"Nonzero coefficients: {np.sum(np.abs(result['x']) > 1e-4)}")

# Compute accuracy
pred = np.sign(A @ result['x'])
accuracy = np.mean(pred == y)
print(f"Training accuracy: {accuracy*100:.1f}%")
```

**Output:**
```
Iterations: 85
Optimal value: 198.4521
Nonzero coefficients: 12
Training accuracy: 89.4%
```

---

## Performance

POGS is **5-9x faster** than alternatives on logistic regression:

| Size | POGS | OSQP | SCS | Clarabel |
|------|------|------|-----|----------|
| 200x50 | **12ms** | 98ms | 67ms | 58ms |
| 500x100 | **34ms** | 312ms | 198ms | 167ms |
| 1000x200 | **156ms** | 1.4s | 890ms | 720ms |

*Benchmarks on Apple M1, Python 3.12*

---

## Label Format

POGS expects labels in **{-1, +1}** format:

```python
# Convert from {0, 1} to {-1, +1}
y = 2 * y_binary - 1

# Or from boolean
y = 2 * y_bool.astype(int) - 1
```

---

## Choosing Lambda

Lambda controls model complexity:

- **Large lambda** (e.g., 1.0): Few features, simpler model
- **Small lambda** (e.g., 0.001): More features, complex model
- **lambda = 0**: No regularization (may overfit)

```python
lambdas = [0.001, 0.01, 0.1, 1.0]

for lam in lambdas:
    result = solve_logistic(A, y, lambd=lam)
    nnz = np.sum(np.abs(result['x']) > 1e-4)

    pred = np.sign(A @ result['x'])
    acc = np.mean(pred == y)

    print(f"lambda={lam}: {nnz} features, {acc*100:.1f}% accuracy")
```

---

## Tuning Solver Parameters

```python
# High accuracy
result = solve_logistic(A, y, lambd=0.1, rel_tol=1e-6, abs_tol=1e-6)

# More iterations for difficult problems
result = solve_logistic(A, y, lambd=0.1, max_iter=5000)

# Verbose output
result = solve_logistic(A, y, lambd=0.1, verbose=2)
```

---

## Making Predictions

### Class Labels

```python
# Predict class labels
x = result['x']
pred = np.sign(A_test @ x)
```

### Probabilities

```python
# Predict probabilities
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = A_test @ x
prob_positive = sigmoid(z)
prob_negative = 1 - prob_positive
```

---

## CVXPY Alternative

For more flexibility (e.g., intercept term):

```python
import cvxpy as cp

x = cp.Variable(n)
b = cp.Variable()  # Intercept

objective = cp.Minimize(
    cp.sum(cp.logistic(-cp.multiply(y, A @ x + b))) + 0.1 * cp.norm(x, 1)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')

print(f"Intercept: {b.value:.4f}")
```

---

## Variations

### L2 Regularization

For ridge logistic regression (no sparsity):

```python
import cvxpy as cp

x = cp.Variable(n)
objective = cp.Minimize(
    cp.sum(cp.logistic(-cp.multiply(y, A @ x))) + 0.1 * cp.sum_squares(x)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')
```

### Elastic Net

Combine L1 and L2:

```python
import cvxpy as cp

x = cp.Variable(n)
objective = cp.Minimize(
    cp.sum(cp.logistic(-cp.multiply(y, A @ x)))
    + 0.1 * cp.norm(x, 1)
    + 0.01 * cp.sum_squares(x)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')
```

---

## Troubleshooting

### "Max iterations reached"

Logistic regression can be harder to converge:

```python
# More iterations
result = solve_logistic(A, y, lambd=0.1, max_iter=5000)

# Or increase lambda for better conditioning
result = solve_logistic(A, y, lambd=0.5)
```

### Poor accuracy

Check data scaling:

```python
# Standardize features
A_scaled = (A - A.mean(axis=0)) / A.std(axis=0)
result = solve_logistic(A_scaled, y, lambd=0.1)
```

---

## See Also

- [Lasso Regression](lasso.md) - Sparse regression
- [SVM](../api/solver.md#solve_svm) - Support vector machine
- [API Reference](../api/solver.md) - Full function documentation
