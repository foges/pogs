# Proximal Operators

Reference for proximal operators and function implementations in POGS.

---

## Overview

POGS uses **proximal operators** to handle the objective functions $f$ and $g$. Each function type has an associated proximal operator that is computed efficiently using closed-form solutions.

The **proximal operator** of a function $h$ with parameter $\rho$ is:

$$
\text{prox}_{h,\rho}(v) = \arg\min_x \left\{ h(x) + \frac{\rho}{2}\|x - v\|^2 \right\}
$$

---

## Supported Functions

### Zero

$$
h(x) = 0
$$

**Proximal operator:**
$$
\text{prox}(v) = v
$$

**Use case:** Unconstrained variables

---

### Identity

$$
h(x) = x
$$

**Proximal operator:**
$$
\text{prox}(v) = v - \frac{1}{\rho}
$$

**Use case:** Linear objectives

---

### Absolute Value

$$
h(x) = |x|
$$

**Proximal operator (soft thresholding):**
$$
\text{prox}(v) = \begin{cases}
v - \frac{1}{\rho} & \text{if } v > \frac{1}{\rho} \\
0 & \text{if } |v| \leq \frac{1}{\rho} \\
v + \frac{1}{\rho} & \text{if } v < -\frac{1}{\rho}
\end{cases}
$$

**Use case:** L1 regularization (Lasso)

**Example:**
```cpp
f.type = pogs::FunctionType::Abs;
f.c = lambda;  // Regularization parameter
```

---

### Square

$$
h(x) = \frac{1}{2}x^2
$$

**Proximal operator:**
$$
\text{prox}(v) = \frac{\rho}{\rho + 1} v
$$

**Use case:** Least squares, L2 regularization

**Example:**
```cpp
f.type = pogs::FunctionType::Square;
f.c = 0.5;
f.d = -b[i];  // For ||Ax - b||^2
```

---

### Indicator Functions

#### Indicator of {0}

$$
h(x) = I_{\{0\}}(x) = \begin{cases}
0 & \text{if } x = 0 \\
\infty & \text{otherwise}
\end{cases}
$$

**Proximal operator:**
$$
\text{prox}(v) = 0
$$

**Use case:** Equality constraints

---

#### Indicator of [0, ∞)

$$
h(x) = I_{[0,\infty)}(x) = \begin{cases}
0 & \text{if } x \geq 0 \\
\infty & \text{otherwise}
\end{cases}
$$

**Proximal operator:**
$$
\text{prox}(v) = \max(0, v)
$$

**Use case:** Non-negativity constraints

---

#### Indicator of (-∞, 0]

$$
h(x) = I_{(-\infty,0]}(x) = \begin{cases}
0 & \text{if } x \leq 0 \\
\infty & \text{otherwise}
\end{cases}
$$

**Proximal operator:**
$$
\text{prox}(v) = \min(0, v)
$$

**Use case:** Non-positivity constraints

---

#### Indicator of [0, 1]

$$
h(x) = I_{[0,1]}(x) = \begin{cases}
0 & \text{if } 0 \leq x \leq 1 \\
\infty & \text{otherwise}
\end{cases}
$$

**Proximal operator:**
$$
\text{prox}(v) = \max(0, \min(1, v))
$$

**Use case:** Box constraints

---

### Huber Loss

$$
h(x) = \begin{cases}
\frac{1}{2}x^2 & \text{if } |x| \leq 1 \\
|x| - \frac{1}{2} & \text{if } |x| > 1
\end{cases}
$$

**Proximal operator:**
Computed via bisection or closed-form formula.

**Use case:** Robust regression (less sensitive to outliers than square loss)

**Example:**
```cpp
f.type = pogs::FunctionType::Huber;
f.c = 1.0;  // Huber parameter
```

---

### Logistic Loss

$$
h(x) = \log(1 + e^x)
$$

**Proximal operator:**
Computed numerically.

**Use case:** Logistic regression

**Example:**
```cpp
f.type = pogs::FunctionType::Logistic;
f.d = -y[i];  // For logistic regression with label y[i]
```

---

### Negative Logarithm

$$
h(x) = -\log(x), \quad x > 0
$$

**Proximal operator:**
$$
\text{prox}(v) = \frac{v + \sqrt{v^2 + 4/\rho}}{2}
$$

**Use case:** Barrier functions, entropy maximization

---

### Exponential

$$
h(x) = e^x
$$

**Proximal operator:**
Computed via Lambert W function.

**Use case:** Exponential objectives

---

### Negative Entropy

$$
h(x) = x \log x, \quad x > 0
$$

**Proximal operator:**
$$
\text{prox}(v) = \frac{1}{\rho} W(\rho v e^{\rho v})
$$

where $W$ is the Lambert W function.

**Use case:** Entropy regularization

---

### Max Functions

#### MaxPos0

$$
h(x) = \max(0, x)
$$

**Proximal operator:**
$$
\text{prox}(v) = \begin{cases}
v - \frac{1}{\rho} & \text{if } v > \frac{1}{\rho} \\
0 & \text{otherwise}
\end{cases}
$$

**Use case:** ReLU activation, hinge loss

---

#### MaxNeg0

$$
h(x) = \max(0, -x)
$$

**Proximal operator:**
$$
\text{prox}(v) = \begin{cases}
v + \frac{1}{\rho} & \text{if } v < -\frac{1}{\rho} \\
0 & \text{otherwise}
\end{cases}
$$

**Use case:** Hinge loss (SVM)

---

### Reciprocal

$$
h(x) = \frac{1}{x}, \quad x > 0
$$

**Proximal operator:**
Computed via cubic equation solution.

**Use case:** Reciprocal penalties

---

## Function Parameterization

Each function can be parameterized using the `FunctionObj` struct:

```cpp
template<typename T>
struct FunctionObj {
    FunctionType type;  // Base function h
    T a = 1.0;         // Input scaling
    T b = 0.0;         // Input shift
    T c = 1.0;         // Output scaling
    T d = 0.0;         // Linear term
    T e = 0.0;         // Constant offset
    T rho = 1.0;       // Penalty parameter
};
```

The full function is:

$$
f(x) = h(ax + b) \cdot c + d \cdot x + e
$$

### Examples

**Weighted L1:**
```cpp
f.type = FunctionType::Abs;
f.c = lambda;  // Weight
```

**Shifted square:**
```cpp
f.type = FunctionType::Square;
f.c = 0.5;
f.d = -b[i];   // Creates (1/2)(x)^2 - b[i]*x
```

**Scaled indicator:**
```cpp
f.type = FunctionType::IndGe0;
f.a = 2.0;     // Constraint is 2*x >= 0
```

---

## Implementation Details

### Numerical Stability

POGS proximal operators are implemented with numerical stability in mind:

- **Soft thresholding:** Uses stable formula avoiding cancellation
- **Log-exp:** Uses log-sum-exp trick
- **Square root:** Checks for negative values
- **Division:** Handles near-zero denominators

### Performance

- Most proximal operators have **O(1)** complexity
- Computed element-wise (embarrassingly parallel)
- GPU implementations available for dense problems

---

## Adding Custom Functions

To add a new function:

1. Add enum value to `FunctionType`
2. Implement proximal operator in `prox_lib.h`
3. Update function evaluation in `evaluator.h`
4. Add tests

---

## See Also

- [Types](types.md) - FunctionType enumeration
- [Basic Usage](../user-guide/basic-usage.md) - Using functions in practice
- [Examples](../examples/lasso.md) - Practical examples
