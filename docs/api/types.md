# Types Reference

Reference for POGS type definitions.

---

## Function Objects

### FunctionObj

```cpp
template<typename T>
struct FunctionObj {
    FunctionType type;
    T a = 1.0;
    T b = 0.0;
    T c = 1.0;
    T d = 0.0;
    T e = 0.0;
    T rho = 1.0;

    explicit FunctionObj(FunctionType t = FunctionType::Zero);
};
```

Represents a function of the form:

$$
h(ax + b) \cdot c + d \cdot x + e
$$

where $h$ is the base function determined by `type`.

**Fields:**
- `type`: Base function type
- `a`: Input scaling
- `b`: Input shift
- `c`: Output scaling
- `d`: Linear term coefficient
- `e`: Constant offset
- `rho`: Penalty parameter (adaptive)

---

## Function Types

### FunctionType Enumeration

```cpp
enum class FunctionType {
    Abs,        // |x|
    Exp,        // e^x
    Huber,      // Huber loss
    Identity,   // x
    IndBox01,   // I_{[0,1]}(x)
    IndEq0,     // I_{x=0}(x)
    IndGe0,     // I_{x>=0}(x)
    IndLe0,     // I_{x<=0}(x)
    Logistic,   // log(1 + e^x)
    MaxNeg0,    // max(0, -x)
    MaxPos0,    // max(0, x)
    NegEntr,    // x log(x)
    NegLog,     // -log(x)
    Recipr,     // 1/x
    Square,     // x^2
    Zero        // 0
};
```

### Mathematical Definitions

| Type | Function h(x) | Domain | Use Case |
|------|---------------|---------|----------|
| `Abs` | $\|x\|$ | $\mathbb{R}$ | L1 regularization |
| `Exp` | $e^x$ | $\mathbb{R}$ | Exponential objectives |
| `Huber` | Huber loss | $\mathbb{R}$ | Robust regression |
| `Identity` | $x$ | $\mathbb{R}$ | Linear terms |
| `IndBox01` | $I_{[0,1]}(x)$ | $[0,1]$ | Box constraints |
| `IndEq0` | $I_{\{0\}}(x)$ | $\{0\}$ | Equality constraints |
| `IndGe0` | $I_{[0,\infty)}(x)$ | $[0,\infty)$ | Non-negativity |
| `IndLe0` | $I_{(-\infty,0]}(x)$ | $(-\infty,0]$ | Non-positivity |
| `Logistic` | $\log(1 + e^x)$ | $\mathbb{R}$ | Logistic regression |
| `MaxNeg0` | $\max(0, -x)$ | $\mathbb{R}$ | Hinge loss |
| `MaxPos0` | $\max(0, x)$ | $\mathbb{R}$ | ReLU |
| `NegEntr` | $x \log x$ | $(0,\infty)$ | Entropy |
| `NegLog` | $-\log x$ | $(0,\infty)$ | Barrier functions |
| `Recipr` | $1/x$ | $(0,\infty)$ | Reciprocal |
| `Square` | $\frac{1}{2}x^2$ | $\mathbb{R}$ | Least squares |
| `Zero` | $0$ | $\mathbb{R}$ | Unconstrained |

---

## Cone Constraints

### ConeType Enumeration

```cpp
enum class ConeType {
    Zero,        // {x : x = 0}
    NonNeg,      // {x : x >= 0}
    NonPos,      // {x : x <= 0}
    SOC,         // {(t,x) : ||x||_2 <= t}
    SDP,         // {X : X ⪰ 0}
    ExpPrimal,   // Exponential cone
    ExpDual      // Dual exponential cone
};
```

### ConeConstraint

```cpp
template<typename T>
struct ConeConstraint {
    ConeType type;
    std::vector<size_t> indices;

    ConeConstraint(ConeType t, std::vector<size_t> idx);
};
```

**Fields:**
- `type`: Type of cone
- `indices`: Variable indices in this cone

**Example:**
```cpp
// x[0], x[1], x[2] must be non-negative
pogs::ConeConstraint<double> cone{
    pogs::ConeType::NonNeg,
    {0, 1, 2}
};
```

---

## Status Codes

### Status Enumeration

```cpp
enum class Status {
    Success = 0,               // Converged successfully
    MaxIterations,             // Maximum iterations reached
    NumericalError,            // Numerical error encountered
    InfeasibleOrUnbounded      // Problem is infeasible or unbounded
};
```

**Usage:**
```cpp
auto result = solver.solve(f, g);

switch (result.status) {
    case pogs::Status::Success:
        // Handle success
        break;
    case pogs::Status::MaxIterations:
        // Handle non-convergence
        break;
    case pogs::Status::NumericalError:
        // Handle numerical issues
        break;
    case pogs::Status::InfeasibleOrUnbounded:
        // Handle infeasibility
        break;
}
```

---

## Matrix Ordering

### Ord Enumeration

```cpp
enum class Ord {
    RowMajor,    // Row-major (C-style)
    ColMajor     // Column-major (Fortran-style)
};
```

---

## Type Aliases

### Common Scalar Types

```cpp
using pogs::Float = float;
using pogs::Double = double;
```

### Vector Types

```cpp
template<typename T>
using Vector = std::vector<T>;

template<typename T>
using Span = std::span<T>;

template<typename T>
using ConstSpan = std::span<const T>;
```

---

## See Also

- [Solver API](solver.md) - Main solver interface
- [Configuration](configuration.md) - Solver configuration
- [Proximal Operators](proximal.md) - Function implementations
