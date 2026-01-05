# Types Reference

Reference for POGS type definitions.

---

## Function Objects

### FunctionObj

```cpp
template<typename T>
struct FunctionObj {
    Function h;       // Base function type
    T a = 1.0;        // Input scaling
    T b = 0.0;        // Input shift
    T c = 1.0;        // Output scaling
    T d = 0.0;        // Linear term coefficient
    T e = 0.0;        // Quadratic term coefficient

    // Constructors
    explicit FunctionObj(Function h);
    FunctionObj(Function h, T a);
    FunctionObj(Function h, T a, T b);
    FunctionObj(Function h, T a, T b, T c);
    FunctionObj(Function h, T a, T b, T c, T d);
    FunctionObj(Function h, T a, T b, T c, T d, T e);
    FunctionObj();  // Default: kZero
};
```

Represents a function of the form:

$$
c \cdot h(ax - b) + d \cdot x + e \cdot x^2
$$

where $h$ is the base function determined by the `h` field.

---

## Function Types

### Function Enumeration

```cpp
enum Function {
    kAbs,       // |x|
    kExp,       // e^x
    kHuber,     // Huber loss
    kIdentity,  // x
    kIndBox01,  // I_{[0,1]}(x)
    kIndEq0,    // I_{x=0}(x)
    kIndGe0,    // I_{x>=0}(x)
    kIndLe0,    // I_{x<=0}(x)
    kLogistic,  // log(1 + e^x)
    kMaxNeg0,   // max(0, -x)
    kMaxPos0,   // max(0, x)
    kNegEntr,   // x log(x)
    kNegLog,    // -log(x)
    kRecipr,    // 1/x
    kSquare,    // (1/2) x^2
    kZero       // 0
};
```

### Mathematical Definitions

| Type | Function h(x) | Domain | Use Case |
|------|---------------|---------|----------|
| `kAbs` | $\|x\|$ | $\mathbb{R}$ | L1 regularization |
| `kExp` | $e^x$ | $\mathbb{R}$ | Exponential objectives |
| `kHuber` | Huber loss | $\mathbb{R}$ | Robust regression |
| `kIdentity` | $x$ | $\mathbb{R}$ | Linear terms |
| `kIndBox01` | $I_{[0,1]}(x)$ | $[0,1]$ | Box constraints |
| `kIndEq0` | $I_{\{0\}}(x)$ | $\{0\}$ | Equality constraints |
| `kIndGe0` | $I_{[0,\infty)}(x)$ | $[0,\infty)$ | Non-negativity |
| `kIndLe0` | $I_{(-\infty,0]}(x)$ | $(-\infty,0]$ | Non-positivity |
| `kLogistic` | $\log(1 + e^x)$ | $\mathbb{R}$ | Logistic regression |
| `kMaxNeg0` | $\max(0, -x)$ | $\mathbb{R}$ | Hinge loss |
| `kMaxPos0` | $\max(0, x)$ | $\mathbb{R}$ | ReLU |
| `kNegEntr` | $x \log x$ | $(0,\infty)$ | Entropy |
| `kNegLog` | $-\log x$ | $(0,\infty)$ | Barrier functions |
| `kRecipr` | $1/x$ | $(0,\infty)$ | Reciprocal |
| `kSquare` | $\frac{1}{2}x^2$ | $\mathbb{R}$ | Least squares |
| `kZero` | $0$ | $\mathbb{R}$ | Unconstrained |

---

## Cone Constraints

### ConeConstraint Structure

```cpp
struct ConeConstraint {
    enum ConeType { kConeZero, kConeNonNeg, kConeNonPos,
                    kConeSoc, kConeSdp,
                    kConeExpPrimal, kConeExpDual };

    ConeType cone;
    std::vector<unsigned int> idx;

    ConeConstraint(ConeType c, const std::vector<unsigned int>& indices);
};
```

**Cone Types:**

| Type | Definition | Use Case |
|------|------------|----------|
| `kConeZero` | $\{x : x = 0\}$ | Equality constraints |
| `kConeNonNeg` | $\{x : x \geq 0\}$ | Non-negativity |
| `kConeNonPos` | $\{x : x \leq 0\}$ | Non-positivity |
| `kConeSoc` | $\{(t,x) : \|x\|_2 \leq t\}$ | Second-order cone |
| `kConeSdp` | $\{X : X \succeq 0\}$ | Semidefinite |
| `kConeExpPrimal` | Exponential cone | Exponential constraints |
| `kConeExpDual` | Dual exponential cone | Dual constraints |

**Example:**
```cpp
// x[0], x[1], x[2] must be non-negative
ConeConstraint cone{ConeConstraint::kConeNonNeg, {0, 1, 2}};
```

---

## Status Codes

### PogsStatus Enumeration

```cpp
enum PogsStatus {
    POGS_SUCCESS,      // Converged successfully
    POGS_INFEASIBLE,   // Problem likely infeasible
    POGS_UNBOUNDED,    // Problem likely unbounded
    POGS_MAX_ITER,     // Maximum iterations reached
    POGS_NAN_FOUND,    // Numerical error (NaN)
    POGS_INVALID_CONE, // Invalid cone specification
    POGS_ERROR         // Generic error
};
```

**Usage:**
```cpp
PogsStatus status = solver.Solve(f, g);

switch (status) {
    case POGS_SUCCESS:
        // Handle success
        break;
    case POGS_MAX_ITER:
        // Handle non-convergence
        break;
    case POGS_NAN_FOUND:
        // Handle numerical issues
        break;
    default:
        // Handle other errors
        break;
}
```

---

## Matrix Ordering

For the C interface:

```c
enum ORD {
    COL_MAJ,  // Column-major (Fortran-style)
    ROW_MAJ   // Row-major (C-style)
};
```

For the C++ interface, the `MatrixDense` and `MatrixSparse` constructors take a `char ord` parameter:
- `'r'` for row-major
- `'c'` for column-major

---

## See Also

- [Solver API](solver.md) - Main solver interface
- [Configuration](configuration.md) - Solver parameters
- [Proximal Operators](proximal.md) - Function implementations
