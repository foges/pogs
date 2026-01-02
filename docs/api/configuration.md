# Solver Configuration

Reference for solver configuration options.

---

## SolverConfig Structure

```cpp
struct SolverConfig {
    double rho = 1.0;
    double abs_tol = 1e-4;
    double rel_tol = 1e-3;
    size_t max_iter = 1000;
    bool verbose = false;
    bool adaptive_rho = true;
    bool gap_stop = true;

    // Reserved for future features
    bool use_anderson = false;
    size_t anderson_mem = 5;
    size_t anderson_start = 10;
};
```

---

## Parameters

### rho

**Type:** `double`
**Default:** `1.0`
**Range:** `(0, ∞)`

The ADMM penalty parameter. Controls the weight of the augmented Lagrangian term.

**Guidelines:**
- **Larger ρ** (5.0-100.0): Faster convergence, potentially less accurate
- **Smaller ρ** (0.01-0.5): Slower convergence, more accurate
- **Default** (1.0): Good starting point for most problems

**Example:**
```cpp
config.rho = 10.0;  // Faster, less accurate
```

---

### abs_tol

**Type:** `double`
**Default:** `1e-4`
**Range:** `(0, 1)`

Absolute tolerance for convergence.

Solver stops when:
$$
\|r\|_2 \leq \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \max(\|Ax\|_2, \|y\|_2)
$$

**Guidelines:**
- **High accuracy:** `1e-6` or smaller
- **Medium accuracy:** `1e-4` (default)
- **Low accuracy:** `1e-2`

**Example:**
```cpp
config.abs_tol = 1e-6;  // High accuracy
```

---

### rel_tol

**Type:** `double`
**Default:** `1e-3`
**Range:** `(0, 1)`

Relative tolerance for convergence.

**Guidelines:**
- **High accuracy:** `1e-5` or smaller
- **Medium accuracy:** `1e-3` (default)
- **Low accuracy:** `1e-2`

**Example:**
```cpp
config.rel_tol = 1e-5;  // High accuracy
```

---

### max_iter

**Type:** `size_t`
**Default:** `1000`
**Range:** `[1, ∞)`

Maximum number of ADMM iterations.

**Guidelines:**
- **Easy problems:** 100-500 iterations sufficient
- **Medium problems:** 1000-2000 iterations
- **Hard problems:** 5000-10000 iterations

**Example:**
```cpp
config.max_iter = 5000;  // For difficult problems
```

---

### verbose

**Type:** `bool`
**Default:** `false`

Enable verbose output showing iteration progress.

**Output format:**
```
Iter   Primal Res   Dual Res     Gap        ρ
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  ...
```

**Example:**
```cpp
config.verbose = true;  // Show progress
```

---

### adaptive_rho

**Type:** `bool`
**Default:** `true`

Enable adaptive adjustment of ρ based on residual balance.

When enabled, ρ is automatically adjusted:
- **Increase ρ** if primal residual >> dual residual
- **Decrease ρ** if dual residual >> primal residual

**Guidelines:**
- **Enable** (recommended): Better convergence for most problems
- **Disable**: When you want fixed ρ or manual control

**Example:**
```cpp
config.adaptive_rho = false;  // Fixed ρ
```

---

### gap_stop

**Type:** `bool`
**Default:** `true`

Enable duality gap stopping criterion.

Stops when both residuals are small AND duality gap is small.

**Guidelines:**
- **Enable** (recommended): More reliable convergence
- **Disable**: For problems where duality gap is not available

**Example:**
```cpp
config.gap_stop = true;  // Use duality gap
```

---

### use_anderson

**Type:** `bool`
**Default:** `false`
**Status:** Experimental

Enable Anderson acceleration (experimental).

!!! warning "Experimental Feature"
    Anderson acceleration is disabled by default and may not improve convergence for all problems.

See [Anderson Acceleration](../examples/anderson.md) for details.

**Example:**
```cpp
config.use_anderson = true;       // Enable Anderson
config.anderson_mem = 10;         // Memory depth
config.anderson_start = 20;       // Start iteration
```

---

### anderson_mem

**Type:** `size_t`
**Default:** `5`
**Range:** `[1, 20]`

Number of past iterates to store for Anderson acceleration.

Only used when `use_anderson = true`.

---

### anderson_start

**Type:** `size_t`
**Default:** `10`
**Range:** `[0, max_iter)`

Iteration number to start applying Anderson acceleration.

Only used when `use_anderson = true`.

---

## Usage Examples

### Fast Approximate Solution

```cpp
auto config = pogs::SolverConfig{
    .rho = 10.0,
    .abs_tol = 1e-2,
    .rel_tol = 1e-2,
    .max_iter = 100
};
```

### High Accuracy Solution

```cpp
auto config = pogs::SolverConfig{
    .rho = 0.1,
    .abs_tol = 1e-6,
    .rel_tol = 1e-6,
    .max_iter = 5000,
    .adaptive_rho = true
};
```

### Debugging Configuration

```cpp
auto config = pogs::SolverConfig{
    .verbose = true,
    .max_iter = 100
};
```

---

## Designated Initializers (C++20)

POGS uses C++20 designated initializers for convenient configuration:

```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4,
    .verbose = true
};
// Unspecified fields use defaults
```

---

## Performance Tuning

### Problem-Specific Guidelines

**Well-Conditioned Problems:**
```cpp
config.rho = 5.0;          // Larger ρ
config.adaptive_rho = true; // Let solver adjust
```

**Ill-Conditioned Problems:**
```cpp
config.rho = 0.1;           // Smaller ρ
config.adaptive_rho = true; // Critical for convergence
config.max_iter = 5000;     // More iterations
```

**Large-Scale Problems:**
```cpp
config.abs_tol = 1e-3;      // Looser tolerances
config.rel_tol = 1e-2;
config.max_iter = 2000;
```

---

## See Also

- [Basic Usage](../user-guide/basic-usage.md) - Using the configuration
- [Advanced Features](../user-guide/advanced-features.md) - Parameter tuning
- [Solver API](solver.md) - Main solver interface
