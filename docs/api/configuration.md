# Solver Configuration

Reference for solver configuration methods.

---

## Configuration Methods

POGS solvers are configured using setter methods on the solver instance. All parameters have sensible defaults.

### SetRho

```cpp
void SetRho(T rho);
T GetRho() const;
```

**Default:** `1.0`
**Range:** `(0, infinity)`

The ADMM penalty parameter. Controls the weight of the augmented Lagrangian term.

**Guidelines:**
- **Larger rho** (5.0-100.0): Faster convergence, potentially less accurate
- **Smaller rho** (0.01-0.5): Slower convergence, more accurate
- **Default** (1.0): Good starting point for most problems

---

### SetAbsTol

```cpp
void SetAbsTol(T abs_tol);
T GetAbsTol() const;
```

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

---

### SetRelTol

```cpp
void SetRelTol(T rel_tol);
T GetRelTol() const;
```

**Default:** `1e-3`
**Range:** `(0, 1)`

Relative tolerance for convergence.

---

### SetMaxIter

```cpp
void SetMaxIter(unsigned int max_iter);
unsigned int GetMaxIter() const;
```

**Default:** `2500`
**Range:** `[1, infinity)`

Maximum number of ADMM iterations.

**Guidelines:**
- **Easy problems:** 100-500 iterations sufficient
- **Medium problems:** 1000-2000 iterations
- **Hard problems:** 5000-10000 iterations

---

### SetVerbose

```cpp
void SetVerbose(unsigned int verbose);
unsigned int GetVerbose() const;
```

**Default:** `2`
**Range:** `0-4`

Verbosity level:
- `0`: Silent
- `1`: Summary only
- `2`: Progress every 10 iterations
- `3`: Progress every iteration
- `4`: Debug output

**Output format:**
```
Iter   Primal Res   Dual Res     Gap        rho
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  ...
```

---

### SetAdaptiveRho

```cpp
void SetAdaptiveRho(bool adaptive_rho);
bool GetAdaptiveRho() const;
```

**Default:** `true`

Enable adaptive adjustment of rho based on residual balance.

When enabled, rho is automatically adjusted:
- **Increase rho** if primal residual >> dual residual
- **Decrease rho** if dual residual >> primal residual

**Guidelines:**
- **Enable** (recommended): Better convergence for most problems
- **Disable**: When you want fixed rho or manual control

---

### SetGapStop

```cpp
void SetGapStop(bool gap_stop);
bool GetGapStop() const;
```

**Default:** `false`

Enable duality gap stopping criterion.

Stops when both residuals are small AND duality gap is small.

---

### SetUseAnderson

```cpp
void SetUseAnderson(bool use_anderson);
bool GetUseAnderson() const;
```

**Default:** `false`

Enable Anderson acceleration for faster convergence.

Anderson acceleration can provide up to 2x speedup on well-conditioned problems.

---

### SetAndersonMem

```cpp
void SetAndersonMem(unsigned int mem);
unsigned int GetAndersonMem() const;
```

**Default:** `5`
**Range:** `[1, 20]`

Number of past iterates to store for Anderson acceleration.

Only used when `SetUseAnderson(true)`.

---

### SetAndersonStart

```cpp
void SetAndersonStart(unsigned int start);
unsigned int GetAndersonStart() const;
```

**Default:** `10`
**Range:** `[0, max_iter)`

Iteration number to start applying Anderson acceleration.

Only used when `SetUseAnderson(true)`.

---

### SetInitX

```cpp
void SetInitX(const T* x);
```

Provide an initial guess for the primal variable x (warm start).

**Usage:**
```cpp
double x_init[n] = { /* initial values */ };
solver.SetInitX(x_init);
```

---

### SetInitLambda

```cpp
void SetInitLambda(const T* lambda);
```

Provide an initial guess for the dual variable lambda (warm start).

---

## Default Values Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 1.0 | ADMM penalty parameter |
| `abs_tol` | 1e-4 | Absolute tolerance |
| `rel_tol` | 1e-3 | Relative tolerance |
| `max_iter` | 2500 | Maximum iterations |
| `verbose` | 2 | Verbosity level |
| `adaptive_rho` | true | Enable adaptive penalty |
| `gap_stop` | false | Stop on duality gap |
| `use_anderson` | false | Anderson acceleration |
| `anderson_mem` | 5 | Anderson memory depth |
| `anderson_start` | 10 | Anderson start iteration |

---

## Usage Examples

### Basic Configuration

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

pogs::MatrixDense<double> A('r', m, n, A_data);
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

// Configure
solver.SetAbsTol(1e-4);
solver.SetRelTol(1e-3);
solver.SetMaxIter(1000);
solver.SetVerbose(2);

// Solve
PogsStatus status = solver.Solve(f, g);
```

### High Accuracy

```cpp
solver.SetAbsTol(1e-6);
solver.SetRelTol(1e-5);
solver.SetMaxIter(5000);
solver.SetRho(0.1);
solver.SetAdaptiveRho(true);
```

### Fast Approximate Solution

```cpp
solver.SetAbsTol(1e-2);
solver.SetRelTol(1e-2);
solver.SetMaxIter(100);
solver.SetRho(10.0);
```

### With Anderson Acceleration

```cpp
solver.SetUseAnderson(true);
solver.SetAndersonMem(10);
solver.SetAndersonStart(20);
solver.SetMaxIter(1000);
```

### Warm Starting

```cpp
// First solve
solver.Solve(f, g);
const double* x_solution = solver.GetX();
const double* lambda_solution = solver.GetLambda();

// Modify problem slightly...

// Warm start second solve
double x_init[n], lambda_init[m];
memcpy(x_init, x_solution, n * sizeof(double));
memcpy(lambda_init, lambda_solution, m * sizeof(double));

solver.SetInitX(x_init);
solver.SetInitLambda(lambda_init);
solver.Solve(f_modified, g_modified);
```

---

## Performance Tuning

### Well-Conditioned Problems

```cpp
solver.SetRho(5.0);
solver.SetAdaptiveRho(true);
solver.SetUseAnderson(true);
```

### Ill-Conditioned Problems

```cpp
solver.SetRho(0.1);
solver.SetAdaptiveRho(true);
solver.SetMaxIter(5000);
```

### Large-Scale Problems

```cpp
solver.SetAbsTol(1e-3);
solver.SetRelTol(1e-2);
solver.SetMaxIter(2000);
solver.SetVerbose(1);  // Less output
```

---

## See Also

- [Solver API](solver.md) - Main solver interface
- [Basic Usage](../user-guide/basic-usage.md) - Usage guide
- [Advanced Features](../user-guide/advanced-features.md) - Parameter tuning
