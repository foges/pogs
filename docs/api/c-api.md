# C API Reference

Complete reference for the POGS C interface.

---

## Main Functions

### PogsConeD

Solve cone form problem with double precision.

```c
int PogsConeD(
    enum ORD ord,
    size_t m,
    size_t n,
    const double *A,
    const double *b,
    const double *c,
    const struct ConeConstraintC *cones_x,
    size_t num_cones_x,
    const struct ConeConstraintC *cones_y,
    size_t num_cones_y,
    double rho,
    double abs_tol,
    double rel_tol,
    unsigned int max_iter,
    unsigned int verbose,
    int adaptive_rho,
    int gap_stop,
    double *x,
    double *y,
    double *l,
    double *optval,
    unsigned int *final_iter
);
```

**Returns:** Status code (0 = success)

---

### PogsConeF

Solve cone form problem with single precision.

```c
int PogsConeF(
    enum ORD ord,
    size_t m,
    size_t n,
    const float *A,
    const float *b,
    const float *c,
    const struct ConeConstraintC *cones_x,
    size_t num_cones_x,
    const struct ConeConstraintC *cones_y,
    size_t num_cones_y,
    float rho,
    float abs_tol,
    float rel_tol,
    unsigned int max_iter,
    unsigned int verbose,
    int adaptive_rho,
    int gap_stop,
    float *x,
    float *y,
    float *l,
    float *optval,
    unsigned int *final_iter
);
```

**Returns:** Status code (0 = success)

---

## Types and Enumerations

### Cone

```c
enum Cone {
    CONE_ZERO,        // Equality: x = 0
    CONE_NON_NEG,     // Non-negativity: x >= 0
    CONE_NON_POS,     // Non-positivity: x <= 0
    CONE_SOC,         // Second-order cone
    CONE_SDP,         // Semidefinite cone
    CONE_EXP_PRIMAL,  // Exponential cone
    CONE_EXP_DUAL     // Dual exponential cone
};
```

---

### ConeConstraintC

```c
struct ConeConstraintC {
    enum Cone cone;           // Cone type
    unsigned int *indices;    // Variable indices
    size_t size;              // Number of variables
};
```

**Example:**
```c
unsigned int idx[] = {0, 1, 2};
struct ConeConstraintC constraint = {CONE_NON_NEG, idx, 3};
```

---

### ORD

```c
enum ORD {
    ROW_MAJ,  // Row-major (C-style)
    COL_MAJ   // Column-major (Fortran-style)
};
```

---

## Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `ord` | `enum ORD` | Matrix storage order |
| `m` | `size_t` | Number of constraints |
| `n` | `size_t` | Number of variables |
| `A` | `const T*` | Constraint matrix (m×n) |
| `b` | `const T*` | RHS vector (m) |
| `c` | `const T*` | Objective coefficients (n) |
| `cones_x` | `const struct ConeConstraintC*` | Cones for x |
| `num_cones_x` | `size_t` | Number of x cones |
| `cones_y` | `const struct ConeConstraintC*` | Cones for y |
| `num_cones_y` | `size_t` | Number of y cones |
| `rho` | `T` | Penalty parameter |
| `abs_tol` | `T` | Absolute tolerance |
| `rel_tol` | `T` | Relative tolerance |
| `max_iter` | `unsigned int` | Maximum iterations |
| `verbose` | `unsigned int` | Verbosity (0=quiet, 1=verbose) |
| `adaptive_rho` | `int` | Enable adaptive ρ (1=yes, 0=no) |
| `gap_stop` | `int` | Enable gap stopping (1=yes, 0=no) |
| `x` | `T*` | [Output] Primal solution (n) |
| `y` | `T*` | [Output] Slack variables (m) |
| `l` | `T*` | [Output] Dual variables (m, can be NULL) |
| `optval` | `T*` | [Output] Optimal value |
| `final_iter` | `unsigned int*` | [Output] Final iteration count |

---

## Return Codes

| Code | Meaning |
|------|---------|
| 0 | Success (converged) |
| 1 | Maximum iterations reached |
| 2 | Numerical error |
| 3 | Infeasible or unbounded |

---

## Complete Example

```c
#include <pogs/c/pogs_c.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Problem dimensions
    const size_t m = 2;  // 2 constraints
    const size_t n = 3;  // 3 variables

    // Problem data: minimize c'x s.t. Ax = b, x >= 0
    double A[] = {
        1.0, 1.0, 1.0,   // First constraint
        1.0, 2.0, 3.0    // Second constraint
    };
    double b[] = {3.0, 6.0};
    double c[] = {1.0, 1.0, 1.0};

    // Cone for x (non-negativity)
    unsigned int x_idx[] = {0, 1, 2};
    struct ConeConstraintC cone_x = {CONE_NON_NEG, x_idx, 3};

    // Cone for y (equality)
    unsigned int y_idx[] = {0, 1};
    struct ConeConstraintC cone_y = {CONE_ZERO, y_idx, 2};

    // Solution arrays
    double x[3];
    double y[2];
    double optval;
    unsigned int iter;

    // Solve
    int status = PogsConeD(
        ROW_MAJ,       // Row-major order
        m, n,          // Dimensions
        A, b, c,       // Problem data
        &cone_x, 1,    // x cones
        &cone_y, 1,    // y cones
        1.0,           // rho
        1e-4, 1e-3,    // Tolerances
        10000,         // max_iter
        1,             // verbose
        1, 1,          // adaptive_rho, gap_stop
        x, y, NULL,    // Solutions
        &optval,       // Optimal value
        &iter          // Iterations
    );

    // Check result
    if (status == 0) {
        printf("Success! Converged in %u iterations\n", iter);
        printf("Optimal value: %.6f\n", optval);
        printf("Solution: x = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);
    } else {
        printf("Solver failed with status %d\n", status);
    }

    return 0;
}
```

---

## Compilation

### GCC/Clang (macOS)

```bash
gcc -o myprogram myprogram.c \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -framework Accelerate
```

### GCC (Linux)

```bash
gcc -o myprogram myprogram.c \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -lopenblas \
    -llapack
```

### CMake

```cmake
find_package(POGS REQUIRED)
add_executable(myprogram myprogram.c)
target_link_libraries(myprogram PRIVATE pogs::cpu)
```

---

## Memory Management

### Allocation

Caller is responsible for allocating output arrays:

```c
double *x = (double*)malloc(n * sizeof(double));
double *y = (double*)malloc(m * sizeof(double));

PogsConeD(..., x, y, NULL, ...);

free(x);
free(y);
```

### Matrix Storage

**Row-major** (default in C):
```c
// A[i,j] at index i*n + j
double A[m*n];
for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
        A[i*n + j] = value;
    }
}
```

**Column-major** (Fortran-style):
```c
// A[i,j] at index j*m + i
double A[m*n];
for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
        A[j*m + i] = value;
    }
}
```

---

## Error Handling

```c
int status = PogsConeD(...);

switch (status) {
    case 0:
        printf("Converged successfully\n");
        break;
    case 1:
        fprintf(stderr, "Warning: Max iterations reached\n");
        break;
    case 2:
        fprintf(stderr, "Error: Numerical error\n");
        return 1;
    case 3:
        fprintf(stderr, "Error: Infeasible or unbounded\n");
        return 1;
    default:
        fprintf(stderr, "Error: Unknown status %d\n", status);
        return 1;
}
```

---

## Thread Safety

- **Single solver call:** Thread-safe
- **Multiple simultaneous calls:** Safe with separate data
- **Shared data:** Not thread-safe (use locking)

---

## See Also

- [C Interface Guide](../user-guide/c-interface.md) - Usage guide
- [Cone Problems](../user-guide/cone-problems.md) - Cone formulation
- Examples: `examples/cpp_cone/test_c_interface.c`
