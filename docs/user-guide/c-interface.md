# C Interface

POGS provides a C interface for maximum compatibility with other languages and systems.

---

## Overview

The C interface supports cone form problems, allowing you to solve:

- Linear programs (LP)
- Quadratic programs (QP)
- Second-order cone programs (SOCP)
- Semidefinite programs (SDP)

---

## Basic Example

```c
#include <pogs/c/pogs_c.h>
#include <stdio.h>

int main() {
    // Problem: minimize x[0] subject to x[0] + x[1] = 2, x >= 0
    double A[] = {1.0, 1.0};  // 1x2 matrix (row-major)
    double b[] = {2.0};       // RHS
    double c[] = {1.0, 0.0};  // Objective coefficients

    // Define cones for variables (x)
    unsigned int x_indices[] = {0, 1};
    struct ConeConstraintC cone_x = {CONE_NON_NEG, x_indices, 2};

    // Define cones for slack variables (y = Ax - b)
    unsigned int y_indices[] = {0};
    struct ConeConstraintC cone_y = {CONE_ZERO, y_indices, 1};

    // Solution arrays
    double x[2], y[1], optval;
    unsigned int final_iter;

    // Solve
    int status = PogsConeD(
        ROW_MAJ,           // Matrix order
        1, 2,              // m, n (dimensions)
        A, b, c,           // Problem data
        &cone_x, 1,        // x cones and count
        &cone_y, 1,        // y cones and count
        1.0,               // rho
        1e-4, 1e-3,        // abs_tol, rel_tol
        10000,             // max_iter
        0,                 // verbose (0=quiet, 1=verbose)
        1, 1,              // adaptive_rho, gap_stop
        x, y, NULL,        // Solutions (lambda can be NULL)
        &optval,           // Optimal value
        &final_iter        // Final iteration count
    );

    if (status == 0) {
        printf("Success!\n");
        printf("Solution: x = [%.6f, %.6f]\n", x[0], x[1]);
        printf("Optimal value: %.6f\n", optval);
        printf("Iterations: %u\n", final_iter);
    } else {
        printf("Solver failed with status %d\n", status);
    }

    return 0;
}
```

---

## API Reference

### Main Function

```c
int PogsConeD(
    enum ORD ord,
    size_t m, size_t n,
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

**Parameters:**

- `ord`: Matrix order (`ROW_MAJ` or `COL_MAJ`)
- `m`, `n`: Problem dimensions
- `A`: Constraint matrix (size m×n)
- `b`: RHS vector (size m)
- `c`: Objective vector (size n)
- `cones_x`: Array of cone constraints for x
- `num_cones_x`: Number of x cone constraints
- `cones_y`: Array of cone constraints for y
- `num_cones_y`: Number of y cone constraints
- `rho`: Penalty parameter (try 1.0)
- `abs_tol`: Absolute tolerance (try 1e-4)
- `rel_tol`: Relative tolerance (try 1e-3)
- `max_iter`: Maximum iterations (try 10000)
- `verbose`: Verbosity level (0=quiet, 1=verbose)
- `adaptive_rho`: Enable adaptive ρ (1=yes, 0=no)
- `gap_stop`: Enable duality gap stopping (1=yes, 0=no)
- `x`: [Output] Primal solution (size n)
- `y`: [Output] Slack variables (size m)
- `l`: [Output] Dual variables (size m, can be NULL)
- `optval`: [Output] Optimal objective value
- `final_iter`: [Output] Final iteration count

**Returns:** 0 on success, non-zero on failure

---

## Cone Types

### Available Cones

```c
enum Cone {
    CONE_ZERO,        // {x : x = 0} - Equality constraints
    CONE_NON_NEG,     // {x : x >= 0} - Non-negativity
    CONE_NON_POS,     // {x : x <= 0} - Non-positivity
    CONE_SOC,         // {(t,x) : ||x||_2 <= t} - Second-order cone
    CONE_SDP,         // {X : X ⪰ 0} - Semidefinite cone
    CONE_EXP_PRIMAL,  // Exponential cone
    CONE_EXP_DUAL     // Dual exponential cone
};
```

### Cone Constraint Structure

```c
struct ConeConstraintC {
    enum Cone cone;           // Cone type
    unsigned int *indices;    // Indices of variables in this cone
    size_t size;              // Number of variables
};
```

---

## Matrix Ordering

### Row-Major (C-style)

```c
// A = [1 2 3]
//     [4 5 6]
double A[] = {1, 2, 3, 4, 5, 6};  // Row-major
enum ORD ord = ROW_MAJ;
```

Element `A[i,j]` is at index `i*n + j`.

### Column-Major (Fortran-style)

```c
// A = [1 2 3]
//     [4 5 6]
double A[] = {1, 4, 2, 5, 3, 6};  // Column-major
enum ORD ord = COL_MAJ;
```

Element `A[i,j]` is at index `j*m + i`.

---

## Examples

### Linear Program

Solve:
$$
\begin{align}
\text{minimize} \quad & x_1 + 2x_2 \\
\text{subject to} \quad & x_1 + x_2 = 3 \\
& x \geq 0
\end{align}
$$

```c
double A[] = {1.0, 1.0};
double b[] = {3.0};
double c[] = {1.0, 2.0};

unsigned int x_idx[] = {0, 1};
struct ConeConstraintC cx = {CONE_NON_NEG, x_idx, 2};

unsigned int y_idx[] = {0};
struct ConeConstraintC cy = {CONE_ZERO, y_idx, 1};

double x[2], y[1], opt;
unsigned int iter;

int status = PogsConeD(ROW_MAJ, 1, 2, A, b, c,
                       &cx, 1, &cy, 1,
                       1.0, 1e-4, 1e-3, 10000, 0, 1, 1,
                       x, y, NULL, &opt, &iter);
```

### Quadratic Program

For quadratic objectives, reformulate using auxiliary variables.

---

## Compiling

### GCC/Clang (macOS)

```bash
gcc -o myprogram myprogram.c \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -framework Accelerate \
    -lm
```

### GCC (Linux)

```bash
gcc -o myprogram myprogram.c \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -lopenblas \
    -llapack \
    -lm
```

### CMake Integration

```cmake
find_package(POGS REQUIRED)

add_executable(myprogram myprogram.c)
target_link_libraries(myprogram PRIVATE pogs::cpu)
```

---

## Error Handling

```c
int status = PogsConeD(...);

if (status == 0) {
    printf("Success: converged\n");
} else if (status == 1) {
    printf("Warning: maximum iterations reached\n");
} else if (status == 2) {
    printf("Error: numerical error\n");
} else {
    printf("Error: unknown status %d\n", status);
}
```

---

## Float Precision

Single precision version:

```c
int PogsConeF(
    enum ORD ord,
    size_t m, size_t n,
    const float *A,
    const float *b,
    const float *c,
    // ... same parameters but with float ...
);
```

Use `PogsConeF` for faster but less accurate solutions.

---

## See Also

- [Cone Problems](cone-problems.md) - Cone formulation details
- [API Reference](../api/c-api.md) - Complete C API documentation
- Examples: `examples/cpp_cone/test_c_interface.c`
