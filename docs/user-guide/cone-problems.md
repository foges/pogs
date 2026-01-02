# Cone Form Problems

POGS supports cone form problems, which provide a unified framework for expressing linear programs, quadratic programs, second-order cone programs, and semidefinite programs.

---

## Standard Cone Form

A cone form problem has the structure:

$$
\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax + s = b \\
& s \in \mathcal{K}
\end{align}
$$

where $\mathcal{K}$ is a product of convex cones (zero, non-negative, SOC, SDP, exponential).

---

## Supported Cone Types

### Zero Cone

$$
\mathcal{K}_\text{zero} = \{x : x = 0\}
$$

Used for **equality constraints**.

**Example**: $Ax = b$ becomes $Ax - b \in \mathcal{K}_\text{zero}$

### Non-Negative Cone

$$
\mathcal{K}_+ = \{x : x \geq 0\}
$$

Used for **inequality constraints**.

**Example**: $x \geq 0$ becomes $x \in \mathcal{K}_+$

### Non-Positive Cone

$$
\mathcal{K}_- = \{x : x \leq 0\}
$$

Used for **upper bound constraints**.

### Second-Order Cone (SOC)

$$
\mathcal{K}_\text{SOC} = \{(t, x) : \|x\|_2 \leq t\}
$$

Used for **quadratic constraints**.

**Example**: $\|x\|_2 \leq t$ becomes $(t, x) \in \mathcal{K}_\text{SOC}$

### Semidefinite Cone (SDP)

$$
\mathcal{K}_\text{SDP} = \{X : X \succeq 0\}
$$

where $X \succeq 0$ means $X$ is positive semidefinite.

Used for **matrix constraints**.

**Example**: Matrix $X$ must be PSD becomes $X \in \mathcal{K}_\text{SDP}$

### Exponential Cone

$$
\mathcal{K}_\text{exp} = \{(x, y, z) : y > 0, ye^{x/y} \leq z\}
$$

Used for **exponential and logarithmic constraints**.

---

## C Interface

### Basic Usage

```c
#include <pogs/c/pogs_c.h>

// Problem data
double A[] = {1.0, 1.0};  // 1x2 matrix
double b[] = {2.0};       // m=1
double c[] = {1.0, 0.0};  // n=2

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
    &cone_x, 1,        // x cones
    &cone_y, 1,        // y cones
    1.0,               // rho
    1e-4, 1e-3,        // abs_tol, rel_tol
    10000,             // max_iter
    0,                 // verbose
    1, 1,              // adaptive_rho, gap_stop
    x, y, NULL,        // Solutions
    &optval,           // Optimal value
    &final_iter        // Iterations
);

printf("Status: %d\n", status);
printf("Solution: x = [%.6f, %.6f]\n", x[0], x[1]);
printf("Optimal value: %.6f\n", optval);
```

### Cone Types in C

```c
enum Cone {
    CONE_ZERO,        // {x : x = 0}
    CONE_NON_NEG,     // {x : x >= 0}
    CONE_NON_POS,     // {x : x <= 0}
    CONE_SOC,         // {(p,x) : ||x||₂ ≤ p}
    CONE_SDP,         // {X : X ⪰ 0}
    CONE_EXP_PRIMAL,  // Exponential cone
    CONE_EXP_DUAL     // Dual exponential cone
};
```

---

## Examples

### Linear Program

Solve:
$$
\begin{align}
\text{minimize} \quad & x_1 \\
\text{subject to} \quad & x_1 + x_2 = 2 \\
& x \geq 0
\end{align}
$$

**Solution**: $x = [0, 2]^T$, optimal value = 0

See `examples/cpp_cone/test_c_interface.c` for complete code.

### Quadratic Program

Solve:
$$
\text{minimize} \quad \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1
$$

This can be reformulated in cone form using auxiliary variables.

### SDP Problem

Solve:
$$
\begin{align}
\text{minimize} \quad & \text{trace}(CX) \\
\text{subject to} \quad & \text{trace}(A_i X) = b_i, \quad i=1,\ldots,m \\
& X \succeq 0
\end{align}
$$

See `examples/cpp_cone/test_sdp.cpp` for implementation.

---

## Implementation Details

### SDP Cone Projection

POGS implements SDP cone projection using eigenvalue decomposition:

1. Compute eigenvalue decomposition: $X = V\Lambda V^T$
2. Project eigenvalues onto non-negative orthant: $\Lambda_+ = \max(\Lambda, 0)$
3. Reconstruct: $X_+ = V\Lambda_+ V^T$

This uses LAPACK routines (`dsyevd` for double precision) for numerical stability.

### Sparse Matrix Support

The cone interface supports both dense and sparse matrices. For sparse problems, use the sparse matrix format for better performance.

---

## Performance Notes

- **Dense problems**: POGS uses optimized BLAS/LAPACK routines
- **Sparse problems**: CSR/CSC formats supported
- **SDP cones**: CPU-only (GPU support planned)
- **Typical convergence**: 100-1000 iterations

---

## See Also

- [C API Reference](../api/c-api.md) - Complete C interface documentation
- [CVXPY Integration](cvxpy-integration.md) - Using POGS with CVXPY
- [Examples](../examples/sdp.md) - SDP problem examples
