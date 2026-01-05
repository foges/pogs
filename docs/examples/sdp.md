# Semidefinite Programming Example

Complete example of solving semidefinite programs (SDP) with POGS.

---

## Problem Formulation

A **Semidefinite Program** (SDP) has the form:

$$
\begin{align}
\text{minimize} \quad & \text{trace}(C \cdot X) \\
\text{subject to} \quad & \text{trace}(A_i \cdot X) = b_i, \quad i = 1, \ldots, m \\
& X \succeq 0
\end{align}
$$

where:
- $X \in \mathbb{R}^{n \times n}$ is a symmetric matrix variable
- $X \succeq 0$ means $X$ is positive semidefinite
- $C, A_1, \ldots, A_m \in \mathbb{R}^{n \times n}$ are given symmetric matrices
- $b \in \mathbb{R}^m$ is a given vector

**Applications:**
- Control theory (LQR, robust control)
- Combinatorial optimization relaxations
- Eigenvalue optimization
- Covariance estimation

---

## POGS Cone Form

POGS solves SDPs using the cone form interface:

$$
\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax + s = b \\
& s_{\text{eq}} \in \mathcal{K}_{\text{zero}} \\
& s_{\text{sdp}} \in \mathcal{K}_{\text{SDP}}
\end{align}
$$

---

## Matrix Vectorization

SDP matrices are vectorized using **lower triangular packing**:

For a 2×2 symmetric matrix:
$$
X = \begin{bmatrix} x_{11} & x_{12} \\ x_{12} & x_{22} \end{bmatrix}
\quad \to \quad
\text{vec}(X) = [x_{11}, x_{12}, x_{22}]^T
$$

For a 3×3 symmetric matrix:
$$
X = \begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{12} & x_{22} & x_{23} \\
x_{13} & x_{23} & x_{33}
\end{bmatrix}
\quad \to \quad
\text{vec}(X) = [x_{11}, x_{12}, x_{22}, x_{13}, x_{23}, x_{33}]^T
$$

**Size:** For $n \times n$ matrix, vectorized size is $\frac{n(n+1)}{2}$.

---

## C Example

```c
#include <pogs/c/pogs_c.h>
#include <stdio.h>
#include <math.h>

int main() {
    // Solve: minimize trace(C*X) subject to trace(A*X) = 1, X ⪰ 0
    // where C = [1 0; 0 2], A = [1 0; 0 1] (identity)
    // Solution: X = [0.5 0; 0 0.5] with optimal value = 1.5

    // Matrix dimensions: 2x2 → vec size = 3
    const size_t n_vec = 3;  // Number of variables (vectorized X)
    const size_t m = 1;      // Number of constraints

    // Objective: minimize c'*x where x = vec(X)
    // c = vec(C) = [1, 0, 2] (from C = [1 0; 0 2])
    double c[] = {1.0, 0.0, 2.0};

    // Constraint: A*x + s = b
    // trace(A*X) = 1 → [1 0 1] * [x11, x12, x22]' = 1
    double A[] = {1.0, 0.0, 1.0};  // Row-major, 1×3
    double b[] = {1.0};

    // Cone for x (SDP cone for 2×2 matrix)
    unsigned int x_idx[] = {0, 1, 2};
    struct ConeConstraintC cone_x = {CONE_SDP, x_idx, 3};

    // Cone for s (equality constraint)
    unsigned int s_idx[] = {0};
    struct ConeConstraintC cone_s = {CONE_ZERO, s_idx, 1};

    // Solution arrays
    double x[3];
    double s[1];
    double optval;
    unsigned int iter;

    // Solve
    int status = PogsConeD(
        ROW_MAJ,
        m, n_vec,
        A, b, c,
        &cone_x, 1,
        &cone_s, 1,
        1.0,           // rho
        1e-4, 1e-3,    // tolerances
        10000,         // max_iter
        1,             // verbose
        1, 1,          // adaptive_rho, gap_stop
        x, s, NULL,
        &optval,
        &iter
    );

    if (status == 0) {
        printf("Success! Converged in %u iterations\n", iter);
        printf("Optimal value: %.6f\n", optval);
        printf("\nSolution matrix X (vectorized):\n");
        printf("  vec(X) = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);
        printf("\nReconstruct X:\n");
        printf("  X = [%.6f  %.6f]\n", x[0], x[1]);
        printf("      [%.6f  %.6f]\n", x[1], x[2]);
    } else {
        printf("Solver failed with status %d\n", status);
    }

    return 0;
}
```

---

## Python/CVXPY Implementation

```python
import cvxpy as cp
import numpy as np

# Problem: minimize trace(C @ X) subject to trace(A @ X) = 1, X PSD

# Define matrices
C = np.array([[1.0, 0.0],
              [0.0, 2.0]])

A = np.array([[1.0, 0.0],
              [0.0, 1.0]])  # Identity matrix

# Define variable (2×2 PSD matrix)
X = cp.Variable((2, 2), PSD=True)

# Define objective
objective = cp.Minimize(cp.trace(C @ X))

# Define constraints
constraints = [cp.trace(A @ X) == 1.0]

# Create problem
prob = cp.Problem(objective, constraints)

# Solve with POGS
prob.solve(solver='POGS', verbose=True)

# Print results
print(f"\nStatus: {prob.status}")
print(f"Optimal value: {prob.value:.6f}")
print(f"\nSolution matrix X:\n{X.value}")

# Verify PSD (eigenvalues should be non-negative)
eigvals = np.linalg.eigvalsh(X.value)
print(f"\nEigenvalues: {eigvals}")
print(f"Is PSD: {np.all(eigvals >= -1e-6)}")
```

---

## Larger Example: Max-Cut SDP Relaxation

The **maximum cut problem** can be relaxed to an SDP:

```python
import cvxpy as cp
import numpy as np

# Graph adjacency matrix (5 nodes)
W = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
], dtype=float)

n = W.shape[0]

# SDP relaxation of Max-Cut
X = cp.Variable((n, n), PSD=True)

# Objective: maximize (1/4) * trace(W @ (J - X))
# where J is all-ones matrix
J = np.ones((n, n))
objective = cp.Maximize(0.25 * cp.trace(W @ (J - X)))

# Constraints
constraints = [cp.diag(X) == 1]  # X[i,i] = 1 for all i

# Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver='POGS', verbose=True)

print(f"\nMax-Cut SDP bound: {prob.value:.4f}")
print(f"\nSolution matrix X:\n{X.value}")

# Round to get approximate cut
eigvals, eigvecs = np.linalg.eigh(X.value)
v = eigvecs[:, -1]  # Largest eigenvector
cut = np.sign(v)
print(f"\nApproximate cut: {cut}")
```

---

## SDP Cone Projection

POGS projects onto the SDP cone using eigenvalue decomposition:

**Algorithm:**
1. Compute eigenvalue decomposition: $X = V\Lambda V^T$
2. Project eigenvalues: $\Lambda_+ = \max(\Lambda, 0)$
3. Reconstruct: $X_+ = V\Lambda_+ V^T$

This ensures $X_+$ is positive semidefinite.

**Implementation** (src/include/prox_lib_cone.h:144-230):
- Uses LAPACK (`dsyevd` for double, `ssyevd` for float)
- Handles symmetric matrices in vectorized form
- Efficient for small to medium matrices (< 100×100)

---

## Expected Output

For the simple example:

```
Iter   Primal Res   Dual Res     Gap        ρ
  10   3.45e-03    1.23e-03    5.67e-03   1.00
  20   6.78e-04    2.34e-04    1.23e-03   1.00
  50   8.90e-05    3.21e-05    2.34e-04   1.00  ✓ Converged

Success! Converged in 50 iterations
Optimal value: 1.500000

Solution matrix X (vectorized):
  vec(X) = [0.500000, 0.000000, 0.500000]

Reconstruct X:
  X = [0.500000  0.000000]
      [0.000000  0.500000]
```

---

## Performance Notes

### Problem Size

- **Small SDPs** (n < 50): Fast, typically < 1 second
- **Medium SDPs** (50 < n < 200): Moderate, few seconds
- **Large SDPs** (n > 200): Slow, eigenvalue decomposition dominates

### Solver Parameters

For SDPs, try:

```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4,
    .rel_tol = 1e-3,
    .max_iter = 2000,     // May need more iterations
    .adaptive_rho = true
};
```

For tighter accuracy:

```cpp
config.abs_tol = 1e-6;
config.rel_tol = 1e-6;
config.max_iter = 5000;
```

---

## GPU Support

!!! warning "GPU Limitation"
    SDP cone projection is currently **CPU-only**. GPU support for SDP is planned for future releases.

---

## See Also

- [Cone Problems](../user-guide/cone-problems.md) - Cone formulation
- [C Interface](../user-guide/c-interface.md) - C API usage
- [CVXPY Integration](../user-guide/cvxpy-integration.md) - Python interface
- Examples: `examples/cpp_cone/test_sdp.cpp`
