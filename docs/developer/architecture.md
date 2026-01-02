# POGS Architecture

Technical overview of POGS architecture and design.

---

## Overview

POGS is built around the **Alternating Direction Method of Multipliers** (ADMM) algorithm for solving convex optimization problems.

---

## ADMM Algorithm

POGS solves problems in graph form:

$$
\begin{align}
\text{minimize} \quad & f(y) + g(x) \\
\text{subject to} \quad & y = Ax
\end{align}
$$

where $f$ and $g$ are convex, separable functions.

### Algorithm Steps

The ADMM algorithm iterates:

1. **x-update** (proximal operator for g):
   $$
   x^{k+1} = \text{prox}_{g,\rho}(x^k - A^T\lambda^k)
   $$

2. **y-update** (proximal operator for f):
   $$
   y^{k+1} = \text{prox}_{f,\rho}(Ax^{k+1} + \lambda^k/\rho)
   $$

3. **Dual update**:
   $$
   \lambda^{k+1} = \lambda^k + \rho(Ax^{k+1} - y^{k+1})
   $$

### Convergence

Stops when both primal and dual residuals are small:

- **Primal residual**: $r = Ax - y$
- **Dual residual**: $s = \rho A^T(y^{k+1} - y^k)$

---

## Code Structure

### High-Level Organization

```
src/
├── common/           # Shared code
│   ├── admm_state.hpp     # RAII memory management
│   └── admm_core.hpp      # Common ADMM logic (future)
├── cpu/              # CPU implementation
│   ├── pogs.cpp           # Main ADMM solver
│   ├── projector.cu       # Projection operations
│   └── include/
│       ├── gsl/           # GSL wrappers (BLAS/LAPACK)
│       └── anderson.h     # Anderson acceleration (experimental)
├── gpu/              # GPU implementation
│   ├── pogs.cu            # GPU ADMM solver
│   └── kernels.cuh        # CUDA kernels
└── interface_c/      # C interface
    ├── pogs_c.h
    └── pogs_c.cpp
```

### Key Classes

**Modern C++20 API** (include/pogs/):
```cpp
template<typename T, Matrix M>
class Solver {
    std::unique_ptr<Impl> impl_;  // PIMPL pattern
};
```

**Legacy API** (src/cpu/pogs.cpp):
```cpp
template <typename T, typename M, typename P>
class Pogs {
    // Direct ADMM implementation
};
```

---

## Matrix Abstraction

### Matrix Interface

```cpp
template<typename M>
concept Matrix = requires(M m) {
    { m.rows() } -> std::convertible_to<size_t>;
    { m.cols() } -> std::convertible_to<size_t>;
};
```

### Implementations

**Dense Matrix** (src/include/matrix/matrix_dense.h):
- Column-major or row-major storage
- BLAS/LAPACK operations
- O(mn) memory

**Sparse Matrix** (src/include/matrix/matrix_sparse.h):
- CSR/CSC format
- Sparse matrix-vector products
- O(nnz) memory

---

## Proximal Operators

### Function Library

Location: `src/include/prox_lib.h`

Each function type has a proximal operator:

```cpp
template<typename T>
void Prox(const FunctionObj<T> *f,
          T rho,
          T *x,
          size_t n);
```

### Supported Functions

- **Indicators**: IndEq0, IndGe0, IndLe0, IndBox01
- **Norms**: Abs, Square
- **Nonlinear**: Logistic, Huber, Exp, NegLog
- **Other**: Identity, Zero, MaxPos0, MaxNeg0, Recipr, NegEntr

### Implementation Pattern

```cpp
template<typename T>
inline void ProxAbs(T rho, T *x) {
    // Soft thresholding
    T threshold = 1.0 / rho;
    if (*x > threshold) {
        *x -= threshold;
    } else if (*x < -threshold) {
        *x += threshold;
    } else {
        *x = 0;
    }
}
```

---

## Cone Projections

### Cone Library

Location: `src/include/prox_lib_cone.h`

Projections onto convex cones:

```cpp
template<typename T>
void ProxCone(const ConeConstraint<T> *cone,
              T *x,
              size_t size);
```

### Supported Cones

- **Zero**: $\{x : x = 0\}$ → projection: $x \gets 0$
- **Non-negative**: $\{x : x \geq 0\}$ → projection: $x \gets \max(x, 0)$
- **SOC**: $\{(t,x) : \|x\|_2 \leq t\}$ → analytical formula
- **SDP**: $\{X : X \succeq 0\}$ → eigenvalue decomposition

### SDP Projection

Algorithm (lines 144-230 in prox_lib_cone.h):

```cpp
1. Compute eigenvalue decomposition: X = V*Λ*V^T
2. Project eigenvalues: Λ_+ = max(Λ, 0)
3. Reconstruct: X_+ = V*Λ_+*V^T
```

Uses LAPACK: `dsyevd` (double) or `ssyevd` (float).

---

## Linear Algebra Backend

### GSL Wrapper

Location: `src/cpu/include/gsl/gsl_linalg.h`

Provides unified interface to BLAS/LAPACK:

- **Matrix-vector**: `gemv`
- **Matrix-matrix**: `gemm`
- **Eigenvalues**: `syevd`
- **QR decomposition**: `geqrf`, `orgqr`

### Platform-Specific

**macOS**: Uses Accelerate framework
**Linux**: Uses OpenBLAS or ATLAS
**Windows**: Uses Intel MKL or OpenBLAS

---

## Memory Management

### Modern C++20 (include/pogs/)

**RAII Everywhere**:
```cpp
class ADMMState {
    std::vector<T> x_, y_, mu_, lambda_;  // Automatic cleanup
};
```

**Smart Pointers**:
```cpp
auto solver = pogs::make_solver<double>(std::move(matrix));
// Automatic cleanup when solver goes out of scope
```

### Legacy Code (src/cpu/)

**Manual Management** (being modernized):
```cpp
Pogs() {
    _x = new T[n]();
    _y = new T[m]();
}

~Pogs() {
    delete[] _x;
    delete[] _y;
}
```

---

## Adaptive Parameters

### Adaptive ρ

Adjusts penalty parameter based on residual balance:

```cpp
if (primal_residual > 10 * dual_residual) {
    rho *= 2.0;  // Increase ρ
} else if (dual_residual > 10 * primal_residual) {
    rho /= 2.0;  // Decrease ρ
}
```

Prevents oscillation and improves convergence.

### Over-Relaxation

Uses relaxation parameter α = 1.7:

```cpp
x_relaxed = alpha * x + (1 - alpha) * x_prev;
```

Can speed up convergence for some problems.

---

## GPU Implementation

### CUDA Kernels

Location: `src/gpu/kernels.cuh`

Parallel proximal operators:

```cpp
__global__ void prox_abs_kernel(T *x, T rho, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Soft thresholding
        ProxAbs(rho, &x[idx]);
    }
}
```

### cuBLAS/cuSOLVER

Uses NVIDIA libraries for linear algebra:
- **cuBLAS**: Matrix operations
- **cuSOLVER**: Eigenvalue decomposition (future)

---

## Build System

### CMake Structure

```cmake
pogs/
├── CMakeLists.txt           # Root
├── src/CMakeLists.txt       # Library
├── tests/CMakeLists.txt     # Tests
└── examples/CMakeLists.txt  # Examples
```

### Targets

- **pogs::cpu**: CPU solver library
- **pogs::gpu**: GPU solver library (optional)
- **test_cone**: CPU cone solver tests
- **test_sdp**: SDP cone projection tests

---

## Testing

### Test Files

- `tests/test_cone.cpp`: Cone projection tests
- `tests/test_sdp.cpp`: SDP-specific tests
- `examples/`: Integration tests

### Running Tests

```bash
cd build
ctest --output-on-failure
```

---

## Future Improvements

### Phase 3+ (Planned)

1. **Code Deduplication**: Extract common ADMM core from CPU/GPU
2. **Policy-Based Design**: Backend traits for CPU/GPU
3. **Concepts**: Template constraints for type safety
4. **std::span**: Safe array views throughout
5. **Coroutines**: Async solver interface (C++20)

### Research Features

1. **Anderson Acceleration**: Improve integration with ADMM
2. **Warm Starting**: Better initialization strategies
3. **Problem Detection**: Automatic parameter selection
4. **Preconditioning**: Improve conditioning

---

## See Also

- [Building from Source](building.md) - Build instructions
- [Contributing](contributing.md) - How to contribute
- [Modernization](modernization.md) - Modernization progress
