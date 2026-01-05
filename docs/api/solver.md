# Solver API

Reference for the POGS solver classes.

---

## Graph Form Solvers

POGS provides templated solver classes for graph form problems:

$$
\begin{align}
\text{minimize} \quad & f(y) + g(x) \\
\text{subject to} \quad & y = Ax
\end{align}
$$

### PogsDirect

Direct factorization solver (recommended for dense problems).

```cpp
namespace pogs {

template<typename T, typename M>
class PogsDirect {
public:
    explicit PogsDirect(const M& A);
    ~PogsDirect();

    // Solve the optimization problem
    PogsStatus Solve(const std::vector<FunctionObj<T>>& f,
                     const std::vector<FunctionObj<T>>& g);

    // Configuration
    void SetRho(T rho);
    void SetAbsTol(T abs_tol);
    void SetRelTol(T rel_tol);
    void SetMaxIter(unsigned int max_iter);
    void SetVerbose(unsigned int verbose);
    void SetAdaptiveRho(bool adaptive_rho);
    void SetGapStop(bool gap_stop);

    // Results
    T GetOptval() const;
    const T* GetX() const;
    const T* GetY() const;
    const T* GetLambda() const;
    const T* GetMu() const;
    unsigned int GetIter() const;
};

} // namespace pogs
```

### PogsCgls

Iterative solver using CGLS (better for large sparse problems).

```cpp
namespace pogs {

template<typename T, typename M>
class PogsCgls {
public:
    explicit PogsCgls(const M& A);
    // Same interface as PogsDirect
};

} // namespace pogs
```

---

## Cone Form Solvers

For cone form problems:

$$
\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax + s = b, \quad s \in \mathcal{K}
\end{align}
$$

### PogsDirectCone

```cpp
namespace pogs {

template<typename T, typename M>
class PogsDirectCone {
public:
    PogsDirectCone(const M& A,
                   const std::vector<ConeConstraint>& Kx,
                   const std::vector<ConeConstraint>& Ky);

    PogsStatus Solve(const std::vector<T>& b,
                     const std::vector<T>& c);

    // Configuration (same as PogsDirect)
    void SetRho(T rho);
    void SetAbsTol(T abs_tol);
    void SetRelTol(T rel_tol);
    void SetMaxIter(unsigned int max_iter);
    void SetVerbose(unsigned int verbose);
    void SetAdaptiveRho(bool adaptive_rho);
    void SetGapStop(bool gap_stop);

    // Results
    T GetOptval() const;
    const T* GetX() const;
    const T* GetY() const;
};

} // namespace pogs
```

---

## Status Codes

```cpp
namespace pogs {

enum PogsStatus {
    POGS_SUCCESS = 0,      // Converged successfully
    POGS_INFEASIBLE,       // Problem infeasible
    POGS_UNBOUNDED,        // Problem unbounded
    POGS_MAX_ITER,         // Maximum iterations reached
    POGS_NAN_FOUND,        // NaN encountered
    POGS_ERROR             // Other error
};

} // namespace pogs
```

---

## Matrix Classes

### MatrixDense

```cpp
namespace pogs {

template<typename T>
class MatrixDense {
public:
    // Constructors
    MatrixDense(char ord, size_t m, size_t n, const T* data);
    MatrixDense(const MatrixDense& A);

    // Dimensions
    size_t Rows() const;
    size_t Cols() const;

    // Data access
    const T* Data() const;
};

} // namespace pogs
```

**Parameters:**
- `ord`: Matrix order ('r' for row-major, 'c' for column-major)
- `m`: Number of rows
- `n`: Number of columns
- `data`: Pointer to matrix data

### MatrixSparse

```cpp
namespace pogs {

template<typename T>
class MatrixSparse {
public:
    MatrixSparse(char ord, size_t m, size_t n, size_t nnz,
                 const T* data, const int* ptr, const int* ind);

    size_t Rows() const;
    size_t Cols() const;
    size_t Nnz() const;
};

} // namespace pogs
```

---

## Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 1.0 | ADMM penalty parameter |
| `abs_tol` | 1e-4 | Absolute tolerance |
| `rel_tol` | 1e-3 | Relative tolerance |
| `max_iter` | 2500 | Maximum iterations |
| `verbose` | 2 | Verbosity (0=quiet, 1=summary, 2=progress) |
| `adaptive_rho` | true | Enable adaptive penalty |
| `gap_stop` | false | Stop on duality gap |

---

## Usage Examples

### Lasso Regression

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

// Create matrix
pogs::MatrixDense<double> A('r', m, n, A_data);

// Create solver
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

// Configure
solver.SetAbsTol(1e-5);
solver.SetRelTol(1e-4);
solver.SetMaxIter(1000);
solver.SetVerbose(2);

// Define f(y) = 0.5 * ||y - b||^2
std::vector<FunctionObj<double>> f(m);
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].c = 0.5;
    f[i].d = -b[i];
}

// Define g(x) = lambda * ||x||_1
std::vector<FunctionObj<double>> g(n);
for (size_t j = 0; j < n; ++j) {
    g[j].h = kAbs;
    g[j].c = lambda;
}

// Solve
pogs::PogsStatus status = solver.Solve(f, g);

// Get results
if (status == pogs::POGS_SUCCESS) {
    const double* x = solver.GetX();
    double optval = solver.GetOptval();
}
```

### Linear Program (Cone Form)

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

// min c'x s.t. Ax = b, x >= 0
pogs::MatrixDense<double> A('r', m, n, A_data);

// Cone constraints
std::vector<ConeConstraint> Kx = {{kConeNonNeg, {0, 1, 2, ..., n-1}}};
std::vector<ConeConstraint> Ky = {{kConeZero, {0, 1, 2, ..., m-1}}};

// Create solver
pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);

// Solve
pogs::PogsStatus status = solver.Solve(b, c);
```

---

## See Also

- [Function Objects](types.md) - FunctionObj and function types
- [Proximal Operators](proximal.md) - Supported functions
- [Configuration](configuration.md) - Solver parameters
- [C API](c-api.md) - C interface for cone problems
