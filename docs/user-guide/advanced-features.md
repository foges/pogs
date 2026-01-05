# Advanced Features

This guide covers advanced POGS features for power users.

---

## Warm Starting

Warm starting can significantly reduce solve time when solving a sequence of related problems.

### Basic Warm Start

```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

pogs::MatrixDense<double> A('r', m, n, A_data);
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

// Solve first problem
PogsStatus status1 = solver.Solve(f1, g1);

// Get solution
const double* x_sol = solver.GetX();
const double* lambda_sol = solver.GetLambda();

// Copy for warm start
std::vector<double> x_init(x_sol, x_sol + n);
std::vector<double> lambda_init(lambda_sol, lambda_sol + m);

// Warm start for next problem
solver.SetInitX(x_init.data());
solver.SetInitLambda(lambda_init.data());

// Solve similar problem (faster!)
PogsStatus status2 = solver.Solve(f2, g2);
```

### Use Cases

- **Parameter sweeps**: Solving problems with varying lambda
- **Online optimization**: Updating solutions as new data arrives
- **Iterative refinement**: Solving with increasing accuracy

---

## Custom Penalty Parameter (rho)

The penalty parameter rho controls the ADMM convergence behavior.

### Manual rho Selection

```cpp
solver.SetRho(10.0);           // Larger for faster convergence
solver.SetAdaptiveRho(false);  // Disable adaptive adjustment
```

### When to Adjust rho

**Increase rho** (larger values like 5.0-100.0):
- When primal residual >> dual residual
- For well-conditioned problems
- When you want faster convergence (may sacrifice accuracy)

**Decrease rho** (smaller values like 0.01-0.5):
- When dual residual >> primal residual
- For ill-conditioned problems
- When you need high accuracy

### Adaptive rho (Recommended)

```cpp
solver.SetRho(1.0);
solver.SetAdaptiveRho(true);  // Automatically adjust rho
```

The solver will automatically increase/decrease rho based on residual balance.

---

## Anderson Acceleration

Anderson acceleration can speed up convergence by 20-50% on well-conditioned problems.

### Enabling Anderson Acceleration

```cpp
solver.SetUseAnderson(true);
solver.SetAndersonMem(5);     // Number of past iterates to store
solver.SetAndersonStart(10);  // Start after 10 iterations
```

### When to Use

- Well-conditioned problems
- Problems that converge slowly without it
- When iterations are the bottleneck (not matrix operations)

### When to Avoid

- Ill-conditioned problems (may be unstable)
- When each iteration is expensive
- Very easy problems (overhead not worth it)

---

## Function Parameterization

### General Function Form

Each function has the form:

$$
c \cdot h(ax - b) + d \cdot x + e \cdot x^2
$$

where:
- `a`: Input scaling
- `b`: Input shift
- `c`: Output scaling
- `d`: Linear term coefficient
- `e`: Quadratic term coefficient
- `h`: Base function type

### Example: Scaled Huber Loss

```cpp
FunctionObj<double> f;
f.h = kHuber;
f.a = 2.0;     // Scale input
f.c = 0.5;     // Scale output
f.d = -b[i];   // Linear term (for data fitting)
```

This creates: $f(x) = 0.5 \cdot \text{huber}(2x) - b_i \cdot x$

---

## Sparse Matrix Operations

### Creating Sparse Matrices

```cpp
#include "pogs.h"
#include "matrix/matrix_sparse.h"

// CSR format (row-major sparse)
// ptr: row pointers (size m+1)
// ind: column indices (size nnz)
// val: values (size nnz)

pogs::MatrixSparse<double> A('r', m, n, nnz, val, ptr, ind);
pogs::PogsCgls<double, pogs::MatrixSparse<double>> solver(A);
```

### Sparse Format Benefits

- **Memory**: O(nnz) instead of O(m*n)
- **Speed**: Faster matrix-vector products when sparse
- **Scalability**: Enables much larger problems

### When to Use Sparse

Use sparse format when:
- Sparsity > 90% (fewer than 10% non-zeros)
- Problem size > 1000 variables
- Memory is constrained

---

## GPU Acceleration

GPU support requires CUDA and is built separately.

### Building with GPU Support

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOGS_BUILD_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

cmake --build build
```

### Using GPU Solver

The GPU solver uses the same API as CPU:

```cpp
// Include GPU-specific headers
#include "pogs.h"
#include "matrix/matrix_dense.h"

// Create solver (GPU version if built with CUDA)
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);
solver.Solve(f, g);
```

### GPU Limitations

- SDP cones not yet supported on GPU
- Requires CUDA 11.0+
- Best for large dense problems (> 10,000 variables)

---

## Precision Control

### Single vs. Double Precision

```cpp
// Double precision (default)
pogs::MatrixDense<double> A_d('r', m, n, A_data_d);
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver_d(A_d);

// Single precision (faster, less accurate)
pogs::MatrixDense<float> A_f('r', m, n, A_data_f);
pogs::PogsDirect<float, pogs::MatrixDense<float>> solver_f(A_f);
```

**Use single precision when:**
- Speed is critical
- Problem is well-conditioned
- You don't need < 1e-5 accuracy

**Use double precision when:**
- High accuracy required
- Problem is ill-conditioned
- Working with financial data

---

## Monitoring Convergence

### Verbose Output

```cpp
solver.SetVerbose(2);  // 0=quiet, 1=summary, 2=progress, 3=detailed
```

Output shows:
```
Iter   Primal Res   Dual Res     Gap        rho
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  ...
 186   9.12e-05    3.45e-05    1.23e-04   1.00
```

### Extracting Results

```cpp
PogsStatus status = solver.Solve(f, g);

// Result information
printf("Iterations: %u\n", solver.GetFinalIter());
printf("Optimal value: %f\n", solver.GetOptval());
printf("Final rho: %f\n", solver.GetRho());

// Solution vectors
const double* x = solver.GetX();
const double* y = solver.GetY();
const double* lambda = solver.GetLambda();
const double* mu = solver.GetMu();
```

---

## Problem Scaling

Proper scaling can dramatically improve convergence.

### Data Matrix Scaling

```cpp
// Normalize columns of A to have unit norm
for (size_t j = 0; j < n; ++j) {
    double norm = compute_column_norm(A, j);
    scale_column(A, j, 1.0 / norm);
}

// Normalize rows of A
for (size_t i = 0; i < m; ++i) {
    double norm = compute_row_norm(A, i);
    scale_row(A, i, 1.0 / norm);
}
```

### Objective Scaling

Scale objectives to have similar magnitudes:

```cpp
// If ||Ax - b||^2 ~ 1000 and ||x||_1 ~ 10
// Scale the L1 term:
double scale = 100.0;  // Balance the terms
for (size_t j = 0; j < n; ++j) {
    g[j].c = lambda * scale;
}
```

---

## Handling Infeasibility

If the solver reports infeasibility:

1. **Check constraints**: Are they mutually compatible?
2. **Relax tolerances**: Some "infeasible" problems are just hard
3. **Scale the problem**: Poor scaling can cause numerical issues
4. **Check for unboundedness**: Add bounds if needed

```cpp
if (status == POGS_INFEASIBLE || status == POGS_UNBOUNDED) {
    // Try with relaxed tolerances
    solver.SetAbsTol(1e-3);
    solver.SetRelTol(1e-2);
    solver.SetMaxIter(5000);

    PogsStatus status2 = solver.Solve(f, g);
}
```

---

## Custom Stopping Criteria

The solver uses both absolute and relative stopping criteria:

$$
\text{stop when } \|r\|_2 \leq \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \max(\|Ax\|_2, \|y\|_2)
$$

Adjust based on problem:

```cpp
// High accuracy
solver.SetAbsTol(1e-6);
solver.SetRelTol(1e-6);

// Fast approximate solution
solver.SetAbsTol(1e-2);
solver.SetRelTol(1e-2);
solver.SetMaxIter(100);
```

---

## Performance Tips

### For Fastest Solve Time

1. Use sparse matrices when applicable
2. Enable adaptive rho
3. Use single precision if accuracy permits
4. Warm start for sequences of problems
5. Scale your data properly
6. Try Anderson acceleration

### For Highest Accuracy

1. Use double precision
2. Tighten tolerances (1e-6 or better)
3. Increase max iterations
4. Disable early stopping (`SetGapStop(false)`)
5. Start with smaller rho

---

## See Also

- [Basic Usage](basic-usage.md) - Fundamentals
- [Cone Problems](cone-problems.md) - LP, QP, SOCP, SDP
- [Examples](../examples/lasso.md) - Complete examples
