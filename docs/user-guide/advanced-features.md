# Advanced Features

This guide covers advanced POGS features for power users.

---

## Warm Starting

Warm starting can significantly reduce solve time when solving a sequence of related problems.

### Basic Warm Start

```cpp
auto solver = pogs::make_solver<double>(std::move(A));

// Solve first problem
auto result1 = solver.solve(f1, g1);

// Warm start from previous solution
solver.set_warm_start(result1.x, result1.y);

// Solve similar problem (faster!)
auto result2 = solver.solve(f2, g2);
```

### Use Cases

- **Parameter sweeps**: Solving problems with varying λ
- **Online optimization**: Updating solutions as new data arrives
- **Iterative refinement**: Solving with increasing accuracy

---

## Custom Penalty Parameter (ρ)

The penalty parameter ρ controls the ADMM convergence behavior.

### Manual ρ Selection

```cpp
auto config = pogs::SolverConfig{
    .rho = 10.0,          // Larger for faster (but less accurate) convergence
    .adaptive_rho = false  // Disable adaptive adjustment
};
```

### When to Adjust ρ

**Increase ρ** (larger values like 5.0-100.0):
- When primal residual >> dual residual
- For well-conditioned problems
- When you want faster convergence (may sacrifice accuracy)

**Decrease ρ** (smaller values like 0.01-0.5):
- When dual residual >> primal residual
- For ill-conditioned problems
- When you need high accuracy

### Adaptive ρ (Recommended)

```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .adaptive_rho = true  // Automatically adjust ρ
};
```

The solver will automatically increase/decrease ρ based on residual balance.

---

## Function Parameterization

### General Function Form

Each function has the form:

$$
h(ax + b) \cdot c + d \cdot x + e
$$

where:
- `a`: Input scaling
- `b`: Input shift
- `c`: Output scaling
- `d`: Linear term coefficient
- `e`: Constant offset
- `h`: Base function type

### Example: Scaled Huber Loss

```cpp
pogs::FunctionObj<double> f;
f.type = pogs::FunctionType::Huber;
f.a = 2.0;   // Scale input
f.c = 0.5;   // Scale output
f.d = -b[i]; // Linear term (for data fitting)
```

This creates: $f(x) = 0.5 \cdot \text{huber}(2x) - b_i \cdot x$

---

## Over-Relaxation

Over-relaxation can improve convergence for some problems.

```cpp
// Not yet exposed in modern API
// Coming in future release
```

The ADMM algorithm uses over-relaxation parameter α = 1.7 by default for better convergence.

---

## Sparse Matrix Operations

### Creating Sparse Matrices

```cpp
// From triplet format (row, col, value)
std::vector<int> rows = {0, 0, 1, 1, 2};
std::vector<int> cols = {0, 1, 1, 2, 2};
std::vector<double> vals = {1.0, 2.0, 3.0, 4.0, 5.0};

auto A = pogs::make_sparse_matrix(m, n, rows, cols, vals);
```

### Sparse Format Benefits

- **Memory**: O(nnz) instead of O(m×n)
- **Speed**: Faster matrix-vector products when sparse
- **Scalability**: Enables much larger problems

### When to Use Sparse

Use sparse format when:
- Sparsity > 90% (fewer than 10% non-zeros)
- Problem size > 1000 variables
- Memory is constrained

---

## GPU Acceleration

!!! note "GPU Support"
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

```cpp
#include <pogs/gpu/pogs_gpu.hpp>

auto A_gpu = pogs::make_gpu_matrix<double>(m, n);
auto solver = pogs::make_gpu_solver<double>(std::move(A_gpu));

auto result = solver.solve(f, g);
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
auto solver_d = pogs::make_solver<double>(std::move(A));

// Single precision (faster, less accurate)
auto solver_f = pogs::make_solver<float>(std::move(A));
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
auto config = pogs::SolverConfig{
    .verbose = true
};
```

Output shows:
```
Iter   Primal Res   Dual Res     Gap        ρ
  10   1.23e-02    4.56e-03    8.90e-02   1.00
  20   3.45e-03    1.23e-03    2.34e-02   1.00
  ...
 186   9.12e-05    3.45e-05    1.23e-04   1.00  ✓ Converged
```

### Extracting Residuals

```cpp
auto result = solver.solve(f, g);

// Result contains convergence information
std::cout << "Iterations: " << result.iterations << "\n";
std::cout << "Primal objective: " << result.primal_obj.value() << "\n";
std::cout << "Dual objective: " << result.dual_obj.value() << "\n";
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
if (result.status == pogs::Status::InfeasibleOrUnbounded) {
    // Try with relaxed tolerances
    config.abs_tol = 1e-3;
    config.rel_tol = 1e-2;
    config.max_iter = 5000;

    auto result2 = solver.solve(f, g, config);
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
config.abs_tol = 1e-6;
config.rel_tol = 1e-6;

// Fast approximate solution
config.abs_tol = 1e-2;
config.rel_tol = 1e-2;
config.max_iter = 100;
```

---

## Performance Tips

### For Fastest Solve Time

1. Use sparse matrices when applicable
2. Enable adaptive ρ
3. Use single precision if accuracy permits
4. Warm start for sequences of problems
5. Scale your data properly

### For Highest Accuracy

1. Use double precision
2. Tighten tolerances (1e-6 or better)
3. Increase max iterations
4. Disable early stopping (`gap_stop = false`)
5. Start with smaller ρ

---

## See Also

- [Basic Usage](basic-usage.md) - Fundamentals
- [Cone Problems](cone-problems.md) - LP, QP, SOCP, SDP
- [Examples](../examples/lasso.md) - Complete examples
