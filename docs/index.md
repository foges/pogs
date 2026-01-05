# POGS - Proximal Operator Graph Solver

**Modern C++20 convex optimization solver using ADMM**

POGS is a high-performance solver for convex optimization problems using the [Alternating Direction Method of Multipliers](https://stanford.edu/~boyd/papers/admm_distr_stats.html) (ADMM). It supports both graph form and cone form problems, with implementations for CPU and GPU.

---

## Features

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } __High Performance__

    ---

    Optimized CPU implementation with BLAS/LAPACK

    GPU acceleration with CUDA support

    Sparse matrix support for large-scale problems

-   :material-language-python:{ .lg .middle } __Multiple Interfaces__

    ---

    Modern C++20 API with type safety

    Python/CVXPY integration for easy modeling

    C interface for maximum compatibility

-   :material-chart-bell-curve:{ .lg .middle } __Rich Problem Support__

    ---

    Lasso, Ridge, Logistic Regression

    Quadratic and Linear Programs

    Second-order cone and SDP constraints

-   :material-code-tags:{ .lg .middle } __Modern Codebase__

    ---

    C++20 with RAII and smart pointers

    CMake build system

    Comprehensive test suite

</div>

---

## Quick Start

=== "C++"

    ```cpp
    #include "pogs.h"
    #include "matrix/matrix_dense.h"

    // Problem: min 0.5||Ax - b||^2 + lambda||x||_1
    pogs::MatrixDense<double> A('r', m, n, A_data);
    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    // Configure solver
    solver.SetRho(1.0);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetMaxIter(1000);
    solver.SetVerbose(2);

    // Define objective functions f and g
    std::vector<FunctionObj<double>> f(m), g(n);
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].d = -b[i];
    }
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kAbs;
        g[j].c = lambda;
    }

    // Solve and get results
    pogs::PogsStatus status = solver.Solve(f, g);
    if (status == pogs::POGS_SUCCESS) {
        const double* x = solver.GetX();
    }
    ```

=== "Python"

    ```python
    import cvxpy as cp
    import numpy as np

    # Define problem
    A = np.random.randn(100, 50)
    b = np.random.randn(100)
    lambda_val = 0.1

    x = cp.Variable(50)
    objective = cp.Minimize(
        0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1)
    )
    prob = cp.Problem(objective)
    prob.solve(solver='POGS', verbose=True)

    print(f"Status: {prob.status}")
    print(f"Optimal value: {prob.value}")
    ```

=== "C"

    ```c
    #include "interface_c/pogs_c.h"

    // Problem: min c'x s.t. Ax = b, x >= 0
    double A[] = {1.0, 1.0};  // 1x2 matrix
    double b[] = {2.0};
    double c[] = {1.0, 0.0};

    // Define cone constraints
    unsigned int x_idx[] = {0, 1};
    unsigned int y_idx[] = {0};
    ConeConstraintC cone_x = {CONE_NON_NEG, x_idx, 2};
    ConeConstraintC cone_y = {CONE_ZERO, y_idx, 1};

    // Solve
    double x[2], y[1], optval;
    unsigned int final_iter;
    int status = PogsConeD(ROW_MAJ, 1, 2, A, b, c,
                          &cone_x, 1, &cone_y, 1,
                          1.0, 1e-4, 1e-3, 1000, 0, 1, 0,
                          x, y, NULL, &optval, &final_iter);
    ```

---

## Installation

### Using CMake

```bash
# Clone repository
git clone https://github.com/foges/pogs.git
cd pogs

# Configure (CPU-only)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF

# Build
cmake --build build --config Release

# Install
sudo cmake --install build
```

### Using in Your Project

```cmake
find_package(POGS REQUIRED)
target_link_libraries(your_target PRIVATE pogs::cpu)
```

---

## Problem Classes

POGS can solve a wide variety of convex optimization problems:

- **Regression**: Lasso, Ridge, Elastic Net, Huber Fitting
- **Classification**: Logistic Regression, SVM
- **Signal Processing**: Total Variation Denoising
- **Control Theory**: Optimal Control, LQR
- **Convex Programs**: Linear Programs (LP), Quadratic Programs (QP)
- **Cone Programs**: Second-Order Cone Programs (SOCP), Semidefinite Programs (SDP)

---

## Graph Form

A graph form problem has the structure:

$$
\begin{align}
\text{minimize} \quad & f(y) + g(x) \\
\text{subject to} \quad & y = Ax
\end{align}
$$

where $f$ and $g$ are convex, separable functions:

$$
f(y) = \sum_{i=1}^m f_i(y_i), \quad g(x) = \sum_{j=1}^n g_j(x_j)
$$

---

## Cone Form

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

## What's New in v0.4

!!! success "Major Modernization Release"

    - **C++20 Support**: Modern C++ with smart pointers, enum classes, RAII
    - **CMake Build System**: Cross-platform, easy integration
    - **Full Cone Support**: Zero, NonNeg, NonPos, SOC, SDP, and Exponential cones
    - **Anderson Acceleration**: Optional acceleration for faster convergence
    - **Python/CVXPY**: High-level modeling interface
    - **130+ Test Assertions**: Comprehensive test coverage

    See the [Changelog](about/changelog.md) for full details.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Get Started__

    ---

    Install POGS and run your first optimization

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } __Learn More__

    ---

    Understand POGS capabilities and API

    [:octicons-arrow-right-24: User Guide](user-guide/basic-usage.md)

-   :material-code-braces:{ .lg .middle } __See Examples__

    ---

    Learn from real optimization problems

    [:octicons-arrow-right-24: Examples](examples/lasso.md)

-   :material-language-python:{ .lg .middle } __Python Users__

    ---

    Use POGS with CVXPY for easy modeling

    [:octicons-arrow-right-24: CVXPY Integration](user-guide/cvxpy-integration.md)

</div>

---

## Community & Support

- **GitHub**: [foges/pogs](https://github.com/foges/pogs)
- **Issues**: [Report bugs or request features](https://github.com/foges/pogs/issues)
- **Documentation**: You're reading it!

---

## License

POGS is licensed under the [Apache 2.0 License](about/license.md).

---

## Citation

If you use POGS in your research, please cite:

```bibtex
@article{fougner2016pogs,
  title={Parameter selection and preconditioning for a graph form solver},
  author={Fougner, Chris and Boyd, Stephen},
  journal={Optimization and Engineering},
  year={2016},
  publisher={Springer}
}
```
