# POGS - Proximal Operator Graph Solver

**Modern C++20 solver for convex optimization using ADMM**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/Build-CMake-green.svg)](https://cmake.org/)

[**Documentation**](https://foges.github.io/pogs/) | [**Paper**](http://stanford.edu/~boyd/papers/pogs.html) | [**Examples**](https://foges.github.io/pogs/examples/)

---

## Overview

POGS is a high-performance solver for convex optimization problems in **graph form** and **cone form** using the Alternating Direction Method of Multipliers (ADMM). Version 0.4 represents a major modernization with C++20, CMake, and comprehensive documentation.

### Graph Form

```
minimize        f(y) + g(x)
subject to      y = Ax
```

where `f` and `g` are separable convex functions:
- `f(y) = sum_{i=1}^m f_i(y_i)`
- `g(x) = sum_{j=1}^n g_j(x_j)`

### Cone Form

```
minimize        c'x
subject to      Ax + s = b, s ∈ K
```

where `K` is a product of convex cones (zero, non-negative, SOC, SDP, exponential).

### Supported Functions

POGS supports a rich library of proximal operators including:
- **Norms**: L1 (abs), L2 (square), Huber
- **Indicators**: box constraints, non-negativity, equality
- **Non-linear**: logistic, exponential, reciprocal, negative log

See the [full list](https://foges.github.io/pogs/api/proximal/) in the documentation.


## Features

- **Fast**: Optimized CPU and GPU implementations using BLAS/LAPACK and CUDA
- **Flexible**: Support for graph form and cone form problems
- **Modern**: C++20 codebase with smart pointers, RAII, and modern CMake
- **Well-Tested**: Comprehensive test suite using Catch2
- **Well-Documented**: Full documentation with examples at [foges.github.io/pogs](https://foges.github.io/pogs/)
- **Multiple Interfaces**: C++, C, and Python/CVXPY bindings

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/foges/pogs.git
cd pogs

# Configure (CPU-only)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF

# Build
cmake --build build

# Install (optional)
sudo cmake --install build
```

For GPU support, use `-DPOGS_BUILD_GPU=ON` and ensure CUDA is installed.

See the [Installation Guide](https://foges.github.io/pogs/getting-started/installation/) for detailed instructions.

### Example: Lasso Regression

**C++ (Graph Form)**:
```cpp
#include "pogs.h"
#include "matrix/matrix_dense.h"

// Problem: min 0.5||Ax - b||^2 + lambda||x||_1
pogs::MatrixDense<double> A('r', m, n, A_data);
pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

std::vector<FunctionObj<double>> f(m), g(n);
for (size_t i = 0; i < m; ++i) {
    f[i].h = kSquare;
    f[i].d = -b[i];
}
for (size_t j = 0; j < n; ++j) {
    g[j].h = kAbs;
    g[j].c = lambda;
}

solver.Solve(f, g);
const double* x = solver.GetX();
```

**Python/CVXPY**:
```python
import cvxpy as cp
import numpy as np

A = np.random.randn(100, 50)
b = np.random.randn(100)
lambda_val = 0.1

x = cp.Variable(50)
objective = cp.Minimize(
    0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1)
)
prob = cp.Problem(objective)
prob.solve(solver='POGS')

print(f"Optimal value: {prob.value}")
print(f"Solution: {x.value}")
```

See more [examples in the documentation](https://foges.github.io/pogs/examples/).

## Interfaces

POGS provides multiple interfaces for different use cases:

1. **C++/BLAS**: High-performance CPU implementation using BLAS/LAPACK
   - Apple Accelerate Framework (macOS)
   - OpenBLAS (Linux/Windows)
   - Intel MKL

2. **C++/CUDA**: GPU-accelerated implementation
   - Requires CUDA Toolkit
   - Supports NVIDIA GPUs with compute capability 3.5+

3. **Python/CVXPY**: High-level modeling interface
   - Integrates with [CVXPY](https://www.cvxpy.org/)
   - Easy problem formulation
   - See [CVXPY Integration Guide](https://foges.github.io/pogs/user-guide/cvxpy-integration/)

4. **C Interface**: Direct C API for cone form
   - See [C API Documentation](https://foges.github.io/pogs/api/c-api/)


## Problem Classes

POGS can solve a wide variety of convex optimization problems, including:

### Machine Learning
- **Lasso Regression**: L1-regularized least squares
- **Ridge Regression**: L2-regularized least squares
- **Elastic Net**: Combined L1/L2 regularization
- **Logistic Regression**: Binary classification
- **Huber Fitting**: Robust regression
- **Support Vector Machines**: Linear and kernel SVMs

### Signal Processing
- **Total Variation Denoising**: Image denoising and reconstruction
- **Compressed Sensing**: Sparse signal recovery
- **Basis Pursuit**: L1 minimization

### Optimization
- **Linear Programs** (LP): Standard form and cone form
- **Quadratic Programs** (QP): Convex quadratic objectives
- **Second-Order Cone Programs** (SOCP): Cone constraints
- **Semidefinite Programs** (SDP): Matrix cone constraints

### Control
- **Optimal Control**: State-space control problems
- **Model Predictive Control**: MPC formulations

See the [Examples](https://foges.github.io/pogs/examples/) for detailed problem formulations.

## What's New in v0.4

Version 0.4 represents a major modernization of the POGS codebase:

- **C++20**: Modern C++ features including smart pointers, RAII, concepts
- **CMake Build System**: Replaced Makefiles with modern CMake
- **Comprehensive Documentation**: New MkDocs Material site with search
- **Test Suite**: Catch2-based testing framework
- **Code Quality**: Eliminated code duplication, improved maintainability
- **MATLAB Removal**: MATLAB interface removed (use Python/CVXPY instead)

See the [Migration Guide](https://foges.github.io/pogs/migration/v0.3-to-v0.4/) for upgrade instructions.

## Documentation

Full documentation is available at **[foges.github.io/pogs](https://foges.github.io/pogs/)**

- [Getting Started](https://foges.github.io/pogs/getting-started/installation/)
- [User Guide](https://foges.github.io/pogs/user-guide/basic-usage/)
- [API Reference](https://foges.github.io/pogs/api/solver/)
- [Examples](https://foges.github.io/pogs/examples/)
- [Developer Guide](https://foges.github.io/pogs/developer/architecture/)
- [Migration Guide](https://foges.github.io/pogs/migration/v0.3-to-v0.4/)


## Requirements

### CPU Version
- **Compiler**: C++20 compatible compiler
  - GCC 10+ / Clang 13+ / AppleClang 13+ / MSVC 19.29+
- **CMake**: 3.20 or later
- **BLAS/LAPACK**: One of the following
  - macOS: Accelerate Framework (built-in)
  - Linux: OpenBLAS, ATLAS, or Intel MKL
  - Windows: Intel MKL or OpenBLAS

### GPU Version (Optional)
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 3.5+

## Contributing

Contributions are welcome! Please see the [Contributing Guide](https://foges.github.io/pogs/developer/contributing/) for details.

## License

POGS is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## References

1. **C. Fougner and S. Boyd**, [*Parameter Selection and Pre-Conditioning for a Graph Form Solver*](http://stanford.edu/~boyd/papers/pogs.html), 2015
2. **N. Parikh and S. Boyd**, [*Block Splitting for Distributed Optimization*](http://www.stanford.edu/~boyd/papers/block_splitting.html), 2013
3. **S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein**, [*Distributed Optimization and Statistical Learning via ADMM*](http://www.stanford.edu/~boyd/papers/admm_distr_stats.html), 2011
4. **N. Parikh and S. Boyd**, [*Proximal Algorithms*](http://www.stanford.edu/~boyd/papers/prox_algs.html), 2014

## Authors

**Chris Fougner** - Original author and maintainer

With input from **Stephen Boyd** (Stanford University)

The core algorithm is based on the block splitting ADMM method from Parikh and Boyd (2013).

See [AUTHORS.md](docs/about/authors.md) for a complete list of contributors.

---

**For questions, bug reports, or feature requests, please open an issue on [GitHub](https://github.com/foges/pogs/issues).**

