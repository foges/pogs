# Installation

Get POGS running in seconds.

---

## Quick Install (Recommended)

```bash
pip install pogs
```

That's it! Works on:

- **macOS**: Intel and Apple Silicon (M1/M2/M3)
- **Linux**: x86_64 and ARM64

---

## Verify Installation

```python
import pogs
print(pogs.__version__)  # Should print "0.4.2"

# Quick test
import numpy as np
from pogs import solve_lasso

A = np.random.randn(100, 50)
b = np.random.randn(100)
result = solve_lasso(A, b, lambd=0.1)
print(f"Solved in {result['iter']} iterations")
```

---

## Optional: CVXPY Integration

To use POGS with CVXPY for more general problems:

```bash
pip install pogs[cvxpy]
```

Then:

```python
import cvxpy as cp
import numpy as np

x = cp.Variable(50)
A = np.random.randn(100, 50)
b = np.random.randn(100)

prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
prob.solve(solver='POGS')
```

---

## Troubleshooting

### Import Error: Library not found

On some Linux systems, you may need to install OpenBLAS:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-base

# Fedora/RHEL
sudo dnf install openblas
```

### macOS: No issues expected

macOS uses the built-in Accelerate framework - no additional dependencies needed.

---

## Building from Source

For developers or if pre-built wheels aren't available:

### Prerequisites

- Python 3.10+
- C++20 compiler (GCC 10+, Clang 13+, or AppleClang 13+)
- CMake 3.20+
- BLAS/LAPACK (OpenBLAS on Linux, Accelerate on macOS)

### Install

```bash
git clone https://github.com/foges/pogs.git
cd pogs
pip install .
```

### Development Install

```bash
pip install -e ".[dev]"
```

---

## C++ Library (Advanced)

If you need the C++ library directly:

=== "macOS"

    ```bash
    brew install cmake
    git clone https://github.com/foges/pogs.git
    cd pogs
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
    cmake --build build
    sudo cmake --install build
    ```

=== "Linux (Ubuntu)"

    ```bash
    sudo apt-get install cmake g++ libopenblas-dev liblapack-dev
    git clone https://github.com/foges/pogs.git
    cd pogs
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
    cmake --build build
    sudo cmake --install build
    ```

---

## Next Steps

- [Quick Start](quick-start.md) - Run your first optimization
- [Examples](../examples/lasso.md) - See complete examples
