# Installation

## Quick Install

```bash
pip install pogs
```

That's it! Pre-built wheels are available for:

| Platform | Architectures |
|:---------|:--------------|
| **macOS** | Intel (x86_64), Apple Silicon (arm64) |
| **Linux** | x86_64, aarch64 |
| **Windows** | x86_64 |

**Requirements:** Python 3.9+ and NumPy (installed automatically).

---

## Verify Installation

```python
import pogs
print(pogs.__version__)
```

```python
from pogs import solve_lasso
import numpy as np

A = np.random.randn(100, 50)
b = np.random.randn(100)
result = solve_lasso(A, b, lambd=0.1)
print(f"Solved in {result['iterations']} iterations")
```

---

## Optional: CVXPY Integration

To use POGS with CVXPY, install CVXPY separately:

```bash
pip install cvxpy
```

Then use `pogs_solve()`:

```python
import cvxpy as cp
from pogs import pogs_solve

x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
pogs_solve(prob)  # Auto-detects Lasso, uses fast solver
print(x.value)
```

---

## Building from Source

For development or if pre-built wheels aren't available:

### Prerequisites

=== "macOS"

    Xcode command line tools (includes Accelerate framework):
    ```bash
    xcode-select --install
    ```

=== "Linux"

    ```bash
    # Ubuntu/Debian
    sudo apt-get install build-essential cmake libblas-dev liblapack-dev

    # Fedora/RHEL
    sudo dnf install gcc-c++ cmake openblas-devel lapack-devel
    ```

=== "Windows"

    Visual Studio Build Tools with C++ workload, or use conda:
    ```bash
    conda install -c conda-forge openblas
    ```

### Install from GitHub

```bash
pip install git+https://github.com/foges/pogs.git
```

### Local Development

```bash
git clone https://github.com/foges/pogs.git
cd pogs
pip install -e .
```

---

## C++ Library

If you need the C++ library directly:

```bash
git clone https://github.com/foges/pogs.git
cd pogs
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

The library will be in `build/lib/`.

---

## Troubleshooting

### "No matching distribution found"

Your platform may not have pre-built wheels. Build from source:

```bash
pip install git+https://github.com/foges/pogs.git
```

### ImportError on Linux

You may need BLAS/LAPACK runtime libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas0

# Fedora/RHEL
sudo dnf install openblas
```

### Slow performance

Ensure NumPy uses optimized BLAS:

```python
import numpy as np
np.show_config()  # Check BLAS backend
```

For best performance, NumPy should use OpenBLAS, MKL, or Accelerate (macOS).

---

## Next Steps

- [Quick Start](quick-start.md) - Run your first optimization
- [API Reference](../api/solver.md) - Full function documentation
