# Installation

This guide covers how to install POGS on your system.

---

## Requirements

### C++ Library

- **Compiler**: C++20 capable compiler
    - GCC 10+
    - Clang 13+
    - AppleClang 13+
    - MSVC 19.29+
- **CMake**: Version 3.20 or higher
- **BLAS/LAPACK**: Linear algebra libraries
    - macOS: Accelerate framework (built-in)
    - Linux: OpenBLAS or ATLAS
    - Windows: Intel MKL or OpenBLAS

### Optional

- **CUDA**: For GPU support (CUDA 11.0+)
- **Python**: For Python/CVXPY interface (Python 3.7+)
- **NumPy**: For Python solver
- **CVXPY**: For high-level modeling

---

## Install from Source

### 1. Clone Repository

```bash
git clone https://github.com/foges/pogs.git
cd pogs
```

### 2. Configure with CMake

=== "CPU Only (Recommended)"

    ```bash
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPOGS_BUILD_GPU=OFF \
        -DPOGS_BUILD_TESTS=ON \
        -DPOGS_BUILD_EXAMPLES=ON
    ```

=== "With GPU Support"

    ```bash
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPOGS_BUILD_GPU=ON \
        -DPOGS_BUILD_TESTS=ON \
        -DPOGS_BUILD_EXAMPLES=ON
    ```

=== "Custom Install Location"

    ```bash
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$HOME/local \
        -DPOGS_BUILD_GPU=OFF
    ```

### 3. Build

```bash
cmake --build build --config Release -j4
```

### 4. Run Tests (Optional)

```bash
cd build
ctest --output-on-failure
```

Or run tests directly:

```bash
./build/bin/test_cone
./build/bin/test_sdp
```

### 5. Install

```bash
sudo cmake --install build
```

Or without sudo for custom location:

```bash
cmake --install build --prefix $HOME/local
```

---

## Platform-Specific Notes

### macOS

macOS includes the Accelerate framework with optimized BLAS/LAPACK:

```bash
# Install dependencies
brew install cmake

# Build as normal
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install cmake g++ libopenblas-dev liblapack-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install cmake gcc-c++ openblas-devel lapack-devel

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

### Windows

```bash
# Using Visual Studio 2022
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
cmake --install build
```

!!! note
    On Windows, you'll need to provide BLAS/LAPACK libraries (e.g., Intel MKL or OpenBLAS).

---

## GPU Support

To build with CUDA support:

### 1. Install CUDA Toolkit

Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### 2. Configure with GPU

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOGS_BUILD_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### 3. Build

```bash
cmake --build build --config Release
```

!!! warning
    GPU support for SDP cones is not yet implemented. Other cone types work on GPU.

---

## Python Installation

### Using pip (Coming Soon)

```bash
pip install pogs
```

!!! note
    PyPI package not yet available. Currently requires building from source.

### From Source

```bash
# Build C++ library first
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
sudo cmake --install build

# Install Python dependencies
pip install numpy cvxpy

# Python interface is in python/ directory
# Add to PYTHONPATH or install locally
cd python
pip install -e .
```

---

## Using POGS in Your Project

### CMake Integration

After installation, use POGS in your CMake project:

```cmake
find_package(POGS REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE pogs::cpu)
```

### Manual Compilation

```bash
# Compile
g++ -std=c++20 -I/usr/local/include my_app.cpp -lpogs_cpu -llapack -lblas -o my_app

# Run
./my_app
```

---

## Verify Installation

Create a test file `test_pogs.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "POGS v0.4.0" << std::endl;
    std::cout << "C++20 installation verified!" << std::endl;
    return 0;
}
```

Compile and run:

```bash
g++ -std=c++20 test_pogs.cpp -o test_pogs
./test_pogs
```

---

## Troubleshooting

### CMake Can't Find BLAS/LAPACK

**macOS**:
```bash
# Accelerate should be found automatically
# If not, specify manually:
cmake -B build -DBLAS_LIBRARIES="-framework Accelerate"
```

**Linux**:
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev liblapack-dev

# Or specify paths manually
cmake -B build \
    -DBLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libopenblas.so" \
    -DLAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so"
```

### Compiler Not C++20 Compatible

```bash
# Check compiler version
g++ --version  # Need GCC 10+
clang++ --version  # Need Clang 13+

# Specify compiler explicitly
cmake -B build -DCMAKE_CXX_COMPILER=g++-11
```

### CUDA Not Found

```bash
# Specify CUDA path
cmake -B build \
    -DPOGS_BUILD_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first optimization
- [User Guide](../user-guide/basic-usage.md) - Learn the API
- [Examples](../examples/lasso.md) - See real problems
