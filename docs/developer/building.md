# Building from Source

Developer guide for building POGS from source.

---

## Prerequisites

### Required

- **C++ compiler** with C++20 support:
  - GCC 10+
  - Clang 13+
  - AppleClang 13+
  - MSVC 19.29+
- **CMake** 3.20 or higher
- **BLAS/LAPACK** libraries

### Optional

- **CUDA Toolkit** 11.0+ (for GPU support)
- **Python** 3.7+ (for Python interface)
- **Doxygen** (for API documentation)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/foges/pogs.git
cd pogs

# Configure (CPU-only, Release build)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOGS_BUILD_GPU=OFF \
    -DPOGS_BUILD_TESTS=ON \
    -DPOGS_BUILD_EXAMPLES=ON

# Build
cmake --build build --config Release -j4

# Run tests
cd build
ctest --output-on-failure

# Install (optional)
sudo cmake --install build
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Release, Debug, RelWithDebInfo) |
| `POGS_BUILD_GPU` | `OFF` | Build GPU support with CUDA |
| `POGS_BUILD_TESTS` | `ON` | Build test suite |
| `POGS_BUILD_EXAMPLES` | `ON` | Build examples |
| `CMAKE_INSTALL_PREFIX` | `/usr/local` | Installation directory |

### Example Configurations

**Debug build:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

**With GPU support:**
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOGS_BUILD_GPU=ON
```

**Custom install location:**
```bash
cmake -B build \
    -DCMAKE_INSTALL_PREFIX=$HOME/local
```

---

## Platform-Specific Instructions

### macOS

macOS includes the Accelerate framework:

```bash
# Install CMake
brew install cmake

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF
cmake --build build
```

**Xcode:**
```bash
cmake -B build -G Xcode
open build/POGS.xcodeproj
```

---

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install \
    cmake \
    gcc-c++ \
    openblas-devel \
    lapack-devel

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

### Windows (Visual Studio)

```bash
# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build
cmake --build build --config Release

# Install
cmake --install build --config Release
```

**Note:** You'll need to provide BLAS/LAPACK (e.g., Intel MKL or OpenBLAS).

---

## GPU Build

### Prerequisites

1. **CUDA Toolkit** 11.0 or higher
2. **NVIDIA GPU** with compute capability 3.5+

### Configuration

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOGS_BUILD_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### Troubleshooting

If CMake can't find CUDA:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DPOGS_BUILD_GPU=ON
```

---

## Testing

### Run All Tests

```bash
cd build
ctest --output-on-failure
```

### Run Specific Tests

```bash
./build/bin/test_cone   # Cone projection tests
./build/bin/test_sdp    # SDP tests
```

### Expected Output

```
Test project /path/to/pogs/build
    Start 1: test_cone
1/2 Test #1: test_cone ........................   Passed    0.15 sec
    Start 2: test_sdp
2/2 Test #2: test_sdp .........................   Passed    0.08 sec

100% tests passed, 0 tests failed out of 2
```

---

## Installation

### System-Wide Install

```bash
sudo cmake --install build
```

Installs to `/usr/local` by default:
- Headers: `/usr/local/include/pogs/`
- Library: `/usr/local/lib/libpogs_cpu.a`
- CMake config: `/usr/local/lib/cmake/POGS/`

### User Install

```bash
cmake --install build --prefix $HOME/local
```

Then add to your environment:

```bash
export CMAKE_PREFIX_PATH=$HOME/local:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```

---

## Using POGS in Your Project

### CMake Integration

```cmake
find_package(POGS REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE pogs::cpu)
```

### Manual Compilation

```bash
g++ -std=c++20 -O3 myapp.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -llapack -lblas \
    -o myapp
```

**macOS:**
```bash
g++ -std=c++20 -O3 myapp.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lpogs_cpu \
    -framework Accelerate \
    -o myapp
```

---

## Development Build

For active development:

```bash
# Debug build with all warnings
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"

# Build and run tests frequently
cmake --build build && cd build && ctest
```

### Code Formatting

Format code before committing:

```bash
find src include -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | \
    xargs clang-format -i
```

---

## Troubleshooting

### CMake Can't Find BLAS/LAPACK

**macOS:**
```bash
cmake -B build -DBLAS_LIBRARIES="-framework Accelerate"
```

**Linux:**
```bash
cmake -B build \
    -DBLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libopenblas.so" \
    -DLAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so"
```

### Compiler Not C++20 Compatible

```bash
# Check version
g++ --version

# Specify compiler explicitly
cmake -B build -DCMAKE_CXX_COMPILER=g++-11
```

### CUDA Not Found

```bash
cmake -B build \
    -DPOGS_BUILD_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8
```

---

## Clean Build

```bash
# Remove build directory
rm -rf build

# Reconfigure and rebuild
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## See Also

- [Architecture](architecture.md) - Code structure
- [Contributing](contributing.md) - How to contribute
- [Installation Guide](../getting-started/installation.md) - User installation
