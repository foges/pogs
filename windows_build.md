# Windows Build Investigation

This document summarizes the attempts to build POGS Python wheels on Windows using scipy-openblas32 for BLAS/LAPACK support.

## Problem Summary

Windows builds fail during linking with unresolved CBLAS symbols:

```
pogs.obj : error LNK2019: unresolved external symbol cblas_sdot
pogs.obj : error LNK2019: unresolved external symbol cblas_ddot
pogs.obj : error LNK2019: unresolved external symbol cblas_snrm2
...
```

The scipy-openblas32 library is correctly found and the paths are valid, but CMake/MSVC fails to actually link against the library.

## Environment

- **CI Runner**: windows-2022
- **Compiler**: MSVC (Visual Studio 2022)
- **BLAS Library**: scipy-openblas32 (provides MSVC-compatible OpenBLAS)
- **Build System**: CMake via scikit-build-core
- **Wheel Builder**: cibuildwheel

## scipy-openblas32 Discovery

The library is correctly discovered via Python:

```cmake
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_lib_dir())"
  OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_include_dir())"
  OUTPUT_VARIABLE SCIPY_OPENBLAS_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_library())"
  OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
```

This correctly returns:
- Library dir: `C:/Users/.../scipy_openblas32/lib`
- Include dir: `C:/Users/.../scipy_openblas32/include`
- Library name: `libscipy_openblas`
- Files in lib dir: `['cmake', 'libscipy_openblas.dll', 'libscipy_openblas.lib', 'pkgconfig']`

## Approaches Tried

### 1. SHARED IMPORTED Target with IMPORTED_IMPLIB

```cmake
add_library(BLAS::BLAS SHARED IMPORTED GLOBAL)
set_target_properties(BLAS::BLAS PROPERTIES
  IMPORTED_IMPLIB "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas.lib"
  IMPORTED_LOCATION "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas.dll"
  INTERFACE_INCLUDE_DIRECTORIES "${SCIPY_OPENBLAS_INCLUDE_DIR}"
)
```

**Result**: Same linker errors. The import library wasn't passed to the linker.

### 2. INTERFACE IMPORTED Target

```cmake
add_library(BLAS::BLAS INTERFACE IMPORTED GLOBAL)
set_target_properties(BLAS::BLAS PROPERTIES
  INTERFACE_LINK_LIBRARIES "${SCIPY_OPENBLAS_LIB_FILE}"
  INTERFACE_INCLUDE_DIRECTORIES "${SCIPY_OPENBLAS_INCLUDE_DIR}"
)
```

**Result**: Same linker errors.

### 3. UNKNOWN IMPORTED Target

```cmake
add_library(BLAS::BLAS UNKNOWN IMPORTED GLOBAL)
set_target_properties(BLAS::BLAS PROPERTIES
  IMPORTED_LOCATION "${SCIPY_OPENBLAS_LIB_FILE}"
  INTERFACE_INCLUDE_DIRECTORIES "${SCIPY_OPENBLAS_INCLUDE_DIR}"
)
```

**Result**: Same linker errors.

### 4. Using scipy-openblas32's CMake Config

scipy-openblas32 provides a CMake config in its `lib/cmake` directory:

```cmake
set(CMAKE_PREFIX_PATH "${SCIPY_OPENBLAS_LIB_DIR}/cmake" ${CMAKE_PREFIX_PATH})
find_package(OpenBLAS REQUIRED CONFIG)

# This finds the library and sets:
# OpenBLAS_LIBRARIES: .../libscipy_openblas.lib
# OpenBLAS_INCLUDE_DIRS: .../include
# But does NOT create an OpenBLAS::OpenBLAS target
```

Then wrapping in INTERFACE target:

```cmake
add_library(BLAS::BLAS INTERFACE IMPORTED GLOBAL)
set_target_properties(BLAS::BLAS PROPERTIES
  INTERFACE_LINK_LIBRARIES OpenBLAS::OpenBLAS  # Target doesn't exist!
)
```

**Result**: CMake error - OpenBLAS::OpenBLAS target not found. The config only sets variables.

### 5. Direct target_link_libraries with Full Path

```cmake
set(POGS_BLAS_LIBRARIES "${OpenBLAS_LIBRARIES}" CACHE STRING "BLAS library path" FORCE)

# In src/CMakeLists.txt:
target_link_libraries(pogs_cpu_shared PUBLIC "${POGS_BLAS_LIBRARIES}")
```

**Result**: Same linker errors.

### 6. target_link_directories + Library Name

```cmake
get_filename_component(POGS_BLAS_LIB_DIR "${POGS_BLAS_LIBRARIES}" DIRECTORY)
get_filename_component(POGS_BLAS_LIB_NAME "${POGS_BLAS_LIBRARIES}" NAME)

target_link_directories(pogs_cpu_shared PUBLIC "${POGS_BLAS_LIB_DIR}")
target_link_libraries(pogs_cpu_shared PUBLIC "${POGS_BLAS_LIB_NAME}")
```

**Result**: Same linker errors.

### 7. target_link_options

```cmake
target_link_options(pogs_cpu_shared PUBLIC "${POGS_BLAS_LIBRARIES}")
```

**Result**: Same linker errors.

### 8. Direct LINK_FLAGS Property

```cmake
set_property(TARGET pogs_cpu_shared APPEND_STRING PROPERTY LINK_FLAGS " \"${POGS_BLAS_LIBRARIES}\"")
```

**Result**: Same linker errors.

## CI Configuration

The CI workflow was configured as:

```yaml
# .github/workflows/ci.yml
wheels:
  name: Wheels (${{ matrix.os }})
  runs-on: ${{ matrix.os }}
  strategy:
    matrix:
      os: [ubuntu-22.04, macos-14, windows-2022]

  steps:
    - uses: actions/checkout@v4

    - name: Build wheels
      uses: pypa/cibuildwheel@v2.22
      env:
        CIBW_BUILD: "cp312-*"
        CIBW_BEFORE_BUILD_WINDOWS: >
          pip install scipy-openblas32 delvewheel
        CIBW_ENVIRONMENT_WINDOWS: >
          CMAKE_ARGS="-DPOGS_BUILD_GPU=OFF -DPOGS_BUILD_TESTS=OFF -DPOGS_USE_SCIPY_OPENBLAS=ON"
        CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: >
          python -c "import scipy_openblas32; print(scipy_openblas32.get_lib_dir())" > openblas_path.txt &&
          set /p OPENBLAS_PATH=<openblas_path.txt &&
          delvewheel repair --add-path %OPENBLAS_PATH% -w {dest_dir} {wheel}
```

And pyproject.toml:

```toml
[build-system]
requires = [
    "scikit-build-core>=0.10",
    "pybind11>=2.12",
    "scipy-openblas32; platform_system == 'Windows'",
]
```

## Key Observations

1. **Static library builds succeed**: The static library `pogs_cpu.lib` builds without errors because static libraries don't resolve symbols at build time.

2. **Shared library fails**: Only `pogs_cpu_shared.dll` fails because shared libraries must resolve all symbols at link time.

3. **Library paths are correct**: CMake output confirms the library is found at the correct path.

4. **Symbols exist in the library**: scipy-openblas32 is used by numpy/scipy which use the same CBLAS interface.

5. **CMake shows the library**: The `message()` calls confirm the library path is being used.

## Possible Root Causes

1. **CMake not passing library to MSVC linker**: Despite all approaches, the `.lib` file may not be getting passed to `link.exe`.

2. **Path format issues**: Windows paths with spaces or forward/back slashes might cause issues.

3. **scikit-build-core isolation**: The build happens in an isolated environment which might affect how libraries are found.

4. **scipy-openblas32 designed for meson**: The library is primarily used by numpy/scipy which use meson, not CMake. The CMake config may be incomplete.

## Potential Solutions to Investigate

1. **Verbose linker output**: Add `/VERBOSE` to linker flags to see what libraries are actually being passed.

2. **Check exported symbols**: Use `dumpbin /exports libscipy_openblas.dll` to verify CBLAS symbols are exported.

3. **Study numpy's build**: Look at how numpy builds on Windows with scipy-openblas32 (uses meson).

4. **Try Intel MKL**: Use Intel MKL instead of OpenBLAS for Windows builds.

5. **Use vcpkg**: Install OpenBLAS via vcpkg which has better CMake integration.

6. **Static linking**: Try statically linking OpenBLAS instead of using the DLL.

## Solution Implemented

The root cause was that scipy-openblas32 exports **prefixed symbols** (e.g., `scipy_cblas_sdot` instead of `cblas_sdot`). The solution uses compile-time macro mapping to remap symbol names.

### Key Changes

1. **Symbol prefix header** (`src/include/openblas_prefix.h`):
   ```c
   #ifdef _WIN32
   #define cblas_sdot scipy_cblas_sdot
   #define cblas_ddot scipy_cblas_ddot
   // ... all CBLAS and LAPACK symbols
   #endif
   ```

2. **CMake changes** (`CMakeLists.txt`):
   - Added `POGS_USE_SCIPY_OPENBLAS` option
   - Created `pogs_openblas` imported target pointing to scipy-openblas32
   - Skips `find_package(BLAS/LAPACK)` on Windows

3. **Force-include** (`src/CMakeLists.txt`):
   ```cmake
   if(POGS_USE_SCIPY_OPENBLAS)
     target_compile_options(pogs_cpu_shared PRIVATE "/FI${POGS_OPENBLAS_PREFIX_HEADER}")
   endif()
   ```

4. **DLL bundling** (via delvewheel in CI):
   - `delvewheel repair --add-path %OPENBLAS_PATH% ...`
   - Creates self-contained wheel with OpenBLAS DLL included

### Current Status

Windows builds are now enabled in CI. Users can `pip install pogs` on Windows with zero extra steps - the wheel is self-contained with OpenBLAS bundled.

```yaml
# .github/workflows/ci.yml
wheels:
  strategy:
    matrix:
      os: [ubuntu-22.04, macos-14, windows-2022]
```
