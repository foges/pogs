# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.4.0] - 2026-01-02

**Major modernization release** - This version represents a complete overhaul of the POGS codebase with significant breaking changes. See the [Migration Guide](https://foges.github.io/pogs/migration/v0.3-to-v0.4/) for detailed upgrade instructions.

### Added

#### Build System (Phase 1)
- **CMake build system** - Modern, cross-platform build system replacing Makefiles
  - Supports CPU and GPU builds with `-DPOGS_BUILD_GPU=ON/OFF`
  - Automatic BLAS/LAPACK detection (Accelerate, OpenBLAS, MKL)
  - Automatic CUDA detection for GPU builds
  - Install target with proper header installation
  - Package configuration for `find_package(POGS)` support
  - Separate targets: `pogs::cpu`, `pogs::gpu`

#### C++20 Modernization (Phase 2)
- **C++20 compiler requirement** - Minimum GCC 10+, Clang 13+, AppleClang 13+, MSVC 19.29+
- **Modern type system foundation**:
  - Enum classes for `FunctionType`, `ConeType`, `Status`, `Ord` (foundation laid)
  - RAII-based memory management with `std::vector` for ADMM state
  - Removed deprecated `std::unary_function` (C++17 removal)
  - Fixed C++20 compatibility issues throughout codebase

#### Cone Form Support
- **Cone form solver** with full support for:
  - Zero cone (equality constraints)
  - Non-negative cone (inequality constraints)
  - Second-order cone (SOC/SOCP)
  - Semidefinite cone (SDP) with CPU and GPU projections
  - Exponential cone (primal and dual)
- **C interface** for cone form problems (`PogsConeD`, `PogsConeS`)
- **Anderson acceleration** for faster convergence

#### Python Integration
- **Python/CVXPY integration** for high-level optimization modeling
  - Subprocess-based interface (no compilation required)
  - Full support for graph form and cone form problems
  - Comprehensive documentation and examples

#### Testing (Phase 4)
- **Catch2 test framework** - Modern C++ testing with v3.5.1
  - Automatic test discovery via CMake
  - 8 integration tests for solver and cone projections
  - BDD-style syntax with `TEST_CASE`, `SECTION`, `REQUIRE`
  - Foundation for future unit tests (type system, proximal operators)

#### Documentation (Phase 3)
- **MkDocs Material documentation** - Modern, comprehensive documentation site
  - 30+ documentation pages covering all aspects of POGS
  - Full-text search functionality
  - Mobile-friendly responsive design
  - Dark mode support
  - Automatic deployment via GitHub Actions
  - **Sections**:
    - Getting Started (installation, quick start)
    - User Guide (basic usage, advanced features, cone problems, CVXPY, C interface)
    - API Reference (solver, types, configuration, proximal operators, C API)
    - Examples (Lasso, logistic regression, SDP, Anderson acceleration)
    - Developer Guide (architecture, building, contributing, modernization)
    - About (changelog, license, authors)
  - **Migration Guide (Phase 5)**: Comprehensive v0.3 → v0.4 upgrade guide
    - Build system migration (Makefile → CMake)
    - MATLAB → Python/CVXPY conversion
    - Compiler requirements (C++11 → C++20)
    - Platform-specific instructions
    - Troubleshooting section
    - Code examples for common scenarios

### Changed

#### Build System
- **Migrated from Makefiles to CMake** (breaking change)
  - All users must update their build process
  - Old: `cd src && make cpu`
  - New: `cmake -B build && cmake --build build`
  - See Migration Guide for integration into user projects

#### C++ Standard
- **Upgraded from C++11 to C++20** (breaking change)
  - Requires modern compiler (GCC 10+, Clang 13+)
  - Users with older compilers must upgrade
  - See Migration Guide for compiler update instructions

#### Code Quality
- Improved code organization and structure
- Fixed memory management patterns (foundation for full RAII)
- Eliminated deprecated C++ features
- Better error handling

#### Documentation
- **Replaced Jekyll site with MkDocs Material**
  - Old site: http://foges.github.io/pogs (outdated, 2014)
  - New site: https://foges.github.io/pogs/ (modern, 2026)
  - Automatic deployment via GitHub Actions

### Deprecated

- **Graph form interface** - Still works but cone form is recommended
  - Cone form provides more flexibility and better performance
  - Graph form will continue to be supported for backward compatibility

### Removed

#### MATLAB Interface (Breaking Change)
- **MATLAB interface completely removed**
  - **Reason**: Unmaintained since 2015, superseded by Python/CVXPY
  - **Migration path**: Use Python with CVXPY instead
  - **Benefits**: Better ecosystem, easier installation, more active development
  - See [Migration Guide](https://foges.github.io/pogs/migration/v0.3-to-v0.4/) for MATLAB → Python examples

#### R Interface
- **R interface removed**
  - Reason: Lack of maintenance and usage

#### Build System
- **Old Makefiles removed** (breaking change)
  - All Makefiles in `src/`, `examples/`, etc. removed
  - Use CMake instead (see Migration Guide)

#### Documentation
- **Old gh-pages Jekyll site removed**
  - Replaced with modern MkDocs Material site

### Fixed

- C++17/C++20 compatibility issues
- Memory management issues (foundation for future improvements)
- Build system portability issues
- Documentation outdated content (2014 → 2026)

### Migration Notes

**Breaking Changes Summary**:
1. **Build System**: Must use CMake (Makefiles removed)
2. **MATLAB**: Must migrate to Python/CVXPY
3. **Compiler**: Must use C++20 compiler (GCC 10+, Clang 13+)
4. **C++ API**: Legacy API still works (no code changes needed, just rebuild with CMake)

**Migration Resources**:
- **Comprehensive Migration Guide**: https://foges.github.io/pogs/migration/v0.3-to-v0.4/
- **MATLAB → Python Examples**: See migration guide
- **CMake Integration**: See migration guide
- **Troubleshooting**: See migration guide

**For v0.3 Users**:
- See the [Migration Guide](https://foges.github.io/pogs/migration/v0.3-to-v0.4/) for step-by-step upgrade instructions
- C++ API is backward compatible (just rebuild with CMake)
- MATLAB users must switch to Python/CVXPY
- Build system must be updated to CMake

**Performance**:
- No performance regression expected
- Same core ADMM algorithm as v0.3
- Potentially better performance with C++20 compiler optimizations

## [0.3.0] - Previous Release

Initial release with graph form solver, MATLAB interface, and basic C++ implementation.
