# Changelog

All notable changes to POGS are documented here.

---

## [Unreleased] - Version 0.4.0

### Added

**Build System:**
- CMake build system - Modern, cross-platform build replacing Makefiles
- Package configuration for `find_package(POGS)` support
- Installation support with proper headers and libraries

**C++20 Features:**
- Full migration to C++20 standard
- Modern type system with enum classes (`FunctionType`, `ConeType`, `Status`, `Ord`)
- RAII-based memory management (`ADMMState` class using `std::vector`)
- Designated initializers for configuration (`SolverConfig`)
- New public API headers in `include/pogs/`:
  - `types.hpp` - Modern type definitions
  - `config.hpp` - Solver configuration
  - `c/pogs_c.h` - C interface

**Optimization Features:**
- Cone form support with SDP projection
- C interface for cone form problems (`PogsConeD`, `PogsConeF`)
- Python/CVXPY integration for high-level optimization modeling
- Comprehensive test suite for cone projections (48 tests passing)

**Documentation:**
- Modern MkDocs Material documentation site
- Comprehensive user guides and API reference
- Practical examples (Lasso, Logistic Regression, SDP)
- Developer documentation and contribution guide

### Changed

- **Build system**: Migrated from Makefiles to CMake
- **C++ standard**: Upgraded from C++11 to C++20
- **Memory management**: Foundation for RAII patterns with ADMMState class
- **Code quality**: Fixed C++17/C++20 compatibility issues (removed `std::unary_function`)
- **Project structure**: Modern header organization
- **Documentation**: Migrated to MkDocs Material with search and mobile support

### Deprecated

- Old Makefile-based build system (use CMake instead)
- Graph form interface (use cone form for new projects)

### Removed

**MATLAB Interface:**
- Removed pedagogical MATLAB implementation (unmaintained)
- **Migration path**: Use Python with CVXPY for high-level modeling
- **Alternative**: Use C/C++ interface for performance-critical applications

**R Interface:**
- Removed due to lack of maintenance

**Old Documentation:**
- Removed outdated gh-pages Jekyll documentation
- Replaced with modern MkDocs Material site

### Migration Notes

**For MATLAB Users:**
- Migrate to Python with CVXPY (see [CVXPY Integration](../user-guide/cvxpy-integration.md))
- Python provides similar high-level modeling with better ecosystem
- Easy installation: `pip install cvxpy`

**For Build System Users:**
- Replace `make cpu` with `cmake --build build`
- See [Installation Guide](../getting-started/installation.md) for details

---

## [0.3.0] - Previous Release

Initial release with:
- Graph form solver
- MATLAB interface
- Basic C++ implementation
- CPU and GPU backends
- Examples and documentation

---

## Version History

- **0.4.0** (Unreleased): C++20 modernization, CMake, MkDocs documentation
- **0.3.0** (2016): Initial public release
- **0.2.0** (2015): Internal development
- **0.1.0** (2014): Research prototype

---

## Release Process

POGS follows [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0): Breaking API changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

---

## See Also

- [Modernization](../developer/modernization.md) - Modernization progress
- [License](license.md) - Project license
- [Authors](authors.md) - Contributors
