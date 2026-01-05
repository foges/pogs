# C++20 Modernization

Overview of the POGS modernization effort.

---

## Overview

POGS has undergone a comprehensive modernization from C++11 to C++20, replacing technical debt accumulated over 10+ years with modern best practices.

---

## Phase 1: Infrastructure (Completed)

### CMake Build System

**Old (Makefiles):**
- Platform-specific Makefiles
- Hardcoded compiler flags
- Manual dependency management
- Difficult integration

**New (CMake):**
- Cross-platform build
- Automatic dependency detection
- `find_package(POGS)` support
- Modern installation

### Cleanup

- Deleted unmaintained MATLAB interface
- Deleted unmaintained R interface
- Removed outdated examples
- Cleaned up directory structure

---

## Phase 2: C++20 Migration (Completed)

### Modern Type System

**Enum Classes:**
```cpp
// Old: Raw enums
enum Function { kAbs, kSquare, kZero };

// New: Enum classes
enum class FunctionType { Abs, Square, Zero };
```

**Benefits:**
- Type safety
- No implicit conversions
- Better error messages
- Scoped names

### RAII Memory Management

**Old:**
```cpp
Pogs() {
    _x = new T[n]();
    _y = new T[m]();
}

~Pogs() {
    delete[] _x;
    delete[] _y;
}
```

**New:**
```cpp
class ADMMState {
    std::vector<T> x_, y_;  // Automatic cleanup!
};
```

**Benefits:**
- No manual `new`/`delete`
- Automatic cleanup
- Exception safe
- Move semantics

### Modern Configuration

**Designated Initializers (C++20):**
```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4,
    .verbose = true
};
```

**Benefits:**
- Clear, readable
- Named parameters
- Default values
- Type safe

### Fixed C++17/C++20 Issues

**Removed deprecated code:**
```cpp
// Old: Deprecated in C++17, removed in C++20
template <typename T>
struct ReciprF : std::unary_function<T, T> {
    T operator()(T x) { return 1.0 / x; }
};

// New: Modern functor
template <typename T>
struct ReciprF {
    T operator()(T x) const { return 1.0 / x; }
};
```

---

## Phase 3: Documentation (In Progress)

### MkDocs Material

Modern, beautiful documentation:

- Clean design
- Mobile responsive
- Search functionality
- Code highlighting
- Mathematical typesetting

### Content Organization

```
docs/
├── getting-started/
├── user-guide/
├── api/
├── examples/
├── developer/
└── about/
```

---

## Future Phases

### Phase 4: Test Suite

- Catch2 integration
- Comprehensive unit tests
- Performance benchmarks
- Continuous integration

### Phase 5: Code Modernization

**Smart Pointers:**
```cpp
// Replace raw pointers throughout
template<typename T, Matrix M>
class Solver {
    std::unique_ptr<Impl> impl_;  // PIMPL
};
```

**std::span:**
```cpp
// Replace raw pointer parameters
void Prox(std::span<FunctionObj<T>> f,
          std::span<T> x);
```

**Concepts:**
```cpp
// Template constraints
template<Numeric T, DenseMatrix M>
class Solver { /* ... */ };
```

### Phase 6: Code Deduplication

Extract common ADMM logic from CPU/GPU implementations:

```cpp
namespace pogs::detail {

template<typename T, typename Backend>
class ADMMAlgorithm {
    // Common ADMM loop
    // Backend-specific operations delegated
};

} // namespace pogs::detail
```

---

## Modernization Benefits

### Code Quality

- ✅ Type safety with enum classes
- ✅ Memory safety with RAII
- ✅ Move semantics for efficiency
- ✅ Const correctness
- ⏳ Smart pointers everywhere (in progress)
- ⏳ std::span for safety (planned)

### Developer Experience

- ✅ Modern C++20 features
- ✅ Better compiler errors
- ✅ Less boilerplate
- ✅ Designated initializers
- ✅ CMake build system
- ✅ Better documentation

### Performance

- No regression (same algorithm)
- Potential improvements with move semantics
- Better optimization opportunities for compilers

---

## Migration Impact

### Breaking Changes

**Build System:**
- Old: `make cpu`
- New: `cmake --build build`

**API (Future):**
- Old graph form will be deprecated
- Transition to modern API

### Compatibility

- Old C++11 code still compiles (for now)
- Gradual migration strategy
- Clear migration guide

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| C++ Standard | C++20 | ✅ Complete |
| Build System | CMake | ✅ Complete |
| Memory Classes | RAII | 🔄 In Progress |
| Documentation | MkDocs | 🔄 In Progress |
| Tests Passing | 100% | ✅ 48/48 |
| Compiler Warnings | 0 critical | ✅ Complete |

---

## Timeline

- **Phase 1** (Infrastructure): ✅ Complete
- **Phase 2** (C++20): ✅ Complete
- **Phase 3** (Documentation): 🔄 In Progress
- **Phase 4** (Tests): ⏳ Planned
- **Phase 5** (Full Modernization): ⏳ Planned
- **Phase 6** (Deduplication): ⏳ Planned

---

## See Also

- [Architecture](architecture.md) - Current architecture
- [Contributing](contributing.md) - How to contribute
