# Contributing to POGS

Thank you for your interest in contributing to POGS!

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/pogs.git
cd pogs

# Add upstream remote
git remote add upstream https://github.com/foges/pogs.git
```

### 2. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 3. Build and Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DPOGS_BUILD_TESTS=ON
cmake --build build
cd build && ctest
```

---

## Contribution Areas

### Code Contributions

- **Bug fixes**: Fix issues from the issue tracker
- **New features**: Add proximal operators, cone types, or optimizations
- **Performance**: Optimize hot paths, improve convergence
- **Testing**: Add test cases for better coverage

### Documentation

- **Examples**: Add practical examples
- **Tutorials**: Write guides for common use cases
- **API docs**: Improve API documentation
- **Translation**: Translate documentation to other languages

### Testing and Validation

- **Bug reports**: Report issues with detailed reproduction steps
- **Benchmarks**: Compare POGS with other solvers
- **Validation**: Test on different platforms and compilers

---

## Code Style

### C++ Style Guide

**Modern C++20:**
```cpp
// YES: Use modern C++20 features
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4
};

auto solver = pogs::make_solver<double>(std::move(matrix));

// NO: Don't use raw pointers
T* x = new T[n];  // Avoid!
```

**Naming conventions:**
- Types: `PascalCase` (e.g., `SolverConfig`, `FunctionType`)
- Functions: `snake_case` (e.g., `make_solver`, `set_warm_start`)
- Variables: `snake_case` (e.g., `abs_tol`, `max_iter`)
- Private members: `snake_case_` with trailing underscore (e.g., `state_`, `matrix_`)

**Memory management:**
- Use `std::unique_ptr` for ownership
- Use `std::vector` for arrays
- Use `std::span` for non-owning array views
- Avoid `new`/`delete`

### Formatting

Use clang-format:

```bash
# Format all source files
find src include -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | \
    xargs clang-format -i
```

---

## Testing

### Writing Tests

Add tests to `tests/` directory:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <pogs/types.hpp>

TEST_CASE("FunctionObj default constructor", "[types]") {
    pogs::FunctionObj<double> f;
    REQUIRE(f.type == pogs::FunctionType::Zero);
    REQUIRE(f.a == 1.0);
    REQUIRE(f.c == 1.0);
}
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commits are atomic and well-described
- [ ] No compiler warnings

### 2. Create Pull Request

**Good PR description:**

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Added proximal operator for new function type
- Updated tests to cover edge cases
- Added documentation example

## Testing
Ran all tests on:
- macOS 14.0 (AppleClang 15)
- Ubuntu 22.04 (GCC 11)

## Related Issues
Fixes #123
```

### 3. Review Process

- Maintainers will review your PR
- Address feedback and update PR
- Once approved, PR will be merged

---

## Commit Message Guidelines

**Format:**
```
<type>: <short summary>

<detailed description>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

**Examples:**

```
feat: Add Huber function proximal operator

Implement ProxHuber() with analytical formula. Includes tests
for various input values and rho parameters.

Closes #45
```

```
fix: Correct SDP cone projection for degenerate matrices

Handle case where all eigenvalues are negative. Previously
would return zero matrix, now correctly projects.

Fixes #67
```

---

## Development Workflow

### Typical Workflow

```bash
# 1. Sync with upstream
git fetch upstream
git checkout master
git merge upstream/master

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit code ...

# 4. Test
cmake --build build && cd build && ctest

# 5. Commit
git add src/my_file.cpp
git commit -m "feat: Add new feature"

# 6. Push and create PR
git push origin feature/my-feature
# Create PR on GitHub
```

### Iterating on Feedback

```bash
# Make requested changes
# ... edit code ...

# Amend commit or add new commit
git commit --amend  # or: git commit -m "Address review feedback"

# Force push (if amended)
git push --force origin feature/my-feature
```

---

## Adding New Features

### Adding a Proximal Operator

1. **Define function** in `include/pogs/types.hpp`:
   ```cpp
   enum class FunctionType {
       // ... existing ...
       MyNewFunction  // Add here
   };
   ```

2. **Implement proximal** in `src/include/prox_lib.h`:
   ```cpp
   template<typename T>
   inline void ProxMyNewFunction(T rho, T *x) {
       // Implementation
   }
   ```

3. **Add to dispatcher** in `src/include/prox_lib.h`:
   ```cpp
   case FunctionType::MyNewFunction:
       ProxMyNewFunction(rho, &x[i]);
       break;
   ```

4. **Add tests** in `tests/test_proximal.cpp`

5. **Document** in `docs/api/proximal.md`

### Adding a Cone Type

1. **Define cone** in `include/pogs/types.hpp`
2. **Implement projection** in `src/include/prox_lib_cone.h`
3. **Add to C interface** in `src/interface_c/pogs_c.h`
4. **Add tests** in `tests/test_cone.cpp`
5. **Document** in `docs/user-guide/cone-problems.md`

---

## Documentation

### Building Documentation

```bash
# Install MkDocs Material
pip install mkdocs-material

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add mathematical formulations where appropriate
- Link to related pages

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with [Feature Request] tag
- **Chat**: Join our community (link TBD)

---

## Code of Conduct

Be respectful and constructive. We want POGS to be welcoming to everyone.

---

## License

By contributing to POGS, you agree that your contributions will be licensed under the Apache 2.0 License.

---

## Thank You!

Your contributions make POGS better for everyone. We appreciate your time and effort!
