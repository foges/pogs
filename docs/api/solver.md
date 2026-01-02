# Core Solver API

Reference for the main POGS solver interface.

---

## Solver Class

```cpp
namespace pogs {

template<typename T, Matrix M>
class Solver {
public:
    explicit Solver(std::unique_ptr<M> matrix);

    void configure(const SolverConfig& config);
    const SolverConfig& config() const;

    Solution<T> solve(std::span<const FunctionObj<T>> f,
                     std::span<const FunctionObj<T>> g);

    void set_warm_start(std::span<const T> x, std::span<const T> y);

    // Move semantics
    Solver(Solver&&) noexcept = default;
    Solver& operator=(Solver&&) noexcept = default;

    // Deleted copy (expensive)
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
};

} // namespace pogs
```

---

## Solution Structure

```cpp
template<typename T>
struct Solution {
    Status status;                    // Solver status
    std::vector<T> x;                 // Primal solution
    std::vector<T> y;                 // Auxiliary variables
    std::optional<T> primal_obj;      // Primal objective value
    std::optional<T> dual_obj;        // Dual objective value
    size_t iterations;                // Number of iterations
};
```

### Status Enumeration

```cpp
enum class Status {
    Success = 0,               // Converged successfully
    MaxIterations,             // Maximum iterations reached
    NumericalError,            // Numerical error encountered
    InfeasibleOrUnbounded      // Problem is infeasible or unbounded
};
```

---

## Factory Function

```cpp
template<typename T, Matrix M>
auto make_solver(std::unique_ptr<M> matrix) -> Solver<T, M>;
```

**Example:**
```cpp
auto A = std::make_unique<pogs::MatrixDense<double>>(m, n);
auto solver = pogs::make_solver<double>(std::move(A));
```

---

## Methods

### configure()

Set solver configuration parameters.

```cpp
void configure(const SolverConfig& config);
```

**Parameters:**
- `config`: Solver configuration (see [Configuration](configuration.md))

**Example:**
```cpp
auto config = pogs::SolverConfig{
    .rho = 1.0,
    .abs_tol = 1e-4,
    .rel_tol = 1e-3,
    .max_iter = 1000,
    .verbose = true
};
solver.configure(config);
```

---

### solve()

Solve the optimization problem.

```cpp
Solution<T> solve(std::span<const FunctionObj<T>> f,
                 std::span<const FunctionObj<T>> g);
```

**Parameters:**
- `f`: Array of m function objects for $f(y)$
- `g`: Array of n function objects for $g(x)$

**Returns:** Solution structure with status and results

**Example:**
```cpp
std::vector<pogs::FunctionObj<double>> f(m);
std::vector<pogs::FunctionObj<double>> g(n);

// ... configure f and g ...

auto result = solver.solve(f, g);

if (result.status == pogs::Status::Success) {
    std::cout << "Optimal value: " << result.primal_obj.value() << "\n";
}
```

---

### set_warm_start()

Provide initial guess to accelerate convergence.

```cpp
void set_warm_start(std::span<const T> x, std::span<const T> y);
```

**Parameters:**
- `x`: Initial guess for primal variables (size n)
- `y`: Initial guess for auxiliary variables (size m)

**Example:**
```cpp
// Solve first problem
auto result1 = solver.solve(f1, g1);

// Warm start for similar problem
solver.set_warm_start(result1.x, result1.y);
auto result2 = solver.solve(f2, g2);  // Faster!
```

---

## Matrix Interface

### Dense Matrix

```cpp
template<typename T>
class MatrixDense {
public:
    MatrixDense(size_t m, size_t n);

    size_t rows() const;
    size_t cols() const;

    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;
};
```

### Sparse Matrix

```cpp
template<typename T>
class MatrixSparse {
public:
    MatrixSparse(size_t m, size_t n, size_t nnz);

    size_t rows() const;
    size_t cols() const;
    size_t nnz() const;  // Number of non-zeros

    // CSR/CSC access methods
};
```

---

## Concepts

The solver uses C++20 concepts for template constraints:

```cpp
template<typename T>
concept Numeric = std::is_floating_point_v<T>;

template<typename M>
concept Matrix = requires(M m) {
    { m.rows() } -> std::convertible_to<size_t>;
    { m.cols() } -> std::convertible_to<size_t>;
};
```

---

## Thread Safety

- **Solver instance**: Not thread-safe
- **Multiple solvers**: Safe to use in different threads
- **Const methods**: Safe for concurrent calls

---

## Memory Management

All memory is managed automatically using RAII:

- Smart pointers for matrix ownership
- `std::vector` for internal state
- No manual `new`/`delete` required
- Move semantics for efficiency

---

## See Also

- [Types](types.md) - Function objects and enumerations
- [Configuration](configuration.md) - Solver configuration options
- [Proximal Operators](proximal.md) - Supported functions
