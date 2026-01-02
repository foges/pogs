# Anderson Acceleration

Anderson acceleration is an experimental feature for accelerating ADMM convergence on certain problem types.

!!! warning "Experimental Feature"
    Anderson acceleration is **disabled by default** and should be used with caution. It may not provide speedup for all problem types.

---

## Overview

Anderson acceleration extrapolates better iterates by solving a least-squares problem over recent iteration history. It can help with:

- Ill-conditioned problems (condition number κ > 100)
- High-accuracy requirements (tolerance < 1e-6)
- Slowly converging ADMM (> 500 iterations)

---

## Algorithm

Given iterates $x_k, x_{k-1}, \ldots, x_{k-m}$:

1. Compute residuals: $r_i = x_i - x_{i-1}$
2. Build matrix: $F = [r_k - r_{k-1}, r_k - r_{k-2}, \ldots, r_k - r_{k-m}]$
3. Solve least-squares: $\min \|F\alpha + r_k\|_2$ (via QR decomposition)
4. Update: $x_{\text{acc}} = x_k + \sum_i \alpha_i (x_{k-i} - x_k)$

**Key Features:**
- QR-based least-squares (numerically stable)
- Circular buffer for memory efficiency
- Safeguarding to prevent destabilization
- Automatic reset when solver parameters change

---

## Usage

!!! note "Disabled by Default"
    Anderson acceleration must be explicitly enabled with `SetUseAnderson(true)`.

### Basic Example

```cpp
#include "pogs.h"

pogs::MatrixDense<double> A('c', m, n, A_data);
pogs::PogsDirect<double, pogs::MatrixDense<double>> pogs_data(A);

// Enable Anderson acceleration
pogs_data.SetUseAnderson(true);
pogs_data.SetAndersonMem(5);       // Memory depth (default: 5)
pogs_data.SetAndersonStart(10);    // Start after iteration 10 (default: 10)

pogs_data.Solve(f, g);
```

### Parameters

**`SetUseAnderson(bool flag)`**
- Enable/disable Anderson acceleration
- Default: `false`

**`SetAndersonMem(size_t m)`**
- Number of past iterates to store
- Default: `5`
- Recommended: `5-10` for ill-conditioned problems

**`SetAndersonStart(size_t k)`**
- Iteration number to start applying Anderson
- Default: `10`
- Recommended: `10-20` (start later for high-accuracy problems)

---

## Problem-Specific Recommendations

### Ill-Conditioned Problems (κ > 100)

```cpp
pogs_data.SetUseAnderson(true);
pogs_data.SetAndersonMem(10);      // Larger memory
pogs_data.SetAndersonStart(5);     // Start earlier
```

### High-Accuracy Requirements (tol < 1e-5)

```cpp
pogs_data.SetUseAnderson(true);
pogs_data.SetAndersonMem(10);      // Larger memory
pogs_data.SetAndersonStart(20);    // Start later (near convergence)
```

### Fast-Converging Problems (< 50 iterations)

```cpp
pogs_data.SetUseAnderson(false);   // Don't use Anderson (overhead not worth it)
```

---

## Benchmark Results

Results from comprehensive benchmarks:

| Problem Type                          | Iter (No AA) | Iter (AA) | Speedup |
|---------------------------------------|--------------|-----------|---------|
| Ill-conditioned (κ=10)               | 54           | 2499      | 0.02    |
| Ill-conditioned (κ=100)              | 61           | 61        | 1.00    |
| Ill-conditioned (κ=1000)             | 31           | 31        | 1.00    |
| Basis Pursuit (m=100, n=500)         | 2499         | 2499      | 1.00    |
| High-Accuracy Lasso (m=300, n=150)   | 249          | 4999      | 0.05    |

**Interpretation:**
- Mixed results: Anderson either doesn't help or makes convergence worse
- Safeguarding prevents destabilization but also limits improvements
- Current implementation does not provide consistent speedup

---

## Safeguarding

The implementation includes safeguards to prevent Anderson from destabilizing ADMM:

```cpp
// Reject Anderson update if change magnitude increases by >10x
if (||z_acc - z_prev||² < 10 * ||z - z_prev||²) {
    Accept Anderson update
} else {
    Reject and use standard ADMM iterate
}
```

---

## Known Limitations

1. **Variable Selection**: Accelerating combined variables may not be optimal
2. **Problem Type Dependency**: Limited success even on target problem types
3. **Overhead**: QR decomposition adds per-iteration cost
4. **Interaction with ADMM Features**: May conflict with over-relaxation and adaptive ρ

---

## Future Work

To improve Anderson acceleration:

1. **Alternative Formulations**:
   - Implement Douglas-Rachford acceleration
   - Try accelerating dual variables only
   - Experiment with accelerating residuals

2. **Better Integration**:
   - Coordinate with over-relaxation parameter
   - Adaptive Anderson parameters based on problem conditioning
   - Problem-type detection for automatic configuration

---

## Building and Testing

### Build POGS with Anderson Support

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The library includes Anderson acceleration (no special flags needed).

### Run Benchmarks

```bash
cd examples/cpp
g++ -I../../include -std=c++20 -O3 \
    anderson_benchmark.cpp -lpogs_cpu \
    -lm -framework Accelerate -o anderson_benchmark

./anderson_benchmark
```

---

## References

**Key Papers:**
1. Walker & Ni (2011) - "Anderson Acceleration for Fixed-Point Iterations" (SIAM J. Numer. Anal.)
2. Fu, Zhang, Boyd (2020) - "Anderson Accelerated Douglas-Rachford Splitting"
3. Ouyang et al. (2020) - "Anderson Acceleration for Nonconvex ADMM"

---

## Conclusion

Anderson acceleration provides solid infrastructure but limited practical effectiveness for ADMM. It is **disabled by default** and should be used experimentally with careful tuning.

**Recommendation**: Use standard ADMM for most problems. Enable Anderson only for specific ill-conditioned cases with careful parameter tuning.
