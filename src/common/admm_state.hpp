// ADMM State Management - RAII-based
// Copyright (c) 2024-2026 POGS Contributors
// Licensed under Apache 2.0

#pragma once

#include <vector>
#include <memory>
#include <cstddef>

namespace pogs {
namespace detail {

// RAII wrapper for ADMM solver state
// Replaces all the raw pointer allocations in the old Pogs class
// with automatic memory management using std::vector
template<typename T>
class ADMMState {
public:
    // Constructor: allocates all arrays automatically
    ADMMState(size_t m, size_t n)
        : m_(m), n_(n),
          // Primal and dual variables
          x_(n), y_(m),
          // Lagrange multipliers
          mu_(n), lambda_(m),
          // Temporary variables for ADMM
          x12_(n), y12_(m),
          z_(m + n), zprev_(m + n), ztemp_(m + n),
          zt_(m + n),
          // Additional workspace
          z12_(m + n) {
        // All memory automatically initialized to zero
    }

    // Destructor: automatic cleanup (no delete needed!)
    ~ADMMState() = default;

    // Move semantics (no copy - solver state should be unique)
    ADMMState(ADMMState&&) noexcept = default;
    ADMMState& operator=(ADMMState&&) noexcept = default;

    // Delete copy operations (expensive and usually not needed)
    ADMMState(const ADMMState&) = delete;
    ADMMState& operator=(const ADMMState&) = delete;

    // Accessors - return raw pointers for compatibility with existing code
    // TODO Phase 3: Convert to std::span for safer access
    T* x() { return x_.data(); }
    T* y() { return y_.data(); }
    T* mu() { return mu_.data(); }
    T* lambda() { return lambda_.data(); }
    T* x12() { return x12_.data(); }
    T* y12() { return y12_.data(); }
    T* z() { return z_.data(); }
    T* zprev() { return zprev_.data(); }
    T* ztemp() { return ztemp_.data(); }
    T* zt() { return zt_.data(); }
    T* z12() { return z12_.data(); }

    // Const accessors
    const T* x() const { return x_.data(); }
    const T* y() const { return y_.data(); }
    const T* mu() const { return mu_.data(); }
    const T* lambda() const { return lambda_.data(); }
    const T* x12() const { return x12_.data(); }
    const T* y12() const { return y12_.data(); }
    const T* z() const { return z_.data(); }
    const T* zprev() const { return zprev_.data(); }
    const T* ztemp() const { return ztemp_.data(); }
    const T* zt() const { return zt_.data(); }
    const T* z12() const { return z12_.data(); }

    // Dimensions
    size_t m() const { return m_; }
    size_t n() const { return n_; }

    // Reset all state (useful for re-solving)
    void reset() {
        std::fill(x_.begin(), x_.end(), T{0});
        std::fill(y_.begin(), y_.end(), T{0});
        std::fill(mu_.begin(), mu_.end(), T{0});
        std::fill(lambda_.begin(), lambda_.end(), T{0});
        std::fill(x12_.begin(), x12_.end(), T{0});
        std::fill(y12_.begin(), y12_.end(), T{0});
        std::fill(z_.begin(), z_.end(), T{0});
        std::fill(zprev_.begin(), zprev_.end(), T{0});
        std::fill(ztemp_.begin(), ztemp_.end(), T{0});
        std::fill(zt_.begin(), zt_.end(), T{0});
        std::fill(z12_.begin(), z12_.end(), T{0});
    }

private:
    size_t m_, n_;

    // All state variables using std::vector (RAII!)
    std::vector<T> x_, y_;           // Primal variables
    std::vector<T> mu_, lambda_;     // Lagrange multipliers
    std::vector<T> x12_, y12_;       // Temporary primal variables
    std::vector<T> z_, zprev_, ztemp_, zt_;  // Combined variables
    std::vector<T> z12_;             // Temporary combined variable
};

} // namespace detail
} // namespace pogs
