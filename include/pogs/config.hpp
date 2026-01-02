// POGS Configuration Header
// Copyright (c) 2024-2026 POGS Contributors
// Licensed under Apache 2.0

#pragma once

#include <cstddef>

namespace pogs {

// Solver configuration using modern C++20 designated initializers
struct SolverConfig {
    // Penalty parameter
    double rho = 1.0;

    // Tolerances
    double abs_tol = 1e-4;
    double rel_tol = 1e-3;

    // Iteration control
    size_t max_iter = 1000;
    bool verbose = false;

    // Adaptive rho
    bool adaptive_rho = true;

    // Gap stopping criterion
    bool gap_stop = true;

    // Reserved for future use (Anderson acceleration, etc.)
    bool use_anderson = false;
    size_t anderson_mem = 5;
    size_t anderson_start = 10;
};

} // namespace pogs
