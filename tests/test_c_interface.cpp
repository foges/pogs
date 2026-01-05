// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>

extern "C" {
#include "interface_c/pogs_c.h"
}

using Catch::Approx;

TEST_CASE("C Interface - Dense Lasso", "[c_interface][dense]") {
    // Problem: min 0.5||Ax - b||^2 + lambda||x||_1
    // Using: A = [1 1; 1 -1], b = [2; 0], lambda = 0.1
    // Solution should be approximately x = [1, 1]

    const size_t m = 2;
    const size_t n = 2;
    const double lambda = 0.1;

    // Matrix A (row-major)
    std::vector<double> A = {1.0, 1.0, 1.0, -1.0};
    std::vector<double> b = {2.0, 0.0};

    // f(y) = 0.5 * (y - b)^2
    std::vector<double> f_a(m, 1.0);
    std::vector<double> f_b = b;  // shift by b
    std::vector<double> f_c(m, 1.0);
    std::vector<double> f_d(m, 0.0);
    std::vector<double> f_e(m, 0.0);
    std::vector<FUNCTION> f_h(m, SQUARE);

    // g(x) = lambda * |x|
    std::vector<double> g_a(n, 1.0);
    std::vector<double> g_b(n, 0.0);
    std::vector<double> g_c(n, lambda);
    std::vector<double> g_d(n, 0.0);
    std::vector<double> g_e(n, 0.0);
    std::vector<FUNCTION> g_h(n, ABS);

    // Output
    std::vector<double> x(n), y(m), l(m);
    double optval;
    unsigned int final_iter;

    int status = PogsD(ROW_MAJ, m, n, A.data(),
                       f_a.data(), f_b.data(), f_c.data(),
                       f_d.data(), f_e.data(), f_h.data(),
                       g_a.data(), g_b.data(), g_c.data(),
                       g_d.data(), g_e.data(), g_h.data(),
                       1.0,      // rho
                       1e-4,     // abs_tol
                       1e-3,     // rel_tol
                       1000,     // max_iter
                       0,        // verbose
                       1,        // adaptive_rho
                       0,        // gap_stop
                       x.data(), y.data(), l.data(),
                       &optval, &final_iter);

    REQUIRE(status == 0);  // POGS_SUCCESS
    REQUIRE(optval >= 0);  // Objective should be non-negative

    // Solution should satisfy Ax approximately equals y
    double residual = std::abs(A[0]*x[0] + A[1]*x[1] - y[0]) +
                      std::abs(A[2]*x[0] + A[3]*x[1] - y[1]);
    REQUIRE(residual < 0.1);
}

TEST_CASE("C Interface - Cone Form LP", "[c_interface][cone]") {
    // Problem: min x[0] s.t. x[0] + x[1] = 2, x >= 0
    // Solution: x = [0, 2], optimal value = 0

    const size_t m = 1;
    const size_t n = 2;

    std::vector<double> A = {1.0, 1.0};  // row-major
    std::vector<double> b = {2.0};
    std::vector<double> c = {1.0, 0.0};

    // Kx: x >= 0 (NonNeg cone for all variables)
    std::vector<unsigned int> kx_indices = {0, 1};
    ConeConstraintC cones_x[1] = {{CONE_NON_NEG, kx_indices.data(), 2}};

    // Ky: Ax = b (Zero cone for equality constraint)
    std::vector<unsigned int> ky_indices = {0};
    ConeConstraintC cones_y[1] = {{CONE_ZERO, ky_indices.data(), 1}};

    // Output
    std::vector<double> x(n), y(m), l(m);
    double optval;
    unsigned int final_iter;

    int status = PogsConeD(ROW_MAJ, m, n, A.data(),
                           b.data(), c.data(),
                           cones_x, 1,  // num_cones_x
                           cones_y, 1,  // num_cones_y
                           1.0,         // rho
                           1e-4,        // abs_tol
                           1e-4,        // rel_tol
                           1000,        // max_iter
                           0,           // verbose
                           1,           // adaptive_rho
                           0,           // gap_stop
                           x.data(), y.data(), l.data(),
                           &optval, &final_iter);

    REQUIRE(status == 0);  // POGS_SUCCESS
    REQUIRE(optval == Approx(0.0).margin(0.01));
    REQUIRE(x[0] == Approx(0.0).margin(0.01));
    REQUIRE(x[1] == Approx(2.0).margin(0.01));
}

TEST_CASE("C Interface - Cone Form Direct", "[c_interface][cone]") {
    // Same LP but using direct projector
    const size_t m = 1;
    const size_t n = 2;

    std::vector<double> A = {1.0, 1.0};
    std::vector<double> b = {2.0};
    std::vector<double> c = {1.0, 0.0};

    std::vector<unsigned int> kx_indices = {0, 1};
    ConeConstraintC cones_x[1] = {{CONE_NON_NEG, kx_indices.data(), 2}};

    std::vector<unsigned int> ky_indices = {0};
    ConeConstraintC cones_y[1] = {{CONE_ZERO, ky_indices.data(), 1}};

    std::vector<double> x(n), y(m), l(m);
    double optval;
    unsigned int final_iter;

    int status = PogsConeDirectD(ROW_MAJ, m, n, A.data(),
                                  b.data(), c.data(),
                                  cones_x, 1,
                                  cones_y, 1,
                                  1.0, 1e-4, 1e-4, 1000, 0, 1, 0,
                                  x.data(), y.data(), l.data(),
                                  &optval, &final_iter);

    REQUIRE(status == 0);
    REQUIRE(optval == Approx(0.0).margin(0.01));
}

TEST_CASE("C Interface - FUNCTION enum values", "[c_interface]") {
    // Verify enum values match prox_lib.h
    REQUIRE(ABS == 0);
    REQUIRE(SQUARE == 14);
    REQUIRE(ZERO == 15);
}

TEST_CASE("C Interface - CONE enum values", "[c_interface]") {
    // Verify cone enum values
    REQUIRE(CONE_ZERO == 0);
    REQUIRE(CONE_NON_NEG == 1);
    REQUIRE(CONE_NON_POS == 2);
    REQUIRE(CONE_SOC == 3);
}
