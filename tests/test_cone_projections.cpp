// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>
#include "pogs.h"
#include "matrix/matrix_dense.h"

using Catch::Approx;

// Test cone form solving end-to-end
TEST_CASE("Simple LP with cone constraints", "[cone][integration]") {
    // Problem: minimize x[0] subject to x[0] + x[1] = 2, x >= 0
    // Solution: x = [0, 2], optimal value = 0

    const size_t m = 1;
    const size_t n = 2;

    std::vector<double> A_data = {1.0, 1.0};
    std::vector<double> b = {2.0};
    std::vector<double> c = {1.0, 0.0};

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<ConeConstraint> Kx = {{kConeNonNeg, {0, 1}}};
    std::vector<ConeConstraint> Ky = {{kConeZero, {0}}};

    pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);

    solver.SetMaxIter(1000);
    solver.SetAbsTol(1e-5);
    solver.SetRelTol(1e-5);
    solver.SetVerbose(0);

    pogs::PogsStatus status = solver.Solve(b, c);

    REQUIRE(status == pogs::POGS_SUCCESS);
    REQUIRE(solver.GetOptval() == Approx(0.0).margin(1e-4));
}

TEST_CASE("LP with inequality constraints", "[cone][integration]") {
    // min x[0] s.t. [-1 -1; 1 -1; 0 1] * x <= [0; 0; 2], x unconstrained

    const size_t m = 3;
    const size_t n = 2;

    std::vector<double> A_data = {-1.0, -1.0, 1.0, -1.0, 0.0, 1.0};
    std::vector<double> b = {0.0, 0.0, 2.0};
    std::vector<double> c = {1.0, 0.0};

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<ConeConstraint> Kx;  // x unconstrained
    std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};

    pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);

    solver.SetMaxIter(1000);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetVerbose(0);

    pogs::PogsStatus status = solver.Solve(b, c);

    REQUIRE(status == pogs::POGS_SUCCESS);

    // Should converge to a finite value
    double optval = solver.GetOptval();
    REQUIRE(std::isfinite(optval));
}

TEST_CASE("Cone constraints work correctly", "[cone]") {
    // Verify that different cone types can be combined

    const size_t m = 3;
    const size_t n = 3;

    // Identity matrix - ensures feasible system Ax = b
    std::vector<double> A_data = {1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 1.0};
    std::vector<double> b_data = {1.0, 1.0, 1.0};  // Feasible: x = [1,1,1]
    std::vector<double> c_data = {0.0, 0.0, 0.0};

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    SECTION("NonNeg cone on variables") {
        std::vector<ConeConstraint> Kx = {{kConeNonNeg, {0, 1, 2}}};
        std::vector<ConeConstraint> Ky = {{kConeZero, {0, 1, 2}}};

        pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);
        solver.SetMaxIter(1000);
        solver.SetAbsTol(1e-4);
        solver.SetRelTol(1e-3);
        solver.SetVerbose(0);

        pogs::PogsStatus status = solver.Solve(b_data, c_data);
        REQUIRE(status == pogs::POGS_SUCCESS);
    }

    SECTION("Mixed cone constraints") {
        // Some variables in NonNeg, some unconstrained
        std::vector<ConeConstraint> Kx = {{kConeNonNeg, {0, 1}}};  // First two non-negative
        std::vector<ConeConstraint> Ky = {{kConeZero, {0, 1, 2}}};

        pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);
        solver.SetMaxIter(1000);
        solver.SetAbsTol(1e-4);
        solver.SetRelTol(1e-3);
        solver.SetVerbose(0);

        pogs::PogsStatus status = solver.Solve(b_data, c_data);
        REQUIRE(status == pogs::POGS_SUCCESS);
    }
}
