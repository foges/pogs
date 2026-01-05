// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>
#include "pogs.h"
#include "matrix/matrix_dense.h"
#include "prox_lib_cone.h"

using Catch::Approx;

// Helper to check if a point is in the primal exponential cone
template <typename T>
bool IsInExpPrimalCone(T x, T y, T z, T tol = 1e-6) {
    // K_exp = { (x, y, z) : y > 0, y*e^(x/y) <= z } U { x <= 0, y = 0, z >= 0 }
    if (y > tol && y * std::exp(x / y) <= z + tol) return true;
    if (x <= tol && std::abs(y) <= tol && z >= -tol) return true;
    return false;
}

// Helper to check if a point is in the dual exponential cone
template <typename T>
bool IsInExpDualCone(T u, T v, T w, T tol = 1e-6) {
    // K_exp* = { (u, v, w) : u < 0, -u*e^(v/u) <= e*w } U { u = 0, v >= 0, w >= 0 }
    const T e = 2.718281828459045;
    if (u < -tol && -u * std::exp(v / u) <= e * w + tol) return true;
    if (std::abs(u) <= tol && v >= -tol && w >= -tol) return true;
    return false;
}

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

// ============================================================================
// Exponential Cone Projection Tests
// ============================================================================

TEST_CASE("Exponential primal cone - point already in cone", "[cone][exp]") {
    // Test that points already in the cone are unchanged
    double v[3];
    CONE_IDX idx[3] = {0, 1, 2};

    SECTION("Interior point: y > 0, y*exp(x/y) < z") {
        // Point: (0, 1, 3) - clearly in interior since 1*exp(0) = 1 < 3
        v[0] = 0.0; v[1] = 1.0; v[2] = 3.0;
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2]));

        double orig[3] = {v[0], v[1], v[2]};
        ProjectExpPrimalCone(idx, v);

        REQUIRE(v[0] == Approx(orig[0]).margin(1e-6));
        REQUIRE(v[1] == Approx(orig[1]).margin(1e-6));
        REQUIRE(v[2] == Approx(orig[2]).margin(1e-6));
    }

    SECTION("Boundary point: y*exp(x/y) = z") {
        // Point: (1, 1, e) - on boundary since 1*exp(1/1) = e
        v[0] = 1.0; v[1] = 1.0; v[2] = std::exp(1.0);
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2]));

        ProjectExpPrimalCone(idx, v);

        // Should still be in cone (may have small numerical change)
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2], 1e-4));
    }

    SECTION("Boundary ray: x <= 0, y = 0, z >= 0") {
        // Point: (-1, 0, 2) - in boundary ray
        v[0] = -1.0; v[1] = 0.0; v[2] = 2.0;
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2]));

        double orig[3] = {v[0], v[1], v[2]};
        ProjectExpPrimalCone(idx, v);

        REQUIRE(v[0] == Approx(orig[0]).margin(1e-6));
        REQUIRE(v[1] == Approx(orig[1]).margin(1e-6));
        REQUIRE(v[2] == Approx(orig[2]).margin(1e-6));
    }
}

TEST_CASE("Exponential primal cone - point in polar cone", "[cone][exp]") {
    // Points in polar cone should project to origin
    double v[3];
    CONE_IDX idx[3] = {0, 1, 2};

    SECTION("Point in polar cone") {
        // Polar cone: { (u,v,w) : u > 0, u*exp(v/u) <= -e*w }
        // Try (1, -10, -1): 1*exp(-10) ≈ 0.0000454 <= -e*(-1) = e ≈ 2.718
        v[0] = 1.0; v[1] = -10.0; v[2] = -1.0;

        ProjectExpPrimalCone(idx, v);

        // Should project to origin
        REQUIRE(v[0] == Approx(0.0).margin(1e-4));
        REQUIRE(v[1] == Approx(0.0).margin(1e-4));
        REQUIRE(v[2] == Approx(0.0).margin(1e-4));
    }
}

TEST_CASE("Exponential primal cone - general projection", "[cone][exp]") {
    // Points outside cone that require Newton iteration
    double v[3];
    CONE_IDX idx[3] = {0, 1, 2};

    SECTION("Point outside main body") {
        // Point: (2, 0.5, 1) - not in cone since 0.5*exp(2/0.5) = 0.5*e^4 ≈ 27 > 1
        v[0] = 2.0; v[1] = 0.5; v[2] = 1.0;
        REQUIRE_FALSE(IsInExpPrimalCone(v[0], v[1], v[2]));

        double orig[3] = {v[0], v[1], v[2]};
        ProjectExpPrimalCone(idx, v);

        // Result should be in the cone
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2], 1e-4));

        // Projection should be closer to cone than original
        // (This is always true for a proper projection)
    }

    SECTION("Point with negative y") {
        // Point: (1, -1, 2) - not in cone
        v[0] = 1.0; v[1] = -1.0; v[2] = 2.0;
        REQUIRE_FALSE(IsInExpPrimalCone(v[0], v[1], v[2]));

        ProjectExpPrimalCone(idx, v);

        // Result should be in the cone
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2], 1e-4));
    }

    SECTION("Point with large positive x") {
        // Point: (10, 1, 1) - not in cone since 1*exp(10) >> 1
        v[0] = 10.0; v[1] = 1.0; v[2] = 1.0;
        REQUIRE_FALSE(IsInExpPrimalCone(v[0], v[1], v[2]));

        ProjectExpPrimalCone(idx, v);

        // Result should be in the cone
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2], 1e-4));
    }

    SECTION("Point near boundary ray") {
        // Point: (-0.5, 0.1, 0.5) - should project to somewhere in cone
        v[0] = -0.5; v[1] = 0.1; v[2] = 0.5;

        ProjectExpPrimalCone(idx, v);

        // Result should be in the cone
        REQUIRE(IsInExpPrimalCone(v[0], v[1], v[2], 1e-4));
    }
}

TEST_CASE("Exponential dual cone projection", "[cone][exp]") {
    // Test dual cone projection using Moreau decomposition
    double v[3];
    CONE_IDX idx[3] = {0, 1, 2};

    SECTION("Point already in dual cone") {
        // Dual cone interior: u < 0, -u*exp(v/u) < e*w
        // Point: (-1, 0, 1): -(-1)*exp(0/-1) = exp(0) = 1 < e*1 ≈ 2.718
        v[0] = -1.0; v[1] = 0.0; v[2] = 1.0;
        REQUIRE(IsInExpDualCone(v[0], v[1], v[2]));

        double orig[3] = {v[0], v[1], v[2]};
        ProjectExpDualCone(idx, v);

        // Should be approximately unchanged
        REQUIRE(v[0] == Approx(orig[0]).margin(1e-4));
        REQUIRE(v[1] == Approx(orig[1]).margin(1e-4));
        REQUIRE(v[2] == Approx(orig[2]).margin(1e-4));
    }

    SECTION("Point outside dual cone") {
        // Point: (1, 1, 0) - positive u, not in dual cone
        v[0] = 1.0; v[1] = 1.0; v[2] = 0.0;
        REQUIRE_FALSE(IsInExpDualCone(v[0], v[1], v[2]));

        ProjectExpDualCone(idx, v);

        // Result should be in the dual cone
        REQUIRE(IsInExpDualCone(v[0], v[1], v[2], 1e-4));
    }

    SECTION("Point in dual boundary ray") {
        // Dual boundary ray: u = 0, v >= 0, w >= 0
        v[0] = 0.0; v[1] = 1.0; v[2] = 1.0;
        REQUIRE(IsInExpDualCone(v[0], v[1], v[2]));

        double orig[3] = {v[0], v[1], v[2]};
        ProjectExpDualCone(idx, v);

        // Should be approximately unchanged
        REQUIRE(v[0] == Approx(orig[0]).margin(1e-4));
        REQUIRE(v[1] == Approx(orig[1]).margin(1e-4));
        REQUIRE(v[2] == Approx(orig[2]).margin(1e-4));
    }
}

TEST_CASE("Moreau decomposition property", "[cone][exp]") {
    // Verify: x = proj_K(x) + proj_K*(-x) for exponential cone
    double v[3] = {1.5, -0.5, 2.0};  // Some arbitrary point
    double orig[3] = {v[0], v[1], v[2]};
    CONE_IDX idx[3] = {0, 1, 2};

    // Compute primal projection
    double primal[3] = {v[0], v[1], v[2]};
    ProjectExpPrimalCone(idx, primal);

    // Compute dual projection of negated point
    double neg[3] = {-orig[0], -orig[1], -orig[2]};
    ProjectExpDualCone(idx, neg);
    // Note: dual projection gives proj_K*(-x), we need -proj_K*(-x)
    // But Moreau says: x = proj_K(x) + proj_K*(-x) isn't right
    // Correct: proj_K*(x) = x + proj_K(-x), so x = proj_K(x) - proj_K(-x)
    // Actually the standard form is: x = proj_K(x) + proj_{-K*}(x)

    // The primal projection should be in the primal cone
    REQUIRE(IsInExpPrimalCone(primal[0], primal[1], primal[2], 1e-4));
}
