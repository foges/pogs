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

TEST_CASE("Simple LP with cone form", "[solver][integration]") {
    // Problem: minimize x[0] subject to x[0] + x[1] = 2, x >= 0
    // Solution: x = [0, 2], optimal value = 0

    const size_t m = 1;  // One constraint
    const size_t n = 2;  // Two variables

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

TEST_CASE("Lasso regression (small)", "[solver][integration]") {
    // Minimize: 0.5 * ||Ax - b||^2 + lambda * ||x||_1
    // Using graph form

    const size_t m = 10;  // Number of samples
    const size_t n = 5;   // Number of features

    // Create simple problem data
    std::vector<double> A_data(m * n);
    std::vector<double> b_data(m);

    // Simple test data: A = I (first m rows), b = ones
    for (size_t i = 0; i < m; ++i) {
        b_data[i] = 1.0;
        for (size_t j = 0; j < n; ++j) {
            A_data[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<FunctionObj<double>> f(m);
    std::vector<FunctionObj<double>> g(n);

    // f_i(y_i) = 0.5 * (y_i - b_i)^2
    // FunctionObj computes: c * h(a*x - b) + d*x + e*x^2
    // With kSquare, h(x) = 0.5*x^2, so we want c=1, a=1, b=b_data[i]
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].a = 1.0;
        f[i].b = b_data[i];  // offset: 0.5*(y - b)^2
        f[i].c = 1.0;
        f[i].d = 0.0;
        f[i].e = 0.0;
    }

    // g_j(x_j) = lambda * |x_j|
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kAbs;
        g[j].c = lambda;
    }

    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    solver.SetMaxIter(1000);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetVerbose(0);

    pogs::PogsStatus status = solver.Solve(f, g);

    REQUIRE(status == pogs::POGS_SUCCESS);

    // Optimal value should be positive (due to L1 penalty)
    double optval = solver.GetOptval();
    REQUIRE(optval > 0.0);
    REQUIRE(optval < 10.0);  // Reasonable bound

    // Get solution
    const double* x = solver.GetX();

    // Solution should exist
    REQUIRE(x != nullptr);

    // First n components should have reasonable values
    for (size_t j = 0; j < n; ++j) {
        REQUIRE(std::isfinite(x[j]));
        REQUIRE(std::abs(x[j]) < 2.0);  // Bounded
    }
}

TEST_CASE("Ridge regression (small)", "[solver][integration]") {
    // Minimize: ||Ax - b||^2 + lambda * ||x||^2

    const size_t m = 10;
    const size_t n = 5;

    std::vector<double> A_data(m * n);
    std::vector<double> b_data(m);

    // Simple identity-like problem
    for (size_t i = 0; i < m; ++i) {
        b_data[i] = 1.0;
        for (size_t j = 0; j < n; ++j) {
            A_data[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<FunctionObj<double>> f(m);
    std::vector<FunctionObj<double>> g(n);

    // f_i(y_i) = (y_i - b_i)^2
    // FunctionObj computes: c * h(a*x - b) + d*x + e*x^2
    // With kSquare, h(x) = 0.5*x^2, so c=2 gives (y - b)^2
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].a = 1.0;
        f[i].b = b_data[i];
        f[i].c = 2.0;  // 2 * 0.5 * (y-b)^2 = (y-b)^2
        f[i].d = 0.0;
        f[i].e = 0.0;
    }

    // g_j(x_j) = lambda * x_j^2
    // c * h(x) = lambda * 0.5 * x^2, so use c = 2*lambda for lambda*x^2
    double lambda = 0.1;
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kSquare;
        g[j].c = 2.0 * lambda;  // 2*lambda * 0.5 * x^2 = lambda * x^2
    }

    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    solver.SetMaxIter(1000);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetVerbose(0);

    pogs::PogsStatus status = solver.Solve(f, g);

    REQUIRE(status == pogs::POGS_SUCCESS);

    // Solution should have finite optimal value
    double optval = solver.GetOptval();
    REQUIRE(std::isfinite(optval));
    REQUIRE(optval >= 0.0);
}

TEST_CASE("Non-negative least squares", "[solver][integration]") {
    // Minimize: ||Ax - b||^2 subject to x >= 0

    const size_t m = 10;
    const size_t n = 5;

    std::vector<double> A_data(m * n, 1.0);  // All ones
    std::vector<double> b_data(m, 2.0);      // b = 2

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<FunctionObj<double>> f(m);
    std::vector<FunctionObj<double>> g(n);

    // f_i(y_i) = (y_i - b_i)^2
    // FunctionObj computes: c * h(a*x - b) + d*x + e*x^2
    // With kSquare, h(x) = 0.5*x^2, so c=2 gives (y - b)^2
    for (size_t i = 0; i < m; ++i) {
        f[i].h = kSquare;
        f[i].a = 1.0;
        f[i].b = b_data[i];
        f[i].c = 2.0;  // 2 * 0.5 * (y-b)^2 = (y-b)^2
        f[i].d = 0.0;
        f[i].e = 0.0;
    }

    // g_j(x_j) = I(x_j >= 0)
    for (size_t j = 0; j < n; ++j) {
        g[j].h = kIndGe0;
    }

    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    solver.SetMaxIter(1000);
    solver.SetAbsTol(1e-4);
    solver.SetRelTol(1e-3);
    solver.SetVerbose(0);

    pogs::PogsStatus status = solver.Solve(f, g);

    REQUIRE(status == pogs::POGS_SUCCESS);

    // Get solution and verify non-negativity
    const double* x = solver.GetX();
    for (size_t j = 0; j < n; ++j) {
        REQUIRE(x[j] >= -1e-6);  // Allow small numerical error
    }
}

TEST_CASE("Solver with different tolerances", "[solver][config]") {
    const size_t m = 5;
    const size_t n = 3;

    std::vector<double> A_data(m * n, 0.0);
    std::vector<double> b_data(m, 0.0);
    std::vector<double> c_data(n, 0.0);

    // Trivial problem: all zeros
    for (size_t i = 0; i < std::min(m, n); ++i) {
        A_data[i * n + i] = 1.0;
    }

    pogs::MatrixDense<double> A('r', m, n, A_data.data());

    std::vector<ConeConstraint> Kx = {{kConeNonNeg, {0, 1, 2}}};
    std::vector<ConeConstraint> Ky = {{kConeZero, {0, 1, 2, 3, 4}}};

    SECTION("Loose tolerances") {
        pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);
        solver.SetMaxIter(1000);
        solver.SetAbsTol(1e-2);
        solver.SetRelTol(1e-2);
        solver.SetVerbose(0);

        pogs::PogsStatus status = solver.Solve(b_data, c_data);
        REQUIRE(status == pogs::POGS_SUCCESS);
    }

    SECTION("Tight tolerances") {
        pogs::PogsDirectCone<double, pogs::MatrixDense<double>> solver(A, Kx, Ky);
        solver.SetMaxIter(2000);
        solver.SetAbsTol(1e-6);
        solver.SetRelTol(1e-6);
        solver.SetVerbose(0);

        pogs::PogsStatus status = solver.Solve(b_data, c_data);
        REQUIRE(status == pogs::POGS_SUCCESS);
    }
}
