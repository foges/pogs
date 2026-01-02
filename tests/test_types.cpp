// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "pogs/types.hpp"
#include "pogs/config.hpp"

using Catch::Approx;

TEST_CASE("FunctionType enum class", "[types]") {
    SECTION("Enum values are distinct") {
        REQUIRE(pogs::FunctionType::Abs != pogs::FunctionType::Square);
        REQUIRE(pogs::FunctionType::Zero != pogs::FunctionType::Identity);
    }
}

TEST_CASE("FunctionObj construction", "[types]") {
    SECTION("Default constructor") {
        pogs::FunctionObj<double> f;

        REQUIRE(f.type == pogs::FunctionType::Zero);
        REQUIRE(f.a == Approx(1.0));
        REQUIRE(f.b == Approx(0.0));
        REQUIRE(f.c == Approx(1.0));
        REQUIRE(f.d == Approx(0.0));
        REQUIRE(f.e == Approx(0.0));
        REQUIRE(f.rho == Approx(1.0));
    }

    SECTION("Explicit constructor") {
        pogs::FunctionObj<double> f(pogs::FunctionType::Abs);

        REQUIRE(f.type == pogs::FunctionType::Abs);
        REQUIRE(f.a == Approx(1.0));
        REQUIRE(f.c == Approx(1.0));
    }

    SECTION("Custom parameters") {
        pogs::FunctionObj<double> f;
        f.type = pogs::FunctionType::Square;
        f.c = 0.5;
        f.d = -2.0;

        REQUIRE(f.type == pogs::FunctionType::Square);
        REQUIRE(f.c == Approx(0.5));
        REQUIRE(f.d == Approx(-2.0));
    }
}

TEST_CASE("FunctionObj for common functions", "[types]") {
    SECTION("L1 regularization (Lasso)") {
        pogs::FunctionObj<double> g;
        g.type = pogs::FunctionType::Abs;
        g.c = 0.1;  // lambda

        REQUIRE(g.type == pogs::FunctionType::Abs);
        REQUIRE(g.c == Approx(0.1));
    }

    SECTION("Least squares") {
        pogs::FunctionObj<double> f;
        f.type = pogs::FunctionType::Square;
        f.c = 0.5;
        f.d = -1.5;  // -b_i

        REQUIRE(f.type == pogs::FunctionType::Square);
        REQUIRE(f.c == Approx(0.5));
        REQUIRE(f.d == Approx(-1.5));
    }

    SECTION("Non-negativity constraint") {
        pogs::FunctionObj<double> g;
        g.type = pogs::FunctionType::IndGe0;

        REQUIRE(g.type == pogs::FunctionType::IndGe0);
    }
}

TEST_CASE("ConeType enum class", "[types]") {
    SECTION("Enum values are distinct") {
        REQUIRE(pogs::ConeType::Zero != pogs::ConeType::NonNeg);
        REQUIRE(pogs::ConeType::SOC != pogs::ConeType::SDP);
    }
}

TEST_CASE("Status enum class", "[types]") {
    SECTION("Success is zero") {
        REQUIRE(static_cast<int>(pogs::Status::Success) == 0);
    }

    SECTION("Error statuses are non-zero") {
        REQUIRE(static_cast<int>(pogs::Status::MaxIterations) != 0);
        REQUIRE(static_cast<int>(pogs::Status::NumericalError) != 0);
    }
}

TEST_CASE("SolverConfig construction", "[config]") {
    SECTION("Default constructor") {
        pogs::SolverConfig config;

        REQUIRE(config.rho == Approx(1.0));
        REQUIRE(config.abs_tol == Approx(1e-4));
        REQUIRE(config.rel_tol == Approx(1e-3));
        REQUIRE(config.max_iter == 1000);
        REQUIRE(config.verbose == false);
        REQUIRE(config.adaptive_rho == true);
        REQUIRE(config.gap_stop == true);
    }

    SECTION("Designated initializers (C++20)") {
        auto config = pogs::SolverConfig{
            .rho = 2.0,
            .abs_tol = 1e-6,
            .verbose = true
        };

        REQUIRE(config.rho == Approx(2.0));
        REQUIRE(config.abs_tol == Approx(1e-6));
        REQUIRE(config.rel_tol == Approx(1e-3));  // Default
        REQUIRE(config.verbose == true);
    }

    SECTION("High accuracy configuration") {
        auto config = pogs::SolverConfig{
            .rho = 0.1,
            .abs_tol = 1e-6,
            .rel_tol = 1e-6,
            .max_iter = 5000
        };

        REQUIRE(config.rho == Approx(0.1));
        REQUIRE(config.abs_tol == Approx(1e-6));
        REQUIRE(config.max_iter == 5000);
    }

    SECTION("Fast approximate configuration") {
        auto config = pogs::SolverConfig{
            .rho = 10.0,
            .abs_tol = 1e-2,
            .rel_tol = 1e-2,
            .max_iter = 100
        };

        REQUIRE(config.rho == Approx(10.0));
        REQUIRE(config.abs_tol == Approx(1e-2));
        REQUIRE(config.max_iter == 100);
    }
}

TEST_CASE("Ord enum class", "[types]") {
    SECTION("Row and column major are different") {
        REQUIRE(pogs::Ord::RowMajor != pogs::Ord::ColMajor);
    }
}
