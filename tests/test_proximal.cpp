// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include "prox_lib.h"

using Catch::Approx;

TEST_CASE("Proximal operator for Zero function", "[proximal]") {
    double x = 5.0;
    double rho = 1.0;

    ProxZero(rho, &x);

    REQUIRE(x == Approx(5.0));  // Identity: prox(v) = v
}

TEST_CASE("Proximal operator for Identity function", "[proximal]") {
    double x = 5.0;
    double rho = 2.0;

    ProxIdentity(rho, &x);

    REQUIRE(x == Approx(4.5));  // prox(v) = v - 1/rho
}

TEST_CASE("Proximal operator for Abs (soft thresholding)", "[proximal]") {
    double rho = 2.0;  // threshold = 1/rho = 0.5

    SECTION("Positive value above threshold") {
        double x = 2.0;
        ProxAbs(rho, &x);
        REQUIRE(x == Approx(1.5));  // 2.0 - 0.5
    }

    SECTION("Positive value below threshold") {
        double x = 0.3;
        ProxAbs(rho, &x);
        REQUIRE(x == Approx(0.0));  // Thresholded to zero
    }

    SECTION("Negative value below threshold") {
        double x = -2.0;
        ProxAbs(rho, &x);
        REQUIRE(x == Approx(-1.5));  // -2.0 + 0.5
    }

    SECTION("Value at threshold") {
        double x = 0.5;
        ProxAbs(rho, &x);
        REQUIRE(std::abs(x) < 1e-10);  // Should be zero
    }

    SECTION("Zero input") {
        double x = 0.0;
        ProxAbs(rho, &x);
        REQUIRE(x == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for Square function", "[proximal]") {
    double rho = 3.0;

    SECTION("Positive value") {
        double x = 6.0;
        ProxSquare(rho, &x);
        REQUIRE(x == Approx(4.5));  // rho/(rho+1) * v = 3/4 * 6 = 4.5
    }

    SECTION("Negative value") {
        double x = -4.0;
        ProxSquare(rho, &x);
        REQUIRE(x == Approx(-3.0));  // 3/4 * (-4) = -3
    }

    SECTION("Zero") {
        double x = 0.0;
        ProxSquare(rho, &x);
        REQUIRE(x == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndEq0 (indicator of {0})", "[proximal]") {
    double x = 5.0;
    double rho = 1.0;

    ProxIndEq0(rho, &x);

    REQUIRE(x == Approx(0.0));  // Projection onto {0}
}

TEST_CASE("Proximal operator for IndGe0 (indicator of [0,∞))", "[proximal]") {
    double rho = 1.0;

    SECTION("Positive value") {
        double x = 3.0;
        ProxIndGe0(rho, &x);
        REQUIRE(x == Approx(3.0));  // Already feasible
    }

    SECTION("Negative value") {
        double x = -2.0;
        ProxIndGe0(rho, &x);
        REQUIRE(x == Approx(0.0));  // Projected to 0
    }

    SECTION("Zero") {
        double x = 0.0;
        ProxIndGe0(rho, &x);
        REQUIRE(x == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndLe0 (indicator of (-∞,0])", "[proximal]") {
    double rho = 1.0;

    SECTION("Negative value") {
        double x = -3.0;
        ProxIndLe0(rho, &x);
        REQUIRE(x == Approx(-3.0));  // Already feasible
    }

    SECTION("Positive value") {
        double x = 2.0;
        ProxIndLe0(rho, &x);
        REQUIRE(x == Approx(0.0));  // Projected to 0
    }

    SECTION("Zero") {
        double x = 0.0;
        ProxIndLe0(rho, &x);
        REQUIRE(x == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndBox01 (indicator of [0,1])", "[proximal]") {
    double rho = 1.0;

    SECTION("Value in [0,1]") {
        double x = 0.5;
        ProxIndBox01(rho, &x);
        REQUIRE(x == Approx(0.5));  // Already feasible
    }

    SECTION("Value below 0") {
        double x = -0.5;
        ProxIndBox01(rho, &x);
        REQUIRE(x == Approx(0.0));  // Projected to 0
    }

    SECTION("Value above 1") {
        double x = 1.5;
        ProxIndBox01(rho, &x);
        REQUIRE(x == Approx(1.0));  // Projected to 1
    }

    SECTION("Boundary values") {
        double x0 = 0.0;
        ProxIndBox01(rho, &x0);
        REQUIRE(x0 == Approx(0.0));

        double x1 = 1.0;
        ProxIndBox01(rho, &x1);
        REQUIRE(x1 == Approx(1.0));
    }
}

TEST_CASE("Proximal operator for MaxPos0", "[proximal]") {
    double rho = 2.0;

    SECTION("Large positive value") {
        double x = 3.0;
        ProxMaxPos0(rho, &x);
        REQUIRE(x == Approx(2.5));  // x - 1/rho
    }

    SECTION("Small positive value") {
        double x = 0.3;
        ProxMaxPos0(rho, &x);
        REQUIRE(std::abs(x) < 1e-10);  // Thresholded to 0
    }

    SECTION("Negative value") {
        double x = -1.0;
        ProxMaxPos0(rho, &x);
        REQUIRE(std::abs(x) < 1e-10);  // Projected to 0
    }
}

TEST_CASE("Proximal operator for MaxNeg0", "[proximal]") {
    double rho = 2.0;

    SECTION("Large negative value") {
        double x = -3.0;
        ProxMaxNeg0(rho, &x);
        REQUIRE(x == Approx(-2.5));  // x + 1/rho
    }

    SECTION("Small negative value") {
        double x = -0.3;
        ProxMaxNeg0(rho, &x);
        REQUIRE(std::abs(x) < 1e-10);  // Thresholded to 0
    }

    SECTION("Positive value") {
        double x = 1.0;
        ProxMaxNeg0(rho, &x);
        REQUIRE(std::abs(x) < 1e-10);  // Projected to 0
    }
}

TEST_CASE("Proximal operator consistency", "[proximal]") {
    SECTION("Double and float precision") {
        double xd = 3.0;
        float xf = 3.0f;
        double rho = 2.0;

        ProxAbs(rho, &xd);
        ProxAbs(static_cast<float>(rho), &xf);

        REQUIRE(xd == Approx(static_cast<double>(xf)).epsilon(1e-5));
    }
}
