// POGS - Proximal Operator Graph Solver
// Copyright 2014-2026 Chris Fougner and Contributors
// Licensed under Apache 2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include "prox_lib.h"

using Catch::Approx;

TEST_CASE("Proximal operator for Zero function", "[proximal]") {
    double v = 5.0;
    double rho = 1.0;

    double result = ProxZero<double>(v, rho);

    REQUIRE(result == Approx(5.0));  // Identity: prox(v) = v
}

TEST_CASE("Proximal operator for Identity function", "[proximal]") {
    double v = 5.0;
    double rho = 2.0;

    double result = ProxIdentity<double>(v, rho);

    REQUIRE(result == Approx(4.5));  // prox(v) = v - 1/rho
}

TEST_CASE("Proximal operator for Abs (soft thresholding)", "[proximal]") {
    double rho = 2.0;  // threshold = 1/rho = 0.5

    SECTION("Positive value above threshold") {
        double result = ProxAbs<double>(2.0, rho);
        REQUIRE(result == Approx(1.5));  // 2.0 - 0.5
    }

    SECTION("Positive value below threshold") {
        double result = ProxAbs<double>(0.3, rho);
        REQUIRE(result == Approx(0.0));  // Thresholded to zero
    }

    SECTION("Negative value below threshold") {
        double result = ProxAbs<double>(-2.0, rho);
        REQUIRE(result == Approx(-1.5));  // -2.0 + 0.5
    }

    SECTION("Value at threshold") {
        double result = ProxAbs<double>(0.5, rho);
        REQUIRE(std::abs(result) < 1e-10);  // Should be zero
    }

    SECTION("Zero input") {
        double result = ProxAbs<double>(0.0, rho);
        REQUIRE(result == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for Square function", "[proximal]") {
    double rho = 3.0;

    SECTION("Positive value") {
        double result = ProxSquare<double>(6.0, rho);
        REQUIRE(result == Approx(4.5));  // rho/(rho+1) * v = 3/4 * 6 = 4.5
    }

    SECTION("Negative value") {
        double result = ProxSquare<double>(-4.0, rho);
        REQUIRE(result == Approx(-3.0));  // 3/4 * (-4) = -3
    }

    SECTION("Zero") {
        double result = ProxSquare<double>(0.0, rho);
        REQUIRE(result == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndEq0 (indicator of {0})", "[proximal]") {
    double v = 5.0;
    double rho = 1.0;

    double result = ProxIndEq0<double>(v, rho);

    REQUIRE(result == Approx(0.0));  // Projection onto {0}
}

TEST_CASE("Proximal operator for IndGe0 (indicator of [0,inf))", "[proximal]") {
    double rho = 1.0;

    SECTION("Positive value") {
        double result = ProxIndGe0<double>(3.0, rho);
        REQUIRE(result == Approx(3.0));  // Already feasible
    }

    SECTION("Negative value") {
        double result = ProxIndGe0<double>(-2.0, rho);
        REQUIRE(result == Approx(0.0));  // Projected to 0
    }

    SECTION("Zero") {
        double result = ProxIndGe0<double>(0.0, rho);
        REQUIRE(result == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndLe0 (indicator of (-inf,0])", "[proximal]") {
    double rho = 1.0;

    SECTION("Negative value") {
        double result = ProxIndLe0<double>(-3.0, rho);
        REQUIRE(result == Approx(-3.0));  // Already feasible
    }

    SECTION("Positive value") {
        double result = ProxIndLe0<double>(2.0, rho);
        REQUIRE(result == Approx(0.0));  // Projected to 0
    }

    SECTION("Zero") {
        double result = ProxIndLe0<double>(0.0, rho);
        REQUIRE(result == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for IndBox01 (indicator of [0,1])", "[proximal]") {
    double rho = 1.0;

    SECTION("Value in [0,1]") {
        double result = ProxIndBox01<double>(0.5, rho);
        REQUIRE(result == Approx(0.5));  // Already feasible
    }

    SECTION("Value below 0") {
        double result = ProxIndBox01<double>(-0.5, rho);
        REQUIRE(result == Approx(0.0));  // Projected to 0
    }

    SECTION("Value above 1") {
        double result = ProxIndBox01<double>(1.5, rho);
        REQUIRE(result == Approx(1.0));  // Projected to 1
    }

    SECTION("Boundary values") {
        double r0 = ProxIndBox01<double>(0.0, rho);
        REQUIRE(r0 == Approx(0.0));

        double r1 = ProxIndBox01<double>(1.0, rho);
        REQUIRE(r1 == Approx(1.0));
    }
}

TEST_CASE("Proximal operator for MaxPos0", "[proximal]") {
    // f(x) = max(0, x) - hinge function
    double rho = 2.0;  // 1/rho = 0.5

    SECTION("Large positive value") {
        double result = ProxMaxPos0<double>(3.0, rho);
        REQUIRE(result == Approx(2.5));  // v - 1/rho
    }

    SECTION("Small positive value") {
        double result = ProxMaxPos0<double>(0.3, rho);
        REQUIRE(std::abs(result) < 1e-10);  // In (0, 1/rho) -> 0
    }

    SECTION("Negative value") {
        double result = ProxMaxPos0<double>(-1.0, rho);
        // Negative inputs pass through (since max(0,x)=0 for x<0)
        REQUIRE(result == Approx(-1.0));
    }
}

TEST_CASE("Proximal operator for MaxNeg0", "[proximal]") {
    // f(x) = max(0, -x)
    double rho = 2.0;  // 1/rho = 0.5

    SECTION("Large negative value") {
        double result = ProxMaxNeg0<double>(-3.0, rho);
        REQUIRE(result == Approx(-2.5));  // v + 1/rho
    }

    SECTION("Small negative value") {
        double result = ProxMaxNeg0<double>(-0.3, rho);
        REQUIRE(std::abs(result) < 1e-10);  // In (-1/rho, 0) -> 0
    }

    SECTION("Positive value") {
        double result = ProxMaxNeg0<double>(1.0, rho);
        // Positive inputs pass through (since max(0,-x)=0 for x>0)
        REQUIRE(result == Approx(1.0));
    }
}

TEST_CASE("Proximal operator for Huber function", "[proximal]") {
    // Huber: f(x) = x^2/2 if |x| <= 1, |x| - 1/2 otherwise
    double rho = 2.0;

    SECTION("Value in quadratic region") {
        double result = ProxHuber<double>(0.5, rho);
        // In quadratic region: prox(v) = v * rho / (1 + rho)
        REQUIRE(result == Approx(0.5 * 2.0 / 3.0));
    }

    SECTION("Value in linear region (positive)") {
        double result = ProxHuber<double>(5.0, rho);
        // In linear region: prox(v) = v - 1/rho
        REQUIRE(result == Approx(4.5));
    }

    SECTION("Value in linear region (negative)") {
        double result = ProxHuber<double>(-5.0, rho);
        // In linear region: prox(v) = v + 1/rho
        REQUIRE(result == Approx(-4.5));
    }

    SECTION("Zero input") {
        double result = ProxHuber<double>(0.0, rho);
        REQUIRE(result == Approx(0.0));
    }
}

TEST_CASE("Proximal operator for Exp function", "[proximal]") {
    // f(x) = e^x
    double rho = 1.0;

    SECTION("Positive value") {
        double v = 2.0;
        double result = ProxExp<double>(v, rho);
        // Result should be less than input (exp pushes down)
        REQUIRE(result < v);
        // Verify optimality: result + (1/rho)*exp(result) = v
        REQUIRE(result + std::exp(result) / rho == Approx(v).margin(1e-6));
    }

    SECTION("Zero input") {
        double v = 0.0;
        double result = ProxExp<double>(v, rho);
        // Verify optimality condition
        REQUIRE(result + std::exp(result) / rho == Approx(v).margin(1e-6));
    }

    SECTION("Negative input") {
        double v = -1.0;
        double result = ProxExp<double>(v, rho);
        REQUIRE(result + std::exp(result) / rho == Approx(v).margin(1e-6));
    }
}

TEST_CASE("Proximal operator for NegLog function", "[proximal]") {
    // f(x) = -log(x) for x > 0
    double rho = 2.0;

    SECTION("Positive value") {
        double v = 3.0;
        double result = ProxNegLog<double>(v, rho);
        // Result should be positive
        REQUIRE(result > 0);
        // Verify optimality: result - 1/(rho*result) = v
        REQUIRE(result - 1.0 / (rho * result) == Approx(v).margin(1e-6));
    }

    SECTION("Small positive value") {
        double v = 0.5;
        double result = ProxNegLog<double>(v, rho);
        REQUIRE(result > 0);
    }

    SECTION("Large value") {
        double v = 10.0;
        double result = ProxNegLog<double>(v, rho);
        // For large v, result should be close to v
        REQUIRE(result == Approx(v).epsilon(0.1));
    }
}

TEST_CASE("Proximal operator for Reciprocal function", "[proximal]") {
    // f(x) = 1/x for x > 0 (barrier function)
    double rho = 1.0;

    SECTION("Positive value") {
        double v = 2.0;
        double result = ProxRecipr<double>(v, rho);
        // Result should be positive
        REQUIRE(result > 0);
    }

    SECTION("Large value") {
        double v = 10.0;
        double result = ProxRecipr<double>(v, rho);
        // Result should be positive and close to v
        REQUIRE(result > 0);
        REQUIRE(result == Approx(v).epsilon(0.05));
    }

    SECTION("Zero input") {
        double v = 0.0;
        double result = ProxRecipr<double>(v, rho);
        // Should handle v = 0 gracefully, result > 0 due to barrier
        REQUIRE(result >= 0);
    }
}

TEST_CASE("Proximal operator for NegEntropy function", "[proximal]") {
    // f(x) = x * log(x) for x > 0 (negative entropy)
    double rho = 1.0;

    SECTION("Positive value") {
        double v = 2.0;
        double result = ProxNegEntr<double>(v, rho);
        // Result should be positive
        REQUIRE(result > 0);
    }

    SECTION("Value near 1") {
        double v = 1.0;
        double result = ProxNegEntr<double>(v, rho);
        REQUIRE(result > 0);
    }

    SECTION("Small positive value") {
        double v = 0.5;
        double result = ProxNegEntr<double>(v, rho);
        REQUIRE(result > 0);
    }
}

TEST_CASE("Proximal operator for Logistic function", "[proximal]") {
    // f(x) = log(1 + e^x)
    double rho = 1.0;

    SECTION("Positive value") {
        double v = 3.0;
        double result = ProxLogistic<double>(v, rho);
        // Result should be less than input
        REQUIRE(result < v);
    }

    SECTION("Negative value") {
        double v = -3.0;
        double result = ProxLogistic<double>(v, rho);
        // For large negative, prox(v) ~= v
        REQUIRE(result == Approx(v).margin(0.1));
    }

    SECTION("Zero input") {
        double v = 0.0;
        double result = ProxLogistic<double>(v, rho);
        // Result should be negative (logistic pushes down)
        REQUIRE(result < 0);
    }

    SECTION("Large positive value") {
        double v = 10.0;
        double result = ProxLogistic<double>(v, rho);
        // For large positive, prox(v) ~= v - 1/rho
        REQUIRE(result == Approx(v - 1.0 / rho).margin(0.1));
    }
}

TEST_CASE("Proximal operator consistency", "[proximal]") {
    SECTION("Double and float precision") {
        double vd = 3.0;
        float vf = 3.0f;
        double rho = 2.0;

        double rd = ProxAbs<double>(vd, rho);
        float rf = ProxAbs<float>(vf, static_cast<float>(rho));

        REQUIRE(rd == Approx(static_cast<double>(rf)).epsilon(1e-5));
    }
}
