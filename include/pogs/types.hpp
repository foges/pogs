// POGS Modern Types Header
// Copyright (c) 2024-2026 POGS Contributors
// Licensed under Apache 2.0

#pragma once

#include <cstddef>
#include <vector>

namespace pogs {

// Function types with modern enum class
enum class FunctionType {
    Abs,
    Exp,
    Huber,
    Identity,
    IndBox01,
    IndEq0,
    IndGe0,
    IndLe0,
    Logistic,
    MaxNeg0,
    MaxPos0,
    NegEntr,
    NegLog,
    Recipr,
    Square,
    Zero
};

// Modern function object
// Represents a function of the form:
//   h(x) = c * f(a * x - b) + d * x + e/2 * x^2
template<typename T>
struct FunctionObj {
    FunctionType type;
    T a = static_cast<T>(1);
    T b = static_cast<T>(0);
    T c = static_cast<T>(1);
    T d = static_cast<T>(0);
    T e = static_cast<T>(0);
    T rho = static_cast<T>(1);

    // Constructor for convenience
    explicit FunctionObj(FunctionType t = FunctionType::Zero) : type(t) {}
};

// Cone types for cone form problems
enum class ConeType {
    Zero,      // Equality constraints: x = 0
    NonNeg,    // Non-negative orthant: x >= 0
    NonPos,    // Non-positive orthant: x <= 0
    SOC,       // Second-order cone: ||x[1:]|| <= x[0]
    SDP,       // Positive semidefinite cone
    ExpPrimal, // Exponential cone (primal)
    ExpDual    // Exponential cone (dual)
};

// Cone constraint specification
template<typename T>
struct ConeConstraint {
    ConeType type;
    std::vector<size_t> indices;

    ConeConstraint(ConeType t = ConeType::Zero) : type(t) {}

    ConeConstraint(ConeType t, std::vector<size_t> idx)
        : type(t), indices(std::move(idx)) {}
};

// Solver status
enum class Status {
    Success = 0,
    MaxIterations,
    NumericalError,
    InfeasibleOrUnbounded
};

// Matrix ordering
enum class Ord {
    RowMajor,
    ColMajor
};

} // namespace pogs
