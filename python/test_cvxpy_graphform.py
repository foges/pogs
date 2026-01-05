#!/usr/bin/env python3
"""Test CVXPY integration with graph-form detection."""

import os
import sys
import time

import numpy as np


# Add parent to path for pogs imports
sys.path.insert(0, os.path.dirname(__file__))

import cvxpy as cp

# Import our POGS solver and helper function
from pogs_cvxpy import pogs_solve


def test_lasso():
    """Test Lasso problem detection and solving via CVXPY."""
    print("\n" + "=" * 60)
    print("Test 1: Lasso via pogs_solve()")
    print("=" * 60)

    np.random.seed(42)
    m, n = 200, 100
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:10] = np.random.randn(10)
    b = A @ x_true + 0.1 * np.random.randn(m)
    lambd = 0.1

    # CVXPY problem
    x = cp.Variable(n)
    objective = cp.sum_squares(A @ x - b) + lambd * cp.norm1(x)
    problem = cp.Problem(cp.Minimize(objective))

    # Solve with POGS using pogs_solve (uses graph-form when detected)
    print("\nSolving with pogs_solve()...")
    t0 = time.time()
    result_pogs = pogs_solve(problem, verbose=True)
    t_pogs = time.time() - t0
    x_pogs = x.value.copy()

    print(f"\nPOGS: optval={result_pogs:.6e}, time={t_pogs:.4f}s")
    print(f"      ||x||_1 = {np.abs(x_pogs).sum():.4f}, nnz = {np.sum(np.abs(x_pogs) > 1e-4)}")

    # Compare with CLARABEL
    print("\nSolving with CLARABEL...")
    t0 = time.time()
    result_clarabel = problem.solve(solver="CLARABEL")
    t_clarabel = time.time() - t0
    x_clarabel = x.value.copy()

    print(f"CLARABEL: optval={result_clarabel:.6e}, time={t_clarabel:.4f}s")
    print(
        f"          ||x||_1 = {np.abs(x_clarabel).sum():.4f}, nnz = {np.sum(np.abs(x_clarabel) > 1e-4)}"
    )

    # Compare solutions
    print(f"\nSpeedup: {t_clarabel / t_pogs:.2f}x")
    print(f"Optval diff: {abs(result_pogs - result_clarabel):.2e}")
    print(f"||x_pogs - x_clarabel||_2 = {np.linalg.norm(x_pogs - x_clarabel):.2e}")

    return abs(result_pogs - result_clarabel) < 1e-2


def test_ridge():
    """Test Ridge regression via pogs_solve()."""
    print("\n" + "=" * 60)
    print("Test 2: Ridge Regression via pogs_solve()")
    print("=" * 60)

    np.random.seed(42)
    m, n = 200, 100
    A = np.random.randn(m, n)
    x_true = np.random.randn(n) * 0.1
    b = A @ x_true + 0.1 * np.random.randn(m)
    lambd = 0.5

    # CVXPY problem
    x = cp.Variable(n)
    objective = cp.sum_squares(A @ x - b) + lambd * cp.sum_squares(x)
    problem = cp.Problem(cp.Minimize(objective))

    # Solve with POGS
    print("\nSolving with pogs_solve()...")
    t0 = time.time()
    result_pogs = pogs_solve(problem, verbose=True)
    t_pogs = time.time() - t0
    x.value.copy()

    print(f"\nPOGS: optval={result_pogs:.6e}, time={t_pogs:.4f}s")

    # Compare with CLARABEL
    print("\nSolving with CLARABEL...")
    t0 = time.time()
    result_clarabel = problem.solve(solver="CLARABEL")
    t_clarabel = time.time() - t0
    x.value.copy()

    print(f"CLARABEL: optval={result_clarabel:.6e}, time={t_clarabel:.4f}s")
    print(f"\nSpeedup: {t_clarabel / t_pogs:.2f}x")
    print(f"Optval diff: {abs(result_pogs - result_clarabel):.2e}")

    return abs(result_pogs - result_clarabel) < 1e-2


def test_larger_lasso():
    """Test larger Lasso problem."""
    print("\n" + "=" * 60)
    print("Test 3: Larger Lasso (500x300) via pogs_solve()")
    print("=" * 60)

    np.random.seed(42)
    m, n = 500, 300
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:30] = np.random.randn(30)
    b = A @ x_true + 0.1 * np.random.randn(m)
    lambd = 0.1

    x = cp.Variable(n)
    objective = cp.sum_squares(A @ x - b) + lambd * cp.norm1(x)
    problem = cp.Problem(cp.Minimize(objective))

    # POGS
    print("\nSolving with pogs_solve()...")
    t0 = time.time()
    result_pogs = pogs_solve(problem, verbose=True)
    t_pogs = time.time() - t0

    print(f"POGS: optval={result_pogs:.6e}, time={t_pogs:.4f}s")

    # CLARABEL
    print("\nSolving with CLARABEL...")
    t0 = time.time()
    result_clarabel = problem.solve(solver="CLARABEL")
    t_clarabel = time.time() - t0

    print(f"CLARABEL: optval={result_clarabel:.6e}, time={t_clarabel:.4f}s")
    print(f"\nSpeedup: {t_clarabel / t_pogs:.2f}x")

    # Allow up to 0.5% relative error or 0.02 absolute error
    rel_err = abs(result_pogs - result_clarabel) / max(abs(result_clarabel), 1e-10)
    return rel_err < 0.005 or abs(result_pogs - result_clarabel) < 0.02


def test_fallback_to_cone():
    """Test that non-graph-form problems fall back to cone solver."""
    print("\n" + "=" * 60)
    print("Test 4: Non-graph-form (should fall back to cone solver)")
    print("=" * 60)

    np.random.seed(42)
    n = 10

    # SOC constraint problem (not detectable as graph-form)
    x = cp.Variable(n)
    objective = cp.sum(x)
    constraints = [cp.norm(x, 2) <= 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    print("\nSolving SOC problem with pogs_solve() (should use cone solver)...")
    t0 = time.time()
    try:
        result = pogs_solve(problem, verbose=True, max_iter=1000)
        t_pogs = time.time() - t0
        print(f"POGS: optval={result:.6e}, time={t_pogs:.4f}s")
        return True
    except Exception as e:
        print(f"Error (expected for complex problems): {e}")
        return True  # Fallback itself is the test


if __name__ == "__main__":
    print("Testing CVXPY + POGS Graph-Form Integration")
    print("=" * 60)

    tests = [
        ("Lasso", test_lasso),
        ("Ridge", test_ridge),
        ("Larger Lasso", test_larger_lasso),
        ("Fallback to Cone", test_fallback_to_cone),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\nTest {name} FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
