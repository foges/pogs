"""
Test CVXPY interface for POGS solver.
"""

import sys
import os

# Add pogs python directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import cvxpy as cp
    import numpy as np
    from pogs_cvxpy import POGS
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install cvxpy: pip install cvxpy")
    sys.exit(1)

SOLVER_OPTS = {
    "max_iter": 50000,
    "abs_tol": 1e-6,
    "rel_tol": 1e-6,
}


def test_lp():
    """Test simple LP problem."""
    print("=" * 70)
    print("Test 1: Linear Program")
    print("=" * 70)
    print("Problem:")
    print("  minimize    x[0]")
    print("  subject to  x[0] + x[1] = 2")
    print("              x >= 0")
    print("Expected: x = [0, 2], optimal value = 0")
    print()

    x = cp.Variable(2)
    objective = cp.Minimize(x[0])
    constraints = [
        x[0] + x[1] == 2,
        x >= 0
    ]
    prob = cp.Problem(objective, constraints)

    # Solve with POGS
    result = prob.solve(solver='POGS', verbose=True, **SOLVER_OPTS)

    print()
    print("Result:")
    print(f"  Status: {prob.status}")
    print(f"  Optimal value: {prob.value:.6f}")
    print(f"  x = {x.value}")
    print()

    # Verify
    assert prob.status == 'optimal', f"Expected optimal, got {prob.status}"
    assert abs(prob.value) < 1e-3, f"Expected optimal value ~0, got {prob.value}"
    assert abs(x.value[0]) < 1e-3, f"Expected x[0] ~0, got {x.value[0]}"
    assert abs(x.value[1] - 2.0) < 1e-3, f"Expected x[1] ~2, got {x.value[1]}"

    print("✓ Test passed!")
    return True


def test_lp_ineq():
    """Test LP with inequalities."""
    print("=" * 70)
    print("Test 2: Linear Program with Inequalities")
    print("=" * 70)
    print("Problem:")
    print("  minimize    x[0]")
    print("  subject to  -x[0] - x[1] <= 0")
    print("               x[0] - x[1] <= 0")
    print("               x[1] <= 2")
    print("Expected: x ≈ [-2, 2], optimal value ≈ -2")
    print()

    x = cp.Variable(2)
    objective = cp.Minimize(x[0])
    constraints = [
        -x[0] - x[1] <= 0,
        x[0] - x[1] <= 0,
        x[1] <= 2
    ]
    prob = cp.Problem(objective, constraints)

    # Solve with POGS
    result = prob.solve(solver='POGS', verbose=True, **SOLVER_OPTS)

    print()
    print("Result:")
    print(f"  Status: {prob.status}")
    print(f"  Optimal value: {prob.value:.6f}")
    print(f"  x = {x.value}")
    print()

    assert prob.status == 'optimal', f"Expected optimal, got {prob.status}"
    assert abs(prob.value + 2.0) < 1e-3, (
        f"Expected optimal value ~-2, got {prob.value}"
    )
    assert abs(x.value[0] + 2.0) < 1e-3, (
        f"Expected x[0] ~-2, got {x.value[0]}"
    )
    assert abs(x.value[1] - 2.0) < 1e-3, (
        f"Expected x[1] ~2, got {x.value[1]}"
    )

    print("✓ Test passed!")
    return True


def test_qp():
    """Test quadratic program."""
    print("=" * 70)
    print("Test 3: Quadratic Program")
    print("=" * 70)
    print("Problem:")
    print("  minimize    0.5 * x^T * x + [1, 0]^T * x")
    print("  subject to  x[0] + x[1] = 1")
    print("              x >= 0")
    print()

    x = cp.Variable(2)
    objective = cp.Minimize(0.5 * cp.sum_squares(x) + x[0])
    constraints = [
        x[0] + x[1] == 1,
        x >= 0
    ]
    prob = cp.Problem(objective, constraints)

    # Solve with POGS
    result = prob.solve(solver='POGS', verbose=True, **SOLVER_OPTS)

    print()
    print("Result:")
    print(f"  Status: {prob.status}")
    print(f"  Optimal value: {prob.value:.6f}")
    print(f"  x = {x.value}")
    print()

    assert prob.status == 'optimal', f"Expected optimal, got {prob.status}"

    print("✓ Test passed!")
    return True


def test_soc():
    """Test second-order cone problem."""
    print("=" * 70)
    print("Test 4: Second-Order Cone Problem")
    print("=" * 70)
    print("Problem:")
    print("  minimize    x[0]")
    print("  subject to  ||x[1:]|| <= x[0]")
    print("              x[1] = 1")
    print()

    x = cp.Variable(3)
    objective = cp.Minimize(x[0])
    constraints = [
        cp.norm(x[1:], 2) <= x[0],
        x[1] == 1
    ]
    prob = cp.Problem(objective, constraints)

    try:
        # Solve with POGS
        result = prob.solve(solver='POGS', verbose=True, **SOLVER_OPTS)

        print()
        print("Result:")
        print(f"  Status: {prob.status}")
        print(f"  Optimal value: {prob.value:.6f}")
        print(f"  x = {x.value}")
        print()

        assert prob.status == 'optimal', f"Expected optimal, got {prob.status}"
        print("✓ Test passed!")
        return True

    except Exception as e:
        print(f"Note: SOC test encountered issue: {e}")
        print("This may be due to cone conversion in CVXPY")
        return False


def test_simple_feasibility():
    """Test simple feasibility problem."""
    print("=" * 70)
    print("Test 5: Feasibility Problem")
    print("=" * 70)
    print("Problem:")
    print("  find    x")
    print("  s.t.    x[0] + x[1] = 1")
    print("          x >= 0")
    print()

    x = cp.Variable(2)
    objective = cp.Minimize(0)  # Feasibility
    constraints = [
        x[0] + x[1] == 1,
        x >= 0
    ]
    prob = cp.Problem(objective, constraints)

    # Solve with POGS
    result = prob.solve(solver='POGS', verbose=True, **SOLVER_OPTS)

    print()
    print("Result:")
    print(f"  Status: {prob.status}")
    print(f"  x = {x.value}")
    print()

    assert prob.status == 'optimal', f"Expected optimal, got {prob.status}"

    print("✓ Test passed!")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  POGS CVXPY Interface Test Suite".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    tests = [
        ("Simple LP", test_lp),
        ("LP with inequalities", test_lp_ineq),
        ("Quadratic Program", test_qp),
        ("Second-Order Cone", test_soc),
        ("Feasibility", test_simple_feasibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
