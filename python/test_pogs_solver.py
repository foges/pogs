"""
Test POGS cone solver without CVXPY dependency.

This demonstrates the core solve_cone_problem function.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pogs_cvxpy import solve_cone_problem, CONE_ZERO, CONE_NON_NEG, CONE_SOC


def test_lp_simple():
    """
    Test simple LP problem:
        minimize    x[0]
        subject to  x[0] + x[1] = 2
                    x >= 0

    Expected solution: x = [0, 2], optimal value = 0
    """
    print("=" * 70)
    print("Test 1: Simple Linear Program")
    print("=" * 70)
    print("Problem:")
    print("  minimize    x[0]")
    print("  subject to  x[0] + x[1] = 2")
    print("              x[0], x[1] >= 0")
    print()
    print("Expected solution: x = [0, 2], optimal value = 0")
    print()

    # Problem data
    # Standard form: minimize c^T*x subject to b - A*x in K
    #
    # Original: min x[0] s.t. x[0] + x[1] = 2, x >= 0
    #
    # In standard form:
    #   c = [1, 0]
    #   A = [[ 1,  1],
    #        [-1,  0],
    #        [ 0, -1]]
    #   b = [2, 0, 0]
    #   K_y = {zero cone} x {R^2_+}

    c = np.array([1.0, 0.0])
    A = np.array([
        [1.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    b = np.array([2.0, 0.0, 0.0])

    dims = {
        'f': 1,  # One equality constraint (zero cone)
        'l': 2,  # x >= 0 encoded via b - A x in non-negative cone
    }

    # Note: In POGS cone form, we have:
    #   minimize c^T*x
    #   subject to b - A*x in K_y, x in K_x
    #
    # For this problem:
    #   - The equality constraint is: 2 - [1,1]*x in {zero}
    #   - The non-negativity constraints are: 0 - [-I]*x in {R^2_+}
    #
    # The standard conic form is:
    #   minimize c^T*x subject to b - Ax in K

    print("Solving with POGS...")
    print()

    result = solve_cone_problem(
        c, A, b, dims,
        abs_tol=1e-6,
        rel_tol=1e-6,
        max_iter=50000,
        verbose=5
    )

    print()
    print("Result:")
    print(f"  Status: {result['status']} (0 = success)")
    print(f"  Iterations: {result['num_iters']}")
    print(f"  Optimal value: {result.get('optval', 'N/A'):.6f}")
    print(f"  x = {result['x']}")
    print()

    # Verify solution
    if result['status'] == 0:
        x_opt = result['x']
        print("Verification:")
        print(f"  x[0] + x[1] = {x_opt[0] + x_opt[1]:.6f} (should be 2.0)")
        print(f"  x[0] = {x_opt[0]:.6f} (should be ≥ 0)")
        print(f"  x[1] = {x_opt[1]:.6f} (should be ≥ 0)")
        print(f"  Objective c^T*x = {np.dot(c, x_opt):.6f}")
        print()

        # Check constraints
        constraint_satisfied = abs(x_opt[0] + x_opt[1] - 2.0) < 1e-3
        non_negative = x_opt[0] >= -1e-6 and x_opt[1] >= -1e-6

        if constraint_satisfied and non_negative:
            print("✓ Test PASSED!")
            return True
        else:
            print("✗ Test FAILED: Constraints not satisfied")
            return False
    else:
        print("✗ Test FAILED: Solver did not converge")
        return False


def test_lp_with_inequalities():
    """
    Test LP with inequality constraints:
        minimize    x[0] + x[1]
        subject to  -x[0] - x[1] <= 0
                     x[0] - x[1] <= 0
                              x[1] <= 2

    Expected solution: x = [0, 0], optimal value = 0
    """
    print("\n")
    print("=" * 70)
    print("Test 2: Linear Program with Inequalities")
    print("=" * 70)
    print("Problem:")
    print("  minimize    x[0] + x[1]")
    print("  subject to  -x[0] - x[1] <= 0")
    print("               x[0] - x[1] <= 0")
    print("                       x[1] <= 2")
    print()
    print("Expected solution: x ≈ [0, 0], optimal value = 0")
    print()

    # In standard conic form: minimize c^T*x subject to b - A*x in K
    #
    # Constraints:
    #   -x[0] - x[1] <= 0  =>  0 - [-1, -1]*x in R_+  =>  b[0] = 0, A[0] = [-1, -1]
    #    x[0] - x[1] <= 0  =>  0 - [ 1, -1]*x in R_+  =>  b[1] = 0, A[1] = [ 1, -1]
    #             x[1] <= 2  =>  2 - [ 0,  1]*x in R_+  =>  b[2] = 2, A[2] = [ 0,  1]

    c = np.array([1.0, 1.0])
    A = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 0.0,  1.0]
    ])
    b = np.array([0.0, 0.0, 2.0])

    dims = {
        'f': 0,  # No equality constraints
        'l': 3,  # Three inequality constraints (non-negative cone)
    }

    print("Solving with POGS...")
    print()

    result = solve_cone_problem(
        c, A, b, dims,
        abs_tol=1e-6,
        rel_tol=1e-6,
        max_iter=50000,
        verbose=5
    )

    print()
    print("Result:")
    print(f"  Status: {result['status']} (0 = success)")
    print(f"  Iterations: {result['num_iters']}")
    print(f"  Optimal value: {result.get('optval', 'N/A'):.6f}")
    print(f"  x = {result['x']}")
    print()

    if result['status'] == 0:
        x_opt = result['x']
        print("Verification:")
        print(f"  -x[0] - x[1] = {-x_opt[0] - x_opt[1]:.6f} (should be ≤ 0)")
        print(f"   x[0] - x[1] = { x_opt[0] - x_opt[1]:.6f} (should be ≤ 0)")
        print(f"          x[1] = { x_opt[1]:.6f} (should be ≤ 2)")
        print(f"  Objective c^T*x = {np.dot(c, x_opt):.6f}")
        print()

        # Check constraints
        c1 = -x_opt[0] - x_opt[1] <= 1e-3
        c2 =  x_opt[0] - x_opt[1] <= 1e-3
        c3 =  x_opt[1] <= 2.0 + 1e-3

        if c1 and c2 and c3:
            print("✓ Test PASSED!")
            return True
        else:
            print("✗ Test FAILED: Constraints not satisfied")
            return False
    else:
        print("✗ Test FAILED: Solver did not converge")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  POGS Cone Solver Test Suite".center(68) + "*")
    print("*" + "  (Without CVXPY dependency)".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    tests = [
        test_lp_simple,
        test_lp_with_inequalities,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    try:
        import numpy as np
    except ImportError:
        print("NumPy is required for these tests.")
        print("Please install: pip install numpy")
        sys.exit(1)

    success = main()
    sys.exit(0 if success else 1)
