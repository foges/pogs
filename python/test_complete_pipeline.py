"""
Complete pipeline test demonstrating POGS cone interface from C++ to Python.

This test verifies:
1. C++ SDP cone projection
2. C interface for cone problems
3. Python solver interface
4. End-to-end problem solving

No external dependencies required (no numpy/cvxpy needed).
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"✗ FAILED with return code {result.returncode}")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        return False
    else:
        print("✓ SUCCESS")
        return True


def test_cpp_sdp_cone():
    """Test C++ SDP cone projection."""
    pogs_root = os.path.join(os.path.dirname(__file__), "..")

    # Check if test exists
    test_exe = os.path.join(pogs_root, "examples", "cpp_cone", "test_sdp")
    if not os.path.exists(test_exe):
        print("Building test_sdp...")
        os.chdir(os.path.join(pogs_root, "examples", "cpp_cone"))
        if not run_command(["make", "clean"], "Clean build directory"):
            return False
        if not run_command(["make", "cpu"], "Build C++ tests"):
            return False

    # Run test
    return run_command([test_exe], "Test 1: C++ SDP Cone Projection")


def test_c_interface():
    """Test C interface."""
    pogs_root = os.path.join(os.path.dirname(__file__), "..")

    test_exe = os.path.join(pogs_root, "examples", "cpp_cone", "test_c")
    if not os.path.exists(test_exe):
        print("Test executable not found. Please build it first.")
        return False

    return run_command([test_exe], "Test 2: C Interface for Cone Problems")


def test_python_interface():
    """Test Python interface without dependencies."""
    test_file = os.path.join(os.path.dirname(__file__), "test_cone_simple.py")

    if not os.path.exists(test_file):
        print("Python test file not found.")
        return False

    return run_command(["python3", test_file], "Test 3: Python Interface (No Dependencies)")


def test_python_solver():
    """Test Python solver (requires numpy)."""
    test_file = os.path.join(os.path.dirname(__file__), "test_pogs_solver.py")

    if not os.path.exists(test_file):
        print("Python solver test file not found.")
        return False

    # Check if numpy is available
    check_numpy = subprocess.run(["python3", "-c", "import numpy"], capture_output=True)
    if check_numpy.returncode != 0:
        print("\nNote: Skipping test (numpy not installed)")
        print("To run this test: pip install numpy")
        return None  # Skip, not fail

    return run_command(["python3", test_file], "Test 4: Python Solver with NumPy")


def verify_installation():
    """Verify all components are installed."""
    verify_script = os.path.join(os.path.dirname(__file__), "verify_cvxpy_interface.py")

    if not os.path.exists(verify_script):
        print("Verification script not found.")
        return False

    return run_command(["python3", verify_script], "Verification: Check All Components")


def main():
    """Run complete pipeline test."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  POGS Complete Pipeline Test".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    print("This test verifies the complete POGS cone implementation:")
    print("  1. C++ SDP cone projection")
    print("  2. C interface for cone problems")
    print("  3. Python interface (basic)")
    print("  4. Python solver (with numpy)")
    print("  5. Component verification")
    print()

    tests = [
        ("Component Verification", verify_installation),
        ("C++ SDP Cone", test_cpp_sdp_cone),
        ("C Interface", test_c_interface),
        ("Python Interface", test_python_interface),
        ("Python Solver", test_python_solver),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
            else:  # None = skipped
                skipped += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Final summary
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Tests Passed:  {passed}")
    print(f"Tests Failed:  {failed}")
    print(f"Tests Skipped: {skipped}")
    print(f"Total Tests:   {len(tests)}")
    print("=" * 70)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print()
        print("The POGS cone implementation is working correctly.")
        print()
        print("Next steps:")
        print("  - Install dependencies: pip install numpy cvxpy")
        print("  - Run CVXPY tests: python3 test_cvxpy_interface.py")
        print("  - See documentation: CVXPY_INTERFACE.md")
        print()
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
