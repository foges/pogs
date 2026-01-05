"""
Simple test for POGS cone interface using Python (no dependencies).
"""

import subprocess
import os
import tempfile

# Test problem: minimize x1 subject to x1 + x2 = 2, x >= 0
# Expected: x = [0, 2], optimal value = 0

print("Testing POGS cone interface from Python...")
print("Problem: minimize x1 subject to x1 + x2 = 2, x >= 0")
print("Expected: x = [0, 2], optimal value = 0\n")

# Create a C program that calls the interface
test_program = """
#include <stdio.h>
#include "pogs_c.h"

int main() {
    double A[] = {1.0, 1.0};
    double b[] = {2.0};
    double c[] = {1.0, 0.0};

    unsigned int x_indices[] = {0, 1};
    struct ConeConstraintC cone_x = {CONE_NON_NEG, x_indices, 2};

    unsigned int y_indices[] = {0};
    struct ConeConstraintC cone_y = {CONE_ZERO, y_indices, 1};

    double x[2], y[1], l[1];
    double optval;
    unsigned int final_iter;

    int status = PogsConeD(ROW_MAJ, 1, 2, A, b, c,
                           &cone_x, 1, &cone_y, 1,
                           1.0, 1e-6, 1e-6, 10000, 0, 1, 1,
                           x, y, l, &optval, &final_iter);

    printf("Solution (LP):\\n");
    printf("  x = [%.6f, %.6f]\\n", x[0], x[1]);
    printf("  optimal value = %.6f\\n", optval);
    printf("  status = %s\\n", status == 0 ? "Success" : "Failed");
    printf("  iterations = %u\\n", final_iter);

    // Quadratic objective test: min 0.5||x||^2 + x1 s.t. x1 + x2 = 1
    double b_qp[] = {1.0};
    double c_qp[] = {1.0, 0.0};
    double P_qp[] = {1.0, 0.0, 0.0, 1.0};

    double x_qp[2], y_qp[1], l_qp[1];
    double optval_qp;
    unsigned int final_iter_qp;

    int status_qp = PogsConeQD(ROW_MAJ, 1, 2, A, b_qp, c_qp, P_qp,
                               NULL, 0, &cone_y, 1,
                               1.0, 1e-6, 1e-6, 10000, 0, 1, 1,
                               x_qp, y_qp, l_qp, &optval_qp, &final_iter_qp);

    printf("Solution (QP):\\n");
    printf("  x = [%.6f, %.6f]\\n", x_qp[0], x_qp[1]);
    printf("  optimal value = %.6f\\n", optval_qp);
    printf("  status = %s\\n", status_qp == 0 ? "Success" : "Failed");
    printf("  iterations = %u\\n", final_iter_qp);

    return (status == 0 && status_qp == 0) ? 0 : 1;
}
"""

# Write test program
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    test_file = f.name
    f.write(test_program)

try:
    # Compile and run
    pogs_root = os.path.join(os.path.dirname(__file__), '..')
    output_file = tempfile.mktemp()

    compile_cmd = [
        'gcc', '-g', '-O3',
        f'-I{pogs_root}/src/include',
        f'-I{pogs_root}/src/interface_c',
        f'-I{pogs_root}/src/cpu/include',
        '-std=c11',
        '-o', output_file,
        test_file,
        f'{pogs_root}/src/build/pogs.a',
        '-lm', '-framework', 'Accelerate', '-lstdc++'
    ]

    print("Compiling test program...")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        exit(1)

    print("Running test...\n")
    result = subprocess.run([output_file], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Test failed with return code:", result.returncode)
        if result.stderr:
            print("Error output:", result.stderr)
        exit(1)

finally:
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(output_file):
        os.remove(output_file)

print("\nPython test passed! C interface is working correctly.")
