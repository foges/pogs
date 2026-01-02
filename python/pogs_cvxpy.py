"""
CVXPY solver interface for POGS.

Allows CVXPY to use POGS as a backend solver for cone programs.
"""

import numpy as np
import subprocess
import os
import sys
import tempfile


class PogsError(Exception):
    """Exception raised for POGS solver errors."""
    pass


# Cone type mappings
CONE_ZERO = 0
CONE_NON_NEG = 1
CONE_NON_POS = 2
CONE_SOC = 3
CONE_SDP = 4
CONE_EXP_PRIMAL = 5
CONE_EXP_DUAL = 6


def solve_cone_problem(c, A, b, dims,
                       rho=1.0, abs_tol=1e-4, rel_tol=1e-3,
                       max_iter=10000, verbose=0):
    """
    Solve a cone problem using POGS.

    minimize    c^T * x
    subject to  b - A*x in K

    where K is a Cartesian product of cones specified by dims.

    Parameters
    ----------
    c : array_like, shape (n,)
        Objective vector
    A : array_like, shape (m, n)
        Constraint matrix
    b : array_like, shape (m,)
        Constraint vector
    dims : dict
        Dictionary specifying cone dimensions with keys:
        - 'f': int, number of free variables (zero cone)
        - 'l': int, number of non-negative variables
        - 'q': list of ints, dimensions of SOC cones
        - 's': list of ints, dimensions of SDP cones
        - 'ep': int, number of primal exponential cones
        - 'ed': int, number of dual exponential cones
    rho : float
        Initial penalty parameter
    abs_tol : float
        Absolute tolerance
    rel_tol : float
        Relative tolerance
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level

    Returns
    -------
    dict
        Solution dictionary with keys 'x', 'y', 's', 'z', 'status', 'num_iters'
    """

    # Convert to numpy arrays
    c = np.asarray(c, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    A = np.asarray(A, dtype=np.float64)

    m, n = A.shape
    assert c.shape == (n,), f"c has wrong shape: {c.shape} vs ({n},)"
    assert b.shape == (m,), f"b has wrong shape: {b.shape} vs ({m},)"

    # Build cone constraints for y (dual variables)
    # Format: list of (cone_type, [indices])
    cones_y = []
    offset = 0

    # Free variables (zero cone)
    if dims.get('f', 0) > 0:
        nf = dims['f']
        cones_y.append((CONE_ZERO, list(range(offset, offset + nf))))
        offset += nf

    # Non-negative cone
    if dims.get('l', 0) > 0:
        nl = dims['l']
        cones_y.append((CONE_NON_NEG, list(range(offset, offset + nl))))
        offset += nl

    # Second-order cones
    if 'q' in dims and dims['q']:
        for q_dim in dims['q']:
            cones_y.append((CONE_SOC, list(range(offset, offset + q_dim))))
            offset += q_dim

    # Semidefinite cones (vectorized)
    if 's' in dims and dims['s']:
        for s_dim in dims['s']:
            # SDP cone: symmetric matrix of size s_dim x s_dim
            # Vectorized as lower triangle: s_dim*(s_dim+1)/2 elements
            vec_dim = s_dim * (s_dim + 1) // 2
            cones_y.append((CONE_SDP, list(range(offset, offset + vec_dim))))
            offset += vec_dim

    # Exponential cones (primal)
    if dims.get('ep', 0) > 0:
        n_exp = dims['ep']
        for i in range(n_exp):
            cones_y.append((CONE_EXP_PRIMAL, list(range(offset, offset + 3))))
            offset += 3

    # Exponential cones (dual)
    if dims.get('ed', 0) > 0:
        n_exp = dims['ed']
        for i in range(n_exp):
            cones_y.append((CONE_EXP_DUAL, list(range(offset, offset + 3))))
            offset += 3

    # For now, assume x is free (no constraints on primal variable)
    # This matches the standard conic form used by most solvers
    cones_x = []

    # Generate C code to call POGS
    c_code = _generate_c_code(c, A, b, cones_x, cones_y,
                             rho, abs_tol, rel_tol, max_iter, verbose)

    # Compile and run
    result = _compile_and_run(c_code)

    return result


def _generate_c_code(c, A, b, cones_x, cones_y, rho, abs_tol, rel_tol, max_iter, verbose):
    """Generate C code to solve the problem."""

    m, n = A.shape

    # Start building C code
    code = """
#include <stdio.h>
#include <stdlib.h>
#include "pogs_c.h"

int main() {
"""

    # Add matrix A (row-major)
    code += f"    // Matrix A ({m} x {n})\n"
    code += f"    double A[{m * n}] = {{\n"
    for i in range(m):
        code += "        "
        for j in range(n):
            code += f"{A[i, j]:.16e}"
            if i < m - 1 or j < n - 1:
                code += ", "
        code += "\n"
    code += "    };\n\n"

    # Add vector b
    code += f"    // Vector b ({m})\n"
    code += f"    double b[{m}] = {{\n        "
    code += ", ".join([f"{b[i]:.16e}" for i in range(m)])
    code += "\n    };\n\n"

    # Add vector c
    code += f"    // Vector c ({n})\n"
    code += f"    double c[{n}] = {{\n        "
    code += ", ".join([f"{c[i]:.16e}" for i in range(n)])
    code += "\n    };\n\n"

    # Add cone constraints for x
    if cones_x:
        code += f"    // Cone constraints for x\n"
        for i, (cone_type, indices) in enumerate(cones_x):
            code += f"    unsigned int x_indices_{i}[] = {{"
            code += ", ".join(map(str, indices))
            code += "};\n"
            code += f"    struct ConeConstraintC cone_x_{i} = {{{cone_type}, x_indices_{i}, {len(indices)}}};\n"
        code += f"    struct ConeConstraintC cones_x[] = {{"
        code += ", ".join([f"cone_x_{i}" for i in range(len(cones_x))])
        code += "};\n\n"
    else:
        code += "    // No cone constraints for x (free)\n"
        code += "    struct ConeConstraintC *cones_x = NULL;\n\n"

    # Add cone constraints for y
    code += f"    // Cone constraints for y\n"
    for i, (cone_type, indices) in enumerate(cones_y):
        code += f"    unsigned int y_indices_{i}[] = {{"
        code += ", ".join(map(str, indices))
        code += "};\n"
        code += f"    struct ConeConstraintC cone_y_{i} = {{{cone_type}, y_indices_{i}, {len(indices)}}};\n"
    code += f"    struct ConeConstraintC cones_y[] = {{"
    code += ", ".join([f"cone_y_{i}" for i in range(len(cones_y))])
    code += "};\n\n"

    # Allocate solution arrays
    code += f"    // Allocate solution arrays\n"
    code += f"    double x[{n}];\n"
    code += f"    double y[{m}];\n"
    code += f"    double l[{m}];\n"
    code += f"    double optval;\n"
    code += f"    unsigned int final_iter;\n\n"

    # Call solver
    code += "    // Solve problem\n"
    code += f"    int status = PogsConeD(\n"
    code += f"        ROW_MAJ, {m}, {n}, A, b, c,\n"
    code += f"        cones_x, {len(cones_x) if cones_x else 0},\n"
    code += f"        cones_y, {len(cones_y)},\n"
    code += f"        {rho}, {abs_tol}, {rel_tol}, {max_iter}, {verbose},\n"
    code += f"        1, 1,  // adaptive_rho, gap_stop\n"
    code += f"        x, y, l, &optval, &final_iter\n"
    code += f"    );\n\n"

    # Output results
    code += "    // Output results in machine-readable format\n"
    code += "    printf(\"STATUS=%d\\n\", status);\n"
    code += "    printf(\"ITERS=%u\\n\", final_iter);\n"
    code += "    printf(\"OPTVAL=%.16e\\n\", optval);\n"
    code += "    \n"
    code += "    printf(\"X=\");\n"
    code += f"    for (int i = 0; i < {n}; i++) {{\n"
    code += "        printf(\"%.16e\", x[i]);\n"
    code += f"        if (i < {n - 1}) printf(\",\");\n"
    code += "    }\n"
    code += "    printf(\"\\n\");\n"
    code += "    \n"
    code += "    printf(\"Y=\");\n"
    code += f"    for (int i = 0; i < {m}; i++) {{\n"
    code += "        printf(\"%.16e\", y[i]);\n"
    code += f"        if (i < {m - 1}) printf(\",\");\n"
    code += "    }\n"
    code += "    printf(\"\\n\");\n"
    code += "    \n"
    code += "    printf(\"L=\");\n"
    code += f"    for (int i = 0; i < {m}; i++) {{\n"
    code += "        printf(\"%.16e\", l[i]);\n"
    code += f"        if (i < {m - 1}) printf(\",\");\n"
    code += "    }\n"
    code += "    printf(\"\\n\");\n"
    code += "    \n"
    code += "    return status;\n"
    code += "}\n"

    return code


def _compile_and_run(c_code):
    """Compile and run the generated C code."""

    # Find POGS root directory
    pogs_root = os.path.join(os.path.dirname(__file__), '..')

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        c_file = f.name
        f.write(c_code)

    exe_file = tempfile.mktemp()

    try:
        # Compile
        compile_cmd = [
            'gcc', '-O3',
            f'-I{pogs_root}/src/include',
            f'-I{pogs_root}/src/interface_c',
            f'-I{pogs_root}/src/cpu/include',
            '-std=c11',
            '-o', exe_file,
            c_file,
            f'{pogs_root}/src/build/pogs.a',
            '-lm', '-framework', 'Accelerate', '-lstdc++'
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PogsError(f"Compilation failed: {result.stderr}")

        # Run
        result = subprocess.run([exe_file], capture_output=True, text=True, timeout=60)

        # Parse output
        output_lines = result.stdout.strip().split('\n')
        parsed = {}
        for line in output_lines:
            if '=' in line:
                key, value = line.split('=', 1)
                if key == 'STATUS':
                    parsed['status'] = int(value)
                elif key == 'ITERS':
                    parsed['num_iters'] = int(value)
                elif key == 'OPTVAL':
                    parsed['optval'] = float(value)
                elif key == 'X':
                    parsed['x'] = np.array([float(v) for v in value.split(',')])
                elif key == 'Y':
                    parsed['y'] = np.array([float(v) for v in value.split(',')])
                elif key == 'L':
                    parsed['z'] = np.array([float(v) for v in value.split(',')])  # dual variable

        # Set s (slack) equal to y for conic solvers
        parsed['s'] = parsed['y']

        return parsed

    finally:
        # Cleanup
        if os.path.exists(c_file):
            os.remove(c_file)
        if os.path.exists(exe_file):
            os.remove(exe_file)


# CVXPY integration
try:
    import cvxpy
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

    class POGS(ConicSolver):
        """CVXPY interface for POGS solver."""

        MIP_CAPABLE = False
        SUPPORTED_CONSTRAINTS = [
            cvxpy.Zero,
            cvxpy.NonNeg,
            cvxpy.SOC,
            cvxpy.PSD,
            cvxpy.ExpCone
        ]

        def name(self):
            return "POGS"

        def import_solver(self):
            """Check that POGS is available."""
            pogs_root = os.path.join(os.path.dirname(__file__), '..')
            pogs_lib = os.path.join(pogs_root, 'src', 'build', 'pogs.a')
            if not os.path.exists(pogs_lib):
                raise ImportError("POGS library not found. Please build it first.")

        def apply(self, problem):
            """Return the problem data in POGS format."""
            data = super(POGS, self).apply(problem)
            return data

        def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
            """
            Solve a problem represented in CVXPY's conic format.

            Parameters
            ----------
            data : dict
                Problem data in conic form
            warm_start : bool
                Whether to warm start (ignored)
            verbose : bool
                Whether to print verbose output
            solver_opts : dict
                Additional solver options
            solver_cache : optional
                Cached solver data (ignored)

            Returns
            -------
            dict
                Solution data
            """

            # Extract problem data
            c = data['c']
            A = data['A']
            b = data['b']
            dims = data['dims']

            # Get solver options
            opts = solver_opts.copy() if solver_opts else {}
            abs_tol = opts.get('abs_tol', 1e-4)
            rel_tol = opts.get('rel_tol', 1e-3)
            max_iter = opts.get('max_iter', 10000)
            rho = opts.get('rho', 1.0)
            verbose_level = 5 if verbose else 0

            # Solve
            result = solve_cone_problem(
                c, A, b, dims,
                rho=rho,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                max_iter=max_iter,
                verbose=verbose_level
            )

            return result

        def invert(self, solution, inverse_data):
            """
            Convert POGS solution to CVXPY format.

            Parameters
            ----------
            solution : dict
                Solution from POGS
            inverse_data : dict
                Metadata for converting solution

            Returns
            -------
            dict
                Solution in CVXPY format
            """

            attr = {}

            if solution['status'] == 0:
                status = cvxpy.OPTIMAL
            else:
                status = cvxpy.SOLVER_ERROR

            attr[cvxpy.settings.SOLVE_TIME] = 0  # Not tracked
            attr[cvxpy.settings.NUM_ITERS] = solution.get('num_iters', 0)

            # Unpack solution
            primal_vars = {
                inverse_data[cvxpy.settings.VAR_ID]: solution['x']
            }

            dual_vars = {}
            if 'z' in solution:
                dual_vars = cvxpy.reductions.solvers.utilities.extract_dual_value(
                    solution['z'],
                    inverse_data[cvxpy.settings.EQ_DUAL],
                    inverse_data.get(cvxpy.settings.NEQ_DUAL, [])
                )

            return cvxpy.Solution(status, solution.get('optval'), primal_vars, dual_vars, attr)

    # Register the solver with CVXPY
    # Note: This happens automatically when the solver is imported

except ImportError:
    # CVXPY not installed, skip integration
    POGS = None
    pass
