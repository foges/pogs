"""
CVXPY solver interface for POGS.

Allows CVXPY to use POGS as a backend solver for cone programs.
"""

import numpy as np
import time
import subprocess
import os
import sys
import tempfile

_pogs_dir = os.path.dirname(__file__)
if _pogs_dir not in sys.path:
    sys.path.insert(0, _pogs_dir)

try:
    from pogs_cone import solve_cone as solve_cone_ctypes
    _CTYPES_AVAILABLE = True
except Exception:
    solve_cone_ctypes = None
    _CTYPES_AVAILABLE = False


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
                       rho=None, abs_tol=1e-4, rel_tol=1e-3,
                       max_iter=10000, verbose=0, adaptive_rho=True,
                       use_direct=None, prefer_ctypes=True, P=None,
                       rho_mode=None, rho_scale=1.0):
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
    rho : float or None
        Initial penalty parameter. If None, automatically selected based on
        problem scaling (recommended for general problems).
    abs_tol : float
        Absolute tolerance
    rel_tol : float
        Relative tolerance
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    use_direct : bool or None
        Use direct projection (dense A). If None, choose automatically.
    prefer_ctypes : bool
        Prefer ctypes-based interface when available.
    P : array_like or sparse matrix, optional
        Quadratic objective matrix for 0.5 * x^T P x + c^T x.
    rho_mode : str or None
        Rho selection mode when rho is None: 'auto', 'ratio', or 'ratio_normA'.
    rho_scale : float
        Multiplier applied to auto-selected rho.

    Returns
    -------
    dict
        Solution dictionary with keys 'x', 'y', 's', 'z', 'status', 'num_iters'
    """

    # Convert to numpy arrays (handle sparse matrices from CVXPY)
    c = np.asarray(c, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()

    # Handle sparse matrices
    try:
        import scipy.sparse as sp
        if sp.issparse(A):
            A = A.toarray()
        if P is not None and sp.issparse(P):
            P = P.toarray()
    except ImportError:
        pass
    A = np.asarray(A, dtype=np.float64)
    if P is not None:
        P = np.asarray(P, dtype=np.float64)

    m, n = A.shape

    # Automatic rho selection based on problem scaling.
    # Use a tighter heuristic for non-separable cones (SOC/SDP/EXP) to avoid
    # overly large rho values that slow convergence.
    if rho is None:
        norm_c = np.linalg.norm(c)
        norm_b = np.linalg.norm(b)
        has_nonsep_cone = bool(dims.get('q')) or bool(dims.get('s')) \
            or dims.get('ep', 0) > 0 or dims.get('ed', 0) > 0
        mode = rho_mode or 'auto'
        if mode == 'auto':
            mode = 'ratio_normA' if has_nonsep_cone else 'ratio'
        if mode == 'ratio_normA':
            norm_A = np.linalg.norm(A, 'fro')
            if norm_b > 1e-10 and norm_c > 1e-10 and norm_A > 1e-10:
                rho = norm_c / (norm_b * norm_A)
                rho = max(1e-4, min(1e1, rho))
            else:
                rho = 1.0
            if verbose > 0:
                print(f"Auto rho (ratio_normA): ||c||={norm_c:.2e}, "
                      f"||b||={norm_b:.2e}, ||A||={norm_A:.2e} -> rho={rho:.2e}")
        elif mode == 'ratio':
            if norm_b > 1e-10 and norm_c > 1e-10:
                rho = norm_c / norm_b
                rho = max(1e-3, min(1e3, rho))
            else:
                rho = 1.0
            if verbose > 0:
                print(f"Auto rho (ratio): ||c||={norm_c:.2e}, ||b||={norm_b:.2e} "
                      f"-> rho={rho:.2e}")
        else:
            raise ValueError(f"Unknown rho_mode: {rho_mode}")
        if rho_scale not in (None, 1.0):
            rho *= rho_scale
            if verbose > 0:
                print(f"Auto rho scaled by {rho_scale:.2e} -> rho={rho:.2e}")
    assert c.shape == (n,), f"c has wrong shape: {c.shape} vs ({n},)"
    assert b.shape == (m,), f"b has wrong shape: {b.shape} vs ({m},)"
    if P is not None:
        assert P.shape == (n, n), f"P has wrong shape: {P.shape} vs ({n}, {n})"

    if use_direct is None:
        min_dim = min(m, n)
        use_direct = (m * n <= 1_000_000 and min_dim <= 1000)

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

    if _CTYPES_AVAILABLE and prefer_ctypes:
        t0 = time.perf_counter()
        result = solve_cone_ctypes(
            A, b, c,
            cones_x, cones_y,
            rho=rho,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            adaptive_rho=adaptive_rho,
            gap_stop=True,
            use_direct=use_direct,
            P=P,
        )
        solve_time = time.perf_counter() - t0
        parsed = {
            'status': result.get('status', 0),
            'num_iters': result.get('iterations', 0),
            'optval': result.get('optval', 0),
            'x': result.get('x'),
            'y': result.get('y'),
            'z': result.get('l'),
            'solve_time': solve_time,
        }
        if parsed['y'] is not None:
            parsed['s'] = b - parsed['y']
        return parsed

    # Generate C code to call POGS
    c_code = _generate_c_code(c, A, b, cones_x, cones_y, P,
                              rho, abs_tol, rel_tol, max_iter, verbose,
                              adaptive_rho, use_direct)

    # Compile and run
    t0 = time.perf_counter()
    result = _compile_and_run(c_code)
    result['solve_time'] = time.perf_counter() - t0
    if 'y' in result:
        # Slack for conic form: b - y, where y ~ Ax
        result['s'] = b - result['y']

    return result


def _generate_c_code(c, A, b, cones_x, cones_y, P,
                     rho, abs_tol, rel_tol, max_iter, verbose, adaptive_rho,
                     use_direct):
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
    if P is not None:
        code += f"    // Matrix P ({n} x {n})\n"
        code += f"    double P[{n * n}] = {{\n"
        for i in range(n):
            code += "        "
            for j in range(n):
                code += f"{P[i, j]:.16e}"
                if i < n - 1 or j < n - 1:
                    code += ", "
            code += "\n"
        code += "    };\n\n"

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
    if use_direct:
        solver_base = "PogsConeDirect"
    else:
        solver_base = "PogsCone"
    code += "    // Solve problem\n"
    if P is None:
        code += f"    int status = {solver_base}D(\n"
        code += f"        ROW_MAJ, {m}, {n}, A, b, c,\n"
        code += f"        cones_x, {len(cones_x) if cones_x else 0},\n"
        code += f"        cones_y, {len(cones_y)},\n"
        code += f"        {rho}, {abs_tol}, {rel_tol}, {max_iter}, {verbose},\n"
        code += f"        {1 if adaptive_rho else 0}, 1,  // adaptive_rho, gap_stop\n"
        code += f"        x, y, l, &optval, &final_iter\n"
        code += f"    );\n\n"
    else:
        code += f"    int status = {solver_base}QD(\n"
        code += f"        ROW_MAJ, {m}, {n}, A, b, c, P,\n"
        code += f"        cones_x, {len(cones_x) if cones_x else 0},\n"
        code += f"        cones_y, {len(cones_y)},\n"
        code += f"        {rho}, {abs_tol}, {rel_tol}, {max_iter}, {verbose},\n"
        code += f"        {1 if adaptive_rho else 0}, 1,  // adaptive_rho, gap_stop\n"
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

        @staticmethod
        def cite():
            """Return citation string for POGS."""
            return "@article{fougner2018pogs,\n  title={POGS: Proximal Operator Graph Solver},\n  author={Fougner, Christopher and Boyd, Stephen},\n  year={2018}\n}"

        def supports_quad_obj(self) -> bool:
            """Report quadratic objective support."""
            return True

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
            import cvxpy.settings as s
            P = data.get(s.P, None)

            # Convert ConeDims object to dict format expected by solve_cone_problem
            # CVXPY ConeDims has: zero, nonneg, exp, soc, psd, p3d
            # POGS expects: f, l, q, s, ep, ed
            cvxpy_dims = data['dims']
            dims = {
                'f': getattr(cvxpy_dims, 'zero', 0),      # zero cone
                'l': getattr(cvxpy_dims, 'nonneg', 0),    # nonneg cone
                'q': list(getattr(cvxpy_dims, 'soc', [])), # SOC cones
                's': list(getattr(cvxpy_dims, 'psd', [])), # SDP cones
                'ep': getattr(cvxpy_dims, 'exp', 0),       # exponential cones
                'ed': 0,                                    # dual exponential cones
            }

            # Get solver options
            opts = solver_opts.copy() if solver_opts else {}
            abs_tol = opts.get('abs_tol', 1e-4)
            rel_tol = opts.get('rel_tol', 1e-3)
            max_iter = opts.get('max_iter', 50000)  # Higher default for convergence
            rho = opts.get('rho', None)  # Use automatic rho selection by default
            adaptive_rho = opts.get('adaptive_rho', True)
            rho_mode = opts.get('rho_mode', None)
            rho_scale = opts.get('rho_scale', 1.0)
            verbose_level = opts.get('verbose', 5 if verbose else 0)
            use_direct = opts.get('use_direct', None)
            prefer_ctypes = opts.get('prefer_ctypes', True)

            # Solve
            result = solve_cone_problem(
                c, A, b, dims,
                rho=rho,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                max_iter=max_iter,
                adaptive_rho=adaptive_rho,
                verbose=verbose_level,
                use_direct=use_direct,
                prefer_ctypes=prefer_ctypes,
                P=P,
                rho_mode=rho_mode,
                rho_scale=rho_scale,
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

            from cvxpy.reductions.solution import Solution, failure_solution
            from cvxpy.reductions.solvers.solver import Solver
            import cvxpy.settings as s

            attr = {}

            # POGS status codes:
            # 0 = optimal, 1 = infeasible, 2 = unbounded, 3 = iteration limit
            pogs_status = solution['status']
            if pogs_status == 0:
                status = s.OPTIMAL
            elif pogs_status == 3:
                # Iteration limit reached - return inaccurate solution
                status = s.OPTIMAL_INACCURATE
            else:
                status = s.SOLVER_ERROR

            attr[s.SOLVE_TIME] = solution.get('solve_time', 0)
            attr[s.SETUP_TIME] = solution.get('setup_time', 0)
            attr[s.NUM_ITERS] = solution.get('num_iters', 0)

            if status in [s.OPTIMAL, s.OPTIMAL_INACCURATE]:
                # Extract optimal value with offset
                opt_val = solution.get('optval', 0)
                if s.OFFSET in inverse_data:
                    opt_val += inverse_data[s.OFFSET]

                # Unpack primal solution
                # VAR_ID is a class constant inherited from Solver
                primal_vars = {
                    inverse_data[self.VAR_ID]: solution['x']
                }

                # Return None for dual variables to skip dual extraction
                # (POGS cone solver doesn't track constraint duals in a compatible way)
                dual_vars = None

                return Solution(status, opt_val, primal_vars, dual_vars, attr)
            else:
                return failure_solution(status, attr)

    # Register the solver with CVXPY so "solver='POGS'" works.
    try:
        from cvxpy.reductions.solvers import defines as cvxpy_defines
        solver_name = "POGS"
        if solver_name not in cvxpy_defines.SOLVER_MAP_CONIC:
            cvxpy_defines.SOLVER_MAP_CONIC[solver_name] = POGS()
        if solver_name not in cvxpy_defines.CONIC_SOLVERS:
            cvxpy_defines.CONIC_SOLVERS.append(solver_name)
        cvxpy_defines.INSTALLED_SOLVERS = cvxpy_defines.installed_solvers()
        cvxpy_defines.INSTALLED_CONIC_SOLVERS = [
            slv for slv in cvxpy_defines.INSTALLED_SOLVERS
            if slv in cvxpy_defines.CONIC_SOLVERS
        ]
    except Exception:
        pass

except ImportError:
    # CVXPY not installed, skip integration
    POGS = None
    pass
