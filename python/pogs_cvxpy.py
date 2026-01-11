"""
CVXPY solver interface for POGS.

Allows CVXPY to use POGS as a backend solver for cone programs.
Key feature: Detects graph-form problems (Lasso, Ridge, logistic, etc.)
and routes them to the fast graph-form solver instead of cone form.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np


_pogs_dir = os.path.dirname(__file__)
if _pogs_dir not in sys.path:
    sys.path.insert(0, _pogs_dir)

try:
    from pogs_cone import solve_cone as solve_cone_ctypes

    _CTYPES_AVAILABLE = True
except Exception:
    solve_cone_ctypes = None
    _CTYPES_AVAILABLE = False

try:
    from pogs_graph import (
        Function,
        FunctionObj,
        _solve_graph_form,
        solve_elastic_net,
        solve_huber,
        solve_lasso,
        solve_nonneg_ls,
        solve_qp,
        solve_ridge,
    )

    _GRAPH_AVAILABLE = True
except Exception:
    _GRAPH_AVAILABLE = False


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


def _compute_primal_residual(A, x, y, abs_tol, rel_tol):
    """Compute primal residual and tolerance in original scale."""
    if x is None or y is None:
        return None, None
    Ax = A @ x
    r = Ax - y
    nrm_r = np.linalg.norm(r)
    ax_norm = np.linalg.norm(Ax)
    y_norm = np.linalg.norm(y)
    eps_pri = np.sqrt(A.shape[0]) * abs_tol + rel_tol * max(ax_norm, y_norm)
    return nrm_r, eps_pri


def solve_cone_problem(
    c,
    A,
    b,
    dims,
    rho=None,
    abs_tol=1e-4,
    rel_tol=1e-3,
    max_iter=10000,
    verbose=0,
    adaptive_rho=True,
    use_direct=None,
    prefer_ctypes=True,
    P=None,
    rho_mode=None,
    rho_scale=1.0,
):
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
        # Check if P is actually used (non-zero)
        if np.linalg.norm(P, ord="fro") > 1e-12:
            import warnings

            warnings.warn(
                "POGS HSDE cone solver does not correctly handle quadratic objectives. "
                "The QP will likely fail to converge. Use OSQP, SCS, or CLARABEL instead, "
                "or use pogs_solve() for graph-form problems (Lasso, Ridge, etc.).",
                RuntimeWarning,
                stacklevel=2,
            )

    m, n = A.shape

    # Automatic rho selection based on problem scaling.
    # Use a tighter heuristic for non-separable cones (SOC/SDP/EXP) to avoid
    # overly large rho values that slow convergence.
    if rho is None:
        norm_c = np.linalg.norm(c)
        norm_b = np.linalg.norm(b)
        has_nonsep_cone = (
            bool(dims.get("q"))
            or bool(dims.get("s"))
            or dims.get("ep", 0) > 0
            or dims.get("ed", 0) > 0
        )
        has_quadratic = P is not None
        mode = rho_mode or "auto"
        if mode == "auto":
            mode = "ratio_normA" if (has_nonsep_cone or has_quadratic) else "ratio"
        if mode == "ratio_normA":
            norm_A = np.linalg.norm(A, "fro")
            if norm_b > 1e-10 and norm_c > 1e-10 and norm_A > 1e-10:
                rho = norm_c / (norm_b * norm_A)
                rho = max(1e-4, min(1e1, rho))
            else:
                rho = 1.0
            if verbose > 0:
                print(
                    f"Auto rho (ratio_normA): ||c||={norm_c:.2e}, "
                    f"||b||={norm_b:.2e}, ||A||={norm_A:.2e} -> rho={rho:.2e}"
                )
        elif mode == "ratio":
            if norm_b > 1e-10 and norm_c > 1e-10:
                rho = norm_c / norm_b
                rho = max(1e-3, min(1e3, rho))
            else:
                rho = 1.0
            if verbose > 0:
                print(f"Auto rho (ratio): ||c||={norm_c:.2e}, ||b||={norm_b:.2e} -> rho={rho:.2e}")
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
        use_direct = m * n <= 1_000_000 and min_dim <= 1000

    # Build cone constraints for y (dual variables)
    # Format: list of (cone_type, [indices])
    cones_y = []
    offset = 0

    # Free variables (zero cone)
    if dims.get("f", 0) > 0:
        nf = dims["f"]
        cones_y.append((CONE_ZERO, list(range(offset, offset + nf))))
        offset += nf

    # Non-negative cone
    if dims.get("l", 0) > 0:
        nl = dims["l"]
        cones_y.append((CONE_NON_NEG, list(range(offset, offset + nl))))
        offset += nl

    # Second-order cones
    if dims.get("q"):
        for q_dim in dims["q"]:
            cones_y.append((CONE_SOC, list(range(offset, offset + q_dim))))
            offset += q_dim

    # Semidefinite cones (vectorized)
    if dims.get("s"):
        for s_dim in dims["s"]:
            # SDP cone: symmetric matrix of size s_dim x s_dim
            # Vectorized as lower triangle: s_dim*(s_dim+1)/2 elements
            vec_dim = s_dim * (s_dim + 1) // 2
            cones_y.append((CONE_SDP, list(range(offset, offset + vec_dim))))
            offset += vec_dim

    # Exponential cones (primal)
    if dims.get("ep", 0) > 0:
        n_exp = dims["ep"]
        for _i in range(n_exp):
            cones_y.append((CONE_EXP_PRIMAL, list(range(offset, offset + 3))))
            offset += 3

    # Exponential cones (dual)
    if dims.get("ed", 0) > 0:
        n_exp = dims["ed"]
        for _i in range(n_exp):
            cones_y.append((CONE_EXP_DUAL, list(range(offset, offset + 3))))
            offset += 3

    # For now, assume x is free (no constraints on primal variable)
    # This matches the standard conic form used by most solvers
    cones_x = []

    if _CTYPES_AVAILABLE and prefer_ctypes:
        t0 = time.perf_counter()
        result = solve_cone_ctypes(
            A,
            b,
            c,
            cones_x,
            cones_y,
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
            "status": result.get("status", 0),
            "num_iters": result.get("iterations", 0),
            "optval": result.get("optval", 0),
            "x": result.get("x"),
            "y": result.get("y"),
            "z": result.get("l"),
            "solve_time": solve_time,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
        }
        if parsed["y"] is not None:
            parsed["s"] = b - parsed["y"]
        if parsed["x"] is not None and parsed["y"] is not None:
            primal_res, eps_pri = _compute_primal_residual(
                A, parsed["x"], parsed["y"], abs_tol, rel_tol
            )
            parsed["primal_res"] = primal_res
            parsed["eps_pri"] = eps_pri
            if eps_pri is not None and eps_pri > 0:
                parsed["primal_res_ratio"] = primal_res / eps_pri
        return parsed

    # Generate C code to call POGS
    c_code = _generate_c_code(
        c,
        A,
        b,
        cones_x,
        cones_y,
        P,
        rho,
        abs_tol,
        rel_tol,
        max_iter,
        verbose,
        adaptive_rho,
        use_direct,
    )

    # Compile and run
    t0 = time.perf_counter()
    result = _compile_and_run(c_code)
    result["solve_time"] = time.perf_counter() - t0
    result["abs_tol"] = abs_tol
    result["rel_tol"] = rel_tol
    if "y" in result:
        # Slack for conic form: b - y, where y ~ Ax
        result["s"] = b - result["y"]
    if "x" in result and "y" in result:
        primal_res, eps_pri = _compute_primal_residual(
            A, result["x"], result["y"], abs_tol, rel_tol
        )
        result["primal_res"] = primal_res
        result["eps_pri"] = eps_pri
        if eps_pri is not None and eps_pri > 0:
            result["primal_res_ratio"] = primal_res / eps_pri

    return result


def _generate_c_code(
    c, A, b, cones_x, cones_y, P, rho, abs_tol, rel_tol, max_iter, verbose, adaptive_rho, use_direct
):
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
        code += "    // Cone constraints for x\n"
        for i, (cone_type, indices) in enumerate(cones_x):
            code += f"    unsigned int x_indices_{i}[] = {{"
            code += ", ".join(map(str, indices))
            code += "};\n"
            code += f"    struct ConeConstraintC cone_x_{i} = {{{cone_type}, x_indices_{i}, {len(indices)}}};\n"
        code += "    struct ConeConstraintC cones_x[] = {"
        code += ", ".join([f"cone_x_{i}" for i in range(len(cones_x))])
        code += "};\n\n"
    else:
        code += "    // No cone constraints for x (free)\n"
        code += "    struct ConeConstraintC *cones_x = NULL;\n\n"

    # Add cone constraints for y
    code += "    // Cone constraints for y\n"
    for i, (cone_type, indices) in enumerate(cones_y):
        code += f"    unsigned int y_indices_{i}[] = {{"
        code += ", ".join(map(str, indices))
        code += "};\n"
        code += f"    struct ConeConstraintC cone_y_{i} = {{{cone_type}, y_indices_{i}, {len(indices)}}};\n"
    code += "    struct ConeConstraintC cones_y[] = {"
    code += ", ".join([f"cone_y_{i}" for i in range(len(cones_y))])
    code += "};\n\n"

    # Allocate solution arrays
    code += "    // Allocate solution arrays\n"
    code += f"    double x[{n}];\n"
    code += f"    double y[{m}];\n"
    code += f"    double l[{m}];\n"
    code += "    double optval;\n"
    code += "    unsigned int final_iter;\n\n"

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
        code += "        x, y, l, &optval, &final_iter\n"
        code += "    );\n\n"
    else:
        code += f"    int status = {solver_base}QD(\n"
        code += f"        ROW_MAJ, {m}, {n}, A, b, c, P,\n"
        code += f"        cones_x, {len(cones_x) if cones_x else 0},\n"
        code += f"        cones_y, {len(cones_y)},\n"
        code += f"        {rho}, {abs_tol}, {rel_tol}, {max_iter}, {verbose},\n"
        code += f"        {1 if adaptive_rho else 0}, 1,  // adaptive_rho, gap_stop\n"
        code += "        x, y, l, &optval, &final_iter\n"
        code += "    );\n\n"

    # Output results
    code += "    // Output results in machine-readable format\n"
    code += '    printf("STATUS=%d\\n", status);\n'
    code += '    printf("ITERS=%u\\n", final_iter);\n'
    code += '    printf("OPTVAL=%.16e\\n", optval);\n'
    code += "    \n"
    code += '    printf("X=");\n'
    code += f"    for (int i = 0; i < {n}; i++) {{\n"
    code += '        printf("%.16e", x[i]);\n'
    code += f'        if (i < {n - 1}) printf(",");\n'
    code += "    }\n"
    code += '    printf("\\n");\n'
    code += "    \n"
    code += '    printf("Y=");\n'
    code += f"    for (int i = 0; i < {m}; i++) {{\n"
    code += '        printf("%.16e", y[i]);\n'
    code += f'        if (i < {m - 1}) printf(",");\n'
    code += "    }\n"
    code += '    printf("\\n");\n'
    code += "    \n"
    code += '    printf("L=");\n'
    code += f"    for (int i = 0; i < {m}; i++) {{\n"
    code += '        printf("%.16e", l[i]);\n'
    code += f'        if (i < {m - 1}) printf(",");\n'
    code += "    }\n"
    code += '    printf("\\n");\n'
    code += "    \n"
    code += "    return status;\n"
    code += "}\n"

    return code


def _compile_and_run(c_code):
    """Compile and run the generated C code."""

    # Find POGS root directory
    pogs_root = os.path.join(os.path.dirname(__file__), "..")

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        c_file = f.name
        f.write(c_code)

    exe_file = tempfile.mktemp()

    try:
        # Compile
        compile_cmd = [
            "gcc",
            "-O3",
            f"-I{pogs_root}/src/include",
            f"-I{pogs_root}/src/interface_c",
            f"-I{pogs_root}/src/cpu/include",
            "-std=c11",
            "-o",
            exe_file,
            c_file,
            f"{pogs_root}/src/build/pogs.a",
            "-lm",
            "-framework",
            "Accelerate",
            "-lstdc++",
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PogsError(f"Compilation failed: {result.stderr}")

        # Run
        result = subprocess.run([exe_file], capture_output=True, text=True, timeout=60)

        # Parse output
        output_lines = result.stdout.strip().split("\n")
        parsed = {}
        for line in output_lines:
            if "=" in line:
                key, value = line.split("=", 1)
                if key == "STATUS":
                    parsed["status"] = int(value)
                elif key == "ITERS":
                    parsed["num_iters"] = int(value)
                elif key == "OPTVAL":
                    parsed["optval"] = float(value)
                elif key == "X":
                    parsed["x"] = np.array([float(v) for v in value.split(",")])
                elif key == "Y":
                    parsed["y"] = np.array([float(v) for v in value.split(",")])
                elif key == "L":
                    parsed["z"] = np.array([float(v) for v in value.split(",")])  # dual variable

        return parsed

    finally:
        # Cleanup
        if os.path.exists(c_file):
            os.remove(c_file)
        if os.path.exists(exe_file):
            os.remove(exe_file)


# =============================================================================
# Graph-form pattern detection and direct solver for CVXPY problems
# =============================================================================


def pogs_solve(problem, verbose=False, **solver_opts):
    """
    Solve a CVXPY problem with POGS, using graph-form solver when possible.

    This function should be used instead of `problem.solve(solver='POGS')`
    to get the best performance. It:
    1. Detects if the problem has graph-form structure (Lasso, Ridge, etc.)
    2. If yes, uses the fast graph-form solver directly
    3. If no, falls back to the cone solver via CVXPY

    Parameters
    ----------
    problem : cvxpy.Problem
        The CVXPY problem to solve
    verbose : bool
        Print solver output
    **solver_opts : dict
        Additional solver options (abs_tol, rel_tol, max_iter, rho, etc.)

    Returns
    -------
    float
        Optimal value of the problem

    Example
    -------
    >>> import cvxpy as cp
    >>> from pogs_cvxpy import pogs_solve
    >>> x = cp.Variable(100)
    >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x)))
    >>> optval = pogs_solve(problem, verbose=True)
    """
    if not _GRAPH_AVAILABLE:
        # Graph-form solver not available, use cone solver
        return problem.solve(solver="POGS", verbose=verbose, **solver_opts)

    # Try to detect graph-form pattern
    detection = _detect_graph_form(problem)

    if detection is not None:
        if verbose:
            print(f"POGS: Detected {detection['type']} pattern, using fast graph-form solver")

        # Solve with graph-form solver
        result = _solve_graph_form_detected(detection, solver_opts)

        if result is not None and result.get("status", 3) in (0, 3):
            # Set the variable value in the CVXPY problem
            variables = problem.variables()
            if len(variables) == 1:
                variables[0].value = result["x"]

            # Apply optimal value scale to convert from POGS to CVXPY objective
            # POGS solves 0.5*||..||^2 + ..., CVXPY may have different scaling
            optval_scale = detection["params"].get("optval_scale", 1.0)
            cvxpy_optval = result["optval"] * optval_scale

            # Set problem status
            if result.get("status", 3) == 0:
                problem._status = "optimal"
            else:
                problem._status = "optimal_inaccurate"

            problem._value = cvxpy_optval
            return cvxpy_optval
        else:
            if verbose:
                print("POGS: Graph-form solver failed, falling back to cone solver")

    else:
        if verbose:
            print("POGS: No graph-form pattern detected, using cone solver")

    # Fall back to cone solver
    return problem.solve(solver="POGS", verbose=verbose, **solver_opts)


def _detect_graph_form(problem):
    """
    Detect if a CVXPY problem has graph-form structure that POGS can solve fast.

    Returns a dict with:
      - 'type': problem type ('lasso', 'ridge', 'elastic_net', 'logistic', etc.)
      - 'params': parameters for the solver (A, b, lambda, etc.)
    Or None if graph-form not detected.
    """
    try:
        import cvxpy as cp  # noqa: F401 (used in isinstance checks below)
    except ImportError:
        return None

    if problem.objective.NAME != "minimize":
        return None

    obj_expr = problem.objective.expr

    # Get the single variable (graph-form assumes one variable for x)
    variables = problem.variables()
    if len(variables) != 1:
        return None
    x = variables[0]

    # Check constraints for non-negativity or bounds
    constraints_type = _detect_constraints(problem.constraints, x)

    # Try to detect common patterns
    detection_funcs = [
        _detect_lasso,
        _detect_ridge,
        _detect_elastic_net,
        _detect_logistic,
        _detect_huber,
        _detect_svm,
        _detect_nonneg_ls,
        _detect_sum_squares_only,
    ]

    for detect_fn in detection_funcs:
        result = detect_fn(obj_expr, x, constraints_type)
        if result is not None:
            return result

    return None


def _detect_constraints(constraints, x):
    """Detect constraint type on variable x."""
    if not constraints:
        return "free"

    try:
        import cvxpy as cp
    except ImportError:
        return "other"

    for constr in constraints:
        # Check for x >= 0
        if isinstance(constr, cp.constraints.nonpos.NonNeg):
            # NonNeg constraint: expr >= 0
            # Check if it's our variable
            if constr.args[0] is x or (hasattr(constr.args[0], "value") and constr.args[0] is x):
                return "nonneg"

    return "other" if constraints else "free"


def _extract_affine_transform(expr, x):
    """
    Try to extract A, b from expr where expr = A @ x - b (or A @ x + c).

    Returns (A, b, scale) or None if not affine in x.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    # Handle A @ x - b  or  A @ x + b
    if isinstance(expr, cp.atoms.affine.add_expr.AddExpression):
        args = expr.args
        # Find the linear part (A @ x) and constant part (-b)
        linear_part = None
        const_part = 0

        for arg in args:
            if arg.is_constant():
                const_part = arg.value if hasattr(arg, "value") else np.array(arg)
            else:
                if linear_part is not None:
                    return None  # Multiple non-constant terms
                linear_part = arg

        if linear_part is None:
            return None

        # Extract A from linear part
        A_result = _extract_linear_operator(linear_part, x)
        if A_result is None:
            return None
        A, _ = A_result

        # The offset b: if expr = A@x - b, then const_part = -b, so b = -const_part
        b = -np.asarray(const_part).flatten() if np.size(const_part) > 0 else np.zeros(A.shape[0])
        return A, b, 1.0

    # Handle just A @ x (no offset)
    A_result = _extract_linear_operator(expr, x)
    if A_result is not None:
        A, _ = A_result
        return A, np.zeros(A.shape[0]), 1.0

    return None


def _extract_linear_operator(expr, x):
    """Extract A from expr = A @ x. Returns (A, scale) or None."""
    try:
        import cvxpy as cp
        import scipy.sparse as sp
    except ImportError:
        return None

    # Direct A @ x
    if isinstance(expr, cp.atoms.affine.binary_operators.MulExpression):
        # Matrix multiplication
        if len(expr.args) == 2:
            A_expr, x_expr = expr.args
            if x_expr is x and A_expr.is_constant():
                A = A_expr.value
                if sp.issparse(A):
                    A = A.toarray()
                return np.asarray(A), 1.0

    # cp.matmul or @ operator
    if hasattr(expr, "args") and len(expr.args) == 2:
        A_expr, x_expr = expr.args
        if x_expr is x and hasattr(A_expr, "value") and A_expr.is_constant():
            A = A_expr.value
            if hasattr(sp, "issparse") and sp.issparse(A):
                A = A.toarray()
            return np.asarray(A), 1.0

    # Just x itself (identity transform)
    if expr is x:
        n = x.size
        return np.eye(n), 1.0

    return None


def _is_sum_squares(expr):
    """Check if expression is sum_squares or equivalent (quad_over_lin)."""
    type_name = type(expr).__name__
    # CVXPY may use sum_squares directly or quad_over_lin internally
    return type_name in ("sum_squares", "quad_over_lin")


def _is_norm1(expr):
    """Check if expression is norm1."""
    type_name = type(expr).__name__
    if type_name == "norm1":
        return True
    # Check for Pnorm with p=1
    if type_name == "Pnorm" and hasattr(expr, "p") and expr.p == 1:
        return True
    return False


def _unwrap_scaled(arg):
    """
    Unwrap a possibly scaled expression.
    Returns (inner_expr, scale).
    """
    type_name = type(arg).__name__

    # Handle multiply (element-wise): constant * expr
    if type_name == "multiply":
        if len(arg.args) == 2:
            if arg.args[0].is_constant():
                return arg.args[1], float(arg.args[0].value)
            if arg.args[1].is_constant():
                return arg.args[0], float(arg.args[1].value)

    # Handle MulExpression: constant * expr (matrix mul)
    if type_name == "MulExpression":
        if len(arg.args) == 2 and arg.args[0].is_constant():
            return arg.args[1], float(arg.args[0].value)

    return arg, 1.0


def _detect_lasso(obj_expr, x, constraints_type):
    """
    Detect Lasso: minimize 0.5||Ax - b||² + λ||x||₁

    CVXPY forms:
    - cp.sum_squares(A @ x - b) + lambda * cp.norm1(x)
    - 0.5 * cp.sum_squares(A @ x - b) + lambda * cp.norm(x, 1)
    - quad_over_lin(A @ x - b, 1) + lambda * norm1(x)  (internal form)
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    if not isinstance(obj_expr, cp.atoms.affine.add_expr.AddExpression):
        return None

    # Look for sum_squares + norm1 pattern
    sum_sq_term = None
    norm1_term = None
    sum_sq_scale = 1.0
    norm1_scale = 1.0

    for arg in obj_expr.args:
        inner, scale = _unwrap_scaled(arg)

        if _is_sum_squares(inner):
            if sum_sq_term is not None:
                return None  # Multiple sum_squares terms
            sum_sq_term = inner
            sum_sq_scale = scale

        elif _is_norm1(inner):
            if norm1_term is not None:
                return None
            norm1_term = inner
            norm1_scale = scale

    if sum_sq_term is None or norm1_term is None:
        return None

    # Extract A, b from sum_squares(A @ x - b) or quad_over_lin(A @ x - b, 1)
    # For quad_over_lin, the first arg is the vector expression
    sq_inner = sum_sq_term.args[0]
    affine = _extract_affine_transform(sq_inner, x)
    if affine is None:
        return None
    A, b, _ = affine

    # Verify norm1 is on x
    if norm1_term.args[0] is not x:
        return None

    # Lambda is norm1_scale, accounting for sum_squares scale (typically 1 or 0.5)
    # sum_squares/quad_over_lin gives ||.||², so factor is sum_sq_scale
    # Note: quad_over_lin(v, 1) = ||v||², so no extra factor
    lambd = norm1_scale / (2 * sum_sq_scale) if sum_sq_scale != 0 else norm1_scale

    return {
        "type": "lasso",
        "params": {
            "A": np.asarray(A, dtype=np.float64),
            "b": np.asarray(b, dtype=np.float64).flatten(),
            "lambd": float(lambd),
            # Store scale factor to convert POGS optval back to CVXPY optval
            # POGS minimizes 0.5*||Ax-b||^2 + lambda*||x||_1
            # CVXPY minimizes sum_sq_scale*||Ax-b||^2 + norm1_scale*||x||_1
            # So CVXPY_optval = 2 * sum_sq_scale * POGS_optval
            "optval_scale": 2.0 * sum_sq_scale,
        },
    }


def _detect_ridge(obj_expr, x, constraints_type):
    """
    Detect Ridge: minimize 0.5||Ax - b||² + λ||x||²

    CVXPY forms:
    - cp.sum_squares(A @ x - b) + lambda * cp.sum_squares(x)
    - quad_over_lin(A @ x - b, 1) + lambda * quad_over_lin(x, 1)
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    if not isinstance(obj_expr, cp.atoms.affine.add_expr.AddExpression):
        return None

    sum_sq_data = None
    sum_sq_reg = None
    data_scale = 1.0
    reg_scale = 1.0

    for arg in obj_expr.args:
        inner, scale = _unwrap_scaled(arg)

        if _is_sum_squares(inner):
            # Check if it's data term (A @ x - b) or regularizer (x)
            if inner.args[0] is x:
                sum_sq_reg = inner
                reg_scale = scale
            else:
                affine = _extract_affine_transform(inner.args[0], x)
                if affine is not None:
                    sum_sq_data = inner
                    data_scale = scale

    if sum_sq_data is None or sum_sq_reg is None:
        return None

    affine = _extract_affine_transform(sum_sq_data.args[0], x)
    if affine is None:
        return None
    A, b, _ = affine

    # For Ridge, POGS uses kSquare (0.5*x^2) for both data and regularizer
    # CVXPY: data_scale * ||Ax-b||^2 + reg_scale * ||x||^2
    # POGS: 0.5 * ||Ax-b||^2 + lambd * 0.5 * ||x||^2
    # For same optimum: (reg_scale / data_scale) = lambd
    # But since kSquare is 0.5*x^2, the effective reg is lambd * 0.5
    # So we need lambd such that lambd * 0.5 / 0.5 = reg_scale / data_scale
    # => lambd = reg_scale / data_scale
    lambd = reg_scale / data_scale if data_scale != 0 else reg_scale

    return {
        "type": "ridge",
        "params": {
            "A": np.asarray(A, dtype=np.float64),
            "b": np.asarray(b, dtype=np.float64).flatten(),
            "lambd": float(lambd),
            "optval_scale": 2.0 * data_scale,
        },
    }


def _detect_elastic_net(obj_expr, x, constraints_type):
    """
    Detect Elastic Net: minimize 0.5||Ax - b||² + λ₁||x||₁ + λ₂||x||²
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    if not isinstance(obj_expr, cp.atoms.affine.add_expr.AddExpression):
        return None

    sum_sq_data = None
    norm1_term = None
    sum_sq_reg = None
    data_scale = 1.0
    l1_scale = 1.0
    l2_scale = 1.0

    for arg in obj_expr.args:
        inner = arg
        scale = 1.0
        if isinstance(arg, cp.atoms.affine.binary_operators.MulExpression):
            if len(arg.args) == 2 and arg.args[0].is_constant():
                scale = float(arg.args[0].value)
                inner = arg.args[1]

        if isinstance(inner, cp.atoms.quad_form.sum_squares):
            if inner.args[0] is x:
                sum_sq_reg = inner
                l2_scale = scale
            else:
                affine = _extract_affine_transform(inner.args[0], x)
                if affine is not None:
                    sum_sq_data = inner
                    data_scale = scale

        elif isinstance(inner, (cp.atoms.norm1.norm1,)) or (
            isinstance(inner, cp.atoms.norm.Pnorm) and inner.p == 1
        ):
            if inner.args[0] is x:
                norm1_term = inner
                l1_scale = scale

    if sum_sq_data is None or (norm1_term is None and sum_sq_reg is None):
        return None

    # Need both L1 and L2 for elastic net
    if norm1_term is None or sum_sq_reg is None:
        return None

    affine = _extract_affine_transform(sum_sq_data.args[0], x)
    if affine is None:
        return None
    A, b, _ = affine

    lambda1 = l1_scale / (2 * data_scale) if data_scale != 0 else l1_scale
    lambda2 = l2_scale / (2 * data_scale) if data_scale != 0 else l2_scale

    return {
        "type": "elastic_net",
        "params": {
            "A": np.asarray(A, dtype=np.float64),
            "b": np.asarray(b, dtype=np.float64).flatten(),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
        },
    }


def _detect_logistic(obj_expr, x, constraints_type):
    """
    Detect Logistic Regression: minimize Σ log(1 + exp(-yᵢ(aᵢ'x))) + λ||x||₁

    This is harder to detect from CVXPY expressions.
    """
    # TODO: Implement logistic detection
    return None


def _detect_huber(obj_expr, x, constraints_type):
    """
    Detect Huber regression: minimize Σ huber(Ax - b)
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    # Look for sum(huber(...))
    if isinstance(obj_expr, cp.atoms.affine.sum.Sum):
        inner = obj_expr.args[0]
        if isinstance(inner, cp.atoms.elementwise.huber.huber):
            affine = _extract_affine_transform(inner.args[0], x)
            if affine is not None:
                A, b, _ = affine
                delta = inner.M if hasattr(inner, "M") else 1.0
                return {
                    "type": "huber",
                    "params": {
                        "A": np.asarray(A, dtype=np.float64),
                        "b": np.asarray(b, dtype=np.float64).flatten(),
                        "delta": float(delta),
                        "lambd": 0.0,
                    },
                }

    return None


def _detect_svm(obj_expr, x, constraints_type):
    """Detect SVM: minimize Σ max(0, 1 - yᵢ(aᵢ'x)) + λ||x||²"""
    # TODO: Implement SVM detection
    return None


def _detect_nonneg_ls(obj_expr, x, constraints_type):
    """
    Detect non-negative least squares: minimize 0.5||Ax - b||² s.t. x >= 0
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type != "nonneg":
        return None

    # Should be just sum_squares (no regularizer)
    if isinstance(obj_expr, cp.atoms.quad_form.sum_squares):
        affine = _extract_affine_transform(obj_expr.args[0], x)
        if affine is not None:
            A, b, _ = affine
            return {
                "type": "nonneg_ls",
                "params": {
                    "A": np.asarray(A, dtype=np.float64),
                    "b": np.asarray(b, dtype=np.float64).flatten(),
                },
            }

    # Also handle scaled: 0.5 * sum_squares(...)
    if isinstance(obj_expr, cp.atoms.affine.binary_operators.MulExpression):
        if len(obj_expr.args) == 2 and obj_expr.args[0].is_constant():
            inner = obj_expr.args[1]
            if isinstance(inner, cp.atoms.quad_form.sum_squares):
                affine = _extract_affine_transform(inner.args[0], x)
                if affine is not None:
                    A, b, _ = affine
                    return {
                        "type": "nonneg_ls",
                        "params": {
                            "A": np.asarray(A, dtype=np.float64),
                            "b": np.asarray(b, dtype=np.float64).flatten(),
                        },
                    }

    return None


def _detect_sum_squares_only(obj_expr, x, constraints_type):
    """
    Detect simple least squares: minimize ||Ax - b||² (no regularization).

    We can solve this with graph-form using kSquare for f and kZero for g.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    # Handle sum_squares directly or scaled
    inner = obj_expr
    if isinstance(obj_expr, cp.atoms.affine.binary_operators.MulExpression):
        if len(obj_expr.args) == 2 and obj_expr.args[0].is_constant():
            inner = obj_expr.args[1]

    if isinstance(inner, cp.atoms.quad_form.sum_squares):
        affine = _extract_affine_transform(inner.args[0], x)
        if affine is not None:
            A, b, _ = affine
            return {
                "type": "least_squares",
                "params": {
                    "A": np.asarray(A, dtype=np.float64),
                    "b": np.asarray(b, dtype=np.float64).flatten(),
                },
            }

    return None


def _solve_graph_form_detected(detection_result, solver_opts):
    """Solve a detected graph-form problem using the fast solver."""
    if not _GRAPH_AVAILABLE:
        return None

    ptype = detection_result["type"]
    params = detection_result["params"]

    opts = solver_opts or {}
    abs_tol = opts.get("abs_tol", 1e-4)
    rel_tol = opts.get("rel_tol", 1e-4)
    max_iter = opts.get("max_iter", 2500)
    verbose = opts.get("verbose", 0)
    rho = opts.get("rho", 1.0)

    t0 = time.perf_counter()

    if ptype == "lasso":
        result = solve_lasso(
            params["A"],
            params["b"],
            params["lambd"],
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            rho=rho,
        )
    elif ptype == "ridge":
        result = solve_ridge(
            params["A"],
            params["b"],
            params["lambd"],
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            rho=rho,
        )
    elif ptype == "elastic_net":
        result = solve_elastic_net(
            params["A"],
            params["b"],
            params["lambda1"],
            params["lambda2"],
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            rho=rho,
        )
    elif ptype == "huber":
        result = solve_huber(
            params["A"],
            params["b"],
            delta=params.get("delta", 1.0),
            lambd=params.get("lambd", 0.0),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            rho=rho,
        )
    elif ptype == "nonneg_ls":
        result = solve_nonneg_ls(
            params["A"],
            params["b"],
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iter=max_iter,
            verbose=verbose,
            rho=rho,
        )
    elif ptype == "least_squares":
        # Plain least squares with graph-form (kSquare for f, kZero for g)
        A = params["A"]
        b = params["b"]
        m, n = A.shape
        f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]
        g = [FunctionObj(Function.kZero) for _ in range(n)]
        result = _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)
    else:
        return None

    result["solve_time"] = time.perf_counter() - t0
    return result


# =============================================================================
# CVXPY integration
# =============================================================================
try:
    import cvxpy
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

    class POGS(ConicSolver):
        """CVXPY interface for POGS solver."""

        MIP_CAPABLE = False
        SUPPORTED_CONSTRAINTS = [cvxpy.Zero, cvxpy.NonNeg, cvxpy.SOC, cvxpy.PSD, cvxpy.ExpCone]

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
            # POGS is available if we can import the graph solver
            if not _GRAPH_AVAILABLE:
                raise ImportError("POGS library not found. Please build it first.")

        def apply(self, problem):
            """
            Return the problem data in POGS cone format.

            Note: For graph-form patterns (Lasso, Ridge, etc.), use pogs_solve()
            instead of problem.solve(solver='POGS') for much better performance.
            """
            return super().apply(problem)

        def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
            """
            Solve a problem represented in CVXPY's conic format.

            For LP/QP problems (only Zero and NonNeg cones), uses the efficient
            graph-form solver. For problems with SOC, SDP, or ExpCone, falls back
            to the cone solver.

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
            opts = solver_opts.copy() if solver_opts else {}
            verbose_level = opts.get("verbose", 5 if verbose else 0)
            opts["verbose"] = verbose_level

            # Extract problem data
            c = data["c"]
            A = data["A"]
            b = data["b"]
            import cvxpy.settings as s

            P = data.get(s.P, None)

            # Convert ConeDims object to dict format expected by solve_cone_problem
            # CVXPY ConeDims has: zero, nonneg, exp, soc, psd, p3d
            # POGS expects: f, l, q, s, ep, ed
            cvxpy_dims = data["dims"]
            dims = {
                "f": getattr(cvxpy_dims, "zero", 0),  # zero cone
                "l": getattr(cvxpy_dims, "nonneg", 0),  # nonneg cone
                "q": list(getattr(cvxpy_dims, "soc", [])),  # SOC cones
                "s": list(getattr(cvxpy_dims, "psd", [])),  # SDP cones
                "ep": getattr(cvxpy_dims, "exp", 0),  # exponential cones
                "ed": 0,  # dual exponential cones
            }

            # Get solver options
            abs_tol = opts.get("abs_tol", 1e-4)
            rel_tol = opts.get("rel_tol", 1e-3)
            max_iter = opts.get("max_iter", 50000)  # Higher default for convergence
            rho = opts.get("rho", None)  # Use automatic rho selection by default
            adaptive_rho = opts.get("adaptive_rho", True)
            rho_mode = opts.get("rho_mode", None)
            rho_scale = opts.get("rho_scale", 1.0)
            use_direct = opts.get("use_direct", None)
            prefer_ctypes = opts.get("prefer_ctypes", True)
            use_graph_form = opts.get("use_graph_form", True)  # Enable by default

            # Check if we can use the graph-form QP/LP solver
            # This is possible when there are only Zero (equality) and NonNeg (inequality) cones
            has_complex_cones = (
                len(dims["q"]) > 0 or  # SOC cones
                len(dims["s"]) > 0 or  # SDP cones
                dims["ep"] > 0         # Exponential cones
            )

            if _GRAPH_AVAILABLE and use_graph_form and not has_complex_cones:
                result = self._solve_via_graph_form(
                    c, A, b, dims, P,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    max_iter=max_iter,
                    verbose=verbose_level,
                    rho=rho if rho is not None else 1.0,
                    adaptive_rho=adaptive_rho,
                    use_dense=use_direct,
                )
                if result is not None and result.get("status", 1) == 0:
                    return result
                elif verbose_level > 0:
                    print("POGS: Graph-form solver failed, falling back to cone solver")

            # Fall back to cone solver
            result = solve_cone_problem(
                c,
                A,
                b,
                dims,
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

        def _solve_via_graph_form(self, c, A, b, dims, P,
                                  abs_tol, rel_tol, max_iter, verbose, rho,
                                  adaptive_rho, use_dense):
            """
            Solve LP/QP using the graph-form solver.

            CVXPY cone format: b - A*x ∈ K
            - Zero cone (dims['f'] rows): A*x == b (equality)
            - NonNeg cone (dims['l'] rows): A*x <= b (inequality, since b - Ax >= 0)
            """
            import scipy.sparse as sp

            try:
                c = np.asarray(c, dtype=np.float64).flatten()
                b = np.asarray(b, dtype=np.float64).flatten()
                if sp.issparse(A):
                    A = A.toarray()
                A = np.asarray(A, dtype=np.float64)

                n = len(c)
                m_eq = dims["f"]  # Zero cone = equality constraints
                m_ineq = dims["l"]  # NonNeg cone = inequality constraints

                # Split A and b into equality and inequality parts
                # CVXPY orders: zero cone first, then nonneg
                A_eq = A[:m_eq, :] if m_eq > 0 else None
                b_eq = b[:m_eq] if m_eq > 0 else None
                A_ineq = A[m_eq:m_eq + m_ineq, :] if m_ineq > 0 else None
                b_ineq = b[m_eq:m_eq + m_ineq] if m_ineq > 0 else None

                # Process quadratic term
                if P is not None:
                    if sp.issparse(P):
                        P = P.toarray()
                    P = np.asarray(P, dtype=np.float64)

                if verbose > 0:
                    prob_type = "QP" if P is not None else "LP"
                    print(f"POGS: Using graph-form solver for {prob_type} "
                          f"(n={n}, m_eq={m_eq}, m_ineq={m_ineq})")

                t0 = time.time()
                result = solve_qp(
                    c=c,
                    P=P,
                    A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    lb=None,  # No variable bounds in cone form
                    ub=None,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    max_iter=max_iter,
                    verbose=verbose,
                    rho=rho,
                    adaptive_rho=adaptive_rho,
                    use_dense=use_dense,
                )
                result["solve_time"] = time.time() - t0
                result["num_iters"] = result.get("iterations", 0)

                return result

            except Exception as e:
                if verbose > 0:
                    print(f"POGS: Graph-form solver error: {e}")
                return None

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

            import cvxpy.settings as s
            from cvxpy.reductions.solution import Solution, failure_solution

            attr = {}

            # POGS status codes:
            # 0 = optimal, 1 = infeasible, 2 = unbounded, 3 = iteration limit
            pogs_status = solution.get("status", 3)
            primal_res = solution.get("primal_res")
            eps_pri = solution.get("eps_pri")
            res_ratio = None
            if primal_res is not None and eps_pri is not None and eps_pri > 0:
                res_ratio = primal_res / eps_pri
                attr["pogs_primal_res"] = primal_res
                attr["pogs_primal_tol"] = eps_pri
                attr["pogs_primal_ratio"] = res_ratio

            if pogs_status == 0:
                if res_ratio is None or res_ratio <= 1.0:
                    status = s.OPTIMAL
                else:
                    status = s.OPTIMAL_INACCURATE
            elif pogs_status == 1:
                status = s.INFEASIBLE
            elif pogs_status == 2:
                status = s.UNBOUNDED
            elif pogs_status == 3:
                # Iteration limit reached - only accept if close to tolerance.
                if res_ratio is not None and res_ratio <= 10.0:
                    status = s.OPTIMAL_INACCURATE
                else:
                    status = s.SOLVER_ERROR
            else:
                status = s.SOLVER_ERROR

            attr[s.SOLVE_TIME] = solution.get("solve_time", 0)
            attr[s.SETUP_TIME] = solution.get("setup_time", 0)
            attr[s.NUM_ITERS] = solution.get("num_iters", 0)

            if status in [s.OPTIMAL, s.OPTIMAL_INACCURATE]:
                # Extract optimal value with offset
                opt_val = solution.get("optval", 0)
                if s.OFFSET in inverse_data:
                    opt_val += inverse_data[s.OFFSET]

                # Cone-form solution
                primal_vars = {inverse_data[self.VAR_ID]: solution["x"]}

                # Return None for dual variables to skip dual extraction
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
            slv for slv in cvxpy_defines.INSTALLED_SOLVERS if slv in cvxpy_defines.CONIC_SOLVERS
        ]
    except Exception:
        pass

except ImportError:
    # CVXPY not installed, skip integration
    POGS = None
    pass
