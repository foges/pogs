"""
CVXPY integration for POGS.

Provides pogs_solve() to solve CVXPY problems with automatic graph-form detection.

Usage:
    import cvxpy as cp
    from pogs import pogs_solve

    x = cp.Variable(100)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x)))
    pogs_solve(problem)  # Auto-detects Lasso, uses fast solver

Or register as a solve method:
    cp.Problem.register_solve("POGS", pogs_solve)
    problem.solve(method="POGS")
"""

from __future__ import annotations

import time

import numpy as np

from pogs.graph import (
    solve_lasso,
    solve_nonneg_ls,
    solve_ridge,
)


def pogs_solve(problem, verbose: bool = False, **solver_opts) -> float:
    """
    Solve a CVXPY problem with POGS, using graph-form solver when possible.

    This function detects if the problem has graph-form structure (Lasso, Ridge, etc.)
    and uses the fast direct solver. Otherwise falls back to CVXPY's default solving.

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
    >>> from pogs import pogs_solve
    >>> x = cp.Variable(100)
    >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + 0.1 * cp.norm1(x)))
    >>> optval = pogs_solve(problem, verbose=True)
    """
    # Try to detect graph-form pattern
    detection = _detect_graph_form(problem)

    if detection is not None:
        if verbose:
            print(f"POGS: Detected {detection['type']} pattern, using fast graph-form solver")

        # Solve with graph-form solver
        result = _solve_graph_form_detected(detection, solver_opts)

        if result is not None and result.get("status", 1) == 0:
            # Set the variable value in the CVXPY problem
            variables = problem.variables()
            if len(variables) == 1:
                variables[0].value = result["x"]

            # Apply optimal value scale
            optval_scale = detection["params"].get("optval_scale", 1.0)
            cvxpy_optval = result["optval"] * optval_scale

            problem._status = "optimal"
            problem._value = cvxpy_optval
            return cvxpy_optval
        elif verbose:
            print("POGS: Graph-form solver failed, falling back to default")

    else:
        if verbose:
            print("POGS: No graph-form pattern detected, using default solver")

    # Fall back to default CVXPY solving
    return problem.solve(verbose=verbose, **solver_opts)


def _detect_graph_form(problem):
    """Detect if a CVXPY problem has graph-form structure."""
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError:
        return None

    if problem.objective.NAME != "minimize":
        return None

    obj_expr = problem.objective.expr
    variables = problem.variables()

    if len(variables) != 1:
        return None

    x = variables[0]
    constraints_type = _detect_constraints(problem.constraints, x)

    # Try detection functions
    for detect_fn in [_detect_lasso, _detect_ridge, _detect_nonneg_ls]:
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

        for constr in constraints:
            if isinstance(constr, cp.constraints.nonpos.NonNeg):
                if constr.args[0] is x:
                    return "nonneg"
    except ImportError:
        pass

    return "other" if constraints else "free"


def _extract_affine_transform(expr, x):
    """Extract A, b from expr where expr = A @ x - b."""
    try:
        import cvxpy as cp
    except ImportError:
        return None

    # Handle A @ x - b
    if isinstance(expr, cp.atoms.affine.add_expr.AddExpression):
        args = expr.args
        linear_part = None
        const_part = 0

        for arg in args:
            if arg.is_constant():
                const_part = arg.value if hasattr(arg, "value") else np.array(arg)
            else:
                if linear_part is not None:
                    return None
                linear_part = arg

        if linear_part is None:
            return None

        A_result = _extract_linear_operator(linear_part, x)
        if A_result is None:
            return None
        A, _ = A_result
        b = -np.asarray(const_part).flatten() if np.size(const_part) > 0 else np.zeros(A.shape[0])
        return A, b, 1.0

    # Handle just A @ x
    A_result = _extract_linear_operator(expr, x)
    if A_result is not None:
        A, _ = A_result
        return A, np.zeros(A.shape[0]), 1.0

    return None


def _extract_linear_operator(expr, x):
    """Extract A from expr = A @ x."""
    try:
        import cvxpy as cp
        import scipy.sparse as sp
    except ImportError:
        return None

    if isinstance(expr, cp.atoms.affine.binary_operators.MulExpression):
        if len(expr.args) == 2:
            A_expr, x_expr = expr.args
            if x_expr is x and A_expr.is_constant():
                A = A_expr.value
                if sp.issparse(A):
                    A = A.toarray()
                return np.asarray(A), 1.0

    if hasattr(expr, "args") and len(expr.args) == 2:
        A_expr, x_expr = expr.args
        if x_expr is x and hasattr(A_expr, "value") and A_expr.is_constant():
            A = A_expr.value
            if hasattr(sp, "issparse") and sp.issparse(A):
                A = A.toarray()
            return np.asarray(A), 1.0

    if expr is x:
        return np.eye(x.size), 1.0

    return None


def _is_sum_squares(expr):
    """Check if expression is sum_squares."""
    type_name = type(expr).__name__
    return type_name in ("sum_squares", "quad_over_lin")


def _is_norm1(expr):
    """Check if expression is norm1."""
    type_name = type(expr).__name__
    if type_name == "norm1":
        return True
    if type_name == "Pnorm" and hasattr(expr, "p") and expr.p == 1:
        return True
    return False


def _unwrap_scaled(arg):
    """Unwrap a possibly scaled expression."""
    type_name = type(arg).__name__

    if type_name == "multiply":
        if len(arg.args) == 2:
            if arg.args[0].is_constant():
                return arg.args[1], float(arg.args[0].value)
            if arg.args[1].is_constant():
                return arg.args[0], float(arg.args[1].value)

    if type_name == "MulExpression":
        if len(arg.args) == 2 and arg.args[0].is_constant():
            return arg.args[1], float(arg.args[0].value)

    return arg, 1.0


def _detect_lasso(obj_expr, x, constraints_type):
    """Detect Lasso: minimize 0.5||Ax - b||² + λ||x||₁"""
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type not in ("free", None):
        return None

    if not isinstance(obj_expr, cp.atoms.affine.add_expr.AddExpression):
        return None

    sum_sq_term = None
    norm1_term = None
    sum_sq_scale = 1.0
    norm1_scale = 1.0

    for arg in obj_expr.args:
        inner, scale = _unwrap_scaled(arg)

        if _is_sum_squares(inner):
            if sum_sq_term is not None:
                return None
            sum_sq_term = inner
            sum_sq_scale = scale

        elif _is_norm1(inner):
            if norm1_term is not None:
                return None
            norm1_term = inner
            norm1_scale = scale

    if sum_sq_term is None or norm1_term is None:
        return None

    sq_inner = sum_sq_term.args[0]
    affine = _extract_affine_transform(sq_inner, x)
    if affine is None:
        return None
    A, b, _ = affine

    if norm1_term.args[0] is not x:
        return None

    lambd = norm1_scale / (2 * sum_sq_scale) if sum_sq_scale != 0 else norm1_scale

    return {
        "type": "lasso",
        "params": {
            "A": np.asarray(A, dtype=np.float64),
            "b": np.asarray(b, dtype=np.float64).flatten(),
            "lambd": float(lambd),
            "optval_scale": 2.0 * sum_sq_scale,
        },
    }


def _detect_ridge(obj_expr, x, constraints_type):
    """Detect Ridge: minimize 0.5||Ax - b||² + λ||x||²"""
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


def _detect_nonneg_ls(obj_expr, x, constraints_type):
    """Detect NNLS: minimize 0.5||Ax - b||² s.t. x >= 0"""
    try:
        import cvxpy as cp
    except ImportError:
        return None

    if constraints_type != "nonneg":
        return None

    inner = obj_expr
    if isinstance(obj_expr, cp.atoms.affine.binary_operators.MulExpression):
        if len(obj_expr.args) == 2 and obj_expr.args[0].is_constant():
            inner = obj_expr.args[1]

    if _is_sum_squares(inner):
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


def _solve_graph_form_detected(detection, solver_opts):
    """Solve a detected graph-form problem."""
    ptype = detection["type"]
    params = detection["params"]

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
    else:
        return None

    result["solve_time"] = time.perf_counter() - t0
    return result


