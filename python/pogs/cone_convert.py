"""
Conversion utilities for CVXPY cone problems to POGS cone format.

CVXPY cone format:
    minimize    c^T x  (or 0.5 x^T P x + c^T x for QP)
    subject to  b - A*x ∈ K

where K is a product of cones specified by dims:
    - zero: equality constraints (A*x = b)
    - nonneg: inequality constraints (A*x <= b)
    - soc: second-order cone constraints
    - psd: positive semidefinite cone constraints
    - exp: exponential cone constraints

POGS cone format:
    minimize    c^T x
    subject to  b - A*x ∈ K_y,  x ∈ K_x

where K_x and K_y are specified as lists of (Cone, indices) tuples.
"""

import numpy as np

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Cone types matching pogs_cone.py
class Cone:
    ZERO = 0        # { x : x = 0 }
    NON_NEG = 1     # { x : x >= 0 }
    NON_POS = 2     # { x : x <= 0 }
    SOC = 3         # { (p, x) : ||x||_2 <= p }
    SDP = 4         # { X : X >= 0 }
    EXP_PRIMAL = 5  # { (x, y, z) : y > 0, y e^(x/y) <= z }
    EXP_DUAL = 6    # { (u, v, w) : u < 0, -u e^(v/u) <= ew }


def cvxpy_dims_to_pogs_cones(dims, m):
    """
    Convert CVXPY cone dimensions to POGS cone constraint format.

    Parameters
    ----------
    dims : dict or object
        CVXPY cone dimensions. Can be:
        - dict with keys: 'f'/'zero', 'l'/'nonneg', 'q'/'soc', 's'/'psd', 'ep'/'exp'
        - CVXPY ConeDims object with attributes: zero, nonneg, soc, psd, exp
    m : int
        Total number of constraint rows (should match sum of all cone dimensions)

    Returns
    -------
    cones_y : list of (Cone, list of int)
        List of cone constraints for y = b - A*x.
        Each tuple is (cone_type, list_of_row_indices).

    Notes
    -----
    CVXPY orders rows as: zero, nonneg, soc[0], soc[1], ..., psd[0], psd[1], ..., exp, exp, ...

    Example
    -------
    >>> dims = {'f': 2, 'l': 3, 'q': [4, 3], 's': [], 'ep': 1}
    >>> cones_y = cvxpy_dims_to_pogs_cones(dims, m=15)
    >>> # Returns: [(ZERO, [0,1]), (NON_NEG, [2,3,4]), (SOC, [5,6,7,8]), (SOC, [9,10,11]), (EXP_PRIMAL, [12,13,14])]
    """
    # Extract dimensions from dict or object
    if isinstance(dims, dict):
        n_zero = dims.get('f', dims.get('zero', 0))
        n_nonneg = dims.get('l', dims.get('nonneg', 0))
        soc_sizes = list(dims.get('q', dims.get('soc', [])))
        psd_sizes = list(dims.get('s', dims.get('psd', [])))
        n_exp = dims.get('ep', dims.get('exp', 0))
    else:
        # CVXPY ConeDims object
        n_zero = getattr(dims, 'zero', 0)
        n_nonneg = getattr(dims, 'nonneg', 0)
        soc_sizes = list(getattr(dims, 'soc', []))
        psd_sizes = list(getattr(dims, 'psd', []))
        n_exp = getattr(dims, 'exp', 0)

    cones_y = []
    row = 0

    # Zero cone (equality constraints): b - Ax = 0, i.e., Ax = b
    if n_zero > 0:
        cones_y.append((Cone.ZERO, list(range(row, row + n_zero))))
        row += n_zero

    # NonNeg cone (inequality constraints): b - Ax >= 0, i.e., Ax <= b
    if n_nonneg > 0:
        cones_y.append((Cone.NON_NEG, list(range(row, row + n_nonneg))))
        row += n_nonneg

    # Second-order cones
    for soc_size in soc_sizes:
        cones_y.append((Cone.SOC, list(range(row, row + soc_size))))
        row += soc_size

    # Positive semidefinite cones
    # Note: CVXPY uses vectorized lower-triangular form with size n*(n+1)/2
    for psd_dim in psd_sizes:
        vec_size = psd_dim * (psd_dim + 1) // 2
        cones_y.append((Cone.SDP, list(range(row, row + vec_size))))
        row += vec_size

    # Exponential cones (each has 3 elements)
    for _ in range(n_exp):
        cones_y.append((Cone.EXP_PRIMAL, list(range(row, row + 3))))
        row += 3

    # Verify we covered all rows
    if row != m:
        raise ValueError(f"Cone dimensions sum to {row}, but m={m}")

    return cones_y


def convert_cvxpy_to_pogs(c, A, b, dims, P=None, use_dense=None):
    """
    Convert a CVXPY cone problem to POGS cone format.

    This is the main entry point for CVXPY integration. It handles:
    - Converting cone dimensions to POGS cone constraints
    - Deciding dense vs sparse based on problem structure
    - Converting sparse matrices to dense if needed

    Parameters
    ----------
    c : array_like, shape (n,)
        Linear objective coefficients
    A : array_like or sparse matrix, shape (m, n)
        Constraint matrix
    b : array_like, shape (m,)
        Constraint RHS
    dims : dict or ConeDims
        CVXPY cone dimensions
    P : array_like or sparse matrix, shape (n, n), optional
        Quadratic objective matrix (for QP)
    use_dense : bool or None
        Force dense (True) or sparse (False). If None, auto-select:
        - Dense if matrix density > 10% or n < 500
        - This heuristic favors dense for small/medium problems where
          dense linear algebra is faster

    Returns
    -------
    dict with keys:
        'c' : ndarray, shape (n,)
        'A' : ndarray, shape (m, n) - always dense for now
        'b' : ndarray, shape (m,)
        'P' : ndarray or None, shape (n, n)
        'cones_x' : list - empty (x is free)
        'cones_y' : list of (Cone, indices)
        'use_direct' : bool - whether to use direct (dense) solver
    """
    # Convert to numpy arrays
    c = np.asarray(c, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()

    n = len(c)
    m = len(b)

    # Handle sparse A
    is_sparse = HAS_SCIPY and sp.issparse(A)
    if is_sparse:
        nnz = A.nnz
        density = nnz / (m * n) if m * n > 0 else 1.0
    else:
        A = np.asarray(A, dtype=np.float64)
        density = np.count_nonzero(A) / A.size if A.size > 0 else 1.0

    # Decide dense vs sparse
    if use_dense is None:
        # Heuristic: use dense for small problems or dense matrices
        # Dense factorization is O(mn^2), but has better constants
        # Sparse is better for large sparse problems
        use_dense = density > 0.1 or n < 500

    # Convert to dense if needed (POGS cone solver currently requires dense)
    if is_sparse:
        A = A.toarray()
    A = np.asarray(A, dtype=np.float64, order='C')

    # Handle P
    if P is not None:
        if HAS_SCIPY and sp.issparse(P):
            P = P.toarray()
        P = np.asarray(P, dtype=np.float64, order='C')

    # Convert cone dimensions
    cones_y = cvxpy_dims_to_pogs_cones(dims, m)

    # x is free (no constraints on x in standard CVXPY form)
    cones_x = []

    return {
        'c': c,
        'A': A,
        'b': b,
        'P': P,
        'cones_x': cones_x,
        'cones_y': cones_y,
        'use_direct': use_dense,
        'density': density,
        'm': m,
        'n': n,
    }


def solve_cvxpy_cone_problem(c, A, b, dims, P=None,
                              abs_tol=1e-4, rel_tol=1e-3, max_iter=10000,
                              rho=1.0, adaptive_rho=True, verbose=0,
                              use_dense=None):
    """
    Solve a CVXPY cone problem using POGS cone solver.

    This is the high-level interface that:
    1. Converts CVXPY format to POGS format
    2. Calls the POGS cone solver
    3. Returns results in a standard format

    Parameters
    ----------
    c, A, b, dims, P : see convert_cvxpy_to_pogs
    abs_tol, rel_tol : float
        Convergence tolerances
    max_iter : int
        Maximum iterations
    rho : float
        Initial penalty parameter
    adaptive_rho : bool
        Use adaptive rho selection
    verbose : int
        Verbosity level (0 = silent, 5 = debug)
    use_dense : bool or None
        Force dense/sparse or auto-select

    Returns
    -------
    dict with keys:
        'x' : solution vector
        'y' : slack vector
        'optval' : optimal objective value
        'status' : 0 for success
        'iterations' : number of iterations
        'solve_time' : solve time in seconds
    """
    import time

    # Import cone solver
    try:
        import sys
        import os
        # Add parent directory to path for pogs_cone import
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from pogs_cone import solve_cone
    except ImportError as e:
        raise ImportError(f"Cannot import pogs_cone solver: {e}")

    # Convert problem
    problem = convert_cvxpy_to_pogs(c, A, b, dims, P, use_dense)

    if verbose > 0:
        prob_type = "QP" if P is not None else "LP/Cone"
        print(f"POGS: Solving {prob_type} problem (m={problem['m']}, n={problem['n']}, "
              f"density={problem['density']:.1%}, {'dense' if problem['use_direct'] else 'sparse'} solver)")

    # Solve
    t0 = time.time()
    result = solve_cone(
        A=problem['A'],
        b=problem['b'],
        c=problem['c'],
        cones_x=problem['cones_x'],
        cones_y=problem['cones_y'],
        P=problem['P'],
        rho=rho,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        max_iter=max_iter,
        verbose=verbose,
        adaptive_rho=adaptive_rho,
        gap_stop=True,
        use_direct=problem['use_direct'],
    )
    result['solve_time'] = time.time() - t0
    result['num_iters'] = result.get('iterations', 0)

    return result
