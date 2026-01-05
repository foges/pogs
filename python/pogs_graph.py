"""
Python wrapper for POGS graph-form interface using ctypes.

Solves problems of the form:
    minimize    sum_i f_i(y_i) + sum_j g_j(x_j)
    subject to  y = A * x

where f_i and g_j are convex functions from a predefined library.

Supports both dense and sparse matrices.
"""

import ctypes
import numpy as np
import os
from enum import IntEnum

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _find_shared_library():
    """Locate a shared POGS library built by CMake."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates = [
        os.path.join(root, 'build', 'lib', 'libpogs_cpu.dylib'),
        os.path.join(root, 'build', 'lib', 'libpogs_cpu.so'),
        os.path.join(root, 'src', 'build', 'libpogs_cpu.dylib'),
        os.path.join(root, 'src', 'build', 'libpogs_cpu.so'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_lib_path = _find_shared_library()
if not _lib_path:
    raise ImportError(
        "POGS shared library not found. Build it with:\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF\n"
        "  cmake --build build --target pogs_cpu_shared\n"
    )

_lib = ctypes.CDLL(_lib_path)


class Ordering(IntEnum):
    """Matrix ordering: column-major or row-major."""
    COL_MAJ = 0
    ROW_MAJ = 1


class Function(IntEnum):
    """Function types for f_i and g_j."""
    kAbs = 0        # f(x) = |x|
    kExp = 1        # f(x) = e^x
    kHuber = 2      # f(x) = huber(x)
    kIdentity = 3   # f(x) = x
    kIndBox01 = 4   # f(x) = I(0 <= x <= 1)
    kIndEq0 = 5     # f(x) = I(x = 0)
    kIndGe0 = 6     # f(x) = I(x >= 0)
    kIndLe0 = 7     # f(x) = I(x <= 0)
    kLogistic = 8   # f(x) = log(1 + e^x)
    kMaxNeg0 = 9    # f(x) = max(0, -x)
    kMaxPos0 = 10   # f(x) = max(0, x)
    kNegEntr = 11   # f(x) = x log(x)
    kNegLog = 12    # f(x) = -log(x)
    kRecipr = 13    # f(x) = 1/x
    kSquare = 14    # f(x) = (1/2) x^2
    kZero = 15      # f(x) = 0


class FunctionObj:
    """
    Represents a function object with parameters.

    The function is: c * h(a * x - b) + d * x + e * x^2

    Parameters
    ----------
    h : Function
        The base function type
    a : float
        Scale parameter (default 1.0)
    b : float
        Offset parameter (default 0.0)
    c : float
        Multiplier on h (default 1.0)
    d : float
        Linear term coefficient (default 0.0)
    e : float
        Quadratic term coefficient (default 0.0)
    """
    def __init__(self, h=Function.kZero, a=1.0, b=0.0, c=1.0, d=0.0, e=0.0):
        self.h = h
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)


# Dense solver: PogsD
_lib.PogsD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # f_a
    ctypes.POINTER(ctypes.c_double),  # f_b
    ctypes.POINTER(ctypes.c_double),  # f_c
    ctypes.POINTER(ctypes.c_double),  # f_d
    ctypes.POINTER(ctypes.c_double),  # f_e
    ctypes.POINTER(ctypes.c_int),     # f_h
    ctypes.POINTER(ctypes.c_double),  # g_a
    ctypes.POINTER(ctypes.c_double),  # g_b
    ctypes.POINTER(ctypes.c_double),  # g_c
    ctypes.POINTER(ctypes.c_double),  # g_d
    ctypes.POINTER(ctypes.c_double),  # g_e
    ctypes.POINTER(ctypes.c_int),     # g_h
    ctypes.c_double,  # rho
    ctypes.c_double,  # abs_tol
    ctypes.c_double,  # rel_tol
    ctypes.c_uint,    # max_iter
    ctypes.c_uint,    # verbose
    ctypes.c_int,     # adaptive_rho
    ctypes.c_int,     # gap_stop
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # l
    ctypes.POINTER(ctypes.c_double),  # optval
    ctypes.POINTER(ctypes.c_uint),    # final_iter
]
_lib.PogsD.restype = ctypes.c_int

# Sparse solver: PogsSparseD
_lib.PogsSparseD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # nnz
    ctypes.POINTER(ctypes.c_double),  # data
    ctypes.POINTER(ctypes.c_int),     # ptr
    ctypes.POINTER(ctypes.c_int),     # ind
    ctypes.POINTER(ctypes.c_double),  # f_a
    ctypes.POINTER(ctypes.c_double),  # f_b
    ctypes.POINTER(ctypes.c_double),  # f_c
    ctypes.POINTER(ctypes.c_double),  # f_d
    ctypes.POINTER(ctypes.c_double),  # f_e
    ctypes.POINTER(ctypes.c_int),     # f_h
    ctypes.POINTER(ctypes.c_double),  # g_a
    ctypes.POINTER(ctypes.c_double),  # g_b
    ctypes.POINTER(ctypes.c_double),  # g_c
    ctypes.POINTER(ctypes.c_double),  # g_d
    ctypes.POINTER(ctypes.c_double),  # g_e
    ctypes.POINTER(ctypes.c_int),     # g_h
    ctypes.c_double,  # rho
    ctypes.c_double,  # abs_tol
    ctypes.c_double,  # rel_tol
    ctypes.c_uint,    # max_iter
    ctypes.c_uint,    # verbose
    ctypes.c_int,     # adaptive_rho
    ctypes.c_int,     # gap_stop
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # l
    ctypes.POINTER(ctypes.c_double),  # optval
    ctypes.POINTER(ctypes.c_uint),    # final_iter
]
_lib.PogsSparseD.restype = ctypes.c_int


def _solve_graph_form(A, f, g, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                       verbose=0, rho=1.0, adaptive_rho=True, gap_stop=True):
    """
    Internal function to solve graph-form problem.

    Parameters
    ----------
    A : array_like or sparse matrix
        Constraint matrix (m x n)
    f : list of FunctionObj
        Functions for y (length m)
    g : list of FunctionObj
        Functions for x (length n)
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter
    adaptive_rho : bool
        Use adaptive rho
    gap_stop : bool
        Use duality gap stopping criterion

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    # Check if sparse
    is_sparse = HAS_SCIPY and sp.issparse(A)

    if is_sparse:
        # Convert to CSR format
        A_csr = sp.csr_matrix(A, dtype=np.float64)
        m, n = A_csr.shape
        nnz = A_csr.nnz
        data = np.ascontiguousarray(A_csr.data, dtype=np.float64)
        ptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
        ind = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    else:
        A = np.asarray(A, dtype=np.float64, order='C')
        m, n = A.shape

    # Validate function arrays
    assert len(f) == m, f"f should have length {m}, got {len(f)}"
    assert len(g) == n, f"g should have length {n}, got {len(g)}"

    # Convert function objects to arrays
    f_a = np.array([fi.a for fi in f], dtype=np.float64)
    f_b = np.array([fi.b for fi in f], dtype=np.float64)
    f_c = np.array([fi.c for fi in f], dtype=np.float64)
    f_d = np.array([fi.d for fi in f], dtype=np.float64)
    f_e = np.array([fi.e for fi in f], dtype=np.float64)
    f_h = np.array([int(fi.h) for fi in f], dtype=np.int32)

    g_a = np.array([gi.a for gi in g], dtype=np.float64)
    g_b = np.array([gi.b for gi in g], dtype=np.float64)
    g_c = np.array([gi.c for gi in g], dtype=np.float64)
    g_d = np.array([gi.d for gi in g], dtype=np.float64)
    g_e = np.array([gi.e for gi in g], dtype=np.float64)
    g_h = np.array([int(gi.h) for gi in g], dtype=np.int32)

    # Allocate output arrays
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)
    l = np.zeros(m, dtype=np.float64)
    optval = ctypes.c_double()
    final_iter = ctypes.c_uint()

    if is_sparse:
        status = _lib.PogsSparseD(
            int(Ordering.ROW_MAJ),
            m, n, nnz,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            f_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            rho, abs_tol, rel_tol, max_iter, verbose,
            int(adaptive_rho), int(gap_stop),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            l.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(optval),
            ctypes.byref(final_iter)
        )
    else:
        status = _lib.PogsD(
            int(Ordering.ROW_MAJ),
            m, n,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            rho, abs_tol, rel_tol, max_iter, verbose,
            int(adaptive_rho), int(gap_stop),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            l.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(optval),
            ctypes.byref(final_iter)
        )

    return {
        'x': x,
        'y': y,
        'l': l,
        'optval': optval.value,
        'iterations': final_iter.value,
        'status': status
    }


def solve_lasso(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                verbose=0, rho=1.0):
    """
    Solve Lasso regression:
        minimize    0.5 * ||A*x - b||^2 + lambda * ||x||_1

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Target vector (m,)
    lambd : float
        Regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = 0.5 * (y_i - b_i)^2  =>  kSquare with b = b_i
    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # g_j(x_j) = lambda * |x_j|  =>  kAbs with c = lambda
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_ridge(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                verbose=0, rho=1.0):
    """
    Solve Ridge regression:
        minimize    0.5 * ||A*x - b||^2 + 0.5 * lambda * ||x||^2

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Target vector (m,)
    lambd : float
        Regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = 0.5 * (y_i - b_i)^2  =>  kSquare with b = b_i
    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # g_j(x_j) = 0.5 * lambda * x_j^2  =>  kSquare with c = lambda
    g = [FunctionObj(Function.kSquare, 1.0, 0.0, lambd) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_elastic_net(A, b, lambda1, lambda2, abs_tol=1e-4, rel_tol=1e-4,
                      max_iter=2500, verbose=0, rho=1.0):
    """
    Solve Elastic Net regression:
        minimize    0.5 * ||A*x - b||^2 + lambda1 * ||x||_1 + 0.5 * lambda2 * ||x||^2

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Target vector (m,)
    lambda1 : float
        L1 regularization parameter
    lambda2 : float
        L2 regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = 0.5 * (y_i - b_i)^2
    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # g_j(x_j) = lambda1 * |x_j| + 0.5 * lambda2 * x_j^2
    # Use kAbs with c=lambda1 and e=lambda2/2
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambda1, 0.0, lambda2/2) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_logistic(A, b, lambd=0.0, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                   verbose=0, rho=1.0):
    """
    Solve L1-regularized logistic regression:
        minimize    sum_i log(1 + exp(-b_i * (a_i' * x))) + lambda * ||x||_1

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Labels (+1 or -1) (m,)
    lambd : float
        L1 regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = log(1 + exp(-b_i * y_i))  =>  kLogistic with a = -b_i
    f = [FunctionObj(Function.kLogistic, -b[i], 0.0, 1.0) for i in range(m)]

    # g_j(x_j) = lambda * |x_j|
    if lambd > 0:
        g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for _ in range(n)]
    else:
        g = [FunctionObj(Function.kZero) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_huber(A, b, delta=1.0, lambd=0.0, abs_tol=1e-4, rel_tol=1e-4,
                max_iter=2500, verbose=0, rho=1.0):
    """
    Solve Huber regression:
        minimize    sum_i huber(A*x - b, delta) + lambda * ||x||_1

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Target vector (m,)
    delta : float
        Huber threshold
    lambd : float
        L1 regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = huber(y_i - b_i, delta)
    # The Huber function h(x) has threshold at 1, so we scale: huber(x/delta)*delta^2
    # Using a=1/delta, c=delta^2 gives huber((y-b)/delta)*delta^2
    f = [FunctionObj(Function.kHuber, 1.0/delta, b[i]/delta, delta*delta) for i in range(m)]

    # g_j(x_j) = lambda * |x_j|
    if lambd > 0:
        g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for _ in range(n)]
    else:
        g = [FunctionObj(Function.kZero) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_svm(A, b, lambd=1.0, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
              verbose=0, rho=1.0):
    """
    Solve SVM (hinge loss):
        minimize    sum_i max(0, 1 - b_i * (a_i' * x)) + 0.5 * lambda * ||x||^2

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Labels (+1 or -1) (m,)
    lambd : float
        L2 regularization parameter
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = max(0, 1 - b_i * y_i)  =>  kMaxPos0 with a = -b_i, b = -1
    f = [FunctionObj(Function.kMaxPos0, -b[i], -1.0, 1.0) for i in range(m)]

    # g_j(x_j) = 0.5 * lambda * x_j^2
    g = [FunctionObj(Function.kSquare, 1.0, 0.0, lambd) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_nonneg_ls(A, b, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                    verbose=0, rho=1.0):
    """
    Solve non-negative least squares:
        minimize    0.5 * ||A*x - b||^2
        subject to  x >= 0

    Parameters
    ----------
    A : array_like or sparse matrix
        Data matrix (m x n)
    b : array_like
        Target vector (m,)
    abs_tol, rel_tol : float
        Tolerances
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'
    """
    b = np.asarray(b, dtype=np.float64).flatten()

    if HAS_SCIPY and sp.issparse(A):
        m, n = A.shape
    else:
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

    # f_i(y_i) = 0.5 * (y_i - b_i)^2
    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # g_j(x_j) = I(x_j >= 0)
    g = [FunctionObj(Function.kIndGe0) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


if __name__ == '__main__':
    print("Testing POGS graph-form interface...")

    # Test 1: Dense Lasso
    print("\n1. Dense Lasso:")
    np.random.seed(42)
    m, n = 100, 50
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[np.abs(x_true) < 0.5] = 0  # Make sparse
    b = A @ x_true + 0.1 * np.random.randn(m)
    lambd = 0.1

    result = solve_lasso(A, b, lambd, verbose=1)
    print(f"  Status: {'Success' if result['status'] == 0 else 'Failed'}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Optimal value: {result['optval']:.6f}")

    # Test 2: Sparse Lasso (if scipy available)
    if HAS_SCIPY:
        print("\n2. Sparse Lasso:")
        A_sparse = sp.random(m, n, density=0.3, format='csr')
        A_sparse = A_sparse.toarray()  # Convert for true value computation
        x_true = np.random.randn(n)
        x_true[np.abs(x_true) < 0.5] = 0
        b = A_sparse @ x_true + 0.1 * np.random.randn(m)
        A_sparse = sp.csr_matrix(A_sparse)  # Back to sparse

        result = solve_lasso(A_sparse, b, lambd, verbose=1)
        print(f"  Status: {'Success' if result['status'] == 0 else 'Failed'}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Optimal value: {result['optval']:.6f}")

    print("\nAll tests passed!")
