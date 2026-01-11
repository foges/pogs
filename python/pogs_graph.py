"""
Python wrapper for POGS graph-form interface using ctypes.

Solves problems of the form:
    minimize    sum_i f_i(y_i) + sum_j g_j(x_j)
    subject to  y = A * x

where f_i and g_j are convex functions from a predefined library.

Supports both dense and sparse matrices.
"""

import ctypes
import os
from enum import IntEnum

import numpy as np


try:
    import scipy.sparse as sp

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _find_shared_library():
    """Locate a shared POGS library built by CMake or installed via wheel."""
    import platform

    pkg_dir = os.path.dirname(__file__)
    source_root = os.path.abspath(os.path.join(pkg_dir, ".."))

    # Determine library extension based on platform
    if platform.system() == "Darwin":
        lib_name = "libpogs_cpu.dylib"
    elif platform.system() == "Windows":
        lib_name = "pogs_cpu.dll"
    else:
        lib_name = "libpogs_cpu.so"

    candidates = [
        # Wheel install: library is in the package directory or lib/ subdirectory
        os.path.join(pkg_dir, lib_name),
        os.path.join(pkg_dir, "lib", lib_name),
        # Linux auditwheel puts libs in .libs with version suffix
        os.path.join(pkg_dir, ".libs", lib_name),
        # Source build locations
        os.path.join(source_root, "build", "lib", lib_name),
        os.path.join(source_root, "src", "build", lib_name),
    ]
    # Also check for versioned .so files (auditwheel renames them)
    if platform.system() == "Linux":
        import glob

        patterns = [
            os.path.join(pkg_dir, ".libs", "libpogs_cpu*.so*"),
            os.path.join(pkg_dir, "lib", "libpogs_cpu*.so*"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                candidates.insert(0, matches[0])
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

    kAbs = 0  # f(x) = |x|
    kExp = 1  # f(x) = e^x
    kHuber = 2  # f(x) = huber(x)
    kIdentity = 3  # f(x) = x
    kIndBox01 = 4  # f(x) = I(0 <= x <= 1)
    kIndEq0 = 5  # f(x) = I(x = 0)
    kIndGe0 = 6  # f(x) = I(x >= 0)
    kIndLe0 = 7  # f(x) = I(x <= 0)
    kLogistic = 8  # f(x) = log(1 + e^x)
    kMaxNeg0 = 9  # f(x) = max(0, -x)
    kMaxPos0 = 10  # f(x) = max(0, x)
    kNegEntr = 11  # f(x) = x log(x)
    kNegLog = 12  # f(x) = -log(x)
    kRecipr = 13  # f(x) = 1/x
    kSquare = 14  # f(x) = (1/2) x^2
    kZero = 15  # f(x) = 0


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
    ctypes.POINTER(ctypes.c_int),  # f_h
    ctypes.POINTER(ctypes.c_double),  # g_a
    ctypes.POINTER(ctypes.c_double),  # g_b
    ctypes.POINTER(ctypes.c_double),  # g_c
    ctypes.POINTER(ctypes.c_double),  # g_d
    ctypes.POINTER(ctypes.c_double),  # g_e
    ctypes.POINTER(ctypes.c_int),  # g_h
    ctypes.c_double,  # rho
    ctypes.c_double,  # abs_tol
    ctypes.c_double,  # rel_tol
    ctypes.c_uint,  # max_iter
    ctypes.c_uint,  # verbose
    ctypes.c_int,  # adaptive_rho
    ctypes.c_int,  # gap_stop
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # l
    ctypes.POINTER(ctypes.c_double),  # optval
    ctypes.POINTER(ctypes.c_uint),  # final_iter
]
_lib.PogsD.restype = ctypes.c_int

# Sparse solver: PogsSparseD
_lib.PogsSparseD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.c_size_t,  # nnz
    ctypes.POINTER(ctypes.c_double),  # data
    ctypes.POINTER(ctypes.c_int),  # ptr
    ctypes.POINTER(ctypes.c_int),  # ind
    ctypes.POINTER(ctypes.c_double),  # f_a
    ctypes.POINTER(ctypes.c_double),  # f_b
    ctypes.POINTER(ctypes.c_double),  # f_c
    ctypes.POINTER(ctypes.c_double),  # f_d
    ctypes.POINTER(ctypes.c_double),  # f_e
    ctypes.POINTER(ctypes.c_int),  # f_h
    ctypes.POINTER(ctypes.c_double),  # g_a
    ctypes.POINTER(ctypes.c_double),  # g_b
    ctypes.POINTER(ctypes.c_double),  # g_c
    ctypes.POINTER(ctypes.c_double),  # g_d
    ctypes.POINTER(ctypes.c_double),  # g_e
    ctypes.POINTER(ctypes.c_int),  # g_h
    ctypes.c_double,  # rho
    ctypes.c_double,  # abs_tol
    ctypes.c_double,  # rel_tol
    ctypes.c_uint,  # max_iter
    ctypes.c_uint,  # verbose
    ctypes.c_int,  # adaptive_rho
    ctypes.c_int,  # gap_stop
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # l
    ctypes.POINTER(ctypes.c_double),  # optval
    ctypes.POINTER(ctypes.c_uint),  # final_iter
]
_lib.PogsSparseD.restype = ctypes.c_int


def _solve_graph_form(
    A,
    f,
    g,
    abs_tol=1e-4,
    rel_tol=1e-4,
    max_iter=2500,
    verbose=0,
    rho=1.0,
    adaptive_rho=True,
    gap_stop=True,
):
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
        A = np.asarray(A, dtype=np.float64, order="C")
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
    dual = np.zeros(m, dtype=np.float64)
    optval = ctypes.c_double()
    final_iter = ctypes.c_uint()

    if is_sparse:
        status = _lib.PogsSparseD(
            int(Ordering.ROW_MAJ),
            m,
            n,
            nnz,
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
            rho,
            abs_tol,
            rel_tol,
            max_iter,
            verbose,
            int(adaptive_rho),
            int(gap_stop),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(optval),
            ctypes.byref(final_iter),
        )
    else:
        status = _lib.PogsD(
            int(Ordering.ROW_MAJ),
            m,
            n,
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
            rho,
            abs_tol,
            rel_tol,
            max_iter,
            verbose,
            int(adaptive_rho),
            int(gap_stop),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(optval),
            ctypes.byref(final_iter),
        )

    return {
        "x": x,
        "y": y,
        "l": dual,
        "optval": optval.value,
        "iterations": final_iter.value,
        "status": status,
    }


def solve_lasso(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0):
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


def solve_ridge(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0):
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


def solve_elastic_net(
    A, b, lambda1, lambda2, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0
):
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
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambda1, 0.0, lambda2 / 2) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_logistic(A, b, lambd=0.0, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0):
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


def solve_huber(
    A, b, delta=1.0, lambd=0.0, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0
):
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
    f = [FunctionObj(Function.kHuber, 1.0 / delta, b[i] / delta, delta * delta) for i in range(m)]

    # g_j(x_j) = lambda * |x_j|
    if lambd > 0:
        g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for _ in range(n)]
    else:
        g = [FunctionObj(Function.kZero) for _ in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_svm(A, b, lambd=1.0, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0):
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


def solve_nonneg_ls(A, b, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500, verbose=0, rho=1.0):
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


def solve_lp(
    c,
    A_ineq=None,
    b_ineq=None,
    A_eq=None,
    b_eq=None,
    lb=None,
    ub=None,
    abs_tol=1e-4,
    rel_tol=1e-4,
    max_iter=2500,
    verbose=0,
    rho=1.0,
    adaptive_rho=True,
    use_dense=None,
):
    """
    Solve a linear program in standard form:
        minimize    c' * x
        subject to  A_ineq * x <= b_ineq
                    A_eq * x == b_eq
                    lb <= x <= ub

    Parameters
    ----------
    c : array_like, shape (n,)
        Linear objective coefficients
    A_ineq : array_like, shape (m_ineq, n), optional
        Inequality constraint matrix
    b_ineq : array_like, shape (m_ineq,), optional
        Inequality constraint RHS
    A_eq : array_like, shape (m_eq, n), optional
        Equality constraint matrix
    b_eq : array_like, shape (m_eq,), optional
        Equality constraint RHS
    lb : array_like or float, optional
        Lower bounds on x (default: -inf)
    ub : array_like or float, optional
        Upper bounds on x (default: +inf)
    abs_tol, rel_tol : float
        Tolerances for convergence
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter
    adaptive_rho : bool
        Use adaptive rho selection
    use_dense : bool or None
        Force dense (True) or sparse (False) solver. If None, auto-select
        based on problem structure.

    Returns
    -------
    dict with 'x', 'y', 'optval', 'status', 'iterations'

    Examples
    --------
    >>> # Simple LP: min -x1 - x2 s.t. x1 + x2 <= 1, x >= 0
    >>> c = np.array([-1.0, -1.0])
    >>> A_ineq = np.array([[1.0, 1.0]])
    >>> b_ineq = np.array([1.0])
    >>> result = solve_lp(c, A_ineq, b_ineq, lb=0)
    """
    return solve_qp(
        c=c,
        P=None,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        lb=lb,
        ub=ub,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        max_iter=max_iter,
        verbose=verbose,
        rho=rho,
        adaptive_rho=adaptive_rho,
        use_dense=use_dense,
    )


def solve_qp(
    c,
    P=None,
    A_ineq=None,
    b_ineq=None,
    A_eq=None,
    b_eq=None,
    lb=None,
    ub=None,
    abs_tol=1e-4,
    rel_tol=1e-4,
    max_iter=2500,
    verbose=0,
    rho=1.0,
    adaptive_rho=True,
    use_dense=None,
):
    """
    Solve a quadratic program:
        minimize    0.5 * x' * P * x + c' * x
        subject to  A_ineq * x <= b_ineq
                    A_eq * x == b_eq
                    lb <= x <= ub

    The quadratic term P can be specified in multiple ways:
    - None: LP (no quadratic term)
    - 1D array (n,): diagonal of P (efficient for diagonal Hessians)
    - 2D array (n, n): full symmetric PSD matrix P (uses Cholesky: P = L @ L.T)
    - Tuple (B, factor): where P = factor * B.T @ B (sum-of-squares form)

    Parameters
    ----------
    c : array_like, shape (n,)
        Linear objective coefficients
    P : array_like or tuple, optional
        Quadratic objective. See above for formats.
    A_ineq : array_like, shape (m_ineq, n), optional
        Inequality constraint matrix
    b_ineq : array_like, shape (m_ineq,), optional
        Inequality constraint RHS
    A_eq : array_like, shape (m_eq, n), optional
        Equality constraint matrix
    b_eq : array_like, shape (m_eq,), optional
        Equality constraint RHS
    lb : array_like or float, optional
        Lower bounds on x (default: -inf, meaning unbounded below)
    ub : array_like or float, optional
        Upper bounds on x (default: +inf, meaning unbounded above)
    abs_tol, rel_tol : float
        Tolerances for convergence
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    rho : float
        Initial penalty parameter
    adaptive_rho : bool
        Use adaptive rho selection
    use_dense : bool or None
        Force dense (True) or sparse (False) solver. If None, auto-select:
        - Dense if constraint matrix is more than 10% nonzero
        - Dense if n < 500 (dense factorization is faster for small problems)
        - Sparse otherwise

    Returns
    -------
    dict
        'x' : solution vector
        'y' : auxiliary variable (constraint values)
        'optval' : optimal objective value
        'status' : 0 for success, nonzero for failure
        'iterations' : number of ADMM iterations

    Examples
    --------
    >>> # QP: min 0.5*x'*P*x + c'*x s.t. Ax <= b, x >= 0
    >>> import numpy as np
    >>> n = 10
    >>> P = np.eye(n)  # Simple quadratic
    >>> c = np.random.randn(n)
    >>> A_ineq = np.random.randn(5, n)
    >>> b_ineq = np.ones(5)
    >>> result = solve_qp(c, P, A_ineq, b_ineq, lb=0)

    >>> # Diagonal QP (more efficient)
    >>> P_diag = np.ones(n)  # Diagonal elements only
    >>> result = solve_qp(c, P_diag, A_ineq, b_ineq, lb=0)

    >>> # Sum-of-squares form: min 0.5*||Bx||^2 + c'x
    >>> B = np.random.randn(20, n)
    >>> result = solve_qp(c, (B, 1.0), A_ineq, b_ineq, lb=0)
    """
    c = np.asarray(c, dtype=np.float64).flatten()
    n = len(c)

    # Process bounds
    lb_arr = _process_bounds(lb, n, -np.inf)
    ub_arr = _process_bounds(ub, n, np.inf)

    # Process quadratic term - get B matrix such that P = B.T @ B
    B, q_factor = _process_quadratic(P, n)
    m_quad = B.shape[0] if B is not None else 0

    # Process inequality constraints
    if A_ineq is not None:
        A_ineq = np.asarray(A_ineq, dtype=np.float64)
        b_ineq = np.asarray(b_ineq, dtype=np.float64).flatten()
        m_ineq = A_ineq.shape[0]
    else:
        m_ineq = 0

    # Process equality constraints
    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=np.float64)
        b_eq = np.asarray(b_eq, dtype=np.float64).flatten()
        m_eq = A_eq.shape[0]
    else:
        m_eq = 0

    # Build stacked constraint matrix: [A_ineq; A_eq; B]
    blocks = []
    if m_ineq > 0:
        blocks.append(A_ineq)
    if m_eq > 0:
        blocks.append(A_eq)
    if m_quad > 0:
        blocks.append(B)

    if len(blocks) == 0:
        # No constraints and no quadratic - need at least identity for POGS
        A_stack = np.eye(n, dtype=np.float64)
        # f: just pass through (identity constraint)
        f = [FunctionObj(Function.kZero) for _ in range(n)]
    else:
        A_stack = np.vstack(blocks)

        # Build f functions for each row
        f = []
        # Inequality rows: I(y <= b_ineq)
        for i in range(m_ineq):
            # kIndLe0: I(a*y - b <= 0), so for I(y <= b_i), use a=1, b=b_i
            f.append(FunctionObj(Function.kIndLe0, 1.0, b_ineq[i], 1.0))

        # Equality rows: I(y == b_eq)
        for i in range(m_eq):
            # kIndEq0: I(a*y - b == 0), so for I(y == h_i), use a=1, b=h_i
            f.append(FunctionObj(Function.kIndEq0, 1.0, b_eq[i], 1.0))

        # Quadratic rows: 0.5 * y^2 (for ||Bx||^2 term)
        for _ in range(m_quad):
            # kSquare with c = q_factor gives q_factor * 0.5 * y^2
            f.append(FunctionObj(Function.kSquare, 1.0, 0.0, q_factor))

    # Build g functions for each variable (objective + bounds)
    g = []
    for j in range(n):
        lb_j = lb_arr[j]
        ub_j = ub_arr[j]

        # Determine the base function type based on bounds
        if np.isfinite(lb_j) and np.isfinite(ub_j):
            # Box constraint: I(lb <= x <= ub)
            # Use kIndBox01 with scaling: I(0 <= a*x - b <= 1)
            # where a = 1/(ub-lb), b = lb/(ub-lb)
            if ub_j > lb_j:
                a = 1.0 / (ub_j - lb_j)
                b = lb_j / (ub_j - lb_j)
                g.append(FunctionObj(Function.kIndBox01, a, b, 1.0, c[j], 0.0))
            else:
                # ub == lb: fixed variable, use equality indicator
                g.append(FunctionObj(Function.kIndEq0, 1.0, lb_j, 1.0, c[j], 0.0))
        elif np.isfinite(lb_j):
            # Lower bound only: I(x >= lb)
            # kIndGe0: I(a*x - b >= 0), so for I(x >= lb), use a=1, b=lb
            g.append(FunctionObj(Function.kIndGe0, 1.0, lb_j, 1.0, c[j], 0.0))
        elif np.isfinite(ub_j):
            # Upper bound only: I(x <= ub)
            # kIndLe0: I(a*x - b <= 0), so for I(x <= ub), use a=1, b=ub
            g.append(FunctionObj(Function.kIndLe0, 1.0, ub_j, 1.0, c[j], 0.0))
        else:
            # Free variable: no indicator, just linear objective
            # kZero with d=c[j] gives c[j]*x
            g.append(FunctionObj(Function.kZero, 1.0, 0.0, 1.0, c[j], 0.0))

    # Decide dense vs sparse
    if use_dense is None:
        # Auto-select based on problem structure
        nnz = np.count_nonzero(A_stack)
        density = nnz / A_stack.size if A_stack.size > 0 else 1.0
        use_dense = density > 0.1 or n < 500
        if verbose > 0:
            print(f"POGS QP: Auto-selected {'dense' if use_dense else 'sparse'} solver "
                  f"(density={density:.2%}, n={n})")

    # Convert to sparse if needed
    if not use_dense and HAS_SCIPY:
        A_stack = sp.csr_matrix(A_stack)

    result = _solve_graph_form(
        A_stack, f, g,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        max_iter=max_iter,
        verbose=verbose,
        rho=rho,
        adaptive_rho=adaptive_rho,
    )

    return result


def _process_bounds(bounds, n, default):
    """Convert bounds to array of length n."""
    if bounds is None:
        return np.full(n, default)
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim == 0:
        return np.full(n, float(bounds))
    return bounds.flatten()


def _process_quadratic(P, n):
    """
    Process quadratic term into B matrix where objective includes 0.5*||Bx||^2.

    Returns (B, factor) where the objective term is factor * 0.5 * ||Bx||^2.
    Returns (None, 0) if P is None (LP case).
    """
    if P is None:
        return None, 0.0

    # Tuple form: (B, factor) where P = factor * B.T @ B
    if isinstance(P, tuple):
        B, factor = P
        B = np.asarray(B, dtype=np.float64)
        return B, float(factor)

    P = np.asarray(P, dtype=np.float64)

    # 1D array: diagonal of P
    if P.ndim == 1:
        # P = diag(p), so P = B.T @ B where B = diag(sqrt(p))
        # Handle negative values (shouldn't happen for PSD, but be safe)
        p = np.maximum(P, 0)
        B = np.diag(np.sqrt(p))
        return B, 1.0

    # 2D array: full matrix, use Cholesky
    if P.ndim == 2:
        # Regularize slightly for numerical stability
        P_reg = P + 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(P_reg)
            # P = L @ L.T, so use B = L.T
            return L.T, 1.0
        except np.linalg.LinAlgError:
            # Not PSD, try eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(P_reg)
            eigvals = np.maximum(eigvals, 0)
            B = eigvecs @ np.diag(np.sqrt(eigvals))
            return B.T, 1.0

    raise ValueError(f"Invalid P format: expected 1D, 2D array, or tuple, got shape {P.shape}")


if __name__ == "__main__":
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
        A_sparse = sp.random(m, n, density=0.3, format="csr")
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
