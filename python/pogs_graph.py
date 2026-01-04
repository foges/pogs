"""
POGS Graph-Form Interface

This module provides a graph-form interface to POGS that:
1. Detects problems with separable structure that POGS excels at
2. Maps them to POGS FunctionObj types
3. Uses the fast graph-form solver instead of the cone solver

Supported problem types:
- Lasso: min 0.5||Ax - b||² + λ||x||₁
- Ridge: min 0.5||Ax - b||² + λ||x||²
- Elastic Net: min 0.5||Ax - b||² + λ₁||x||₁ + λ₂||x||²
- Logistic regression: min Σ log(1 + exp(-yᵢaᵢ'x)) + λ||x||₁
- Huber regression: min Σ huber(Ax - b) + λ||x||₁
- Non-negative least squares: min 0.5||Ax - b||² s.t. x >= 0
- Bounded least squares: min 0.5||Ax - b||² s.t. l <= x <= u
"""

import numpy as np
import ctypes
import os
from enum import IntEnum


def _find_shared_library():
    """Locate the shared POGS library."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates = [
        os.path.join(root, 'build', 'lib', 'libpogs_cpu.dylib'),
        os.path.join(root, 'build', 'lib', 'libpogs_cpu.so'),
        # Old Makefile build location (from src/)
        os.path.join(root, 'src', 'build', 'libpogs_cpu.dylib'),
        os.path.join(root, 'src', 'build', 'libpogs_cpu.so'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class Function(IntEnum):
    """POGS function types matching prox_lib.h"""
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


class FunctionObj(ctypes.Structure):
    """
    C structure matching POGS FunctionObj.

    Represents: c * h(a*x - b) + d*x + (e/2)*x²
    """
    _fields_ = [
        ("h", ctypes.c_int),      # Function type
        ("a", ctypes.c_double),   # Scale
        ("b", ctypes.c_double),   # Offset
        ("c", ctypes.c_double),   # Weight
        ("d", ctypes.c_double),   # Linear term
        ("e", ctypes.c_double),   # Quadratic term
    ]

    def __init__(self, h=Function.kZero, a=1.0, b=0.0, c=1.0, d=0.0, e=0.0):
        super().__init__()
        self.h = int(h)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)


def solve_lasso(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                verbose=0, rho=1.0):
    """
    Solve Lasso: minimize 0.5||Ax - b||² + λ||x||₁

    Parameters
    ----------
    A : ndarray (m, n)
        Design matrix
    b : ndarray (m,)
        Target vector
    lambd : float
        L1 regularization parameter

    Returns
    -------
    dict with keys:
        'x': solution vector
        'status': 0 for success
        'optval': optimal value
        'num_iters': iterations used
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    # f_i(y_i) = 0.5 * (y_i - b_i)² -> kSquare with a=1, b=b_i, c=1
    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # g_j(x_j) = λ|x_j| -> kAbs with a=1, b=0, c=λ
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for j in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_ridge(A, b, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                verbose=0, rho=1.0):
    """
    Solve Ridge regression: minimize 0.5||Ax - b||² + λ||x||²
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]
    g = [FunctionObj(Function.kSquare, 1.0, 0.0, lambd) for j in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_elastic_net(A, b, lambda1, lambda2, abs_tol=1e-4, rel_tol=1e-4,
                      max_iter=2500, verbose=0, rho=1.0):
    """
    Solve Elastic Net: minimize 0.5||Ax - b||² + λ₁||x||₁ + λ₂||x||²
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]
    # kAbs with e parameter for quadratic: c*|ax-b| + d*x + (e/2)*x²
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambda1, 0.0, lambda2) for j in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_nonneg_ls(A, b, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                    verbose=0, rho=1.0):
    """
    Solve non-negative least squares: minimize 0.5||Ax - b||² s.t. x >= 0
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]
    g = [FunctionObj(Function.kIndGe0) for j in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_bounded_ls(A, b, lb, ub, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                     verbose=0, rho=1.0):
    """
    Solve bounded least squares: minimize 0.5||Ax - b||² s.t. lb <= x <= ub

    Uses indicator function I(lb <= x <= ub) via kIndBox01 with scaling.
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    m, n = A.shape

    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]

    # Transform x to [0, 1]: t = (x - lb) / (ub - lb)
    # Then use kIndBox01 on t, and scale A accordingly
    scale = ub - lb
    offset = lb

    # Scale columns of A
    A_scaled = A * scale[np.newaxis, :]
    b_adjusted = b - A @ offset

    f = [FunctionObj(Function.kSquare, 1.0, b_adjusted[i], 1.0) for i in range(m)]
    g = [FunctionObj(Function.kIndBox01) for j in range(n)]

    result = _solve_graph_form(A_scaled, f, g, abs_tol, rel_tol, max_iter, verbose, rho)

    # Transform solution back
    if result['status'] == 0:
        result['x'] = result['x'] * scale + offset

    return result


def solve_logistic(A, y, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
                   verbose=0, rho=1.0):
    """
    Solve L1-regularized logistic regression:
    minimize Σ log(1 + exp(-yᵢaᵢ'x)) + λ||x||₁

    Parameters
    ----------
    A : ndarray (m, n)
        Feature matrix (each row is a sample)
    y : ndarray (m,)
        Labels in {-1, +1}
    lambd : float
        L1 regularization parameter
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    y = np.asarray(y, dtype=np.float64)
    m, n = A.shape

    # Scale rows by -y to get: log(1 + exp((-y_i * a_i') * x))
    A_scaled = -y[:, np.newaxis] * A

    # f_i(t) = log(1 + exp(t)) -> kLogistic
    f = [FunctionObj(Function.kLogistic, 1.0, 0.0, 1.0) for i in range(m)]

    # g_j(x_j) = λ|x_j| -> kAbs
    g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for j in range(n)]

    return _solve_graph_form(A_scaled, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_huber(A, b, delta=1.0, lambd=0.0, abs_tol=1e-4, rel_tol=1e-4,
                max_iter=2500, verbose=0, rho=1.0):
    """
    Solve Huber regression: minimize Σ huber_δ(Ax - b) + λ||x||₁

    Huber loss: h(r) = r²/2 if |r| <= δ, else δ|r| - δ²/2
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    # kHuber with parameter controlled by scaling
    # POGS huber has transition at 1, so we scale: huber(x/δ) * δ²
    f = [FunctionObj(Function.kHuber, 1.0/delta, b[i]/delta, delta*delta) for i in range(m)]

    if lambd > 0:
        g = [FunctionObj(Function.kAbs, 1.0, 0.0, lambd) for j in range(n)]
    else:
        g = [FunctionObj(Function.kZero) for j in range(n)]

    return _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def solve_svm(A, y, lambd, abs_tol=1e-4, rel_tol=1e-4, max_iter=2500,
              verbose=0, rho=1.0):
    """
    Solve L2-regularized SVM (hinge loss):
    minimize Σ max(0, 1 - yᵢaᵢ'x) + λ||x||²

    Parameters
    ----------
    A : ndarray (m, n)
        Feature matrix
    y : ndarray (m,)
        Labels in {-1, +1}
    lambd : float
        L2 regularization parameter
    """
    A = np.asarray(A, dtype=np.float64, order='C')
    y = np.asarray(y, dtype=np.float64)
    m, n = A.shape

    # Hinge loss: max(0, 1 - y*a'x) = max(0, 1 - t) where t = y*a'x
    # This is kMaxPos0 with a=-y, b=-1: max(0, -y*a'x - (-1)) = max(0, 1 - y*a'x)
    # Actually: max(0, -t + 1) = max(0, -(t - 1)) which needs kMaxNeg0
    # Or: max(0, 1-t) with t = y_i * a_i' * x

    # Scale rows by y
    A_scaled = y[:, np.newaxis] * A

    # f_i(t) = max(0, 1 - t) -> use kMaxNeg0 with offset
    # kMaxNeg0(t) = max(0, -t), so kMaxNeg0(t - 1) = max(0, 1 - t)
    f = [FunctionObj(Function.kMaxNeg0, 1.0, 1.0, 1.0) for i in range(m)]

    # g_j(x_j) = λ * x_j² / 2 -> kSquare
    g = [FunctionObj(Function.kSquare, 1.0, 0.0, lambd) for j in range(n)]

    return _solve_graph_form(A_scaled, f, g, abs_tol, rel_tol, max_iter, verbose, rho)


def _solve_graph_form(A, f, g, abs_tol, rel_tol, max_iter, verbose, rho):
    """
    Internal: solve graph-form problem using POGS C library.

    minimize Σf_i(y_i) + Σg_j(x_j)  s.t. y = Ax

    The C interface takes separate arrays for each FunctionObj parameter:
    f_a, f_b, f_c, f_d, f_e, f_h (and similarly for g)
    """
    lib_path = _find_shared_library()
    if lib_path is None:
        raise ImportError("POGS shared library not found. Build with: "
                          "cmake --build build --target pogs_cpu_shared")

    lib = ctypes.CDLL(lib_path)

    A = np.asarray(A, dtype=np.float64, order='C')  # Row-major
    m, n = A.shape

    # Convert function lists to separate parameter arrays
    f_a = np.array([fo.a for fo in f], dtype=np.float64)
    f_b = np.array([fo.b for fo in f], dtype=np.float64)
    f_c = np.array([fo.c for fo in f], dtype=np.float64)
    f_d = np.array([fo.d for fo in f], dtype=np.float64)
    f_e = np.array([fo.e for fo in f], dtype=np.float64)
    f_h = np.array([fo.h for fo in f], dtype=np.int32)

    g_a = np.array([go.a for go in g], dtype=np.float64)
    g_b = np.array([go.b for go in g], dtype=np.float64)
    g_c = np.array([go.c for go in g], dtype=np.float64)
    g_d = np.array([go.d for go in g], dtype=np.float64)
    g_e = np.array([go.e for go in g], dtype=np.float64)
    g_h = np.array([go.h for go in g], dtype=np.int32)

    # Output arrays
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)
    lambd_out = np.zeros(m, dtype=np.float64)

    optval = ctypes.c_double()
    final_iter = ctypes.c_uint()

    # Signature from pogs_c.h:
    # int PogsD(enum ORD ord, size_t m, size_t n, const double *A,
    #           const double *f_a, const double *f_b, const double *f_c,
    #           const double *f_d, const double *f_e, const enum FUNCTION *f_h,
    #           const double *g_a, const double *g_b, const double *g_c,
    #           const double *g_d, const double *g_e, const enum FUNCTION *g_h,
    #           double rho, double abs_tol, double rel_tol, unsigned int max_iter,
    #           unsigned int verbose, int adaptive_rho, int gap_stop,
    #           double *x, double *y, double *l, double *optval, unsigned int *final_iter);

    lib.PogsD.argtypes = [
        ctypes.c_int,           # ord (0=col, 1=row)
        ctypes.c_size_t,        # m
        ctypes.c_size_t,        # n
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
        ctypes.c_double,        # rho
        ctypes.c_double,        # abs_tol
        ctypes.c_double,        # rel_tol
        ctypes.c_uint,          # max_iter
        ctypes.c_uint,          # verbose
        ctypes.c_int,           # adaptive_rho
        ctypes.c_int,           # gap_stop
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.POINTER(ctypes.c_double),  # y
        ctypes.POINTER(ctypes.c_double),  # l
        ctypes.POINTER(ctypes.c_double),  # optval
        ctypes.POINTER(ctypes.c_uint),    # final_iter
    ]
    lib.PogsD.restype = ctypes.c_int

    status = lib.PogsD(
        1,  # Row major
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
        rho,
        abs_tol,
        rel_tol,
        max_iter,
        verbose,
        1,  # adaptive_rho
        1,  # gap_stop
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lambd_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(optval),
        ctypes.byref(final_iter),
    )

    return {
        'x': x,
        'y': y,
        'lambda': lambd_out,
        'status': status,
        'optval': optval.value,
        'num_iters': final_iter.value,
    }


# Quick test
if __name__ == '__main__':
    import time

    # Generate Lasso problem
    np.random.seed(42)
    m, n = 500, 200
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:10] = np.random.randn(10)
    b = A @ x_true + 0.1 * np.random.randn(m)
    lambd = 0.1

    print("Testing POGS Graph-Form Interface")
    print("=" * 50)
    print(f"Problem: Lasso {m}x{n}, lambda={lambd}")

    t0 = time.time()
    result = solve_lasso(A, b, lambd, verbose=1)
    t1 = time.time()

    print(f"\nStatus: {result['status']} (0=success)")
    print(f"Optimal value: {result['optval']:.6e}")
    print(f"Iterations: {result['num_iters']}")
    print(f"Time: {t1-t0:.4f}s")
    print(f"||x||_1: {np.abs(result['x']).sum():.4f}")
    print(f"Nonzeros: {np.sum(np.abs(result['x']) > 1e-4)}")
