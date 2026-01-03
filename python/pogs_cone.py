"""
Python wrapper for POGS cone form interface using ctypes.

Solves problems of the form:
    minimize    c^T * x
    subject to  b - A*x ∈ K_y,  x ∈ K_x

where K_x and K_y are Cartesian products of cones.
"""

import ctypes
import numpy as np
import os
from enum import IntEnum


def _find_shared_library():
    """Locate a shared POGS library built by CMake or the Makefile."""
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
        "  make -C src cpu\n"
        "or\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOGS_BUILD_GPU=OFF\n"
        "  cmake --build build\n"
    )

_lib = ctypes.CDLL(_lib_path)


class Ordering(IntEnum):
    """Matrix ordering: column-major or row-major."""
    COL_MAJ = 0
    ROW_MAJ = 1


class Cone(IntEnum):
    """Cone types for cone constraints."""
    ZERO = 0        # { x : x = 0 }
    NON_NEG = 1     # { x : x >= 0 }
    NON_POS = 2     # { x : x <= 0 }
    SOC = 3         # { (p, x) : ||x||_2 <= p }
    SDP = 4         # { X : X >= 0 } (PSD matrix)
    EXP_PRIMAL = 5  # { (x, y, z) : y > 0, y e^(x/y) <= z }
    EXP_DUAL = 6    # { (u, v, w) : u < 0, -u e^(v/u) <= ew }


class ConeConstraintC(ctypes.Structure):
    """C structure for cone constraints."""
    _fields_ = [
        ("cone", ctypes.c_int),
        ("indices", ctypes.POINTER(ctypes.c_uint)),
        ("size", ctypes.c_uint)
    ]


# Define function signatures
_lib.PogsConeD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ConeConstraintC),  # cones_x
    ctypes.c_size_t,  # num_cones_x
    ctypes.POINTER(ConeConstraintC),  # cones_y
    ctypes.c_size_t,  # num_cones_y
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
_lib.PogsConeD.restype = ctypes.c_int

_lib.PogsConeQD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ctypes.c_double),  # P
    ctypes.POINTER(ConeConstraintC),  # cones_x
    ctypes.c_size_t,  # num_cones_x
    ctypes.POINTER(ConeConstraintC),  # cones_y
    ctypes.c_size_t,  # num_cones_y
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
_lib.PogsConeQD.restype = ctypes.c_int

_lib.PogsConeDirectD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ConeConstraintC),  # cones_x
    ctypes.c_size_t,  # num_cones_x
    ctypes.POINTER(ConeConstraintC),  # cones_y
    ctypes.c_size_t,  # num_cones_y
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
_lib.PogsConeDirectD.restype = ctypes.c_int

_lib.PogsConeDirectQD.argtypes = [
    ctypes.c_int,  # ord
    ctypes.c_size_t,  # m
    ctypes.c_size_t,  # n
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ctypes.c_double),  # P
    ctypes.POINTER(ConeConstraintC),  # cones_x
    ctypes.c_size_t,  # num_cones_x
    ctypes.POINTER(ConeConstraintC),  # cones_y
    ctypes.c_size_t,  # num_cones_y
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
_lib.PogsConeDirectQD.restype = ctypes.c_int

def solve_cone(A, b, c, cones_x, cones_y,
               rho=1.0, abs_tol=1e-4, rel_tol=1e-3, max_iter=10000,
               verbose=0, adaptive_rho=True, gap_stop=True,
               use_direct=False, P=None):
    """
    Solve a cone form problem using POGS.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Constraint matrix
    b : array_like, shape (m,)
        Constraint vector
    c : array_like, shape (n,)
        Objective vector
    P : array_like, shape (n, n), optional
        Quadratic objective matrix for 0.5 * x^T P x + c^T x
    cones_x : list of tuples (Cone, list of int)
        Cone constraints for x, where each tuple is (cone_type, indices)
    cones_y : list of tuples (Cone, list of int)
        Cone constraints for y, where each tuple is (cone_type, indices)
    rho : float, optional
        Initial penalty parameter
    abs_tol : float, optional
        Absolute tolerance
    rel_tol : float, optional
        Relative tolerance
    max_iter : int, optional
        Maximum iterations
    verbose : int, optional
        Verbosity level (0 = silent)
    adaptive_rho : bool, optional
        Use adaptive penalty parameter
    gap_stop : bool, optional
        Use gap stopping criterion
    use_direct : bool, optional
        Use direct projection (dense A) instead of CGLS

    Returns
    -------
    dict with keys:
        x : ndarray, shape (n,)
            Solution vector
        y : ndarray, shape (m,)
            Slack vector
        l : ndarray, shape (m,)
            Dual vector
        optval : float
            Optimal objective value
        iterations : int
            Number of iterations
        status : int
            Status code (0 = success)
    """
    # Convert inputs to numpy arrays
    A = np.asarray(A, dtype=np.float64, order='C')
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if P is not None:
        try:
            import scipy.sparse as sp
            if sp.issparse(P):
                P = P.toarray()
        except ImportError:
            pass
        P = np.asarray(P, dtype=np.float64, order='C')

    m, n = A.shape

    # Validate dimensions
    assert b.shape == (m,), f"b should have shape ({m},), got {b.shape}"
    assert c.shape == (n,), f"c should have shape ({n},), got {c.shape}"
    if P is not None:
        assert P.shape == (n, n), f"P should have shape ({n}, {n}), got {P.shape}"

    # Convert cone constraints to C structures
    def make_cone_constraints(cones):
        c_cones = []
        c_indices = []  # Keep references to prevent garbage collection
        for cone_type, indices in cones:
            idx_array = (ctypes.c_uint * len(indices))(*indices)
            c_indices.append(idx_array)
            c_cone = ConeConstraintC(
                cone=int(cone_type),
                indices=ctypes.cast(idx_array, ctypes.POINTER(ctypes.c_uint)),
                size=len(indices)
            )
            c_cones.append(c_cone)
        return (ConeConstraintC * len(c_cones))(*c_cones) if c_cones else None, c_indices

    c_cones_x, c_indices_x = make_cone_constraints(cones_x)
    c_cones_y, c_indices_y = make_cone_constraints(cones_y)

    # Allocate output arrays
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(m, dtype=np.float64)
    l = np.zeros(m, dtype=np.float64)
    optval = ctypes.c_double()
    final_iter = ctypes.c_uint()

    # Call C function
    if P is None:
        solver_fn = _lib.PogsConeDirectD if use_direct else _lib.PogsConeD
        status = solver_fn(
            int(Ordering.ROW_MAJ),
            m, n,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_cones_x, len(cones_x) if cones_x else 0,
            c_cones_y, len(cones_y) if cones_y else 0,
            rho, abs_tol, rel_tol, max_iter, verbose,
            int(adaptive_rho), int(gap_stop),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            l.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(optval),
            ctypes.byref(final_iter)
        )
    else:
        solver_fn = _lib.PogsConeDirectQD if use_direct else _lib.PogsConeQD
        status = solver_fn(
            int(Ordering.ROW_MAJ),
            m, n,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_cones_x, len(cones_x) if cones_x else 0,
            c_cones_y, len(cones_y) if cones_y else 0,
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


if __name__ == '__main__':
    # Simple test: minimize x1 subject to x1 + x2 = 2, x >= 0
    print("Testing Python POGS cone interface...")
    print("Problem: minimize x1 subject to x1 + x2 = 2, x >= 0")
    print("Expected: x = [0, 2], optimal value = 0\n")

    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])
    c = np.array([1.0, 0.0])

    cones_x = [(Cone.NON_NEG, [0, 1])]  # x >= 0
    cones_y = [(Cone.ZERO, [0])]  # b - Ax = 0

    result = solve_cone(A, b, c, cones_x, cones_y, verbose=5)

    print("\nResult:")
    print(f"  x = {result['x']}")
    print(f"  Optimal value = {result['optval']}")
    print(f"  Status: {'Success' if result['status'] == 0 else 'Failed'}")
    print(f"  Iterations: {result['iterations']}")
