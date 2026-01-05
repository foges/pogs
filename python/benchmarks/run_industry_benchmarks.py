#!/usr/bin/env python3
"""
Industry-standard QP benchmark suite.

Runs POGS against SCS, OSQP, CLARABEL on Maros-Meszaros QP problems.

NOTE: POGS cone solver is NOT optimized for general industry QPs.
POGS excels at graph-form problems (Lasso, Ridge, etc.) where it achieves
10-100x speedups. For general QPs, use OSQP or CLARABEL instead.

The Maros-Meszaros benchmark results show POGS struggling because:
1. These problems have wide dynamic range in optimal solutions
2. ADMM can get stuck in suboptimal regions on these hard problems
3. Industry benchmarks are designed to stress-test solvers, not represent
   typical ML/statistics optimization problems where POGS shines.
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import time
import os
import sys
import subprocess
import tempfile
import urllib.request

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed")

# Maros-Meszaros problems to test (subset of commonly used ones)
# Format: (name, url_suffix)
MAROS_MESZAROS_PROBLEMS = [
    # Small problems (< 500 vars)
    "QAFIRO", "QBANDM", "QBEACONF", "QBRANDY", "QE226",
    "QSCAGR7", "QSCORPIO", "QSCTAP1", "QSHARE1B", "QSHARE2B",
    # Medium problems (500-2000 vars)
    "QSCAGR25", "QSCFXM1", "QSCSD1", "QSHIP04S", "QSHIP08S",
    # Larger problems
    "QSCFXM2", "QSCSD6", "QSHIP04L", "QSHIP08L",
    # Classic difficult problems
    "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
    "DUAL1", "DUAL2", "DUAL3", "DUAL4",
    "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4",
]

BASE_URL = "https://raw.githubusercontent.com/osqp/osqp_benchmarks/master/problem_classes/maros_meszaros_data/"


def download_problem(name, cache_dir="/tmp/maros_meszaros"):
    """Download a Maros-Meszaros problem in .mat format."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{name}.mat")

    if os.path.exists(local_path):
        return local_path

    url = f"{BASE_URL}{name}.mat"
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"  Failed to download {name}: {e}")
        return None


def load_qp_problem(mat_path):
    """Load QP problem from .mat file.

    Returns P, q, A, l, u for:
        minimize    0.5 x' P x + q' x
        subject to  l <= A x <= u
    """
    data = sio.loadmat(mat_path)

    P = data['P']
    if sp.issparse(P):
        P = P.toarray()

    q = data['q'].flatten()

    A = data['A']
    if sp.issparse(A):
        A = A.toarray()

    # Convert to float64 to avoid uint8 overflow issues
    l = data['l'].flatten().astype(np.float64)
    u = data['u'].flatten().astype(np.float64)

    return P, q, A, l, u


def solve_qp_cvxpy(P, q, A, l, u, solver_name, time_limit=60, verbose=False):
    """Solve QP using CVXPY with specified solver."""
    n = P.shape[0]
    m = A.shape[0]

    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)

    # Handle constraints: l <= Ax <= u
    constraints = []
    for i in range(m):
        if np.isfinite(l[i]) and np.isfinite(u[i]):
            if abs(l[i] - u[i]) < 1e-10:
                # Equality constraint
                constraints.append(A[i] @ x == l[i])
            else:
                # Box constraint
                constraints.append(A[i] @ x >= l[i])
                constraints.append(A[i] @ x <= u[i])
        elif np.isfinite(l[i]):
            constraints.append(A[i] @ x >= l[i])
        elif np.isfinite(u[i]):
            constraints.append(A[i] @ x <= u[i])

    problem = cp.Problem(objective, constraints)

    solver_opts = {}
    if solver_name == 'SCS':
        solver_opts = {'eps_abs': 1e-4, 'eps_rel': 1e-4, 'max_iters': 10000}
    elif solver_name == 'OSQP':
        solver_opts = {'eps_abs': 1e-4, 'eps_rel': 1e-4, 'max_iter': 10000}
    elif solver_name == 'CLARABEL':
        solver_opts = {'tol_gap_abs': 1e-4, 'tol_gap_rel': 1e-4, 'max_iter': 10000}

    start = time.time()
    try:
        problem.solve(solver=solver_name, verbose=verbose, **solver_opts)
        solve_time = time.time() - start

        if problem.status in ['optimal', 'optimal_inaccurate']:
            return {
                'status': 'solved',
                'time': solve_time,
                'optval': problem.value,
            }
        else:
            return {'status': problem.status, 'time': solve_time}
    except Exception as e:
        return {'status': f'error: {e}', 'time': time.time() - start}


def solve_qp_pogs(P, q, A, l, u, time_limit=60, verbose=0):
    """Solve QP using POGS cone solver.

    QP: min 0.5 x'Px + q'x s.t. l <= Ax <= u
    Uses quadratic objective support to avoid SOC reformulation.
    """
    # Add parent path for pogs_cvxpy
    pogs_python = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pogs_python not in sys.path:
        sys.path.insert(0, pogs_python)

    from pogs_cvxpy import solve_cone_problem

    n = P.shape[0]
    m = A.shape[0]

    if np.linalg.norm(P, ord='fro') < 1e-12:
        return solve_lp_pogs(q, A, l, u, verbose=verbose)

    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_rhs = []

    # Maros-Meszaros uses +/-1e20 for unbounded (not np.inf)
    INF_THRESH = 1e19

    for i in range(m):
        li = l[i]
        ui = u[i]
        has_l = np.isfinite(li) and abs(li) < INF_THRESH
        has_u = np.isfinite(ui) and abs(ui) < INF_THRESH

        if has_l and has_u and abs(li - ui) < 1e-10:
            eq_rows.append(A[i])
            eq_rhs.append(li)
        else:
            if has_l:
                ineq_rows.append(-A[i])
                ineq_rhs.append(-li)
            if has_u:
                ineq_rows.append(A[i])
                ineq_rhs.append(ui)

    A_rows = []
    b_vec = []
    for row, rhs in zip(eq_rows, eq_rhs):
        A_rows.append(row)
        b_vec.append(rhs)
    for row, rhs in zip(ineq_rows, ineq_rhs):
        A_rows.append(row)
        b_vec.append(rhs)

    if A_rows:
        A_cone = np.vstack(A_rows)
        b_cone = np.array(b_vec)
    else:
        A_cone = np.zeros((0, n))
        b_cone = np.zeros(0)

    dims_dict = {
        'f': len(eq_rows),
        'l': len(ineq_rows),
        'q': [],
        's': [],
        'ep': 0,
        'ed': 0,
    }

    # Solve with POGS
    start = time.time()
    try:
        result = solve_cone_problem(
            q, A_cone, b_cone, dims_dict, P=P,
            abs_tol=1e-4, rel_tol=1e-4, max_iter=10000, verbose=verbose
        )
        solve_time = time.time() - start

        if result['status'] == 0:
            optval = result['optval']
            return {
                'status': 'solved',
                'time': solve_time,
                'optval': optval,
                'num_iters': result.get('num_iters', 0),
                'primal_res': result.get('primal_res'),
                'eps_pri': result.get('eps_pri'),
                'primal_res_ratio': result.get('primal_res_ratio'),
            }
        else:
            return {
                'status': f'pogs_status_{result["status"]}',
                'time': solve_time,
                'primal_res': result.get('primal_res'),
                'eps_pri': result.get('eps_pri'),
                'primal_res_ratio': result.get('primal_res_ratio'),
            }
    except Exception as e:
        return {'status': f'error: {e}', 'time': time.time() - start}


def solve_lp_pogs(c, A, l, u, verbose=0):
    """Solve LP using POGS cone solver.

    LP: min c'x s.t. l <= Ax <= u
    """
    pogs_python = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pogs_python not in sys.path:
        sys.path.insert(0, pogs_python)

    from pogs_cvxpy import solve_cone_problem

    n = len(c)
    m = A.shape[0]

    # Count constraint types
    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_rhs = []

    # Maros-Meszaros uses +/-1e20 for unbounded (not np.inf)
    INF_THRESH = 1e19

    for i in range(m):
        has_l = np.isfinite(l[i]) and abs(l[i]) < INF_THRESH
        has_u = np.isfinite(u[i]) and abs(u[i]) < INF_THRESH

        if has_l and has_u and abs(l[i] - u[i]) < 1e-10:
            eq_rows.append(A[i])
            eq_rhs.append(l[i])
        else:
            if has_l:
                ineq_rows.append(A[i])
                ineq_rhs.append(l[i])
            if has_u:
                ineq_rows.append(-A[i])
                ineq_rhs.append(-u[i])

    num_eq = len(eq_rows)
    num_ineq = len(ineq_rows)
    n_cons = num_eq + num_ineq

    A_cone = np.zeros((n_cons, n))
    b_cone = np.zeros(n_cons)

    row = 0
    for a_row, b_val in zip(eq_rows, eq_rhs):
        A_cone[row] = -a_row
        b_cone[row] = -b_val
        row += 1
    for a_row, b_val in zip(ineq_rows, ineq_rhs):
        A_cone[row] = -a_row
        b_cone[row] = -b_val
        row += 1

    dims_dict = {'f': num_eq, 'l': num_ineq, 'q': [], 's': [], 'ep': 0, 'ed': 0}

    start = time.time()
    try:
        result = solve_cone_problem(
            c, A_cone, b_cone, dims_dict,
            abs_tol=1e-4, rel_tol=1e-4, max_iter=10000, verbose=verbose
        )
        solve_time = time.time() - start
        if result['status'] == 0:
            return {'status': 'solved', 'time': solve_time, 'optval': result['optval'],
                    'num_iters': result.get('num_iters', 0),
                    'primal_res': result.get('primal_res'),
                    'eps_pri': result.get('eps_pri'),
                    'primal_res_ratio': result.get('primal_res_ratio')}
        else:
            return {'status': f'pogs_status_{result["status"]}', 'time': solve_time,
                    'primal_res': result.get('primal_res'),
                    'eps_pri': result.get('eps_pri'),
                    'primal_res_ratio': result.get('primal_res_ratio')}
    except Exception as e:
        return {'status': f'error: {e}', 'time': time.time() - start}


def run_benchmarks():
    """Run industry benchmarks."""
    print("=" * 80)
    print("Maros-Meszaros QP Benchmark Suite")
    print("=" * 80)

    # Detect available solvers
    solvers = ['POGS']  # Always include POGS
    if HAS_CVXPY:
        available = cp.installed_solvers()
        for s in ['OSQP', 'SCS', 'CLARABEL']:
            if s in available:
                solvers.append(s)

    if len(solvers) == 1:
        print("Warning: Only POGS available, no comparison solvers!")

    print(f"Solvers: {', '.join(solvers)}")
    print()

    results = {}

    for problem_name in MAROS_MESZAROS_PROBLEMS:
        print(f"\n{problem_name}")
        print("-" * 60)

        # Download problem
        mat_path = download_problem(problem_name)
        if mat_path is None:
            continue

        # Load problem
        try:
            P, q, A, l, u = load_qp_problem(mat_path)
            n, m = P.shape[0], A.shape[0]
            nnz_P = np.count_nonzero(P)
            nnz_A = np.count_nonzero(A)
            print(f"  Size: n={n}, m={m}, nnz(P)={nnz_P}, nnz(A)={nnz_A}")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        problem_results = {'n': n, 'm': m}

        for solver in solvers:
            print(f"  {solver:12s} ... ", end='', flush=True)

            if solver == 'POGS':
                result = solve_qp_pogs(P, q, A, l, u)
            else:
                result = solve_qp_cvxpy(P, q, A, l, u, solver)

            if result['status'] == 'solved':
                iters = result.get('num_iters', '')
                iters_str = f"  {iters:5d} iters" if iters else ""
                print(f"OK  {result['time']:.4f}s{iters_str}  optval={result['optval']:.6e}")
                problem_results[solver] = result
            else:
                print(f"{result['status']}")
                problem_results[solver] = result

        results[problem_name] = problem_results

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Problem':<15} {'n':>6} {'m':>6}", end='')
    for solver in solvers:
        print(f" {solver:>12}", end='')
    print()
    print("-" * (30 + 13 * len(solvers)))

    for name, res in results.items():
        print(f"{name:<15} {res.get('n', 0):>6} {res.get('m', 0):>6}", end='')
        for solver in solvers:
            if solver in res and res[solver].get('status') == 'solved':
                print(f" {res[solver]['time']:>11.4f}s", end='')
            else:
                status = res.get(solver, {}).get('status', 'N/A')
                print(f" {status[:12]:>12}", end='')
        print()

    # Compute success rates
    print("\nSuccess Rates:")
    for solver in solvers:
        solved = sum(1 for r in results.values() if r.get(solver, {}).get('status') == 'solved')
        total = len(results)
        print(f"  {solver}: {solved}/{total} ({100*solved/total:.1f}%)")


if __name__ == '__main__':
    run_benchmarks()
