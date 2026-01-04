#!/usr/bin/env python3
"""
Direct comparison of POGS vs other solvers.
Uses POGS C interface directly via ctypes instead of CVXPY integration.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import CVXPY for comparison solvers
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed, only POGS results will be shown")

# Try to load POGS C library
import ctypes
from ctypes import c_int, c_uint, c_double, POINTER, byref

def load_pogs_library():
    """Load the POGS shared library."""
    pogs_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Try different library paths
    lib_paths = [
        os.path.join(pogs_root, 'build', 'lib', 'libpogs_cpu.a'),
        os.path.join(pogs_root, 'src', 'build', 'pogs.a'),
    ]

    for path in lib_paths:
        if os.path.exists(path):
            return path

    return None


def solve_lasso_pogs(A, b, lambda_val, abs_tol=1e-4, rel_tol=1e-3, max_iter=2500, verbose=0):
    """
    Solve Lasso problem using POGS graph form solver.

    minimize 0.5 * ||Ax - b||^2 + lambda * ||x||_1

    This uses the compiled POGS library via subprocess since ctypes
    can't easily load static libraries.
    """
    import subprocess
    import tempfile

    m, n = A.shape
    pogs_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Generate C++ code for the problem
    code = f'''
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

int main() {{
    const size_t m = {m};
    const size_t n = {n};

    // Matrix A (column-major for POGS)
    std::vector<double> A_data = {{
'''
    # Write A in column-major order
    for j in range(n):
        for i in range(m):
            code += f"        {A[i, j]:.16e}"
            if i < m - 1 or j < n - 1:
                code += ","
            code += "\n"

    code += f'''    }};

    // Vector b
    std::vector<double> b_data = {{
'''
    for i in range(m):
        code += f"        {b[i]:.16e}"
        if i < m - 1:
            code += ","
        code += "\n"

    code += f'''    }};

    double lambda_val = {lambda_val:.16e};

    pogs::MatrixDense<double> A('c', m, n, A_data.data());
    pogs::PogsDirect<double, pogs::MatrixDense<double>> solver(A);

    solver.SetAbsTol({abs_tol});
    solver.SetRelTol({rel_tol});
    solver.SetMaxIter({max_iter});
    solver.SetVerbose({verbose});

    std::vector<FunctionObj<double>> f(m);
    std::vector<FunctionObj<double>> g(n);

    // f(y) = 0.5 * (y - b)^2
    for (size_t i = 0; i < m; ++i) {{
        f[i].h = kSquare;
        f[i].a = 1.0;
        f[i].b = b_data[i];
        f[i].c = 1.0;
        f[i].d = 0.0;
        f[i].e = 0.0;
    }}

    // g(x) = lambda * |x|
    for (size_t j = 0; j < n; ++j) {{
        g[j].h = kAbs;
        g[j].c = lambda_val;
    }}

    double t = timer<double>();
    pogs::PogsStatus status = solver.Solve(f, g);
    double solve_time = timer<double>() - t;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "STATUS=" << status << std::endl;
    std::cout << "TIME=" << solve_time << std::endl;
    std::cout << "ITERS=" << solver.GetFinalIter() << std::endl;
    std::cout << "OPTVAL=" << solver.GetOptval() << std::endl;

    return 0;
}}
'''

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        cpp_file = f.name
        f.write(code)

    exe_file = tempfile.mktemp()

    try:
        # Compile
        include_dirs = [
            f'-I{pogs_root}/src/include',
            f'-I{pogs_root}/src/cpu/include',
        ]

        compile_cmd = [
            'g++', '-O3', '-std=c++20',
            *include_dirs,
            '-o', exe_file,
            cpp_file,
            f'{pogs_root}/build/lib/libpogs_cpu.a',
            '-framework', 'Accelerate',
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr}")

        # Run
        result = subprocess.run([exe_file], capture_output=True, text=True, timeout=60)

        # Parse output
        output = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                if key in ['STATUS', 'ITERS']:
                    output[key] = int(float(value))
                else:
                    output[key] = float(value)

        return output

    finally:
        if os.path.exists(cpp_file):
            os.remove(cpp_file)
        if os.path.exists(exe_file):
            os.remove(exe_file)


def solve_lasso_cvxpy(A, b, lambda_val, solver_name, abs_tol=1e-4, rel_tol=1e-3, max_iter=2500, verbose=False):
    """Solve Lasso using CVXPY with specified solver."""
    m, n = A.shape

    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambda_val * cp.norm(x, 1))
    problem = cp.Problem(objective)

    solver_opts = {}
    if solver_name == 'SCS':
        solver_opts = {'eps_abs': abs_tol, 'eps_rel': rel_tol, 'max_iters': max_iter}
    elif solver_name == 'ECOS':
        solver_opts = {'abstol': abs_tol, 'reltol': rel_tol, 'max_iters': max_iter}
    elif solver_name == 'OSQP':
        solver_opts = {'eps_abs': abs_tol, 'eps_rel': rel_tol, 'max_iter': max_iter}
    elif solver_name == 'CLARABEL':
        solver_opts = {'tol_gap_abs': abs_tol, 'tol_gap_rel': rel_tol, 'max_iter': max_iter}

    start_time = time.time()
    try:
        problem.solve(solver=solver_name, verbose=verbose, **solver_opts)
        solve_time = time.time() - start_time

        # Get iteration count from solver stats
        iters = 0
        if hasattr(problem, 'solver_stats') and problem.solver_stats:
            stats = problem.solver_stats
            if hasattr(stats, 'num_iters'):
                iters = stats.num_iters
            elif hasattr(stats, 'iter'):
                iters = stats.iter

        return {
            'STATUS': 0 if problem.status == 'optimal' else 1,
            'TIME': solve_time,
            'ITERS': iters,
            'OPTVAL': problem.value if problem.value is not None else float('nan'),
        }
    except Exception as e:
        return {
            'STATUS': -1,
            'TIME': time.time() - start_time,
            'ITERS': 0,
            'OPTVAL': float('nan'),
            'ERROR': str(e),
        }


def generate_lasso_problem(m, n, sparsity=0.1, condition_number=1.0, density=1.0, seed=42):
    """Generate a Lasso regression problem.

    Args:
        m: Number of samples
        n: Number of features
        sparsity: Sparsity of true solution (0.1 = 10% nonzero)
        condition_number: Condition number of A
        density: Density of A (1.0 = dense, 0.01 = 1% nonzero)
        seed: Random seed
    """
    np.random.seed(seed)

    # Generate matrix A (sparse or dense)
    if density < 1.0:
        from scipy import sparse
        nnz = int(m * n * density)
        rows = np.random.randint(0, m, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.randn(nnz)
        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(m, n))
        A = A_sparse.toarray()
        # For POGS, store sparse format
        A_csc = A_sparse.tocsc()
    else:
        A = np.random.randn(m, n)
        A_csc = None

    # Apply conditioning if requested
    if condition_number > 1.0:
        for j in range(n):
            scale = condition_number ** (-j / (n - 1))
            A[:, j] *= scale

    # Generate sparse true solution
    x_true = np.random.randn(n)
    x_true[np.random.rand(n) < (1 - sparsity)] = 0

    # Generate observations with noise
    b = A @ x_true + 0.1 * np.random.randn(m)

    # Set regularization parameter
    lambda_max = np.linalg.norm(A.T @ b, np.inf)
    lambda_val = 0.1 * lambda_max

    return A, b, lambda_val, A_csc


def generate_basis_pursuit_problem(m, n, sparsity=0.1, seed=42):
    """Generate a Basis Pursuit problem (sparse signal recovery).

    Problem: minimize ||x||_1 subject to Ax = b

    This is the classic compressed sensing problem where we want to
    recover a sparse signal from underdetermined measurements.

    Args:
        m: Number of measurements (m < n for underdetermined)
        n: Signal dimension
        sparsity: Fraction of nonzero elements in true signal
        seed: Random seed
    """
    np.random.seed(seed)

    # Measurement matrix (typically Gaussian random)
    A = np.random.randn(m, n) / np.sqrt(m)

    # Sparse true signal
    k = max(1, int(n * sparsity))  # Number of nonzeros
    support = np.random.choice(n, k, replace=False)
    x_true = np.zeros(n)
    x_true[support] = np.random.randn(k)

    # Noiseless measurements
    b = A @ x_true

    return A, b, x_true


def run_lasso_benchmark(solvers, sizes, sparse=False):
    """Run Lasso benchmarks."""
    density = 0.05 if sparse else 1.0
    sparse_label = " (Sparse A, 5% density)" if sparse else ""

    print(f"\n{'='*80}")
    print(f"Lasso Regression Benchmark{sparse_label}")
    print("="*80)

    for m, n, size_name in sizes:
        print(f"\n{size_name} Problem (m={m}, n={n})")
        print("-" * 60)

        A, b, lambda_val, _ = generate_lasso_problem(m, n, density=density)

        results = {}
        for solver in solvers:
            print(f"  {solver:12s} ... ", end='', flush=True)
            try:
                if solver == 'POGS':
                    result = solve_lasso_pogs(A, b, lambda_val)
                else:
                    result = solve_lasso_cvxpy(A, b, lambda_val, solver)

                if result['STATUS'] == 0:
                    print(f"OK  {result['TIME']:.4f}s  {result['ITERS']:5d} iters  optval={result['OPTVAL']:.6e}")
                else:
                    print(f"FAIL (status={result['STATUS']})")
                results[solver] = result
            except Exception as e:
                print(f"ERROR: {e}")
                results[solver] = {'STATUS': -1, 'ERROR': str(e)}

        # Print speedup comparison
        if 'POGS' in results and results['POGS']['STATUS'] == 0:
            pogs_time = results['POGS']['TIME']
            print(f"\n  Speedup vs POGS:")
            for solver, result in results.items():
                if solver != 'POGS' and result['STATUS'] == 0:
                    speedup = result['TIME'] / pogs_time
                    if speedup > 1:
                        print(f"    {solver}: POGS is {speedup:.1f}x faster")
                    else:
                        print(f"    {solver}: POGS is {1/speedup:.1f}x slower")


def run_basis_pursuit_benchmark(solvers, sizes):
    """Run Basis Pursuit benchmarks (compressed sensing)."""
    print(f"\n{'='*80}")
    print("Basis Pursuit Benchmark (Compressed Sensing)")
    print("minimize ||x||_1 subject to Ax = b")
    print("="*80)

    for m, n, size_name in sizes:
        print(f"\n{size_name} Problem (m={m}, n={n})")
        print("-" * 60)

        A, b, x_true = generate_basis_pursuit_problem(m, n, sparsity=0.1)

        results = {}
        for solver in solvers:
            if solver == 'POGS':
                continue  # Skip POGS for now - BP needs cone form
            print(f"  {solver:12s} ... ", end='', flush=True)
            try:
                # Basis Pursuit via CVXPY
                x = cp.Variable(n)
                objective = cp.Minimize(cp.norm(x, 1))
                constraints = [A @ x == b]
                problem = cp.Problem(objective, constraints)

                solver_opts = {'max_iters': 5000} if solver == 'SCS' else {}
                start = time.time()
                problem.solve(solver=solver, verbose=False, **solver_opts)
                solve_time = time.time() - start

                if problem.status == 'optimal':
                    # Check recovery quality
                    recovery_error = np.linalg.norm(x.value - x_true) / np.linalg.norm(x_true)
                    print(f"OK  {solve_time:.4f}s  recovery_err={recovery_error:.2e}")
                else:
                    print(f"FAIL ({problem.status})")
                results[solver] = {'TIME': solve_time, 'STATUS': problem.status}
            except Exception as e:
                print(f"ERROR: {e}")

    return results


def main():
    print("=" * 80)
    print("POGS vs Other Solvers - Comprehensive Benchmark Suite")
    print("=" * 80)

    # Detect available solvers
    solvers = ['POGS']
    if HAS_CVXPY:
        available = cp.installed_solvers()
        for s in ['SCS', 'ECOS', 'CLARABEL']:
            if s in available:
                solvers.append(s)

    print(f"Solvers: {', '.join(solvers)}")

    # Dense Lasso problems
    dense_sizes = [
        (100, 50, "Small"),
        (500, 250, "Medium"),
        (1000, 500, "Large"),
    ]
    run_lasso_benchmark(solvers, dense_sizes, sparse=False)

    # Sparse Lasso problems (larger scale)
    sparse_sizes = [
        (1000, 500, "Medium"),
        (5000, 2500, "Large"),
        (10000, 5000, "XLarge"),
    ]
    run_lasso_benchmark(solvers, sparse_sizes, sparse=True)

    # Basis Pursuit (compressed sensing)
    bp_sizes = [
        (50, 200, "Small"),
        (100, 500, "Medium"),
        (200, 1000, "Large"),
    ]
    cvxpy_solvers = [s for s in solvers if s != 'POGS']
    if cvxpy_solvers:
        run_basis_pursuit_benchmark(cvxpy_solvers, bp_sizes)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
