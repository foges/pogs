#!/usr/bin/env python3
"""
Benchmark POGS on graph-form problems where it excels.

POGS is designed for problems of the form:
    min f(Ax) + g(x)

This includes:
- Lasso: min ||Ax - b||^2 + λ||x||_1
- Ridge: min ||Ax - b||^2 + λ||x||^2
- Elastic Net: min ||Ax - b||^2 + λ1||x||_1 + λ2||x||^2
- Constrained least squares with box constraints
- Huber regression

This script compares POGS against standard solvers (OSQP, SCS, CLARABEL)
on these problems.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np


# Add pogs to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed")

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pogs_graph import solve_elastic_net, solve_lasso, solve_ridge

    HAS_POGS = True
except ImportError as e:
    HAS_POGS = False
    print(f"Warning: POGS not available: {e}")


@dataclass
class BenchmarkResult:
    """Result from solving a benchmark problem."""

    solver: str
    time_sec: float
    optval: float
    status: str
    iterations: int | None = None
    error: str | None = None


@dataclass
class BenchmarkProblem:
    """A graph-form benchmark problem."""

    name: str
    m: int  # Number of rows in A
    n: int  # Number of columns in A
    problem_type: str  # lasso, ridge, elastic_net, etc.

    # Data
    A: np.ndarray
    b: np.ndarray
    lambda1: float = 0.1  # L1 regularization
    lambda2: float = 0.0  # L2 regularization

    # Optional: true solution for validation
    x_true: np.ndarray | None = None


def generate_lasso_problem(
    m: int, n: int, sparsity: float = 0.1, noise: float = 0.01, seed: int = 42
) -> BenchmarkProblem:
    """Generate a Lasso problem: min ||Ax - b||^2 + λ||x||_1"""
    np.random.seed(seed)

    A = np.random.randn(m, n)

    # Sparse true solution
    x_true = np.zeros(n)
    nnz = max(1, int(n * sparsity))
    x_true[:nnz] = np.random.randn(nnz)

    # Generate observations with noise
    b = A @ x_true + noise * np.random.randn(m)

    # Regularization parameter (scaled to problem)
    lambda1 = 0.1 * np.linalg.norm(A.T @ b, np.inf)

    return BenchmarkProblem(
        name=f"lasso_{m}x{n}",
        m=m,
        n=n,
        problem_type="lasso",
        A=A,
        b=b,
        lambda1=lambda1,
        x_true=x_true,
    )


def generate_ridge_problem(m: int, n: int, noise: float = 0.1, seed: int = 42) -> BenchmarkProblem:
    """Generate a Ridge problem: min ||Ax - b||^2 + λ||x||^2"""
    np.random.seed(seed)

    A = np.random.randn(m, n)
    x_true = np.random.randn(n) * 0.1
    b = A @ x_true + noise * np.random.randn(m)

    # Regularization parameter
    lambda2 = 0.5

    return BenchmarkProblem(
        name=f"ridge_{m}x{n}",
        m=m,
        n=n,
        problem_type="ridge",
        A=A,
        b=b,
        lambda2=lambda2,
        x_true=x_true,
    )


def generate_elastic_net_problem(
    m: int, n: int, sparsity: float = 0.1, noise: float = 0.01, seed: int = 42
) -> BenchmarkProblem:
    """Generate an Elastic Net problem: min ||Ax - b||^2 + λ1||x||_1 + λ2||x||^2"""
    np.random.seed(seed)

    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    nnz = max(1, int(n * sparsity))
    x_true[:nnz] = np.random.randn(nnz)
    b = A @ x_true + noise * np.random.randn(m)

    lambda1 = 0.1 * np.linalg.norm(A.T @ b, np.inf)
    lambda2 = 0.1

    return BenchmarkProblem(
        name=f"elastic_net_{m}x{n}",
        m=m,
        n=n,
        problem_type="elastic_net",
        A=A,
        b=b,
        lambda1=lambda1,
        lambda2=lambda2,
        x_true=x_true,
    )


def solve_with_pogs(problem: BenchmarkProblem, verbose: bool = False) -> BenchmarkResult:
    """Solve using POGS graph-form solver."""
    if not HAS_POGS:
        return BenchmarkResult(
            solver="pogs",
            time_sec=0,
            optval=float("nan"),
            status="unavailable",
            error="POGS not installed",
        )

    try:
        start = time.perf_counter()

        if problem.problem_type == "lasso":
            result = solve_lasso(problem.A, problem.b, problem.lambda1, verbose=1 if verbose else 0)
        elif problem.problem_type == "ridge":
            result = solve_ridge(problem.A, problem.b, problem.lambda2, verbose=1 if verbose else 0)
        elif problem.problem_type == "elastic_net":
            result = solve_elastic_net(
                problem.A, problem.b, problem.lambda1, problem.lambda2, verbose=1 if verbose else 0
            )
        else:
            raise ValueError(f"Unknown problem type: {problem.problem_type}")

        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            solver="pogs",
            time_sec=elapsed,
            optval=result["optval"],
            status="optimal" if result["status"] == 0 else "error",
            iterations=result["num_iters"],
        )
    except Exception as e:
        import traceback

        return BenchmarkResult(
            solver="pogs",
            time_sec=0,
            optval=float("nan"),
            status="error",
            error=str(e) + "\n" + traceback.format_exc(),
        )


def solve_with_cvxpy(
    problem: BenchmarkProblem, solver_name: str, verbose: bool = False
) -> BenchmarkResult:
    """Solve using CVXPY with specified solver."""
    if not HAS_CVXPY:
        return BenchmarkResult(
            solver=solver_name,
            time_sec=0,
            optval=float("nan"),
            status="unavailable",
            error="CVXPY not installed",
        )

    solver_map = {
        "osqp": cp.OSQP,
        "scs": cp.SCS,
        "clarabel": cp.CLARABEL,
        "ecos": cp.ECOS,
    }

    if solver_name not in solver_map:
        return BenchmarkResult(
            solver=solver_name,
            time_sec=0,
            optval=float("nan"),
            status="unavailable",
            error=f"Unknown solver: {solver_name}",
        )

    try:
        # Check if solver is available
        if solver_map[solver_name] not in cp.installed_solvers():
            return BenchmarkResult(
                solver=solver_name,
                time_sec=0,
                optval=float("nan"),
                status="unavailable",
                error=f"{solver_name} not installed",
            )

        # Build CVXPY problem
        x = cp.Variable(problem.n)

        if problem.problem_type == "lasso":
            objective = cp.sum_squares(problem.A @ x - problem.b) + problem.lambda1 * cp.norm1(x)
        elif problem.problem_type == "ridge":
            objective = cp.sum_squares(
                problem.A @ x - problem.b
            ) + problem.lambda2 * cp.sum_squares(x)
        elif problem.problem_type == "elastic_net":
            objective = (
                cp.sum_squares(problem.A @ x - problem.b)
                + problem.lambda1 * cp.norm1(x)
                + problem.lambda2 * cp.sum_squares(x)
            )
        else:
            raise ValueError(f"Unknown problem type: {problem.problem_type}")

        cvxpy_prob = cp.Problem(cp.Minimize(objective))

        start = time.perf_counter()
        cvxpy_prob.solve(solver=solver_map[solver_name], verbose=verbose)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            solver=solver_name,
            time_sec=elapsed,
            optval=cvxpy_prob.value if cvxpy_prob.value is not None else float("nan"),
            status=cvxpy_prob.status,
            iterations=getattr(cvxpy_prob.solver_stats, "num_iters", None)
            if cvxpy_prob.solver_stats
            else None,
        )
    except Exception as e:
        return BenchmarkResult(
            solver=solver_name, time_sec=0, optval=float("nan"), status="error", error=str(e)
        )


def run_benchmark_suite():
    """Run the full benchmark suite."""
    print("=" * 80)
    print("POGS Graph-Form Benchmark Suite")
    print("=" * 80)
    print()

    # Problem sizes to test
    sizes = [
        (100, 50),  # Small
        (200, 100),  # Small-medium
        (500, 200),  # Medium
        (1000, 500),  # Medium-large
        (2000, 1000),  # Large
        (5000, 2000),  # Very large
    ]

    problem_types = ["lasso", "ridge", "elastic_net"]
    solvers = ["pogs", "osqp", "scs", "clarabel"]

    all_results = []

    for ptype in problem_types:
        print(f"\n{'=' * 60}")
        print(f"Problem Type: {ptype.upper()}")
        print("=" * 60)
        print()

        # Header
        print(f"{'Size':>15} |", end="")
        for solver in solvers:
            print(f" {solver:>12}", end="")
        print(" | Winner")
        print("-" * (16 + 13 * len(solvers) + 10))

        for m, n in sizes:
            # Generate problem
            if ptype == "lasso":
                problem = generate_lasso_problem(m, n)
            elif ptype == "ridge":
                problem = generate_ridge_problem(m, n)
            else:
                problem = generate_elastic_net_problem(m, n)

            results = {}

            # Solve with each solver
            for solver_name in solvers:
                if solver_name == "pogs":
                    result = solve_with_pogs(problem)
                else:
                    result = solve_with_cvxpy(problem, solver_name)
                results[solver_name] = result
                all_results.append((problem, result))

            # Print row
            print(f"{m:>6}x{n:<6} |", end="")
            times = {}
            for solver in solvers:
                r = results[solver]
                if r.status in ["optimal", "optimal_inaccurate"]:
                    print(f" {r.time_sec * 1000:>10.1f}ms", end="")
                    times[solver] = r.time_sec
                else:
                    print(f" {'FAIL':>12}", end="")

            # Determine winner
            if times:
                winner = min(times, key=times.get)
                # Calculate speedup
                if winner == "pogs" and len(times) > 1:
                    others = [t for s, t in times.items() if s != "pogs"]
                    speedup = min(others) / times["pogs"]
                    print(f" | {winner} ({speedup:.1f}x)")
                else:
                    print(f" | {winner}")
            else:
                print(" | N/A")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count wins and compute geometric mean
    wins = dict.fromkeys(solvers, 0)
    times_by_solver = {s: [] for s in solvers}

    for problem, result in all_results:
        if result.status in ["optimal", "optimal_inaccurate"]:
            times_by_solver[result.solver].append(result.time_sec)

    # For each problem, find winner
    from collections import defaultdict

    problems_results = defaultdict(dict)
    for problem, result in all_results:
        problems_results[problem.name][result.solver] = result

    for _pname, presults in problems_results.items():
        valid_times = {
            s: r.time_sec
            for s, r in presults.items()
            if r.status in ["optimal", "optimal_inaccurate"]
        }
        if valid_times:
            winner = min(valid_times, key=valid_times.get)
            wins[winner] += 1

    print(f"\n{'Solver':<12} {'Wins':>6} {'Geom Mean':>12}")
    print("-" * 32)
    for solver in solvers:
        times = times_by_solver[solver]
        if times:
            geom_mean = np.exp(np.mean(np.log(times))) * 1000  # ms
            print(f"{solver:<12} {wins[solver]:>6} {geom_mean:>10.1f}ms")
        else:
            print(f"{solver:<12} {wins[solver]:>6} {'N/A':>12}")

    # POGS speedup summary
    print("\nPOGS Speedups (vs second-best solver):")
    pogs_speedups = []
    for _pname, presults in problems_results.items():
        valid_times = {
            s: r.time_sec
            for s, r in presults.items()
            if r.status in ["optimal", "optimal_inaccurate"]
        }
        if "pogs" in valid_times and len(valid_times) > 1:
            pogs_time = valid_times["pogs"]
            other_times = [t for s, t in valid_times.items() if s != "pogs"]
            if other_times:
                speedup = min(other_times) / pogs_time
                pogs_speedups.append(speedup)

    if pogs_speedups:
        print(f"  Min speedup:  {min(pogs_speedups):.2f}x")
        print(f"  Max speedup:  {max(pogs_speedups):.2f}x")
        print(f"  Mean speedup: {np.mean(pogs_speedups):.2f}x")
        print(f"  Geom speedup: {np.exp(np.mean(np.log(pogs_speedups))):.2f}x")


if __name__ == "__main__":
    run_benchmark_suite()
