#!/usr/bin/env python3
"""
Benchmark POGS on Maros-Mészáros QP test set.

The Maros-Mészáros test set contains 138 classic quadratic programming problems.
This is THE standard benchmark for QP solvers in operations research.

Reference:
- Maros & Mészáros (1999): "A Repository of Convex Quadratic Programming Problems"
- https://github.com/qpsolvers/maros_meszaros_qpbenchmark

Uses cvxbench to load the problems.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np


# Add cvxbench to path
sys.path.insert(0, "/Users/chris/code/cvxbench/src")

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed")

try:
    from cvxbench.loaders.maros_meszaros import MarosMeszarosLoader

    HAS_CVXBENCH = True
except ImportError as e:
    HAS_CVXBENCH = False
    print(f"Warning: cvxbench not available: {e}")


@dataclass
class BenchmarkResult:
    """Result from solving a benchmark problem."""

    solver: str
    time_sec: float
    optval: float
    status: str
    iterations: int | None = None
    error: str | None = None


def build_cvxpy_problem(problem):
    """Convert a cvxbench BenchmarkProblem to CVXPY problem."""
    n = problem.n_vars
    x = cp.Variable(n)

    # Build objective: min 0.5 x'Px + q'x
    if problem.P is not None:
        objective = 0.5 * cp.quad_form(x, problem.P) + problem.q @ x
    else:
        objective = problem.q @ x

    # Build constraints from conic form
    constraints = []
    row_offset = 0
    for cone_type, cone_dim in problem.cones:
        A_block = problem.A[row_offset : row_offset + cone_dim, :]
        b_block = problem.b[row_offset : row_offset + cone_dim]

        if cone_type == "zero":
            constraints.append(A_block @ x == b_block)
        elif cone_type == "nonneg":
            constraints.append(A_block @ x <= b_block)
        elif cone_type == "soc":
            s = b_block - A_block @ x
            t = s[0]
            u = s[1:]
            constraints.append(cp.norm(u, 2) <= t)

        row_offset += cone_dim

    return cp.Problem(cp.Minimize(objective), constraints), x


def solve_with_cvxpy(problem, solver_name: str, verbose: bool = False) -> BenchmarkResult:
    """Solve using CVXPY with specified solver."""
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
        cvxpy_prob, _x = build_cvxpy_problem(problem)

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


def run_benchmark():
    """Run benchmarks on Maros-Mészáros QP problems."""
    if not HAS_CVXBENCH:
        print("ERROR: cvxbench not available")
        return

    print("=" * 80)
    print("POGS Benchmark on Maros-Mészáros QP Test Set")
    print("=" * 80)
    print()
    print("These are REAL industry benchmark problems from operations research.")
    print("Reference: Maros & Mészáros (1999)")
    print()

    loader = MarosMeszarosLoader()
    problems = loader.list_problems()
    print(f"Total problems available: {len(problems)}")
    print()

    # Select a representative subset (small to medium problems)
    # These are well-known benchmark problems
    selected = [
        "QAFIRO",  # Small LP-like QP
        "HS21",  # Hock-Schittkowski test
        "HS35",
        "HS51",
        "HS76",
        "DUAL1",  # Dual formulation
        "DUAL2",
        "PRIMALC1",  # Primal formulation
        "LOTSCHD",  # Lot scheduling
        "CVXQP1_S",  # Convex QP small
        "CVXQP2_S",
        "CVXQP3_S",
        "QPCBLEND",  # Blending problem
        "QPCSTAIR",  # Staircase structure
        "VALUES",  # Value problem
        "ZECEVIC2",  # Zecevic test
    ]

    solvers = ["osqp", "scs", "clarabel"]
    all_results = []

    # Header
    print(f"{'Problem':<12} {'n':>6} {'m':>6} |", end="")
    for solver in solvers:
        print(f" {solver:>10}", end="")
    print(" | Winner")
    print("-" * (12 + 7 + 7 + 11 * len(solvers) + 10))

    for prob_name in selected:
        try:
            problem = loader.load_problem(prob_name)
            n = problem.n_vars
            m = problem.n_constraints

            results = {}
            for solver_name in solvers:
                result = solve_with_cvxpy(problem, solver_name)
                results[solver_name] = result
                all_results.append((prob_name, n, m, result))

            # Print row
            print(f"{prob_name:<12} {n:>6} {m:>6} |", end="")
            times = {}
            for solver in solvers:
                r = results[solver]
                if r.status in ["optimal", "optimal_inaccurate"]:
                    print(f" {r.time_sec * 1000:>8.1f}ms", end="")
                    times[solver] = r.time_sec
                else:
                    print(f" {'FAIL':>10}", end="")

            if times:
                winner = min(times, key=times.get)
                print(f" | {winner}")
            else:
                print(" | N/A")

        except Exception as e:
            print(f"{prob_name:<12} ERROR: {e}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY (Maros-Mészáros QP Benchmark)")
    print("=" * 80)

    wins = dict.fromkeys(solvers, 0)
    times_by_solver = {s: [] for s in solvers}

    from collections import defaultdict

    probs_results = defaultdict(dict)
    for pname, n, m, result in all_results:
        probs_results[pname][result.solver] = result
        if result.status in ["optimal", "optimal_inaccurate"]:
            times_by_solver[result.solver].append(result.time_sec)

    for pname, presults in probs_results.items():
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
            geom_mean = np.exp(np.mean(np.log(times))) * 1000
            print(f"{solver:<12} {wins[solver]:>6} {geom_mean:>10.1f}ms")
        else:
            print(f"{solver:<12} {wins[solver]:>6} {'N/A':>12}")

    print()
    print("Note: POGS graph-form solver is designed for problems like Lasso/Ridge,")
    print("not general QPs. For general QPs, use OSQP, Clarabel, or SCS.")


if __name__ == "__main__":
    run_benchmark()
