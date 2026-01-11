#!/usr/bin/env python3
"""
Benchmark POGS against other QP solvers on standard QP problems.

Generates problems inspired by the Maros-Mészáros test set categories:
- DUAL: Dual formulation problems (typically well-conditioned)
- PRIMAL: Primal formulation problems
- CVXQP: Random convex QPs of varying conditioning
- LP: Linear programs (QP with P=0)

Reference:
- Maros & Mészáros (1999): "A Repository of Convex Quadratic Programming Problems"
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not installed")

# Check for POGS CVXPY integration
try:
    from pogs_cvxpy import POGS

    HAS_POGS = True
except ImportError:
    HAS_POGS = False
    print("Warning: POGS CVXPY integration not available")


@dataclass
class QPProblem:
    """A quadratic programming problem."""

    name: str
    P: np.ndarray | None  # Quadratic term (can be None for LP)
    q: np.ndarray  # Linear term
    A_eq: np.ndarray | None  # Equality constraints
    b_eq: np.ndarray | None
    A_ineq: np.ndarray | None  # Inequality constraints
    b_ineq: np.ndarray | None

    @property
    def n(self):
        return len(self.q)

    @property
    def m(self):
        m = 0
        if self.A_eq is not None:
            m += len(self.b_eq)
        if self.A_ineq is not None:
            m += len(self.b_ineq)
        return m


@dataclass
class BenchmarkResult:
    """Result from solving a benchmark problem."""

    solver: str
    time_sec: float
    optval: float
    status: str
    iterations: int | None = None
    error: str | None = None


def generate_dual_qp(n: int, m: int, seed: int) -> QPProblem:
    """
    Generate a DUAL-style QP (well-conditioned, from dual formulation).

    These problems typically have:
    - Diagonal or near-diagonal P (well-conditioned)
    - Dense constraints
    - Both equality and inequality constraints
    """
    rng = np.random.default_rng(seed)

    # Well-conditioned diagonal P
    P = np.diag(rng.uniform(1.0, 10.0, n))
    q = rng.standard_normal(n)

    # Mix of equality and inequality constraints
    n_eq = m // 3
    n_ineq = m - n_eq

    A_eq = rng.standard_normal((n_eq, n)) if n_eq > 0 else None
    b_eq = rng.standard_normal(n_eq) if n_eq > 0 else None

    A_ineq = rng.standard_normal((n_ineq, n)) if n_ineq > 0 else None
    b_ineq = rng.uniform(1, 5, n_ineq) if n_ineq > 0 else None  # Ensure feasibility

    return QPProblem(f"DUAL_{n}x{m}", P, q, A_eq, b_eq, A_ineq, b_ineq)


def generate_primal_qp(n: int, m: int, seed: int) -> QPProblem:
    """
    Generate a PRIMAL-style QP (from primal formulation).

    These typically have sparse structure.
    """
    rng = np.random.default_rng(seed)

    # Sparse P with moderate conditioning
    P = np.eye(n) * rng.uniform(0.5, 2.0, n)
    # Add some off-diagonal elements
    for _ in range(n // 2):
        i, j = rng.integers(0, n, 2)
        if i != j:
            val = rng.uniform(-0.1, 0.1)
            P[i, j] = val
            P[j, i] = val

    q = rng.standard_normal(n)

    # Only inequality constraints (box + general)
    n_box = min(n, m // 2)
    n_gen = m - n_box

    # Box constraints
    A_box = np.eye(n)[:n_box]
    b_box = rng.uniform(2, 5, n_box)

    # General constraints
    A_gen = rng.standard_normal((n_gen, n)) if n_gen > 0 else np.zeros((0, n))
    b_gen = rng.uniform(1, 10, n_gen) if n_gen > 0 else np.zeros(0)

    A_ineq = np.vstack([A_box, A_gen])
    b_ineq = np.concatenate([b_box, b_gen])

    return QPProblem(f"PRIMAL_{n}x{m}", P, q, None, None, A_ineq, b_ineq)


def generate_cvxqp(n: int, m: int, seed: int, conditioning: str = "good") -> QPProblem:
    """
    Generate a CVXQP-style random convex QP.

    conditioning: "good" (kappa~10), "medium" (kappa~100), "bad" (kappa~1000)
    """
    rng = np.random.default_rng(seed)

    # Generate P = Q @ diag(eigvals) @ Q.T
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    if conditioning == "good":
        eigvals = rng.uniform(1.0, 10.0, n)
    elif conditioning == "medium":
        eigvals = rng.uniform(0.1, 10.0, n)
    else:  # bad
        eigvals = rng.uniform(0.01, 10.0, n)

    P = Q @ np.diag(eigvals) @ Q.T
    P = (P + P.T) / 2  # Ensure symmetry

    q = rng.standard_normal(n)

    # Mixed constraints
    n_eq = m // 4
    n_ineq = m - n_eq

    A_eq = rng.standard_normal((n_eq, n)) if n_eq > 0 else None
    b_eq = rng.standard_normal(n_eq) if n_eq > 0 else None

    A_ineq = rng.standard_normal((n_ineq, n)) if n_ineq > 0 else None
    b_ineq = rng.uniform(1, 10, n_ineq) if n_ineq > 0 else None

    cond_suffix = {"good": "G", "medium": "M", "bad": "B"}[conditioning]
    return QPProblem(f"CVXQP_{cond_suffix}_{n}x{m}", P, q, A_eq, b_eq, A_ineq, b_ineq)


def generate_lp(n: int, m: int, seed: int) -> QPProblem:
    """Generate a bounded linear program (no quadratic term)."""
    rng = np.random.default_rng(seed)

    q = rng.standard_normal(n)

    # Box constraints (0 <= x <= 1) + general inequality constraints
    # This ensures bounded feasible region
    n_gen = m

    A_box = np.eye(n)
    b_box = np.ones(n)  # x <= 1

    A_gen = rng.standard_normal((n_gen, n))
    b_gen = np.abs(A_gen @ np.ones(n) * 0.5) + 1  # Ensure feasibility with x=0.5*ones

    A_ineq = np.vstack([A_box, A_gen])
    b_ineq = np.concatenate([b_box, b_gen])

    return QPProblem(f"LP_{n}x{m}", None, q, None, None, A_ineq, b_ineq)


def build_cvxpy_problem(qp: QPProblem):
    """Convert QPProblem to CVXPY problem."""
    x = cp.Variable(qp.n)

    # Objective
    if qp.P is not None:
        objective = 0.5 * cp.quad_form(x, qp.P) + qp.q @ x
    else:
        objective = qp.q @ x

    # Constraints
    constraints = []
    if qp.A_eq is not None and len(qp.b_eq) > 0:
        constraints.append(qp.A_eq @ x == qp.b_eq)
    if qp.A_ineq is not None and len(qp.b_ineq) > 0:
        constraints.append(qp.A_ineq @ x <= qp.b_ineq)

    # Add x >= 0 for LP problems to ensure boundedness
    if qp.P is None:
        constraints.append(x >= 0)

    return cp.Problem(cp.Minimize(objective), constraints), x


def solve_with_solver(qp: QPProblem, solver_name: str, verbose: bool = False) -> BenchmarkResult:
    """Solve QP using specified solver via CVXPY."""
    solver_map = {
        "OSQP": cp.OSQP,
        "SCS": cp.SCS,
        "POGS": POGS() if HAS_POGS else None,  # POGS needs instance, not class
    }

    # Check for Clarabel
    try:
        solver_map["CLARABEL"] = cp.CLARABEL
    except AttributeError:
        pass

    if solver_name not in solver_map or solver_map[solver_name] is None:
        return BenchmarkResult(
            solver=solver_name,
            time_sec=0,
            optval=float("nan"),
            status="unavailable",
            error=f"Solver {solver_name} not available",
        )

    try:
        prob, _ = build_cvxpy_problem(qp)

        start = time.perf_counter()
        prob.solve(solver=solver_map[solver_name], verbose=verbose)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            solver=solver_name,
            time_sec=elapsed,
            optval=prob.value if prob.value is not None else float("nan"),
            status=prob.status,
            iterations=getattr(prob.solver_stats, "num_iters", None)
            if prob.solver_stats
            else None,
        )
    except Exception as e:
        return BenchmarkResult(
            solver=solver_name,
            time_sec=0,
            optval=float("nan"),
            status="error",
            error=str(e),
        )


def run_benchmark():
    """Run QP benchmark suite."""
    if not HAS_CVXPY:
        print("ERROR: CVXPY not installed")
        return

    print("=" * 80)
    print("POGS QP Solver Benchmark")
    print("=" * 80)
    print()
    print("Problems inspired by Maros-Mészáros QP test set categories.")
    print("Reference: Maros & Mészáros (1999)")
    print()

    # Generate test problems
    problems = []

    # DUAL-style (well-conditioned) - POGS should excel here
    problems.append(generate_dual_qp(50, 30, seed=1))
    problems.append(generate_dual_qp(100, 50, seed=2))
    problems.append(generate_dual_qp(200, 100, seed=3))

    # PRIMAL-style (sparse structure)
    problems.append(generate_primal_qp(50, 40, seed=4))
    problems.append(generate_primal_qp(100, 80, seed=5))
    problems.append(generate_primal_qp(200, 150, seed=6))

    # CVXQP-style (varying conditioning)
    problems.append(generate_cvxqp(50, 30, seed=7, conditioning="good"))
    problems.append(generate_cvxqp(100, 50, seed=8, conditioning="good"))
    problems.append(generate_cvxqp(50, 30, seed=9, conditioning="medium"))
    problems.append(generate_cvxqp(100, 50, seed=10, conditioning="medium"))

    # LP (no quadratic term)
    problems.append(generate_lp(50, 30, seed=11))
    problems.append(generate_lp(100, 50, seed=12))
    problems.append(generate_lp(200, 100, seed=13))

    solvers = ["POGS", "OSQP", "SCS"]
    all_results = []

    # Header
    print(f"{'Problem':<18} {'n':>5} {'m':>5} |", end="")
    for solver in solvers:
        print(f" {solver:>10}", end="")
    print(" | Winner")
    print("-" * (18 + 6 + 6 + 11 * len(solvers) + 10))

    for qp in problems:
        results = {}
        for solver_name in solvers:
            result = solve_with_solver(qp, solver_name)
            results[solver_name] = result
            all_results.append((qp.name, qp.n, qp.m, result))

        # Print row
        print(f"{qp.name:<18} {qp.n:>5} {qp.m:>5} |", end="")
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

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    wins = dict.fromkeys(solvers, 0)
    times_by_solver = {s: [] for s in solvers}

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

    print(f"\n{'Solver':<12} {'Wins':>6} {'Solved':>8} {'Geom Mean':>12}")
    print("-" * 42)
    for solver in solvers:
        times = times_by_solver[solver]
        if times:
            geom_mean = np.exp(np.mean(np.log(times))) * 1000
            print(f"{solver:<12} {wins[solver]:>6} {len(times):>8} {geom_mean:>10.1f}ms")
        else:
            print(f"{solver:<12} {wins[solver]:>6} {0:>8} {'N/A':>12}")

    # Relative performance
    print()
    print("Relative Performance (POGS vs others):")
    print("-" * 42)

    pogs_times = times_by_solver.get("POGS", [])
    for other in ["OSQP", "SCS"]:
        other_times = times_by_solver.get(other, [])
        if pogs_times and other_times and len(pogs_times) == len(other_times):
            ratios = [o / p for p, o in zip(pogs_times, other_times, strict=True) if p > 0]
            if ratios:
                geom_ratio = np.exp(np.mean(np.log(ratios)))
                print(f"  POGS vs {other}: {geom_ratio:.2f}x (>1 means POGS faster)")


if __name__ == "__main__":
    run_benchmark()
