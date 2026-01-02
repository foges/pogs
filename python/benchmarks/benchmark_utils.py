"""
Utility functions for POGS benchmarks.
"""

import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    problem_name: str
    problem_size: Dict[str, int]
    solver: str
    solve_time: float
    setup_time: float
    total_time: float
    iterations: int
    optimal_value: float
    status: str
    primal_residual: Optional[float] = None
    dual_residual: Optional[float] = None
    error: Optional[str] = None


def benchmark_solver(problem, solver_name, verbose=False, timeout=300):
    """
    Benchmark a single solver on a problem.

    Args:
        problem: CVXPY Problem instance
        solver_name: Name of solver to use
        verbose: Print solver output
        timeout: Maximum solve time in seconds

    Returns:
        BenchmarkResult object
    """
    start_time = time.time()

    try:
        # Solve the problem
        result = problem.solve(
            solver=solver_name,
            verbose=verbose,
            # solver_opts={'max_iters': 10000}  # May need solver-specific options
        )

        total_time = time.time() - start_time

        # Extract solver statistics
        setup_time = 0
        iterations = 0
        primal_res = None
        dual_res = None

        if hasattr(problem, 'solver_stats'):
            stats = problem.solver_stats
            setup_time = getattr(stats, 'setup_time', 0)
            iterations = getattr(stats, 'num_iters', 0)

        solve_time = total_time - setup_time

        return BenchmarkResult(
            problem_name=problem.name,
            problem_size=problem.size_metrics,
            solver=solver_name,
            solve_time=solve_time,
            setup_time=setup_time,
            total_time=total_time,
            iterations=iterations,
            optimal_value=result if np.isfinite(result) else np.nan,
            status=problem.status,
            primal_residual=primal_res,
            dual_residual=dual_res,
        )

    except KeyboardInterrupt:
        raise

    except Exception as e:
        total_time = time.time() - start_time

        return BenchmarkResult(
            problem_name=problem.name,
            problem_size=problem.size_metrics,
            solver=solver_name,
            solve_time=total_time,
            setup_time=0,
            total_time=total_time,
            iterations=0,
            optimal_value=np.nan,
            status="ERROR",
            error=str(e)
        )


def save_results(results: List[BenchmarkResult], filename='results/latest.json'):
    """Save benchmark results to JSON file."""
    with open(filename, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_results(filename='results/latest.json') -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return [BenchmarkResult(**r) for r in data]


def generate_text_report(results: List[BenchmarkResult]) -> str:
    """
    Generate a human-readable text report from benchmark results.

    Returns:
        Formatted string report
    """
    # Group results by problem
    by_problem = defaultdict(list)
    for r in results:
        by_problem[r.problem_name].append(r)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("POGS BENCHMARK RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    total_problems = len(by_problem)
    total_runs = len(results)
    solvers = sorted(set(r.solver for r in results))

    report_lines.append(f"Total problems: {total_problems}")
    report_lines.append(f"Total runs: {total_runs}")
    report_lines.append(f"Solvers tested: {', '.join(solvers)}")
    report_lines.append("")

    # Per-problem results
    for problem_name in sorted(by_problem.keys()):
        problem_results = by_problem[problem_name]

        report_lines.append("-" * 80)
        report_lines.append(f"Problem: {problem_name}")
        report_lines.append("-" * 80)

        # Get problem size info
        if problem_results:
            size_info = problem_results[0].problem_size
            size_str = ", ".join(f"{k}={v}" for k, v in size_info.items())
            report_lines.append(f"Size: {size_str}")
            report_lines.append("")

        # Group by solver
        by_solver = defaultdict(list)
        for r in problem_results:
            by_solver[r.solver].append(r)

        # Table header
        report_lines.append(f"{'Solver':<15} {'Avg Time (s)':<15} {'Iterations':<12} {'Success':<10} {'Opt Value':<15}")
        report_lines.append(f"{'-'*15} {'-'*15} {'-'*12} {'-'*10} {'-'*15}")

        for solver in sorted(by_solver.keys()):
            solver_results = by_solver[solver]
            successful = [r for r in solver_results if r.status == 'optimal']

            if successful:
                avg_time = np.mean([r.solve_time for r in successful])
                avg_iters = np.mean([r.iterations for r in successful])
                success_rate = len(successful) / len(solver_results) * 100
                avg_optval = np.mean([r.optimal_value for r in successful if np.isfinite(r.optimal_value)])

                report_lines.append(
                    f"{solver:<15} {avg_time:<15.4f} {avg_iters:<12.0f} "
                    f"{success_rate:<9.0f}% {avg_optval:<15.6e}"
                )
            else:
                errors = [r.error for r in solver_results if r.error]
                error_msg = errors[0][:20] if errors else "Failed"
                report_lines.append(
                    f"{solver:<15} {'--':<15} {'--':<12} "
                    f"{'0%':<10} {error_msg:<15}"
                )

        report_lines.append("")

    # Speedup analysis
    report_lines.append("=" * 80)
    report_lines.append("SPEEDUP ANALYSIS (vs POGS)")
    report_lines.append("=" * 80)
    report_lines.append("")

    for problem_name in sorted(by_problem.keys()):
        problem_results = by_problem[problem_name]
        by_solver = defaultdict(list)
        for r in problem_results:
            if r.status == 'optimal':
                by_solver[r.solver].append(r.solve_time)

        if 'POGS' in by_solver and by_solver['POGS']:
            pogs_time = np.mean(by_solver['POGS'])

            report_lines.append(f"{problem_name}:")
            for solver in sorted(by_solver.keys()):
                if solver == 'POGS':
                    continue
                if by_solver[solver]:
                    solver_time = np.mean(by_solver[solver])
                    speedup = solver_time / pogs_time
                    if speedup > 1:
                        report_lines.append(f"  POGS {speedup:.2f}x faster than {solver}")
                    else:
                        report_lines.append(f"  {solver} {1/speedup:.2f}x faster than POGS")
            report_lines.append("")

    return "\n".join(report_lines)


def print_summary(results: List[BenchmarkResult]):
    """Print a summary of benchmark results to console."""
    report = generate_text_report(results)
    print(report)

    # Save to file
    with open('results/summary.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to results/summary.txt")
