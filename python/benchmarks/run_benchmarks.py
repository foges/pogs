#!/usr/bin/env python3
"""
POGS Benchmark Suite

Compare POGS against other CVXPY solvers on standard optimization problems.

Usage:
    python run_benchmarks.py                    # Run all benchmarks
    python run_benchmarks.py --quick            # Quick test (small problems, 2 trials)
    python run_benchmarks.py --problem lasso    # Run specific problem class
    python run_benchmarks.py --solver POGS SCS  # Test specific solvers only
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cvxpy as cp
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("\nPlease install required packages:")
    print("  pip install cvxpy numpy scipy")
    sys.exit(1)

from benchmark_utils import (
    benchmark_solver,
    save_results,
    print_summary,
    BenchmarkResult
)
from problems import lasso, logistic, portfolio, lp, qp, socp, sdp


# Problem configurations
PROBLEM_CONFIGS = {
    'lasso': [
        ("Lasso Small", lasso.generate_small, ['POGS', 'ECOS', 'SCS', 'OSQP']),
        ("Lasso Medium", lasso.generate_medium, ['POGS', 'ECOS', 'SCS', 'OSQP']),
        ("Lasso Large", lasso.generate_large, ['POGS', 'SCS', 'OSQP']),
    ],
    'logistic': [
        ("Logistic Small", logistic.generate_small, ['POGS', 'ECOS', 'SCS']),
        ("Logistic Medium", logistic.generate_medium, ['POGS', 'SCS']),
    ],
    'portfolio': [
        ("Portfolio Small", portfolio.generate_small, ['POGS', 'ECOS', 'SCS', 'OSQP']),
        ("Portfolio Medium", portfolio.generate_medium, ['POGS', 'SCS', 'OSQP']),
    ],
    'lp': [
        ("LP Small", lp.generate_small, ['POGS', 'ECOS', 'SCS', 'OSQP']),
        ("LP Medium", lp.generate_medium, ['POGS', 'SCS', 'OSQP']),
    ],
    'qp': [
        ("QP Small", qp.generate_small, ['POGS', 'ECOS', 'SCS', 'OSQP']),
        ("QP Medium", qp.generate_medium, ['POGS', 'SCS', 'OSQP']),
    ],
    'socp': [
        ("SOCP Small", socp.generate_small, ['POGS', 'ECOS', 'SCS']),
        ("SOCP Medium", socp.generate_medium, ['POGS', 'SCS']),
    ],
    'sdp': [
        ("SDP Small", sdp.generate_small, ['POGS', 'SCS']),
        ("SDP Medium", sdp.generate_medium, ['POGS', 'SCS']),
    ],
}


# Quick test configuration (smaller problems, fewer trials)
QUICK_CONFIGS = {
    'lasso': [
        ("Lasso Small", lasso.generate_small, ['POGS', 'ECOS', 'SCS']),
    ],
    'portfolio': [
        ("Portfolio Small", portfolio.generate_small, ['POGS', 'ECOS', 'SCS']),
    ],
    'lp': [
        ("LP Small", lp.generate_small, ['POGS', 'ECOS', 'SCS']),
    ],
}


def check_solver_available(solver_name):
    """Check if a solver is installed and available."""
    try:
        # Simple test problem
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve(solver=solver_name, verbose=False)
        return True
    except:
        return False


def run_problem_benchmarks(problem_generator, solvers, n_trials=5, verbose=False):
    """
    Run benchmarks for a specific problem across multiple solvers.

    Args:
        problem_generator: Function that generates a CVXPY problem
        solvers: List of solver names to test
        n_trials: Number of trials for each solver
        verbose: Print solver output

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for trial in range(n_trials):
        # Generate fresh problem instance for each trial
        problem = problem_generator(seed=trial)

        print(f"  Trial {trial + 1}/{n_trials}:")

        for solver_name in solvers:
            # Check if solver is available
            if not check_solver_available(solver_name):
                print(f"    {solver_name:<10} ... SKIPPED (not installed)")
                continue

            print(f"    {solver_name:<10} ... ", end='', flush=True)
            result = benchmark_solver(problem, solver_name, verbose=verbose)
            results.append(result)

            # Print result
            if result.status == 'optimal':
                print(f"✓ {result.solve_time:.3f}s ({result.iterations} iters)")
            elif result.error:
                print(f"✗ ERROR: {result.error[:40]}")
            else:
                print(f"✗ {result.status}")

    return results


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description='POGS Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all benchmarks
  %(prog)s --quick                  # Quick test
  %(prog)s --problem lasso          # Only Lasso problems
  %(prog)s --solver POGS SCS        # Only test POGS and SCS
  %(prog)s --trials 3               # 3 trials per solver
  %(prog)s --verbose                # Show solver output
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick test (small problems, 2 trials)')
    parser.add_argument('--problem', type=str, choices=list(PROBLEM_CONFIGS.keys()),
                       help='Run specific problem class only')
    parser.add_argument('--solver', nargs='+',
                       help='Test specific solvers only (e.g., POGS SCS ECOS)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per solver (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print solver output')

    args = parser.parse_args()

    # Select configuration
    if args.quick:
        configs = QUICK_CONFIGS
        n_trials = 2
        print("Running QUICK benchmarks (small problems, 2 trials)\n")
    else:
        configs = PROBLEM_CONFIGS
        n_trials = args.trials
        print(f"Running FULL benchmarks ({n_trials} trials per solver)\n")

    # Filter by problem class if specified
    if args.problem:
        configs = {args.problem: configs[args.problem]}

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run benchmarks
    all_results = []

    for problem_class, problem_list in configs.items():
        print(f"\n{'='*80}")
        print(f"Problem Class: {problem_class.upper()}")
        print(f"{'='*80}\n")

        for problem_name, generator, default_solvers in problem_list:
            # Use specified solvers or default
            solvers = args.solver if args.solver else default_solvers

            print(f"\n{problem_name}")
            print(f"{'-'*80}")

            results = run_problem_benchmarks(
                generator,
                solvers,
                n_trials=n_trials,
                verbose=args.verbose
            )
            all_results.extend(results)

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")

    save_results(all_results, 'results/latest.json')
    print(f"✓ Saved to results/latest.json")

    # Generate and print summary
    print_summary(all_results)

    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}\n")

    # Print recommendation
    print("Next steps:")
    print("  - View detailed results: results/latest.json")
    print("  - View summary: results/summary.txt")
    print("  - Run specific problems: python run_benchmarks.py --problem lasso")
    print("  - Quick test: python run_benchmarks.py --quick")


if __name__ == '__main__':
    main()
