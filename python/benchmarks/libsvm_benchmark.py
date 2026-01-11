#!/usr/bin/env python3
"""
Benchmark POGS on real industry datasets from LIBSVM.

LIBSVM datasets are standard benchmarks used in machine learning research.
These are REAL datasets, not randomly generated.

Datasets used:
- a1a-a9a: Adult income prediction (UCI)
- w1a-w8a: Web page classification
- rcv1: Reuters news classification
- real-sim: Text classification
- news20: 20 Newsgroups
- E2006-tfidf: Financial prediction

Reference: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
"""

from __future__ import annotations

import bz2
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
    from pogs_graph import solve_lasso, solve_ridge, solve_svm

    HAS_POGS = True
except ImportError as e:
    HAS_POGS = False
    print(f"Warning: POGS not available: {e}")


# LIBSVM dataset URLs (binary classification, sparse format)
LIBSVM_DATASETS = {
    # Small datasets
    "a1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a",
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
    "w1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w1a",
    "w8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
    # Medium datasets
    "mushrooms": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
    "phishing": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
    "madelon": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon",
    "gisette": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
    # Large datasets
    "rcv1": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
    "real-sim": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
    "news20": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
}

# Datasets to use in benchmark (ordered by size)
BENCHMARK_DATASETS = [
    "a1a",  # 1,605 x 123
    "a9a",  # 32,561 x 123
    "w1a",  # 2,477 x 300
    "mushrooms",  # 8,124 x 112
    "phishing",  # 11,055 x 68
]


@dataclass
class BenchmarkResult:
    """Result from solving a benchmark problem."""

    solver: str
    time_sec: float
    optval: float
    status: str
    iterations: int | None = None
    error: str | None = None


def get_cache_dir() -> Path:
    """Get cache directory for datasets."""
    cache = Path.home() / ".cache" / "pogs_benchmarks" / "libsvm"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_dataset(name: str) -> Path:
    """Download a LIBSVM dataset if not cached."""
    cache_dir = get_cache_dir()

    url = LIBSVM_DATASETS[name]
    is_compressed = url.endswith(".bz2")

    # Local filename
    local_name = name + (".bz2" if is_compressed else "")
    local_path = cache_dir / local_name
    final_path = cache_dir / name

    if final_path.exists():
        return final_path

    print(f"Downloading {name} from LIBSVM...")
    urllib.request.urlretrieve(url, local_path)

    # Decompress if needed
    if is_compressed:
        print(f"Decompressing {name}...")
        with bz2.open(local_path, "rt") as f_in:
            with open(final_path, "w") as f_out:
                f_out.write(f_in.read())
        local_path.unlink()  # Remove compressed file

    return final_path


def load_libsvm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a LIBSVM format file.

    Format: label index1:value1 index2:value2 ...

    Returns:
        X: Dense feature matrix (m, n)
        y: Label vector (m,) with values in {-1, +1}
    """
    rows = []
    labels = []
    max_idx = 0

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            label = float(parts[0])
            # Convert to {-1, +1}
            if label == 0:
                label = -1
            elif label > 0:
                label = 1
            else:
                label = -1
            labels.append(label)

            features = {}
            for item in parts[1:]:
                if ":" in item:
                    idx, val = item.split(":")
                    idx = int(idx)
                    features[idx] = float(val)
                    max_idx = max(max_idx, idx)
            rows.append(features)

    # Build dense matrix
    m = len(rows)
    n = max_idx
    X = np.zeros((m, n))
    for i, features in enumerate(rows):
        for idx, val in features.items():
            X[i, idx - 1] = val  # LIBSVM is 1-indexed

    y = np.array(labels)
    return X, y


def solve_lasso_cvxpy(
    X: np.ndarray, y: np.ndarray, lambd: float, solver_name: str, verbose: bool = False
) -> BenchmarkResult:
    """Solve Lasso using CVXPY with specified solver."""
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
        _m, n = X.shape
        w = cp.Variable(n)

        # Lasso: min 0.5||Xw - y||^2 + lambda * ||w||_1
        objective = 0.5 * cp.sum_squares(X @ w - y) + lambd * cp.norm1(w)
        prob = cp.Problem(cp.Minimize(objective))

        start = time.perf_counter()
        prob.solve(solver=solver_map[solver_name], verbose=verbose)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            solver=solver_name,
            time_sec=elapsed,
            optval=prob.value if prob.value is not None else float("nan"),
            status=prob.status,
            iterations=getattr(prob.solver_stats, "num_iters", None) if prob.solver_stats else None,
        )
    except Exception as e:
        return BenchmarkResult(
            solver=solver_name, time_sec=0, optval=float("nan"), status="error", error=str(e)
        )


def solve_lasso_pogs(
    X: np.ndarray, y: np.ndarray, lambd: float, verbose: bool = False
) -> BenchmarkResult:
    """Solve Lasso using POGS."""
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
        result = solve_lasso(X, y, lambd, verbose=1 if verbose else 0)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            solver="pogs",
            time_sec=elapsed,
            optval=result["optval"],
            status="optimal" if result["status"] == 0 else "error",
            iterations=result.get("iterations", result.get("num_iters")),
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


def run_benchmark():
    """Run benchmarks on LIBSVM datasets."""
    print("=" * 80)
    print("POGS Benchmark on LIBSVM Industry Datasets")
    print("=" * 80)
    print()
    print("These are REAL datasets from UCI ML Repository and other sources.")
    print("Reference: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")
    print()

    solvers = ["pogs", "osqp", "scs", "clarabel"]
    all_results = []

    # Header
    print(f"{'Dataset':<15} {'Size':>12} |", end="")
    for solver in solvers:
        print(f" {solver:>10}", end="")
    print(" | Winner")
    print("-" * (15 + 13 + 11 * len(solvers) + 10))

    for dataset_name in BENCHMARK_DATASETS:
        try:
            # Download and load dataset
            path = download_dataset(dataset_name)
            X, y = load_libsvm(path)
            m, n = X.shape

            # Regularization parameter (typical choice)
            lambd = 0.1 * np.linalg.norm(X.T @ y, np.inf) / m

            results = {}

            # Solve with each solver
            for solver_name in solvers:
                if solver_name == "pogs":
                    result = solve_lasso_pogs(X, y, lambd)
                else:
                    result = solve_lasso_cvxpy(X, y, lambd, solver_name)
                results[solver_name] = result
                all_results.append((dataset_name, m, n, result))

            # Print row
            print(f"{dataset_name:<15} {m:>5}x{n:<5} |", end="")
            times = {}
            for solver in solvers:
                r = results[solver]
                if r.status in ["optimal", "optimal_inaccurate"]:
                    print(f" {r.time_sec * 1000:>8.1f}ms", end="")
                    times[solver] = r.time_sec
                else:
                    print(f" {'FAIL':>10}", end="")

            # Determine winner
            if times:
                winner = min(times, key=times.get)
                if winner == "pogs" and len(times) > 1:
                    others = [t for s, t in times.items() if s != "pogs"]
                    speedup = min(others) / times["pogs"]
                    print(f" | {winner} ({speedup:.1f}x)")
                else:
                    print(f" | {winner}")
            else:
                print(" | N/A")

        except Exception as e:
            print(f"{dataset_name:<15} ERROR: {e}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count wins
    wins = dict.fromkeys(solvers, 0)
    times_by_solver = {s: [] for s in solvers}

    from collections import defaultdict

    datasets_results = defaultdict(dict)
    for dname, m, n, result in all_results:
        datasets_results[dname][result.solver] = result
        if result.status in ["optimal", "optimal_inaccurate"]:
            times_by_solver[result.solver].append(result.time_sec)

    for dname, dresults in datasets_results.items():
        valid_times = {
            s: r.time_sec
            for s, r in dresults.items()
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

    # POGS speedup summary
    print("\nPOGS Speedups (vs second-best solver):")
    pogs_speedups = []
    for dname, dresults in datasets_results.items():
        valid_times = {
            s: r.time_sec
            for s, r in dresults.items()
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
    run_benchmark()
