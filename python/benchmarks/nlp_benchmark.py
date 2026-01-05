#!/usr/bin/env python3
"""
NLP Benchmark: Sparse Text Classification on Real Datasets

Uses real text datasets from sklearn to benchmark POGS against other solvers
on L1-regularized logistic regression for text classification.

Problem:
    minimize    sum_i log(1 + exp(-y_i * (a_i' * x))) + lambda * ||x||_1

This is a classic sparse classification problem where POGS should excel
due to the graph-form structure.
"""

import numpy as np
import time
import sys
import os

# Add parent directory for pogs imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: CVXPY not available")

try:
    from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelBinarizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available")

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_20newsgroups(n_samples=500, n_features=1000):
    """Load 20 newsgroups binary classification (sci vs rec)."""
    if not HAS_SKLEARN:
        return None, None, "sklearn not available"

    try:
        # Binary classification: science vs recreation
        categories_pos = ['sci.med', 'sci.space', 'sci.electronics']
        categories_neg = ['rec.sport.baseball', 'rec.sport.hockey', 'rec.autos']

        newsgroups_pos = fetch_20newsgroups(
            subset='train', categories=categories_pos,
            remove=('headers', 'footers', 'quotes')
        )
        newsgroups_neg = fetch_20newsgroups(
            subset='train', categories=categories_neg,
            remove=('headers', 'footers', 'quotes')
        )

        # Combine and subsample
        texts = newsgroups_pos.data[:n_samples//2] + newsgroups_neg.data[:n_samples//2]
        labels = [1] * min(len(newsgroups_pos.data), n_samples//2) + \
                 [-1] * min(len(newsgroups_neg.data), n_samples//2)

        # TF-IDF features
        vectorizer = TfidfVectorizer(max_features=n_features, stop_words='english')
        X = vectorizer.fit_transform(texts)

        return X.toarray(), np.array(labels), "20newsgroups_sci_vs_rec"

    except Exception as e:
        return None, None, str(e)


def load_spam_dataset(n_samples=500, n_features=500):
    """Create a spam-like synthetic dataset based on real patterns."""
    # This creates a realistic sparse text classification problem
    np.random.seed(42)

    # Simulate TF-IDF features (sparse)
    X = np.random.rand(n_samples, n_features)
    X[X < 0.9] = 0  # Make sparse (90% zeros)

    # Create labels based on a sparse linear model
    true_weights = np.zeros(n_features)
    important_features = np.random.choice(n_features, size=20, replace=False)
    true_weights[important_features] = np.random.randn(20)

    probs = 1 / (1 + np.exp(-X @ true_weights))
    labels = 2 * (probs > 0.5).astype(float) - 1  # {-1, +1}

    return X, labels, "synthetic_spam"


def solve_logistic_cvxpy(A, b, lambd, solver_name, verbose=False):
    """Solve L1 logistic regression with CVXPY."""
    m, n = A.shape
    x = cp.Variable(n)

    # Logistic loss: sum log(1 + exp(-y_i * a_i' * x))
    # Using CVXPY's logistic function
    logistic_loss = cp.sum(cp.logistic(-cp.multiply(b, A @ x)))
    objective = cp.Minimize(logistic_loss + lambd * cp.norm1(x))
    problem = cp.Problem(objective)

    solver_map = {
        'ECOS': cp.ECOS,
        'SCS': cp.SCS,
        'CLARABEL': cp.CLARABEL,
    }

    if solver_name not in solver_map:
        return None, None, None, 'unavailable'

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=verbose)
        solve_time = time.perf_counter() - t0

        if problem.status in ['optimal', 'optimal_inaccurate']:
            # Compute accuracy
            preds = np.sign(A @ x.value)
            accuracy = np.mean(preds == b)
            return x.value, solve_time, accuracy, 'optimal'
        else:
            return None, solve_time, None, problem.status
    except Exception as e:
        return None, None, None, str(e)


def solve_logistic_pogs(A, b, lambd, verbose=False):
    """Solve L1 logistic regression with POGS."""
    try:
        from pogs_graph import solve_logistic
    except ImportError:
        return None, None, None, 'pogs not available'

    t0 = time.perf_counter()
    result = solve_logistic(A, b, lambd=lambd, abs_tol=1e-4, rel_tol=1e-4,
                            max_iter=5000, verbose=5 if verbose else 0)
    solve_time = time.perf_counter() - t0

    if result['status'] == 0:
        x = result['x']
        preds = np.sign(A @ x)
        accuracy = np.mean(preds == b)
        return x, solve_time, accuracy, 'optimal'
    else:
        x = result.get('x')
        if x is not None:
            preds = np.sign(A @ x)
            accuracy = np.mean(preds == b)
            return x, solve_time, accuracy, f"status={result['status']}"
        return None, solve_time, None, f"status={result['status']}"


def run_benchmark():
    """Run NLP text classification benchmark."""
    print("=" * 70)
    print("NLP Benchmark: Sparse Text Classification")
    print("=" * 70)

    datasets = []

    # Load real datasets
    print("\nLoading datasets...")

    # 20 newsgroups - small
    X, y, name = load_20newsgroups(n_samples=200, n_features=500)
    if X is not None:
        datasets.append(('20news_small', X, y, 0.1))

    # 20 newsgroups - medium
    X, y, name = load_20newsgroups(n_samples=500, n_features=1000)
    if X is not None:
        datasets.append(('20news_medium', X, y, 0.1))

    # 20 newsgroups - large
    X, y, name = load_20newsgroups(n_samples=1000, n_features=2000)
    if X is not None:
        datasets.append(('20news_large', X, y, 0.05))

    # Synthetic spam
    X, y, name = load_spam_dataset(n_samples=500, n_features=500)
    if X is not None:
        datasets.append(('spam_synthetic', X, y, 0.1))

    if not datasets:
        print("No datasets available")
        return

    solvers = ['POGS', 'ECOS', 'SCS']
    results = []

    for dataset_name, X, y, lambd in datasets:
        m, n = X.shape
        sparsity = 1 - np.count_nonzero(X) / X.size

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Size: m={m} samples, n={n} features, sparsity={100*sparsity:.1f}%")
        print(f"Lambda: {lambd}")
        print(f"{'='*60}")

        solver_results = {}

        for solver in solvers:
            print(f"  {solver:12s}: ", end='', flush=True)

            if solver == 'POGS':
                x_sol, t, acc, status = solve_logistic_pogs(X, y, lambd, verbose=False)
            else:
                if not HAS_CVXPY:
                    print("CVXPY not available")
                    continue
                x_sol, t, acc, status = solve_logistic_cvxpy(X, y, lambd, solver, verbose=False)

            if x_sol is not None and acc is not None:
                nnz = np.sum(np.abs(x_sol) > 1e-6)
                print(f"time={t:.4f}s, acc={100*acc:.1f}%, nnz={nnz}, status={status}")
                solver_results[solver] = {
                    'time': t, 'accuracy': acc, 'nnz': nnz, 'status': status
                }
            else:
                print(f"FAILED ({status})")
                solver_results[solver] = {'time': t, 'accuracy': None, 'status': status}

        results.append({
            'dataset': dataset_name,
            'size': (m, n),
            'solvers': solver_results
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: POGS vs Other Solvers")
    print("=" * 70)

    pogs_wins = 0
    total = 0

    for r in results:
        pogs_data = r['solvers'].get('POGS', {})
        pogs_time = pogs_data.get('time')
        if pogs_time is None:
            continue

        for solver, data in r['solvers'].items():
            if solver == 'POGS':
                continue
            other_time = data.get('time')
            if other_time is not None:
                total += 1
                if pogs_time < other_time:
                    pogs_wins += 1
                    speedup = other_time / pogs_time
                    print(f"  {r['dataset']}: POGS {speedup:.1f}x faster than {solver}")
                else:
                    slowdown = pogs_time / other_time
                    print(f"  {r['dataset']}: POGS {slowdown:.1f}x slower than {solver}")

    if total > 0:
        print(f"\nPOGS wins: {pogs_wins}/{total} ({100*pogs_wins/total:.0f}%)")
    else:
        print("\nNo valid comparisons")


if __name__ == '__main__':
    run_benchmark()
