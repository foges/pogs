#!/usr/bin/env python3
"""
POGS Graph-Form Benchmark on Real Datasets

Benchmarks POGS graph-form solver against other solvers on real UCI/finance data.
Uses the graph-form solver directly for fair comparison.

POGS graph-form excels at:
- Dense matrices
- Separable objectives (Lasso, Ridge, Logistic, etc.)
- Modest accuracy (1e-4)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    from pogs_graph import solve_lasso, solve_ridge, solve_logistic, solve_elastic_net
    HAS_POGS_GRAPH = True
except ImportError:
    HAS_POGS_GRAPH = False
    print("Error: pogs_graph not available")

try:
    from sklearn.datasets import load_diabetes, load_wine, fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def load_uci_datasets():
    """Load real UCI datasets."""
    datasets = []

    if HAS_SKLEARN:
        # Diabetes
        try:
            data = load_diabetes()
            X = StandardScaler().fit_transform(data.data)
            y = (data.target - data.target.mean()) / data.target.std()
            datasets.append(('Diabetes', X, y))
        except:
            pass

        # Wine (classification -> regression on quality)
        try:
            data = load_wine()
            X = StandardScaler().fit_transform(data.data)
            y = (data.target - data.target.mean()) / (data.target.std() + 1e-8)
            datasets.append(('Wine', X, y))
        except:
            pass

    return datasets


def load_stock_data(n_stocks=50, period='2y'):
    """Load real S&P 500 stock returns."""
    if not HAS_YFINANCE:
        return None, None

    # Top S&P 500 stocks
    sp500_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
        'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
        'PEP', 'KO', 'COST', 'AVGO', 'MCD', 'WMT', 'CSCO', 'TMO', 'ACN', 'ABT',
        'DHR', 'NEE', 'DIS', 'PM', 'VZ', 'ADBE', 'CMCSA', 'NKE', 'TXN', 'CRM',
        'WFC', 'BMY', 'RTX', 'UPS', 'HON', 'QCOM', 'COP', 'MS', 'AMGN', 'T'
    ][:n_stocks]

    try:
        data = yf.download(sp500_tickers, period=period, progress=False)['Adj Close']
        returns = data.pct_change().dropna()

        if len(returns) < 100:
            return None, None

        # Covariance matrix
        cov = returns.cov().values
        mean_ret = returns.mean().values

        return cov, mean_ret

    except Exception as e:
        print(f"Stock data error: {e}")
        return None, None


def solve_lasso_cvxpy(X, y, lambd, solver_name):
    """Solve Lasso with CVXPY."""
    m, n = X.shape
    w = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lambd * cp.norm1(w))
    problem = cp.Problem(objective)

    solver_map = {'OSQP': cp.OSQP, 'SCS': cp.SCS, 'CLARABEL': cp.CLARABEL}
    if solver_name not in solver_map:
        return None, None

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=False)
        t = time.perf_counter() - t0

        if problem.status in ['optimal', 'optimal_inaccurate']:
            return problem.value, t
    except:
        pass
    return None, None


def solve_ridge_cvxpy(X, y, lambd, solver_name):
    """Solve Ridge with CVXPY."""
    m, n = X.shape
    w = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + 0.5 * lambd * cp.sum_squares(w))
    problem = cp.Problem(objective)

    solver_map = {'OSQP': cp.OSQP, 'SCS': cp.SCS, 'CLARABEL': cp.CLARABEL}
    if solver_name not in solver_map:
        return None, None

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=False)
        t = time.perf_counter() - t0

        if problem.status in ['optimal', 'optimal_inaccurate']:
            return problem.value, t
    except:
        pass
    return None, None


def solve_portfolio_cvxpy(cov, lambd, solver_name):
    """Solve L1-regularized portfolio optimization."""
    n = cov.shape[0]
    w = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.quad_form(w, cov) + lambd * cp.norm1(w))
    constraints = [cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)

    solver_map = {'OSQP': cp.OSQP, 'SCS': cp.SCS, 'CLARABEL': cp.CLARABEL}
    if solver_name not in solver_map:
        return None, None

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=False)
        t = time.perf_counter() - t0

        if problem.status in ['optimal', 'optimal_inaccurate']:
            return problem.value, t
    except:
        pass
    return None, None


def run_benchmark():
    """Run comprehensive graph-form benchmark."""
    print("=" * 75)
    print("POGS GRAPH-FORM BENCHMARK ON REAL DATA")
    print("=" * 75)
    print()

    if not HAS_POGS_GRAPH:
        print("ERROR: pogs_graph not available")
        return

    results = []
    solvers = ['POGS', 'OSQP', 'SCS', 'CLARABEL']

    # === LASSO on UCI datasets ===
    print("=" * 75)
    print("LASSO REGRESSION: min 0.5||Xw - y||² + λ||w||₁")
    print("=" * 75)

    datasets = load_uci_datasets()
    for name, X, y in datasets:
        m, n = X.shape
        lambd = 0.1 * np.linalg.norm(X.T @ y, np.inf)  # Standard lambda choice

        print(f"\n{name} ({m}x{n}), λ={lambd:.4f}")

        times = {}
        for solver in solvers:
            if solver == 'POGS':
                t0 = time.perf_counter()
                result = solve_lasso(X, y, lambd, verbose=0)
                t = time.perf_counter() - t0
                if result['status'] == 0:
                    times[solver] = t
                    print(f"  {solver:12s}: {t*1000:8.1f}ms (iter={result['iterations']})")
                else:
                    print(f"  {solver:12s}: FAILED")
            else:
                if HAS_CVXPY:
                    _, t = solve_lasso_cvxpy(X, y, lambd, solver)
                    if t:
                        times[solver] = t
                        print(f"  {solver:12s}: {t*1000:8.1f}ms")
                    else:
                        print(f"  {solver:12s}: FAILED")

        if times:
            winner = min(times, key=times.get)
            print(f"  Winner: {winner}")
            results.append({
                'problem': f'Lasso_{name}',
                'times': times,
                'winner': winner
            })

    # === RIDGE on UCI datasets ===
    print("\n" + "=" * 75)
    print("RIDGE REGRESSION: min 0.5||Xw - y||² + 0.5λ||w||²")
    print("=" * 75)

    for name, X, y in datasets:
        m, n = X.shape
        lambd = 1.0

        print(f"\n{name} ({m}x{n}), λ={lambd:.4f}")

        times = {}
        for solver in solvers:
            if solver == 'POGS':
                t0 = time.perf_counter()
                result = solve_ridge(X, y, lambd, verbose=0)
                t = time.perf_counter() - t0
                if result['status'] == 0:
                    times[solver] = t
                    print(f"  {solver:12s}: {t*1000:8.1f}ms (iter={result['iterations']})")
                else:
                    print(f"  {solver:12s}: FAILED")
            else:
                if HAS_CVXPY:
                    _, t = solve_ridge_cvxpy(X, y, lambd, solver)
                    if t:
                        times[solver] = t
                        print(f"  {solver:12s}: {t*1000:8.1f}ms")
                    else:
                        print(f"  {solver:12s}: FAILED")

        if times:
            winner = min(times, key=times.get)
            print(f"  Winner: {winner}")
            results.append({
                'problem': f'Ridge_{name}',
                'times': times,
                'winner': winner
            })

    # === Portfolio optimization on S&P 500 ===
    print("\n" + "=" * 75)
    print("SPARSE PORTFOLIO: min 0.5 w'Σw + λ||w||₁  s.t. 1'w = 1")
    print("=" * 75)

    for n_stocks in [20, 50]:
        cov, _ = load_stock_data(n_stocks, '2y')
        if cov is None:
            continue

        for lambd in [0.01, 0.1]:
            print(f"\nS&P500 ({n_stocks} stocks), λ={lambd}")

            times = {}
            for solver in solvers:
                if solver == 'POGS':
                    # For portfolio, we need to handle the constraint 1'w = 1
                    # This is harder with graph-form, skip for now
                    print(f"  {solver:12s}: (constrained, using CVXPY)")
                    continue
                else:
                    if HAS_CVXPY:
                        _, t = solve_portfolio_cvxpy(cov, lambd, solver)
                        if t:
                            times[solver] = t
                            print(f"  {solver:12s}: {t*1000:8.1f}ms")
                        else:
                            print(f"  {solver:12s}: FAILED")

            if times:
                winner = min(times, key=times.get)
                print(f"  Winner: {winner}")

    # === Summary ===
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    pogs_wins = sum(1 for r in results if r['winner'] == 'POGS')
    total = len(results)

    print(f"\nPOGS wins: {pogs_wins}/{total} ({100*pogs_wins/total:.0f}%)" if total > 0 else "No results")

    # Show speedups
    print("\nPOGS speedups:")
    for r in results:
        pogs_time = r['times'].get('POGS')
        if pogs_time:
            for solver, t in r['times'].items():
                if solver != 'POGS' and t:
                    ratio = t / pogs_time
                    if ratio > 1:
                        print(f"  {r['problem']}: {ratio:.1f}x faster than {solver}")
                    else:
                        print(f"  {r['problem']}: {1/ratio:.1f}x slower than {solver}")


if __name__ == '__main__':
    run_benchmark()
