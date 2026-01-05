#!/usr/bin/env python3
"""
Challenging Constrained Optimization Benchmarks

Problems where graph-form structure + constraints matter:
1. Constrained Lasso - Lasso with box constraints and sum constraints
2. Robust regression with outlier bounds - Huber + constraints
3. Long-short portfolio - Turnover limits, sector constraints
4. Quantile regression - Non-smooth loss with constraints
5. Basis pursuit with measurement noise bounds
"""

import os
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("CVXPY required for this benchmark")
    sys.exit(1)

try:
    from pogs_graph import Function, FunctionObj, _solve_graph_form

    HAS_POGS = True
except ImportError:
    HAS_POGS = False
    print("Warning: pogs_graph not available")


def solve_with_cvxpy(problem, solver_name):
    """Solve a CVXPY problem with given solver."""
    solver_map = {
        "OSQP": cp.OSQP,
        "SCS": cp.SCS,
        "CLARABEL": cp.CLARABEL,
        "ECOS": cp.ECOS,
    }
    if solver_name not in solver_map:
        return None, None, "unavailable"

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=False)
        t = time.perf_counter() - t0
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return problem.value, t, "optimal"
        return None, t, problem.status
    except Exception as e:
        return None, None, str(e)[:50]


# =============================================================================
# Problem 1: Constrained Lasso with box + simplex constraints
# =============================================================================
def constrained_lasso_cvxpy(A, b, lambd, lb, ub, solver):
    """
    min  0.5||Ax - b||^2 + λ||x||_1
    s.t. lb <= x <= ub
         sum(x) = 1
    """
    _m, n = A.shape
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm1(x))
    constraints = [x >= lb, x <= ub, cp.sum(x) == 1]
    prob = cp.Problem(obj, constraints)
    return solve_with_cvxpy(prob, solver)


def constrained_lasso_pogs(A, b, lambd, lb, ub):
    """
    Reformulate as graph-form with auxiliary variables.

    min  0.5||y - b||^2 + λ||z||_1
    s.t. y = Ax, z = x, lb <= x <= ub, sum(x) = 1

    Stack: [y; z; s] = [A; I; 1'] x
    where s is scalar for sum constraint
    """
    m, n = A.shape

    # Build stacked matrix [A; I; 1']
    ones_row = np.ones((1, n))
    A_stacked = np.vstack([A, np.eye(n), ones_row])
    m + n + 1

    # f functions:
    # - First m: 0.5*(y_i - b_i)^2  (kSquare)
    # - Next n: λ|z_i|  (kAbs)
    # - Last 1: I(s = 1)  (kIndEq0 with b=1)
    f = []
    for i in range(m):
        f.append(FunctionObj(Function.kSquare, 1.0, b[i], 1.0))
    for i in range(n):
        f.append(FunctionObj(Function.kAbs, 1.0, 0.0, lambd))
    # Equality constraint: s = 1 => I(s - 1 = 0)
    f.append(FunctionObj(Function.kIndEq0, 1.0, 1.0, 1.0))

    # g functions: box constraints lb <= x <= ub
    # Use kIndBox01 scaled: I(lb <= x <= ub) = I(0 <= (x-lb)/(ub-lb) <= 1)
    g = []
    for i in range(n):
        # Scale to [0,1]: a*(x - b) where a = 1/(ub-lb), b = lb
        scale = 1.0 / (ub - lb) if ub > lb else 1.0
        g.append(FunctionObj(Function.kIndBox01, scale, lb * scale, 1.0))

    t0 = time.perf_counter()
    result = _solve_graph_form(
        A_stacked, f, g, abs_tol=1e-4, rel_tol=1e-4, max_iter=10000, verbose=0
    )
    t = time.perf_counter() - t0

    if result["status"] == 0:
        return result["optval"], t, "optimal"
    return None, t, f"status={result['status']}"


# =============================================================================
# Problem 2: Robust Huber regression with bounded residuals
# =============================================================================
def robust_regression_cvxpy(A, b, delta, max_resid, solver):
    """
    min  sum huber(Ax - b, delta)
    s.t. ||Ax - b||_inf <= max_resid
    """
    _m, n = A.shape
    x = cp.Variable(n)
    obj = cp.Minimize(cp.sum(cp.huber(A @ x - b, delta)))
    constraints = [cp.norm_inf(A @ x - b) <= max_resid]
    prob = cp.Problem(obj, constraints)
    return solve_with_cvxpy(prob, solver)


def robust_regression_pogs(A, b, delta, max_resid):
    """
    Reformulate with auxiliary:
    min  sum huber(y, delta)
    s.t. y = Ax - b, -max_resid <= y <= max_resid

    This is tricky - need to handle offset b and bound constraints.
    Use: y = Ax, then huber(y - b) with bounds on y.
    """
    _m, _n = A.shape

    # f: huber(y_i - b_i) with |y_i - b_i| <= max_resid
    # The bound constraint can be encoded in the function if we're clever
    # For now, this is hard to do purely in graph-form without slack variables

    # Alternative: augment with slack
    # [y; s] = [A; A] x, minimize huber(y-b) + I(|s-b| <= max_resid)
    # But this doubles the problem size...

    return None, None, "reformulation complex"


# =============================================================================
# Problem 3: Long-Short Portfolio with Turnover Constraint
# =============================================================================
def longshort_portfolio_cvxpy(Sigma, mu, w_prev, gamma, max_turnover, solver):
    """
    min  0.5 * w'Σw - γ * μ'w
    s.t. sum(w) = 0  (market neutral)
         ||w||_1 <= 2  (leverage constraint)
         ||w - w_prev||_1 <= max_turnover  (turnover constraint)
    """
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(w, Sigma) - gamma * mu @ w)
    constraints = [cp.sum(w) == 0, cp.norm1(w) <= 2, cp.norm1(w - w_prev) <= max_turnover]
    prob = cp.Problem(obj, constraints)
    return solve_with_cvxpy(prob, solver)


# =============================================================================
# Problem 4: Quantile Regression
# =============================================================================
def quantile_regression_cvxpy(A, b, tau, solver):
    """
    min  sum rho_tau(Ax - b)
    where rho_tau(u) = u*(tau - I(u<0)) = tau*max(u,0) + (1-tau)*max(-u,0)
    """
    _m, n = A.shape
    x = cp.Variable(n)
    residual = A @ x - b
    # Quantile loss: tau*pos(r) + (1-tau)*neg(r) = tau*max(r,0) + (1-tau)*max(-r,0)
    obj = cp.Minimize(tau * cp.sum(cp.pos(residual)) + (1 - tau) * cp.sum(cp.neg(residual)))
    prob = cp.Problem(obj)
    return solve_with_cvxpy(prob, solver)


def quantile_regression_pogs(A, b, tau):
    """
    Graph form: y = Ax
    f_i(y_i) = tau*max(y_i - b_i, 0) + (1-tau)*max(-(y_i - b_i), 0)

    Using kMaxPos0 and kMaxNeg0:
    = tau * maxpos(y - b) + (1-tau) * maxneg(y - b)

    But we can only have one function per component. Need reformulation.
    """
    m, n = A.shape

    # Reformulate: [y; z] = [A; A] x
    # f for y: tau * max(y - b, 0)  => kMaxPos0 with offset
    # f for z: (1-tau) * max(-(z - b), 0) = (1-tau) * max(b - z, 0)  => kMaxPos0 with sign flip

    A_stacked = np.vstack([A, A])

    f = []
    for i in range(m):
        # tau * max(y_i - b_i, 0): kMaxPos0 with a=1, b=b_i, c=tau
        f.append(FunctionObj(Function.kMaxPos0, 1.0, b[i], tau))
    for i in range(m):
        # (1-tau) * max(b_i - z_i, 0) = (1-tau) * max(-z_i + b_i, 0)
        # kMaxPos0 with a=-1, b=-b_i, c=(1-tau)
        f.append(FunctionObj(Function.kMaxPos0, -1.0, -b[i], 1 - tau))

    g = [FunctionObj(Function.kZero) for _ in range(n)]

    t0 = time.perf_counter()
    result = _solve_graph_form(
        A_stacked, f, g, abs_tol=1e-4, rel_tol=1e-4, max_iter=5000, verbose=0
    )
    t = time.perf_counter() - t0

    if result["status"] == 0:
        return result["optval"], t, "optimal"
    return None, t, f"status={result['status']}"


# =============================================================================
# Problem 5: Basis Pursuit Denoising with noise bound
# =============================================================================
def bpdn_cvxpy(A, b, epsilon, solver):
    """
    min  ||x||_1
    s.t. ||Ax - b||_2 <= epsilon
    """
    _m, n = A.shape
    x = cp.Variable(n)
    obj = cp.Minimize(cp.norm1(x))
    constraints = [cp.norm2(A @ x - b) <= epsilon]
    prob = cp.Problem(obj, constraints)
    return solve_with_cvxpy(prob, solver)


# =============================================================================
# Problem 6: Elastic Net with non-negativity
# =============================================================================
def nonneg_elastic_net_cvxpy(A, b, lambda1, lambda2, solver):
    """
    min  0.5||Ax - b||^2 + λ1||x||_1 + 0.5*λ2||x||^2
    s.t. x >= 0
    """
    _m, n = A.shape
    x = cp.Variable(n)
    obj = cp.Minimize(
        0.5 * cp.sum_squares(A @ x - b) + lambda1 * cp.norm1(x) + 0.5 * lambda2 * cp.sum_squares(x)
    )
    constraints = [x >= 0]
    prob = cp.Problem(obj, constraints)
    return solve_with_cvxpy(prob, solver)


def nonneg_elastic_net_pogs(A, b, lambda1, lambda2):
    """
    Graph form: y = Ax
    f_i(y_i) = 0.5*(y_i - b_i)^2
    g_j(x_j) = λ1|x_j| + 0.5*λ2*x_j^2 + I(x_j >= 0)

    For non-negative elastic net, use kAbs with quadratic term and non-neg indicator.
    This requires combining functions which isn't directly supported.

    Alternative: since x >= 0, |x| = x, so:
    g_j(x_j) = λ1*x_j + 0.5*λ2*x_j^2 + I(x_j >= 0)
             = 0.5*λ2*x_j^2 + λ1*x_j + I(x_j >= 0)

    Use kSquare with linear term d=λ1 and indicator... but kSquare doesn't have indicator.
    Need to use kIndGe0 with quadratic penalty e.

    g_j = I(x >= 0) with d=λ1 and e=λ2/2
    """
    m, n = A.shape

    f = [FunctionObj(Function.kSquare, 1.0, b[i], 1.0) for i in range(m)]
    # kIndGe0 with linear term d and quadratic term e
    g = [FunctionObj(Function.kIndGe0, 1.0, 0.0, 1.0, lambda1, lambda2 / 2) for _ in range(n)]

    t0 = time.perf_counter()
    result = _solve_graph_form(A, f, g, abs_tol=1e-4, rel_tol=1e-4, max_iter=5000, verbose=0)
    t = time.perf_counter() - t0

    if result["status"] == 0:
        return result["optval"], t, "optimal"
    return None, t, f"status={result['status']}"


# =============================================================================
# Main benchmark
# =============================================================================
def run_benchmark():
    print("=" * 75)
    print("CHALLENGING CONSTRAINED OPTIMIZATION BENCHMARKS")
    print("=" * 75)
    print()

    np.random.seed(42)
    solvers = ["POGS", "OSQP", "SCS", "CLARABEL"]
    results = []

    # Problem sizes
    sizes = [(100, 50), (500, 100), (1000, 200)]

    # === Problem 1: Constrained Lasso ===
    print("=" * 75)
    print("1. CONSTRAINED LASSO: min ||Ax-b||² + λ||x||₁  s.t. lb≤x≤ub, Σx=1")
    print("=" * 75)

    for m, n in sizes:
        A = np.random.randn(m, n)
        x_true = np.abs(np.random.randn(n))
        x_true = x_true / x_true.sum()  # Normalize to sum=1
        b = A @ x_true + 0.1 * np.random.randn(m)
        lambd = 0.01
        lb, ub = 0.0, 1.0

        print(f"\n  Size: {m}x{n}")
        times = {}

        for solver in solvers:
            if solver == "POGS":
                if HAS_POGS:
                    val, t, status = constrained_lasso_pogs(A, b, lambd, lb, ub)
                else:
                    val, t, status = None, None, "unavailable"
            else:
                val, t, status = constrained_lasso_cvxpy(A, b, lambd, lb, ub, solver)

            if t is not None:
                times[solver] = t
                print(f"    {solver:12s}: {t * 1000:8.1f}ms  ({status})")
            else:
                print(f"    {solver:12s}: FAILED ({status})")

        if times:
            winner = min(times, key=times.get)
            print(f"    Winner: {winner}")
            results.append(("ConstrainedLasso", m, n, times, winner))

    # === Problem 4: Quantile Regression ===
    print("\n" + "=" * 75)
    print("4. QUANTILE REGRESSION: min Σ ρ_τ(Ax - b)  (τ = 0.25, 0.75)")
    print("=" * 75)

    for m, n in sizes[:2]:  # Smaller sizes for this problem
        A = np.random.randn(m, n)
        x_true = np.random.randn(n)
        b = A @ x_true + 0.5 * np.random.randn(m)

        for tau in [0.25, 0.75]:
            print(f"\n  Size: {m}x{n}, τ={tau}")
            times = {}

            for solver in solvers:
                if solver == "POGS":
                    if HAS_POGS:
                        val, t, status = quantile_regression_pogs(A, b, tau)
                    else:
                        val, t, status = None, None, "unavailable"
                else:
                    val, t, status = quantile_regression_cvxpy(A, b, tau, solver)

                if t is not None:
                    times[solver] = t
                    print(f"    {solver:12s}: {t * 1000:8.1f}ms  ({status})")
                else:
                    print(f"    {solver:12s}: FAILED ({status})")

            if times:
                winner = min(times, key=times.get)
                print(f"    Winner: {winner}")
                results.append((f"Quantile_tau{tau}", m, n, times, winner))

    # === Problem 6: Non-negative Elastic Net ===
    print("\n" + "=" * 75)
    print("6. NON-NEGATIVE ELASTIC NET: min ||Ax-b||² + λ₁||x||₁ + λ₂||x||²  s.t. x≥0")
    print("=" * 75)

    for m, n in sizes:
        A = np.random.randn(m, n)
        x_true = np.abs(np.random.randn(n))
        x_true[x_true < 0.5] = 0
        b = A @ x_true + 0.1 * np.random.randn(m)
        lambda1, lambda2 = 0.1, 0.1

        print(f"\n  Size: {m}x{n}")
        times = {}

        for solver in solvers:
            if solver == "POGS":
                if HAS_POGS:
                    val, t, status = nonneg_elastic_net_pogs(A, b, lambda1, lambda2)
                else:
                    _val, t, status = None, None, "unavailable"
            else:
                _val, t, status = nonneg_elastic_net_cvxpy(A, b, lambda1, lambda2, solver)

            if t is not None:
                times[solver] = t
                print(f"    {solver:12s}: {t * 1000:8.1f}ms  ({status})")
            else:
                print(f"    {solver:12s}: FAILED ({status})")

        if times:
            winner = min(times, key=times.get)
            print(f"    Winner: {winner}")
            results.append(("NonnegElasticNet", m, n, times, winner))

    # === Summary ===
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    pogs_wins = sum(1 for r in results if r[4] == "POGS")
    total = len(results)

    print(f"\nResults: {len(results)} benchmarks")
    print(f"POGS wins: {pogs_wins}/{total} ({100 * pogs_wins / total:.0f}%)" if total > 0 else "")

    # Speedups
    print("\nPOGS performance:")
    for name, m, n, times, winner in results:
        pogs_t = times.get("POGS")
        if pogs_t:
            best_other = min(t for s, t in times.items() if s != "POGS")
            ratio = best_other / pogs_t
            if ratio > 1:
                print(f"  {name} ({m}x{n}): POGS {ratio:.1f}x faster")
            else:
                print(f"  {name} ({m}x{n}): POGS {1 / ratio:.1f}x slower")


if __name__ == "__main__":
    run_benchmark()
