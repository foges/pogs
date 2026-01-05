#!/usr/bin/env python3
"""
Model Predictive Control (MPC) Benchmark

MPC problems have graph-form structure:
    minimize    sum_t (x_t'Qx_t + u_t'Ru_t)
    subject to  x_{t+1} = Ax_t + Bu_t
                x_min <= x_t <= x_max
                u_min <= u_t <= u_max

This is a structured QP that POGS should handle well.
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
    print("CVXPY required")
    sys.exit(1)

try:
    from pogs_graph import Function, FunctionObj, _solve_graph_form

    HAS_POGS = True
except ImportError:
    HAS_POGS = False


def generate_mpc_problem(n_state, n_input, horizon, seed=42):
    """Generate a random stable MPC problem."""
    np.random.seed(seed)

    # Random stable dynamics (spectral radius < 1)
    A = np.random.randn(n_state, n_state)
    A = A / (1.1 * np.max(np.abs(np.linalg.eigvals(A))))

    B = np.random.randn(n_state, n_input)

    # Cost matrices (positive definite)
    Q = np.eye(n_state)
    R = 0.1 * np.eye(n_input)

    # Constraints
    x_max = 10 * np.ones(n_state)
    x_min = -x_max
    u_max = 1.0 * np.ones(n_input)
    u_min = -u_max

    # Initial state
    x0 = np.random.randn(n_state)

    return A, B, Q, R, x_min, x_max, u_min, u_max, x0


def solve_mpc_cvxpy(A, B, Q, R, x_min, x_max, u_min, u_max, x0, horizon, solver):
    """Solve MPC with CVXPY."""
    n_state = A.shape[0]
    n_input = B.shape[1]

    # Decision variables
    x = cp.Variable((horizon + 1, n_state))
    u = cp.Variable((horizon, n_input))

    # Cost
    cost = 0
    for t in range(horizon):
        cost += cp.quad_form(x[t], Q) + cp.quad_form(u[t], R)
    cost += cp.quad_form(x[horizon], Q)  # Terminal cost

    # Constraints
    constraints = [x[0] == x0]  # Initial condition
    for t in range(horizon):
        constraints.append(x[t + 1] == A @ x[t] + B @ u[t])  # Dynamics
        constraints.append(x[t] >= x_min)
        constraints.append(x[t] <= x_max)
        constraints.append(u[t] >= u_min)
        constraints.append(u[t] <= u_max)
    constraints.append(x[horizon] >= x_min)
    constraints.append(x[horizon] <= x_max)

    prob = cp.Problem(cp.Minimize(cost), constraints)

    solver_map = {"OSQP": cp.OSQP, "SCS": cp.SCS, "CLARABEL": cp.CLARABEL}
    if solver not in solver_map:
        return None, None, "unavailable"

    try:
        t0 = time.perf_counter()
        prob.solve(solver=solver_map[solver], verbose=False)
        t = time.perf_counter() - t0
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return prob.value, t, "optimal"
        return None, t, prob.status
    except Exception as e:
        return None, None, str(e)[:50]


def solve_mpc_pogs(A, B, Q, R, x_min, x_max, u_min, u_max, x0, horizon):
    """
    Solve MPC by reformulating as graph-form.

    Stack all variables z = [x_0; u_0; x_1; u_1; ... ; x_T]
    Create constraint matrix for dynamics.

    This is more complex but shows POGS flexibility.
    """
    n_state = A.shape[0]
    n_input = B.shape[1]

    # Total variables: (T+1)*n_state + T*n_input
    n_x = (horizon + 1) * n_state
    n_u = horizon * n_input
    n_total = n_x + n_u

    # Build the big constraint matrix for dynamics
    # x_{t+1} = A*x_t + B*u_t  =>  x_{t+1} - A*x_t - B*u_t = 0

    # Variable ordering: [x_0, x_1, ..., x_T, u_0, u_1, ..., u_{T-1}]

    # Dynamics constraints: T * n_state equations
    n_dynamics = horizon * n_state
    # Initial condition: n_state equations (x_0 = x0)
    n_init = n_state

    # Total rows in constraint matrix
    m = n_dynamics + n_init

    # Build constraint matrix
    C = np.zeros((m, n_total))
    d = np.zeros(m)

    # Initial condition: x_0 = x0
    C[:n_init, :n_state] = np.eye(n_state)
    d[:n_init] = x0

    # Dynamics: x_{t+1} - A*x_t - B*u_t = 0
    for t in range(horizon):
        row_start = n_init + t * n_state
        # x_{t+1}
        C[row_start : row_start + n_state, (t + 1) * n_state : (t + 2) * n_state] = np.eye(n_state)
        # -A*x_t
        C[row_start : row_start + n_state, t * n_state : (t + 1) * n_state] = -A
        # -B*u_t
        u_col_start = n_x + t * n_input
        C[row_start : row_start + n_state, u_col_start : u_col_start + n_input] = -B

    # Now we have equality constraints C*z = d

    # Cost: sum_t x_t'*Q*x_t + u_t'*R*u_t
    # This is separable! Each x_t has cost x_t'*Q*x_t, each u_t has cost u_t'*R*u_t

    # Graph form: y = C*z, with f_i(y_i) = I(y_i = d_i), g_j = cost + box constraints

    # f: equality constraints
    f = [FunctionObj(Function.kIndEq0, 1.0, d[i], 1.0) for i in range(m)]

    # g: cost + box constraints
    g = []

    # For x variables: cost = 0.5*x'*Q*x (diagonal Q = I), box constraints
    Q_diag = np.diag(Q)
    for t in range(horizon + 1):
        for i in range(n_state):
            # 0.5*Q_ii*x_i^2 + I(x_min <= x_i <= x_max)
            # Use kSquare with c=Q_ii and box via scaling
            # Actually need to combine... use kIndBox01 with quadratic penalty e
            scale = 1.0 / (x_max[i] - x_min[i])
            g.append(
                FunctionObj(Function.kIndBox01, scale, x_min[i] * scale, 1.0, 0.0, Q_diag[i] / 2)
            )

    # For u variables: cost = 0.5*u'*R*u, box constraints
    R_diag = np.diag(R)
    for t in range(horizon):
        for i in range(n_input):
            scale = 1.0 / (u_max[i] - u_min[i])
            g.append(
                FunctionObj(Function.kIndBox01, scale, u_min[i] * scale, 1.0, 0.0, R_diag[i] / 2)
            )

    t0 = time.perf_counter()
    result = _solve_graph_form(C, f, g, abs_tol=1e-4, rel_tol=1e-4, max_iter=10000, verbose=0)
    solve_time = time.perf_counter() - t0

    if result["status"] == 0:
        return result["optval"], solve_time, "optimal"
    return None, solve_time, f"status={result['status']}"


def run_benchmark():
    print("=" * 75)
    print("MODEL PREDICTIVE CONTROL (MPC) BENCHMARK")
    print("=" * 75)
    print()
    print("Problem: min Σ(x'Qx + u'Ru)  s.t. x_{t+1}=Ax_t+Bu_t, bounds on x,u")
    print()

    solvers = ["POGS", "OSQP", "SCS", "CLARABEL"]
    results = []

    # Test configurations: (n_state, n_input, horizon)
    configs = [
        (4, 2, 10),  # Small
        (4, 2, 50),  # Medium horizon
        (10, 4, 20),  # Medium state
        (10, 4, 50),  # Large
        (20, 8, 30),  # Larger state
    ]

    for n_state, n_input, horizon in configs:
        n_vars = (horizon + 1) * n_state + horizon * n_input
        n_eq = horizon * n_state + n_state

        print(f"\nState={n_state}, Input={n_input}, Horizon={horizon}")
        print(f"  Variables: {n_vars}, Equality constraints: {n_eq}")

        A, B, Q, R, x_min, x_max, u_min, u_max, x0 = generate_mpc_problem(n_state, n_input, horizon)

        times = {}
        for solver in solvers:
            if solver == "POGS":
                if HAS_POGS:
                    val, t, status = solve_mpc_pogs(
                        A, B, Q, R, x_min, x_max, u_min, u_max, x0, horizon
                    )
                else:
                    _val, t, status = None, None, "unavailable"
            else:
                _val, t, status = solve_mpc_cvxpy(
                    A, B, Q, R, x_min, x_max, u_min, u_max, x0, horizon, solver
                )

            if t is not None:
                times[solver] = t
                print(f"  {solver:12s}: {t * 1000:8.1f}ms  ({status})")
            else:
                print(f"  {solver:12s}: FAILED ({status})")

        if times:
            winner = min(times, key=times.get)
            print(f"  Winner: {winner}")
            results.append((n_state, n_input, horizon, times, winner))

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    pogs_wins = sum(1 for r in results if r[4] == "POGS")
    total = len(results)

    print(f"\nPOGS wins: {pogs_wins}/{total} ({100 * pogs_wins / total:.0f}%)" if total else "")

    print("\nPOGS vs best competitor:")
    for n_state, n_input, horizon, times, winner in results:
        pogs_t = times.get("POGS")
        if pogs_t:
            best_other = min(t for s, t in times.items() if s != "POGS")
            ratio = best_other / pogs_t
            config = f"({n_state},{n_input},{horizon})"
            if ratio > 1:
                print(f"  MPC{config}: POGS {ratio:.1f}x faster")
            else:
                print(f"  MPC{config}: POGS {1 / ratio:.1f}x slower")


if __name__ == "__main__":
    run_benchmark()
