#!/usr/bin/env python3
"""
Signal Processing Benchmark: Total Variation Denoising on Real Images

Uses real images from scikit-image's data module to benchmark
POGS against other solvers on total variation denoising problems.

TV denoising solves:
    minimize    0.5 * ||x - b||^2 + lambda * ||Dx||_1

where D is the finite difference operator (gradient).
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
    from skimage import data as skimage_data
    from skimage.color import rgb2gray
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available, using synthetic data")


def get_real_images():
    """Get real test images from scikit-image."""
    images = {}

    if HAS_SKIMAGE:
        # Camera - classic test image (512x512 grayscale)
        try:
            img = skimage_data.camera()
            images['camera_64'] = resize(img, (64, 64), anti_aliasing=True)
            images['camera_128'] = resize(img, (128, 128), anti_aliasing=True)
        except Exception:
            pass

        # Astronaut - color image converted to grayscale
        try:
            img = rgb2gray(skimage_data.astronaut())
            images['astronaut_64'] = resize(img, (64, 64), anti_aliasing=True)
            images['astronaut_128'] = resize(img, (128, 128), anti_aliasing=True)
        except Exception:
            pass

        # Coins - grayscale image with edges
        try:
            img = skimage_data.coins()
            images['coins_64'] = resize(img, (64, 64), anti_aliasing=True)
        except Exception:
            pass

        # Moon - grayscale with smooth regions
        try:
            img = skimage_data.moon()
            images['moon_64'] = resize(img, (64, 64), anti_aliasing=True)
        except Exception:
            pass

    # Fallback: synthetic piecewise constant signal
    if not images:
        print("Using synthetic test signals")
        # 1D piecewise constant
        n = 256
        x = np.zeros(n)
        x[50:100] = 1.0
        x[150:200] = -0.5
        x[220:240] = 0.8
        images['synthetic_1d'] = x

        # 2D blocks
        img = np.zeros((64, 64))
        img[10:30, 10:30] = 1.0
        img[35:55, 35:55] = 0.7
        img[15:45, 40:60] = -0.5
        images['synthetic_2d'] = img

    return images


def create_1d_difference_matrix(n):
    """Create 1D finite difference matrix D such that Dx gives gradients."""
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1
    return D


def create_2d_difference_matrix(height, width):
    """Create 2D finite difference matrix for image gradients."""
    n = height * width

    # Horizontal differences
    rows_h = []
    for i in range(height):
        for j in range(width - 1):
            idx = i * width + j
            rows_h.append((idx, idx, -1))
            rows_h.append((idx, idx + 1, 1))

    # Vertical differences
    rows_v = []
    for i in range(height - 1):
        for j in range(width):
            idx = i * width + j
            row_idx = len(rows_h) // 2 + i * width + j
            rows_v.append((row_idx, idx, -1))
            rows_v.append((row_idx, idx + width, 1))

    # Build sparse matrix
    m = (height - 1) * width + height * (width - 1)
    D = np.zeros((m, n))

    row = 0
    for i in range(height):
        for j in range(width - 1):
            idx = i * width + j
            D[row, idx] = -1
            D[row, idx + 1] = 1
            row += 1

    for i in range(height - 1):
        for j in range(width):
            idx = i * width + j
            D[row, idx] = -1
            D[row, idx + width] = 1
            row += 1

    return D


def solve_tv_cvxpy(b, D, lambd, solver_name, verbose=False):
    """Solve TV denoising with CVXPY."""
    n = b.size
    x = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.sum_squares(x - b.flatten()) + lambd * cp.norm1(D @ x))
    problem = cp.Problem(objective)

    solver_map = {
        'ECOS': cp.ECOS,
        'SCS': cp.SCS,
        'OSQP': cp.OSQP,
        'CLARABEL': cp.CLARABEL,
    }

    if solver_name not in solver_map:
        return None, None, 'unavailable'

    try:
        t0 = time.perf_counter()
        problem.solve(solver=solver_map[solver_name], verbose=verbose)
        solve_time = time.perf_counter() - t0

        if problem.status in ['optimal', 'optimal_inaccurate']:
            return x.value, solve_time, 'optimal'
        else:
            return None, solve_time, problem.status
    except Exception as e:
        return None, None, str(e)


def solve_tv_pogs(b, D, lambd, verbose=False):
    """Solve TV denoising with POGS graph-form solver."""
    try:
        from pogs_graph import solve_lasso, FunctionObj, Function, _solve_graph_form
    except ImportError:
        return None, None, 'pogs not available'

    # TV denoising: min 0.5||x - b||^2 + lambda*||Dx||_1
    # Reformulate as: min 0.5||x - b||^2 + lambda*||y||_1  s.t. y = Dx
    # This is graph form with A = [I; D], f = [square; abs], g = zero

    n = b.size
    m_D = D.shape[0]
    b_flat = b.flatten()

    # Stack: A = [I; D]
    A = np.vstack([np.eye(n), D])
    m = A.shape[0]

    # f_i for i < n: 0.5*(y_i - b_i)^2  => kSquare with b=b_i
    # f_i for i >= n: lambda*|y_i|  => kAbs with c=lambda
    f = []
    for i in range(n):
        f.append(FunctionObj(Function.kSquare, 1.0, b_flat[i], 1.0))
    for i in range(m_D):
        f.append(FunctionObj(Function.kAbs, 1.0, 0.0, lambd))

    # g_j: no constraint on x => kZero
    g = [FunctionObj(Function.kZero) for _ in range(n)]

    t0 = time.perf_counter()
    result = _solve_graph_form(A, f, g, abs_tol=1e-4, rel_tol=1e-4,
                                max_iter=5000, verbose=5 if verbose else 0)
    solve_time = time.perf_counter() - t0

    if result['status'] == 0:
        return result['x'], solve_time, 'optimal'
    else:
        return result['x'], solve_time, f"status={result['status']}"


def run_benchmark():
    """Run TV denoising benchmark on real images."""
    print("=" * 70)
    print("Signal Processing Benchmark: Total Variation Denoising")
    print("=" * 70)

    images = get_real_images()
    if not images:
        print("No images available for benchmark")
        return

    solvers = ['POGS', 'ECOS', 'SCS', 'OSQP']
    results = []

    # Noise level and regularization
    noise_std = 0.1

    for img_name, img in images.items():
        print(f"\n{'='*60}")
        print(f"Image: {img_name} (shape: {img.shape})")
        print(f"{'='*60}")

        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)

        # Add noise
        np.random.seed(42)
        noisy = img + noise_std * np.random.randn(*img.shape)

        # Create difference matrix
        if img.ndim == 1:
            D = create_1d_difference_matrix(len(img))
            n = len(img)
        else:
            D = create_2d_difference_matrix(img.shape[0], img.shape[1])
            n = img.size

        # Set lambda based on noise level
        lambd = 0.1 * noise_std * np.sqrt(n)

        print(f"Problem size: n={n}, m_D={D.shape[0]}, lambda={lambd:.4f}")
        print(f"Noisy PSNR: {-10*np.log10(np.mean((noisy - img)**2)):.2f} dB")
        print()

        solver_results = {}

        for solver in solvers:
            print(f"  {solver:12s}: ", end='', flush=True)

            if solver == 'POGS':
                x_sol, t, status = solve_tv_pogs(noisy, D, lambd, verbose=False)
            else:
                if not HAS_CVXPY:
                    print("CVXPY not available")
                    continue
                x_sol, t, status = solve_tv_cvxpy(noisy, D, lambd, solver, verbose=False)

            if x_sol is not None:
                x_sol = x_sol.reshape(img.shape)
                mse = np.mean((x_sol - img) ** 2)
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                print(f"time={t:.4f}s, PSNR={psnr:.2f}dB, status={status}")
                solver_results[solver] = {'time': t, 'psnr': psnr, 'status': status}
            else:
                print(f"FAILED ({status})")
                solver_results[solver] = {'time': None, 'psnr': None, 'status': status}

        results.append({
            'image': img_name,
            'size': n,
            'solvers': solver_results
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pogs_wins = 0
    total = 0

    for r in results:
        pogs_time = r['solvers'].get('POGS', {}).get('time')
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
                    print(f"  {r['image']}: POGS {speedup:.1f}x faster than {solver}")
                else:
                    slowdown = pogs_time / other_time
                    print(f"  {r['image']}: POGS {slowdown:.1f}x slower than {solver}")

    if total > 0:
        print(f"\nPOGS wins: {pogs_wins}/{total} ({100*pogs_wins/total:.0f}%)")


if __name__ == '__main__':
    run_benchmark()
