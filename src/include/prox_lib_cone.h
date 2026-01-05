#ifndef PROX_LIB_CONE_H_
#define PROX_LIB_CONE_H_

#include <assert.h>

#include <set>
#include <vector>
#include <iostream>

#ifdef __CUDACC__
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#endif

#include "interface_defs.h"
#include "prox_tools.h"
#include "util.h"

typedef unsigned int CONE_IDX;

enum Cone { kConeZero,       // { x : x = 0 }
            kConeNonNeg,     // { x : x >= 0 }
            kConeNonPos,     // { x : x <= 0 }
            kConeSoc,        // { (p, x) : ||x||_2 <= p }
            kConeSdp,        // { X : X >= 0 }
            kConeExpPrimal,  // { (x, y, z) : y > 0, y e^(x/y) <= z }
            kConeExpDual };  // { (u, v, w) : u < 0, -u e^(v/u) <= ew }

struct ConeConstraint {
  Cone cone;
  std::vector<CONE_IDX> idx;
  ConeConstraint(Cone cone, const std::vector<CONE_IDX>& idx)
      : cone(cone), idx(idx) { };
};

struct ConeConstraintRaw {
  Cone cone;
  CONE_IDX *idx;
  CONE_IDX size;
};

inline bool IsSeparable(Cone cone) {
  if (cone == kConeZero || cone == kConeNonNeg || cone == kConeNonPos)
    return true;
  return false;
}

inline bool ValidCone(const std::vector<ConeConstraint>& cones, size_t dim) {
  std::set<CONE_IDX> idx;
  for (const auto &cone : cones) {
    for (auto i : cone.idx) {
      if (idx.count(i) > 0) {
        Printf("ERROR: Cone index %d in multiple cones.\n", i);
        return false;
      }
      if (i >= dim) {
        Printf("ERROR: Cone index %d exceeds dimension of cone.\n", i);
        return false;
      }
      idx.insert(i);
    }
  }
  return true;
}

// Shared GPU/CPU code.
const double kExp1 = 2.718281828459045;

// Helper: solve t*exp(t) = rhs using Newton's method (Lambert W function)
template <typename T>
__DEVICE__ T SolveTExpT(T rhs) {
  // Find t such that t * exp(t) = rhs
  // This is the Lambert W function: t = W(rhs)
  if (rhs < static_cast<T>(0)) return static_cast<T>(0);
  if (rhs < static_cast<T>(1e-10)) return rhs;  // t ≈ rhs for small rhs

  // Initial guess
  T t = (rhs < static_cast<T>(1)) ? rhs : Log(rhs + static_cast<T>(1));

  for (int i = 0; i < 50; ++i) {
    T exp_t = Exp(t);
    T f = t * exp_t - rhs;
    T df = exp_t * (t + static_cast<T>(1));
    if (Abs(df) < static_cast<T>(1e-15)) break;
    T step = f / df;
    t -= step;
    if (Abs(step) < static_cast<T>(1e-12) * (static_cast<T>(1) + Abs(t))) break;
  }
  return t;
}

template <typename T>
__DEVICE__ void ProjectExpPrimalCone(const CONE_IDX *idx, T *v) {
  // Exponential cone: K_exp = { (r, s, t) : s > 0, s*e^(r/s) <= t }
  //                         U { (r, s, t) : r <= 0, s = 0, t >= 0 }
  // Following SCS notation and algorithm
  T r = v[idx[0]], s = v[idx[1]], t = v[idx[2]];
  const T tol = static_cast<T>(1e-8);

  // Case 1: Point is in the cone
  if ((s > tol && s * Exp(r / s) <= t + tol) ||
      (Abs(s) <= tol && r <= tol && t >= -tol)) {
    // Already in cone, ensure boundary conditions
    if (Abs(s) <= tol && r <= tol) {
      v[idx[0]] = Min(r, static_cast<T>(0));
      v[idx[1]] = static_cast<T>(0);
      v[idx[2]] = Max(t, static_cast<T>(0));
    }
    return;
  }

  // Case 2: Point is in polar/dual cone - project to origin
  // Polar cone: { (r, s, t) : r > 0, r*e^(s/r) <= -e*t }
  if (r > tol && t < -tol) {
    T val = r * Exp(s / r);
    if (val <= -kExp1 * t + tol) {
      v[idx[0]] = v[idx[1]] = v[idx[2]] = static_cast<T>(0);
      return;
    }
  }

  // Case 3: General projection using the algorithm from SCS
  // The projection onto exp cone boundary satisfies:
  //   s* * exp(r*/s*) = t*  (on boundary)
  //   (r - r*, s - s*, t - t*) = mu * (exp(r*/s*), exp(r*/s*)*(1-r*/s*), -1)
  // for some mu >= 0
  //
  // This gives: t* = t + mu, and exp(r*/s*) = t*/s*
  // Let u = r*/s*, then exp(u) = t*/s*, so s* = t*/exp(u)
  // And r* = u * s* = u * t* / exp(u)
  //
  // From r - r* = mu * exp(u) = mu * t*/s*:
  //   r - u*t*/exp(u) = mu * t* / (t*/exp(u)) = mu * exp(u)
  //
  // From t* = t + mu:
  //   r - u*(t+mu)/exp(u) = mu * exp(u)
  //   r*exp(u) - u*t - u*mu = mu * exp(2u)
  //   r*exp(u) - u*t = mu * (exp(2u) + u)
  //   mu = (r*exp(u) - u*t) / (exp(2u) + u)
  //
  // From s - s* = mu * exp(u) * (1 - u):
  //   s - (t+mu)/exp(u) = mu * exp(u) * (1 - u)
  //   s*exp(u) - t - mu = mu * exp(2u) * (1 - u)
  //   s*exp(u) - t = mu * (1 + exp(2u)*(1-u))
  //
  // We need to find u. Substituting mu:
  //   s*exp(u) - t = ((r*exp(u) - u*t) / (exp(2u) + u)) * (1 + exp(2u)*(1-u))
  //
  // This is a nonlinear equation in u. Use bisection for robustness.

  // Bisection bounds: u typically in [-50, 50] for reasonable inputs
  T u_lo = static_cast<T>(-50);
  T u_hi = static_cast<T>(50);

  // Residual function: given u, compute the constraint violation
  auto residual = [r, s, t](T u) -> T {
    T exp_u = Exp(u);
    T exp_2u = exp_u * exp_u;
    T denom = exp_2u + u;
    if (Abs(denom) < static_cast<T>(1e-15)) return static_cast<T>(1e10);

    T mu = (r * exp_u - u * t) / denom;
    if (mu < static_cast<T>(0)) return static_cast<T>(1e10) * (static_cast<T>(1) - mu);

    T lhs = s * exp_u - t;
    T rhs = mu * (static_cast<T>(1) + exp_2u * (static_cast<T>(1) - u));
    return lhs - rhs;
  };

  // Find sign change
  T f_lo = residual(u_lo);
  T f_hi = residual(u_hi);

  // If no sign change, try to find better bounds
  if (f_lo * f_hi > 0) {
    // Try Newton from a central point
    T u = static_cast<T>(0);
    for (int i = 0; i < 50; ++i) {
      T exp_u = Exp(u);
      T exp_2u = exp_u * exp_u;
      T denom = exp_2u + u;
      if (Abs(denom) < static_cast<T>(1e-15)) break;

      T mu = (r * exp_u - u * t) / denom;
      if (mu < 0) mu = static_cast<T>(0);

      T t_star = t + mu;
      T s_star = t_star / exp_u;
      T r_star = u * s_star;

      // Check if this is a valid projection
      if (s_star > 0 && Abs(s_star * Exp(r_star / s_star) - t_star) < tol) {
        v[idx[0]] = r_star;
        v[idx[1]] = s_star;
        v[idx[2]] = t_star;
        return;
      }

      // Newton step based on KKT conditions
      // Gradient of objective: 2*(r_star - r, s_star - s, t_star - t)
      // must equal lambda * gradient of constraint
      T grad_r = static_cast<T>(2) * (r_star - r);
      T grad_s = static_cast<T>(2) * (s_star - s);

      T target_ratio = (static_cast<T>(1) - u);
      T actual_ratio = (Abs(grad_r) > tol) ? grad_s / grad_r : target_ratio;

      T du = (target_ratio - actual_ratio) * static_cast<T>(0.5);
      u += du;
      u = Max(static_cast<T>(-50), Min(static_cast<T>(50), u));

      if (Abs(du) < static_cast<T>(1e-10)) break;
    }
  }

  // Bisection
  T u = (u_lo + u_hi) / static_cast<T>(2);
  for (int i = 0; i < 100; ++i) {
    T f = residual(u);
    if (Abs(f) < tol || (u_hi - u_lo) < tol) break;

    if (f * f_lo < 0) {
      u_hi = u;
      f_hi = f;
    } else {
      u_lo = u;
      f_lo = f;
    }
    u = (u_lo + u_hi) / static_cast<T>(2);
  }

  // Compute projection from u
  T exp_u = Exp(u);
  T exp_2u = exp_u * exp_u;
  T denom = exp_2u + u;
  T mu = Max((r * exp_u - u * t) / denom, static_cast<T>(0));

  T t_star = t + mu;
  T s_star = Max(t_star / exp_u, static_cast<T>(1e-12));
  T r_star = u * s_star;

  v[idx[0]] = r_star;
  v[idx[1]] = s_star;
  v[idx[2]] = t_star;
}

template <typename T>
__DEVICE__ void ProjectExpDualCone(const CONE_IDX *idx, T *v) {
  // Dual exponential cone: K_exp* = { (u, v, w) : u < 0, -u*e^(v/u) <= e*w }
  //                               U { (u, v, w) : u = 0, v >= 0, w >= 0 }
  T u = v[idx[0]], s = v[idx[1]], w = v[idx[2]];
  const T tol = static_cast<T>(1e-8);

  // Check if already in dual cone
  // Case 1: Main body - u < 0, -u*e^(v/u) <= e*w
  if (u < -tol && w > tol) {
    T val = -u * Exp(s / u);
    if (val <= kExp1 * w + tol) {
      return;  // Already in dual cone
    }
  }

  // Case 2: Boundary ray - u = 0, v >= 0, w >= 0
  if (Abs(u) <= tol && s >= -tol && w >= -tol) {
    v[idx[0]] = static_cast<T>(0);
    v[idx[1]] = Max(s, static_cast<T>(0));
    v[idx[2]] = Max(w, static_cast<T>(0));
    return;
  }

  // Case 3: Polar cone of K_exp* (which is -K_exp) - project to origin
  // -K_exp = { (-r, -s, -t) : s > 0, s*e^(r/s) <= t } U { r >= 0, s = 0, t <= 0 }
  // So if (u, s, w) is in -K_exp, project to (0,0,0)
  if ((s < -tol && (-s) * Exp(u / (-s)) <= (-w) + tol) ||
      (Abs(s) <= tol && u >= -tol && w <= tol)) {
    v[idx[0]] = v[idx[1]] = v[idx[2]] = static_cast<T>(0);
    return;
  }

  // Case 4: General case - use Moreau decomposition
  // proj_K*(x) = x + proj_K(-x) where K is the primal exponential cone
  T neg[3] = {-u, -s, -w};
  CONE_IDX temp_idx[3] = {0, 1, 2};

  // Project negated point onto primal cone
  ProjectExpPrimalCone(temp_idx, neg);

  // Add the projection to original point
  v[idx[0]] = u + neg[0];
  v[idx[1]] = s + neg[1];
  v[idx[2]] = w + neg[2];
}

// CPU code.
#ifndef __CUDACC__
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#endif

namespace {
template <typename T, typename F>
void ApplyCpu(const F& f, const ConeConstraintRaw& cone_constr, T *v) {
  for (CONE_IDX i = 0; i < cone_constr.size; ++i)
    v[cone_constr.idx[i]] = f(v[cone_constr.idx[i]]);
}
}  // namespace

template <typename T>
inline void ProxConeZeroCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](T) { return static_cast<T>(0); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeNonNegCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](T x) { return std::max(x, static_cast<T>(0)); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeNonPosCpu(const ConeConstraintRaw& cone_constr, T *v) {
  auto f = [](T x) { return std::min(x, static_cast<T>(0)); };
  ApplyCpu(f, cone_constr, v);
}

template <typename T>
inline void ProxConeSocCpu(const ConeConstraintRaw& cone_constr, T *v) {
  T nrm = static_cast<T>(0);
  for (CONE_IDX i = 1; i < cone_constr.size; ++i)
    nrm += v[cone_constr.idx[i]] * v[cone_constr.idx[i]];
  nrm = std::sqrt(nrm);

  T p = v[cone_constr.idx[0]];
  if (nrm <= -p) {
    auto f = [](T) { return static_cast<T>(0); };
    ApplyCpu(f, cone_constr, v);
  } else if (nrm >= std::abs(p)) {
    T scale = (static_cast<T>(1) + p / nrm) / 2;
    v[cone_constr.idx[0]] = nrm;
    auto f = [scale](T x) { return scale * x; };
    ApplyCpu(f, cone_constr, v);
  }
}

template <typename T>
inline void ProxConeSdpCpu(const ConeConstraintRaw& cone_constr, T *v) {
#ifndef __CUDACC__
  using namespace gsl;

  // Compute matrix dimension n from vectorized size n(n+1)/2
  CONE_IDX vec_size = cone_constr.size;
  CONE_IDX n = static_cast<CONE_IDX>((-1.0 + std::sqrt(1.0 + 8.0 * vec_size)) / 2.0);

  // Verify size is consistent
  if (n * (n + 1) / 2 != vec_size) {
    // Invalid size for symmetric matrix
    return;
  }

  // Allocate matrix and eigenvalue storage
  T* A_data = new T[n * n];
  T* w_data = new T[n];

  // Build symmetric matrix from vectorized lower triangular form
  // Vectorization format: column-major lower triangle
  // For 3x3: [a11, a21, a31, a22, a32, a33]
  CONE_IDX vec_idx = 0;
  for (CONE_IDX col = 0; col < n; ++col) {
    for (CONE_IDX row = col; row < n; ++row) {
      T val = v[cone_constr.idx[vec_idx]];
      // Store in column-major full matrix
      A_data[col * n + row] = val;  // Lower triangle
      A_data[row * n + col] = val;  // Upper triangle (symmetric)
      ++vec_idx;
    }
  }

  // Create GSL matrix and vector views
  matrix<T, CblasColMajor> A;
  A.size1 = n;
  A.size2 = n;
  A.tda = n;
  A.data = A_data;

  vector<T> w;
  w.size = n;
  w.stride = 1;
  w.data = w_data;

  // Compute eigenvalue decomposition
  // After this: A contains eigenvectors (columns), w contains eigenvalues
  linalg_syevd(&A, &w);

  // Project eigenvalues to non-negative orthant
  for (CONE_IDX i = 0; i < n; ++i) {
    if (w_data[i] < static_cast<T>(0)) {
      w_data[i] = static_cast<T>(0);
    }
  }

  // Reconstruct matrix: X = V * diag(w) * V^T
  // Allocate temporary storage for result
  T* X_data = new T[n * n];

  // Compute X[i,j] = sum_k V[i,k] * w[k] * V[j,k]
  for (CONE_IDX i = 0; i < n; ++i) {
    for (CONE_IDX j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      for (CONE_IDX k = 0; k < n; ++k) {
        // V is stored column-major: V[i,k] = A_data[k*n + i]
        sum += A_data[k * n + i] * w_data[k] * A_data[k * n + j];
      }
      X_data[j * n + i] = sum;
    }
  }

  // Extract lower triangular part back to vectorized form
  vec_idx = 0;
  for (CONE_IDX col = 0; col < n; ++col) {
    for (CONE_IDX row = col; row < n; ++row) {
      v[cone_constr.idx[vec_idx]] = X_data[col * n + row];
      ++vec_idx;
    }
  }

  delete[] X_data;
  delete[] A_data;
  delete[] w_data;
#else
  assert(false && "SDP Not implemented on CPU");
#endif
}


template <typename T>
inline void ProxConeExpPrimalCpu(const ConeConstraintRaw& cone_constr, T *v) {
  ProjectExpPrimalCone(cone_constr.idx, v);
}

template <typename T>
inline void ProxConeExpDualCpu(const ConeConstraintRaw& cone_constr, T *v) {
  ProjectExpDualCone(cone_constr.idx, v);
}

template <typename T>
void ProxEvalConeCpu(const std::vector<ConeConstraintRaw>& cone_constr_vec,
                     CONE_IDX size, const T *x_in, T *x_out) {
  if (x_in != x_out)
    memcpy(x_out, x_in, size * sizeof(T));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (const auto& cone_constr : cone_constr_vec) {
    switch (cone_constr.cone) {
      case kConeZero: default: ProxConeZeroCpu(cone_constr, x_out); break;
      case kConeNonNeg: ProxConeNonNegCpu(cone_constr, x_out); break;
      case kConeNonPos: ProxConeNonPosCpu(cone_constr, x_out); break;
      case kConeSoc: ProxConeSocCpu(cone_constr, x_out); break;
      case kConeSdp: ProxConeSdpCpu(cone_constr, x_out); break;
      case kConeExpPrimal: ProxConeExpPrimalCpu(cone_constr, x_out); break;
      case kConeExpDual: ProxConeExpDualCpu(cone_constr, x_out); break;
    }
  }
}

// GPU code.
#ifdef __CUDACC__

// Helper functions
const CONE_IDX kBlockSize = 256u;
#if __CUDA_ARCH__ >= 300
const CONE_IDX kMaxGridSize = 2147483647u;  // 2^31 - 1
#else
const CONE_IDX kMaxGridSize = 65535u;  // 2^16 - 1
#endif

template <typename F>
__global__
void __Execute(F f) {
  f();
}

template <typename T, typename F>
__global__
void __Apply(F f, const CONE_IDX *idx, CONE_IDX size, T *v) {
  CONE_IDX tid = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 300
  v[idx[tid]] = f(v[idx[tid]]);
#else
  for (CONE_IDX i = tid; i < size; i += gridDim.x * blockDim.x)
    v[idx[i]] = f(v[idx[i]]);
#endif
}

template <typename T, typename F>
void inline ApplyGpu(F f, const ConeConstraintRaw& cone_constr, T *v,
                     const cudaStream_t& stream) {
  CONE_IDX block_size = std::min<CONE_IDX>(kBlockSize, cone_constr.size);
  CONE_IDX grid_dim = std::min(kMaxGridSize,
      (cone_constr.size + block_size - 1) / block_size);
  __Apply<<<grid_dim, block_size, 0, stream>>>(f, cone_constr.idx,
      cone_constr.size, v);
}

// Functors
template <typename T>
struct Zero {
  __DEVICE__ T operator()(T) const { return static_cast<T>(0); }
};

template <typename T>
struct Max0 {
  __DEVICE__ T operator()(T x) const { return Max(static_cast<T>(0), x); }
};

template <typename T>
struct Min0 {
  __DEVICE__ T operator()(T x) const { return Min(static_cast<T>(0), x); }
};

template <typename T>
struct SquareIdx {
  T *v;
  __DEVICE__ SquareIdx(T *v) : v(v) { };
  __DEVICE__ T operator()(CONE_IDX i) const { return v[i] * v[i]; }
};

template <typename T>
struct Scale {
  T a;
  __DEVICE__ Scale(T a) : a(a) { }
  __DEVICE__ T operator()(T x) const { return a * x; }
};

// Proximal operators
template <typename T>
inline void ProxConeZeroGpu(const ConeConstraintRaw& cone_constr, T *v,
                            const cudaStream_t &stream) {
  ApplyGpu(Zero<T>(), cone_constr, v, stream);
}

// Prox
template <typename T>
inline void ProxConeNonNegGpu(const ConeConstraintRaw& cone_constr, T *v,
                              const cudaStream_t &stream) {

  ApplyGpu(Max0<T>(), cone_constr, v, stream);
}

template <typename T>
inline void ProxConeNonPosGpu(const ConeConstraintRaw& cone_constr, T *v,
                              const cudaStream_t &stream) {
  ApplyGpu(Min0<T>(), cone_constr, v, stream);
}

// TODO: Move this to the GPU, the sync here will make this slow.
template <typename T>
inline void ProxConeSocGpu(const ConeConstraintRaw& cone_constr, T *v,
                           const cudaStream_t &stream) {
  // Compute nrm(v[1:end])
  T nrm = thrust::transform_reduce(thrust::cuda::par.on(stream),
      thrust::device_pointer_cast(cone_constr.idx + 1u),
      thrust::device_pointer_cast(cone_constr.idx + cone_constr.size),
      SquareIdx<T>(v), static_cast<T>(0), thrust::plus<T>());
  nrm = std::sqrt(nrm);

//  if (nrm > 0) {
//    printf("%e\n", nrm);
//    T *x = new T[4];
//    cudaMemcpy(x, v, 4 * sizeof(T), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 4; ++i)
//      printf(".. %f\n", x[i]);
//    exit(1);
//  }

  // Get p from GPU.
  CONE_IDX i;
  T p;
  cudaMemcpyAsync(&i, cone_constr.idx, sizeof(CONE_IDX),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(&p, v + i, sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Project if ||x||_2 > p
  if (nrm <= -p) {
    ApplyGpu(Zero<T>(), cone_constr, v, stream);
  } else if (nrm >= std::abs(p)) {
    cudaMemcpyAsync(v + i, &nrm, sizeof(T), cudaMemcpyHostToDevice, stream);
    T scale = (static_cast<T>(1) + p / nrm) / 2;
    ApplyGpu(Scale<T>(scale), cone_constr, v, stream);
  }
}

template <typename T>
inline void ProxConeSdpGpu(const ConeConstraintRaw& cone_constr, T *v,
                           const cudaStream_t &stream) {
  assert(false && "SDP Not implemented on GPU");
}

// GPU kernel for exponential primal cone projection
template <typename T>
__global__
void __ProjectExpPrimalConeKernel(const CONE_IDX *idx, T *v) {
  ProjectExpPrimalCone(idx, v);
}

// GPU kernel for exponential dual cone projection
template <typename T>
__global__
void __ProjectExpDualConeKernel(const CONE_IDX *idx, T *v) {
  ProjectExpDualCone(idx, v);
}

template <typename T>
inline void ProxConeExpPrimalGpu(const ConeConstraintRaw& cone_constr, T *v,
                                 const cudaStream_t &stream) {
  // Launch single thread kernel - exponential cone has only 3 elements
  __ProjectExpPrimalConeKernel<<<1, 1, 0, stream>>>(cone_constr.idx, v);
}

template <typename T>
inline void ProxConeExpDualGpu(const ConeConstraintRaw& cone_constr, T *v,
                               const cudaStream_t &stream) {
  // Launch single thread kernel - exponential cone has only 3 elements
  __ProjectExpDualConeKernel<<<1, 1, 0, stream>>>(cone_constr.idx, v);
}

template <typename T>
void ProxEvalConeGpu(const std::vector<ConeConstraintRaw>& cone_constr_vec,
                     const std::vector<cudaStream_t> streams,
                     CONE_IDX size, const T *x_in, T *x_out) {
  cudaMemcpy(x_out, x_in, size * sizeof(T), cudaMemcpyDeviceToDevice);

  size_t idx = 0;
  for (const auto& cone_constr : cone_constr_vec) {
    const cudaStream_t& s = streams[idx++];
    switch (cone_constr.cone) {
      case kConeZero: default: ProxConeZeroGpu(cone_constr, x_out, s); break;
      case kConeNonNeg: ProxConeNonNegGpu(cone_constr, x_out, s); break;
      case kConeNonPos: ProxConeNonPosGpu(cone_constr, x_out, s); break;
      case kConeSoc: ProxConeSocGpu(cone_constr, x_out, s); break;
      case kConeSdp: ProxConeSdpGpu(cone_constr, x_out, s); break;
      case kConeExpPrimal: ProxConeExpPrimalGpu(cone_constr, x_out, s); break;
      case kConeExpDual: ProxConeExpDualGpu(cone_constr, x_out, s); break;
    }
  }
}


#endif  // __CUDACC__

#endif  // PROX_LIB_CONE_H_

