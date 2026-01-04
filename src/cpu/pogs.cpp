#include "pogs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

#include "anderson.h"
#include "equil_helper.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "interface_defs.h"
#include "matrix/matrix.h"
#include "matrix/matrix_dense.h"
#include "matrix/matrix_sparse.h"
#include "projector/projector.h"
#include "projector/projector_direct.h"
#include "projector/projector_cgls.h"
#include "util.h"

#include "timer.h"

#define __HBAR__ \
"----------------------------------------------------------------------------\n"

namespace pogs {

template <typename T, typename M, typename P>
PogsImplementation<T, M, P>::PogsImplementation(const M &A)
    : _A(A), _P(_A),
      _de(0), _z(0), _zt(0),
      _rho(static_cast<T>(kRhoInit)),
      _done_init(false),
      _x(0), _y(0), _mu(0), _lambda(0), _optval(static_cast<T>(0.)),
      _final_iter(0),
      _abs_tol(static_cast<T>(kAbsTol)),
      _rel_tol(static_cast<T>(kRelTol)),
      _max_iter(kMaxIter),
      _init_iter(kInitIter),
      _verbose(kVerbose),
      _adaptive_rho(kAdaptiveRho),
      _gap_stop(kGapStop),
      _init_x(false), _init_lambda(false),
      _use_anderson(false),
      _anderson_mem(5u),
      _anderson_start(10u) {
  _x = new T[_A.Cols()]();
  _y = new T[_A.Rows()]();
  _mu = new T[_A.Cols()]();
  _lambda = new T[_A.Rows()]();
}

template <typename T, typename M, typename P>
int PogsImplementation<T, M, P>::_Init(const PogsObjective<T> *objective) {
  DEBUG_EXPECT(!_done_init);
  if (_done_init)
    return 1;
  _done_init = true;

  size_t m = _A.Rows();
  size_t n = _A.Cols();

  _de = new T[m + n];
  ASSERT(_de != 0);
  _z = new T[m + n];
  ASSERT(_z != 0);
  _zt = new T[m + n];
  ASSERT(_zt != 0);
  memset(_de, 0, (m + n) * sizeof(T));
  memset(_z, 0, (m + n) * sizeof(T));
  memset(_zt, 0, (m + n) * sizeof(T));

  _A.Init();
  _A.Equil(_de, _de + m,
           std::function<void(T*)>([objective](T *v){
               objective->constrain_d(v);
           }),
           std::function<void(T*)>([objective](T *v){
               objective->constrain_e(v);
           }));
  _nrmA = Norm2Est(&_A);

  _P.Init();

  return 0;
}

template <typename T, typename M, typename P>
PogsStatus PogsImplementation<T, M, P>::Solve(PogsObjective<T> *objective) {
  double t0 = timer<double>();
  // Constants for adaptive-rho and over-relaxation.
  const T kDeltaMin       = static_cast<T>(1.05);
  const T kGamma          = static_cast<T>(1.01);
  const T kTau            = static_cast<T>(0.8);
  const T kRhoMin         = static_cast<T>(1e-4);
  const T kRhoMax         = static_cast<T>(1e4);
  const T kKappa          = static_cast<T>(0.9);
  const T kOne            = static_cast<T>(1);
  const T kZero           = static_cast<T>(0);
  const bool kUseExactTol = objective->UseExactTol();
  const T kProjTolMax     = kUseExactTol ? static_cast<T>(1e-10)
                                         : static_cast<T>(1e-8);
  const T kProjTolMin     = kUseExactTol ? static_cast<T>(1e-3)
                                         : static_cast<T>(1e-2);
  const T kProjTolIni     = kUseExactTol ? static_cast<T>(1e-6)
                                         : static_cast<T>(1e-5);
  const T kAlpha          = kUseExactTol ? static_cast<T>(1.0)
                                         : static_cast<T>(1.7);

  // Initialize Projector P and Matrix A.
  if (!_done_init)
    _Init(objective);

  // Extract values from pogs_data
  size_t m = _A.Rows();
  size_t n = _A.Cols();

  // Allocate data for ADMM variables.
  gsl::vector<T> de    = gsl::vector_view_array(_de, m + n);
  gsl::vector<T> z     = gsl::vector_view_array(_z, m + n);
  gsl::vector<T> zt    = gsl::vector_view_array(_zt, m + n);
  gsl::vector<T> zprev = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> ztemp = gsl::vector_calloc<T>(m + n);
  gsl::vector<T> z12   = gsl::vector_calloc<T>(m + n);

  // Create views for x and y components.
  gsl::vector<T> d     = gsl::vector_subvector(&de, 0, m);
  gsl::vector<T> e     = gsl::vector_subvector(&de, m, n);
  gsl::vector<T> x     = gsl::vector_subvector(&z, 0, n);
  gsl::vector<T> y     = gsl::vector_subvector(&z, n, m);
  gsl::vector<T> x12   = gsl::vector_subvector(&z12, 0, n);
  gsl::vector<T> y12   = gsl::vector_subvector(&z12, n, m);
  gsl::vector<T> xprev = gsl::vector_subvector(&zprev, 0, n);
  gsl::vector<T> yprev = gsl::vector_subvector(&zprev, n, m);
  gsl::vector<T> xtemp = gsl::vector_subvector(&ztemp, 0, n);
  gsl::vector<T> ytemp = gsl::vector_subvector(&ztemp, n, m);

  // Scale objective to account for diagonal scaling e and d.
  objective->scale(d.data, e.data);

  // Initialize (x, lambda) from (x0, lambda0).
  if (_init_x) {
    gsl::vector_memcpy(&xtemp, _x);
    gsl::vector_div(&xtemp, &e);
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data);
    gsl::vector_memcpy(&z, &ztemp);
  }
  if (_init_lambda) {
    gsl::vector_memcpy(&ytemp, _lambda);
    gsl::vector_div(&ytemp, &d);
    _A.Mul('t', -kOne, ytemp.data, kZero, xtemp.data);
    gsl::blas_scal(-kOne / _rho, &ztemp);
    gsl::vector_memcpy(&zt, &ztemp);
  }

  // Make an initial guess for (x0 or lambda0).
  if (_init_x && !_init_lambda) {
    // Alternating projections to satisfy
    //   1. \lambda \in \partial f(y), \mu \in \partial g(x)
    //   2. \mu = -A^T\lambda
    gsl::vector_set_all(&zprev, kZero);
    for (unsigned int i = 0; i < kInitIter; ++i) {
      ASSERT(false);
      // TODO: Make part of PogsObj
//      ProjSubgradEval(g, xprev.data, x.data, xtemp.data);
//      ProjSubgradEval(f, yprev.data, y.data, ytemp.data);
      _P.Project(xtemp.data, ytemp.data, kOne, xprev.data, yprev.data,
          kProjTolIni);
      gsl::blas_axpy(-kOne, &ztemp, &zprev);
      gsl::blas_scal(-kOne, &zprev);
    }
    // xt = -1 / \rho * \mu, yt = -1 / \rho * \lambda.
    gsl::vector_memcpy(&zt, &zprev);
    gsl::blas_scal(-kOne / _rho, &zt);
  } else if (_init_lambda && !_init_x) {
    ASSERT(false);
  }
  _init_x = _init_lambda = false;

  // Save initialization time.
  double time_init = timer<double>() - t0;

  // Signal start of execution.
  if (_verbose > 0) {
    Printf(__HBAR__
        "           POGS v%s - Proximal Graph Solver (CPU)                \n"
        "           (c) Christopher Fougner, Stanford University 2014-2015\n",
        POGS_VERSION);
  }
  if (_verbose > 1) {
    Printf(__HBAR__
        " Iter | pri res | pri tol | dua res | dua tol |   gap   | eps gap |"
        " pri obj\n" __HBAR__);
  }

  // Initialize scalars.
  T sqrtn_atol = std::sqrt(static_cast<T>(n)) * _abs_tol;
  T sqrtm_atol = std::sqrt(static_cast<T>(m)) * _abs_tol;
  T sqrtmn_atol = std::sqrt(static_cast<T>(m + n)) * _abs_tol;
  T delta = kDeltaMin, xi = static_cast<T>(1);
  unsigned int k = 0u, kd = 0u, ku = 0u;
  bool converged = false;
  T nrm_r, nrm_s, gap, eps_gap, eps_pri, eps_dua;
  const bool profile = _verbose > 3;
  double time_prox = 0.0;
  double time_proj = 0.0;
  double time_res = 0.0;
  auto orig_primal_metrics = [&](const T *r_scaled, const T *y_scaled,
                                 const T *x_scaled, T *nrm_r_out,
                                 T *ax_norm_out, T *y_norm_out,
                                 T *x_norm_out) {
    T r_sq = kZero;
    T ax_sq = kZero;
    T y_sq = kZero;
    for (size_t i = 0; i < m; ++i) {
      const T di = d.data[i];
      if (di == kZero)
        continue;
      const T r_i = r_scaled[i] / di;
      r_sq += r_i * r_i;
      const T y_i = y_scaled[i] / di;
      y_sq += y_i * y_i;
      const T ax_i = (r_scaled[i] + y_scaled[i]) / di;
      ax_sq += ax_i * ax_i;
    }
    T x_sq = kZero;
    for (size_t i = 0; i < n; ++i) {
      const T xi = x_scaled[i] * e.data[i];
      x_sq += xi * xi;
    }
    *nrm_r_out = std::sqrt(r_sq);
    *ax_norm_out = std::sqrt(ax_sq);
    *y_norm_out = std::sqrt(y_sq);
    *x_norm_out = std::sqrt(x_sq);
  };
  auto orig_dual_norm = [&](const T *s_scaled) {
    T s_sq = kZero;
    for (size_t i = 0; i < n; ++i) {
      const T ei = e.data[i];
      if (ei == kZero)
        continue;
      const T si = s_scaled[i] / ei;
      s_sq += si * si;
    }
    return std::sqrt(s_sq);
  };

  // Anderson acceleration for primal variable z.
  // We accelerate only the primal fixed-point iteration, not the dual.
  // This is more stable than accelerating the combined primal-dual state.
  std::unique_ptr<AndersonAccelerator<T>> anderson;
  gsl::vector<T> z_acc;
  T prev_nrm_r = std::numeric_limits<T>::max();  // For safeguarding
  if (_use_anderson && _anderson_mem > 0) {
    anderson = std::make_unique<AndersonAccelerator<T>>(m + n, _anderson_mem);
    z_acc = gsl::vector_calloc<T>(m + n);
    if (_verbose > 0) {
      Printf("Anderson acceleration enabled: mem=%u, start=%u\n",
             _anderson_mem, _anderson_start);
    }
  }

  for (;; ++k) {
    gsl::vector_memcpy(&zprev, &z);

    // Evaluate Proximal Operators
    gsl::blas_axpy(-kOne, &zt, &z);
    if (profile) {
      double t = timer<double>();
      objective->prox(x.data, y.data, x12.data, y12.data, _rho);
      time_prox += timer<double>() - t;
    } else {
      objective->prox(x.data, y.data, x12.data, y12.data, _rho);
    }

    // Compute gap, optval, and tolerances.
    gsl::blas_axpy(-kOne, &z12, &z);
    gsl::blas_dot(&z, &z12, &gap);
    gap = std::abs(gap);
    eps_gap = sqrtmn_atol + _rel_tol * gsl::blas_nrm2(&z) *
        gsl::blas_nrm2(&z12);
    eps_pri = sqrtm_atol + _rel_tol * gsl::blas_nrm2(&y12);
    eps_dua = _rho * (sqrtn_atol + _rel_tol * gsl::blas_nrm2(&x));

    // Apply over relaxation.
    gsl::vector_memcpy(&ztemp, &zt);
    gsl::blas_axpy(kAlpha, &z12, &ztemp);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &ztemp);

    // Warm start projection with previous x iterate.
    gsl::vector_memcpy(&x, &xprev);

    // Project onto y = Ax.
    // Residual-based tolerance: ties inner accuracy to outer progress.
    // Early iterations can be sloppy; late iterations need high accuracy.
    // tol = η0 * residual^p, clamped to [η_max, η_min]
    const T kProjResidualPow = kUseExactTol ? static_cast<T>(1.0)
                                            : static_cast<T>(0.5);
    T proj_tol = kProjTolMin * std::pow(std::min(prev_nrm_r, kOne), kProjResidualPow);
    proj_tol = std::max(proj_tol, kProjTolMax);
    if (profile) {
      double t = timer<double>();
      _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
      time_proj += timer<double>() - t;
    } else {
      _P.Project(xtemp.data, ytemp.data, kOne, x.data, y.data, proj_tol);
    }

    // Calculate residuals.
    if (profile) {
      double t = timer<double>();
      gsl::vector_memcpy(&ztemp, &zprev);
      gsl::blas_axpy(-kOne, &z, &ztemp);
      nrm_s = _rho * (_nrmA * gsl::blas_nrm2(&ytemp) + gsl::blas_nrm2(&xtemp));

      gsl::vector_memcpy(&ztemp, &z12);
      gsl::blas_axpy(-kOne, &z, &ztemp);
      nrm_r = _nrmA * gsl::blas_nrm2(&xtemp) + gsl::blas_nrm2(&ytemp);

      // Calculate exact residuals only if necessary.
      bool exact = false;
      if ((nrm_r < 10 * eps_pri && nrm_s < 10 * eps_dua) || kUseExactTol) {
        gsl::vector_memcpy(&ztemp, &z12);
        _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
        if (kUseExactTol) {
          T ax_norm = kZero;
          T y_norm = kZero;
          T x_norm = kZero;
          orig_primal_metrics(ytemp.data, y12.data, x12.data, &nrm_r,
                              &ax_norm, &y_norm, &x_norm);
          eps_pri = sqrtm_atol + _rel_tol * std::max(ax_norm, y_norm);
          eps_dua = _rho * (sqrtn_atol + _rel_tol * x_norm);
        } else {
          nrm_r = gsl::blas_nrm2(&ytemp);
        }
        gsl::vector_memcpy(&ztemp, &z12);
        gsl::blas_axpy(kOne, &zt, &ztemp);
        gsl::blas_axpy(-kOne, &zprev, &ztemp);
        _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
        if (kUseExactTol) {
          nrm_s = _rho * orig_dual_norm(xtemp.data);
        } else {
          nrm_s = _rho * gsl::blas_nrm2(&xtemp);
        }
        exact = true;
      }
      time_res += timer<double>() - t;
      // Evaluate stopping criteria.
      converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
          (!_gap_stop || gap < eps_gap);
    } else {
      gsl::vector_memcpy(&ztemp, &zprev);
      gsl::blas_axpy(-kOne, &z, &ztemp);
      nrm_s = _rho * (_nrmA * gsl::blas_nrm2(&ytemp) + gsl::blas_nrm2(&xtemp));

      gsl::vector_memcpy(&ztemp, &z12);
      gsl::blas_axpy(-kOne, &z, &ztemp);
      nrm_r = _nrmA * gsl::blas_nrm2(&xtemp) + gsl::blas_nrm2(&ytemp);

      // Calculate exact residuals only if necessary.
      bool exact = false;
      if ((nrm_r < 10 * eps_pri && nrm_s < 10 * eps_dua) || kUseExactTol) {
        gsl::vector_memcpy(&ztemp, &z12);
        _A.Mul('n', kOne, x12.data, -kOne, ytemp.data);
        if (kUseExactTol) {
          T ax_norm = kZero;
          T y_norm = kZero;
          T x_norm = kZero;
          orig_primal_metrics(ytemp.data, y12.data, x12.data, &nrm_r,
                              &ax_norm, &y_norm, &x_norm);
          eps_pri = sqrtm_atol + _rel_tol * std::max(ax_norm, y_norm);
          eps_dua = _rho * (sqrtn_atol + _rel_tol * x_norm);
        } else {
          nrm_r = gsl::blas_nrm2(&ytemp);
        }
        gsl::vector_memcpy(&ztemp, &z12);
        gsl::blas_axpy(kOne, &zt, &ztemp);
        gsl::blas_axpy(-kOne, &zprev, &ztemp);
        _A.Mul('t', kOne, ytemp.data, kOne, xtemp.data);
        if (kUseExactTol) {
          nrm_s = _rho * orig_dual_norm(xtemp.data);
        } else {
          nrm_s = _rho * gsl::blas_nrm2(&xtemp);
        }
        exact = true;
      }

      // Evaluate stopping criteria.
      converged = exact && nrm_r < eps_pri && nrm_s < eps_dua &&
          (!_gap_stop || gap < eps_gap);
    }
    if ((_verbose > 2 && k % 10  == 0) ||
        (_verbose > 1 && k % 100 == 0) ||
        (_verbose > 1 && converged)) {
      T optval = objective->evaluate(x12.data, y12.data);
      Printf("%5d : %.2e  %.2e  %.2e  %.2e  %.2e  %.2e % .2e\n",
          k, nrm_r, eps_pri, nrm_s, eps_dua, gap, eps_gap, optval);
    }

    // Break if converged or there are nans
    if (converged || k == _max_iter - 1){
      _final_iter = k;
      break;
    }

    // Update dual variable.
    gsl::blas_axpy(kAlpha, &z12, &zt);
    gsl::blas_axpy(kOne - kAlpha, &zprev, &zt);
    gsl::blas_axpy(-kOne, &z, &zt);

    // Rescale rho using spectral (AADMM-style) or residual balancing.
    if (_adaptive_rho) {
      // Conservative spectral update: only adjust rho when there's a large imbalance
      // between normalized primal and dual residuals (> 10x difference).
      // This avoids corrupting the solution by over-adjusting rho.
      const unsigned int kRhoUpdateFreq = kUseExactTol ? 10u : 50u;
      const T kRhoChangeMax = kUseExactTol ? static_cast<T>(2.0)
                                           : static_cast<T>(1.5);
      const T kRhoChangeMin = kUseExactTol ? static_cast<T>(0.5)
                                           : static_cast<T>(0.67);
      const T kImbalanceThresh = kUseExactTol ? static_cast<T>(5)
                                              : static_cast<T>(10);

      if (k > 0 && k % kRhoUpdateFreq == 0 && eps_pri > kZero && eps_dua > kZero) {
        T pri_normalized = nrm_r / eps_pri;
        T dua_normalized = nrm_s / eps_dua;

        if (pri_normalized > kZero && dua_normalized > kZero) {
          T imbalance = pri_normalized / dua_normalized;

          // Only update if there's a significant imbalance
          if (imbalance > kImbalanceThresh || imbalance < kOne / kImbalanceThresh) {
            T rho_ratio = std::sqrt(imbalance);
            rho_ratio = std::max(kRhoChangeMin, std::min(kRhoChangeMax, rho_ratio));

            T rho_new = _rho * rho_ratio;
            rho_new = std::max(kRhoMin, std::min(kRhoMax, rho_new));

            if (std::abs(rho_new - _rho) / _rho > static_cast<T>(0.05)) {
              T scale = _rho / rho_new;
              _rho = rho_new;
              gsl::blas_scal(scale, &zt);
              if (anderson) anderson->Reset();
              if (_verbose > 3)
                Printf("spectral rho update: %e (imbalance=%.1f)\n", _rho, imbalance);
            }
          }
        }
      }

      // Fallback to residual balancing if spectral update not triggered
      else if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(kd)) {
        if (_rho < kRhoMax) {
          _rho *= delta;
          gsl::blas_scal(1 / delta, &zt);
          delta = kGamma * delta;
          ku = k;
          if (anderson) anderson->Reset();  // Reset Anderson on rho change
          if (_verbose > 3)
            Printf("+ rho %e\n", _rho);
        }
      } else if (nrm_s > xi * eps_dua && nrm_r < xi * eps_pri &&
          kTau * static_cast<T>(k) > static_cast<T>(ku)) {
        if (_rho > kRhoMin) {
          _rho /= delta;
          gsl::blas_scal(delta, &zt);
          delta = kGamma * delta;
          kd = k;
          if (anderson) anderson->Reset();  // Reset Anderson on rho change
          if (_verbose > 3)
            Printf("- rho %e\n", _rho);
        }
      } else if (nrm_s < xi * eps_dua && nrm_r < xi * eps_pri) {
        xi *= kKappa;
      } else {
        delta = kDeltaMin;
      }
    }

    // Apply Anderson acceleration to primal variable z only.
    // Use residual-based safeguarding: only accept if residual decreases.
    if (anderson && k >= _anderson_start) {
      // Apply Anderson to the primal iterate z
      if (anderson->Apply(z.data, zprev.data, z_acc.data)) {
        // Compute norm of acceleration step
        T acc_step = kZero;
        for (size_t i = 0; i < m + n; ++i) {
          T diff = z_acc.data[i] - z.data[i];
          acc_step += diff * diff;
        }
        acc_step = std::sqrt(acc_step);

        // Compute norm of regular ADMM step
        T admm_step = kZero;
        for (size_t i = 0; i < m + n; ++i) {
          T diff = z.data[i] - zprev.data[i];
          admm_step += diff * diff;
        }
        admm_step = std::sqrt(admm_step);

        // Type-I safeguarding: accept if acceleration step is smaller than ADMM step
        // This ensures we don't overshoot. Use a relaxation factor of 2.0.
        const T kSafetyFactor = static_cast<T>(2.0);
        if (acc_step < kSafetyFactor * admm_step + _abs_tol) {
          // Accept accelerated iterate
          gsl::vector_memcpy(&z, &z_acc);
          if (_verbose > 3) {
            Printf("Anderson: accepted (acc_step=%.2e, admm_step=%.2e)\n",
                   acc_step, admm_step);
          }
        } else {
          if (_verbose > 3) {
            Printf("Anderson: rejected (acc_step=%.2e > %.1f*admm_step=%.2e)\n",
                   acc_step, kSafetyFactor, kSafetyFactor * admm_step);
          }
        }
      }
    }

    // Track residual for potential future safeguarding
    prev_nrm_r = nrm_r;
  }

  // Get optimal value
  _optval = objective->evaluate(x12.data, y12.data);

  // Check status
  PogsStatus status;
  if (!converged && k == _max_iter - 1)
    status = POGS_MAX_ITER;
  else if (!converged && k < _max_iter - 1)
    status = POGS_NAN_FOUND;
  else
    status = POGS_SUCCESS;

  // Print summary
  if (_verbose > 0) {
    Printf(__HBAR__
        "Status: %s\n"
        "Timing: Total = %3.2e s, Init = %3.2e s\n"
        "Iter  : %u\n",
        PogsStatusString(status).c_str(), timer<double>() - t0, time_init, k);
    Printf(__HBAR__
        "Error Metrics:\n"
        "Pri: "
        "|Ax - y|    / (abs_tol sqrt(m)     / rel_tol + |y|)          = %.2e\n"
        "Dua: "
        "|A'l + u|   / (abs_tol sqrt(n)     / rel_tol + |u|)          = %.2e\n"
        "Gap: "
        "|x'u + y'l| / (abs_tol sqrt(m + n) / rel_tol + |x,u| |y,l|)  = %.2e\n"
        __HBAR__, _rel_tol * nrm_r / eps_pri, _rel_tol * nrm_s / eps_dua,
        _rel_tol * gap / eps_gap);
    if (profile) {
      const double iters = static_cast<double>(_final_iter + 1u);
      Printf("Timing breakdown (per-iter avg): prox = %3.2e s, "
             "proj = %3.2e s, residual = %3.2e s\n",
             time_prox / iters, time_proj / iters, time_res / iters);
    }
  }

  // Scale x, y, lambda and mu for output.
  gsl::vector_memcpy(&ztemp, &zt);
  gsl::blas_axpy(-kOne, &zprev, &ztemp);
  gsl::blas_axpy(kOne, &z12, &ztemp);
  gsl::blas_scal(-_rho, &ztemp);
  gsl::vector_mul(&ytemp, &d);
  gsl::vector_div(&xtemp, &e);

  gsl::vector_div(&y12, &d);
  gsl::vector_mul(&x12, &e);

  // Final verification: recompute residual in original space and verify convergence.
  // This catches any bugs where we declared convergence but the solution is poor.
  if (kUseExactTol && status == POGS_SUCCESS) {
    // Compute x_scaled = x_orig / e
    gsl::vector_memcpy(&xtemp, &x12);
    gsl::vector_div(&xtemp, &e);
    // Compute Ax_scaled = A_scaled * x_scaled, store in ytemp
    _A.Mul('n', kOne, xtemp.data, kZero, ytemp.data);
    // Compute Ax_orig = Ax_scaled / d
    gsl::vector_div(&ytemp, &d);
    // Compute r_orig = Ax_orig - y_orig
    gsl::blas_axpy(-kOne, &y12, &ytemp);
    T final_primal_res = gsl::blas_nrm2(&ytemp);

    // Compute tolerance in original scale
    T ax_norm = kZero, y_norm = kZero, x_norm = kZero;
    for (size_t i = 0; i < m; ++i) {
      T yi = y12.data[i];
      y_norm += yi * yi;
      T axi = ytemp.data[i] + yi;  // Ax = r + y
      ax_norm += axi * axi;
    }
    for (size_t i = 0; i < n; ++i) {
      T xi = x12.data[i];
      x_norm += xi * xi;
    }
    ax_norm = std::sqrt(ax_norm);
    y_norm = std::sqrt(y_norm);
    x_norm = std::sqrt(x_norm);
    T final_eps_pri = sqrtm_atol + _rel_tol * std::max(ax_norm, y_norm);

    // Check if solution actually meets tolerance
    if (final_primal_res > final_eps_pri) {
      if (_verbose > 0) {
        Printf("Warning: Post-solve verification failed.\n"
               "  Final primal residual: %.2e > tolerance %.2e\n"
               "  Downgrading status to MAX_ITER (inaccurate).\n",
               final_primal_res, final_eps_pri);
      }
      status = POGS_MAX_ITER;  // Downgrade to inaccurate
    } else if (_verbose > 2) {
      Printf("Post-solve verification passed: primal_res=%.2e < tol=%.2e\n",
             final_primal_res, final_eps_pri);
    }
  }

  // Copy results to output.
  gsl::vector_memcpy(_x, &x12);
  gsl::vector_memcpy(_y, &y12);
  gsl::vector_memcpy(_mu, &xtemp);
  gsl::vector_memcpy(_lambda, &ytemp);

  // Store z.
  gsl::vector_memcpy(&z, &zprev);

  // Free memory.
  gsl::vector_free(&z12);
  gsl::vector_free(&zprev);
  gsl::vector_free(&ztemp);
  if (_use_anderson && _anderson_mem > 0) {
    gsl::vector_free(&z_acc);
  }

  return status;
}

template <typename T, typename M, typename P>
PogsImplementation<T, M, P>::~PogsImplementation() {
  delete [] _de;
  delete [] _z;
  delete [] _zt;
  _de = _z = _zt = 0;

  delete [] _x;
  delete [] _y;
  delete [] _mu;
  delete [] _lambda;
  _x = _y = _mu = _lambda = 0;
}

// Pogs for separable problems
namespace {
template <typename T>
class PogsObjectiveSeparable : public PogsObjective<T> {
 private:
   std::vector<FunctionObj<T> > f, g;
 public:
  PogsObjectiveSeparable(const std::vector<FunctionObj<T> >& f,
                         const std::vector<FunctionObj<T> >& g)
      : f(f), g(g) { }

  T evaluate(const T *x, const T *y) const {
    return FuncEval(f, y) + FuncEval(g, x);
  }

  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    ProxEval(g, rho, x_in, x_out);
    ProxEval(f, rho, y_in, y_out);
  }

  void scale(const T *d, const T *e) {
    auto divide = [](FunctionObj<T> fi, T di) {
      fi.a /= di; fi.d /= di; fi.e /= di * di; return fi;
    };
    std::transform(f.begin(), f.end(), d, f.begin(), divide);
    auto multiply = [](FunctionObj<T> gi, T ei) {
      gi.a *= ei; gi.d *= ei; gi.e *= ei * ei; return gi;
    };
    std::transform(g.begin(), g.end(), e, g.begin(), multiply);
  }

  void constrain_d(T *d) const { }
  void constrain_e(T *e) const { }
};
}  // namespace

// Implementation of PogsSeparable
template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::PogsSeparable(const M& A)
    : PogsImplementation<T, M, P>(A) { }

template <typename T, typename M, typename P>
PogsSeparable<T, M, P>::~PogsSeparable() { }

template <typename T, typename M, typename P>
PogsStatus PogsSeparable<T, M, P>::Solve(const std::vector<FunctionObj<T>>& f,
                                         const std::vector<FunctionObj<T>>& g) {
  PogsObjectiveSeparable<T> pogs_obj(f, g);
  return this->PogsImplementation<T, M, P>::Solve(&pogs_obj);
}

// Pogs for cone problems
namespace {
template <typename T>
class PogsObjectiveCone : public PogsObjective<T> {
 private:
  T c_scale;
  std::vector<T> b, c;
  std::vector<T> P;
  const std::vector<ConeConstraintRaw> &Kx, &Ky;
  mutable std::vector<T> L;
  mutable T rho_cached;
  mutable bool factorized;
 public:
  PogsObjectiveCone(const std::vector<T>& b,
                    const std::vector<T>& c,
                    const std::vector<T>& P,
                    const std::vector<ConeConstraintRaw>& Kx,
                    const std::vector<ConeConstraintRaw>& Ky)
      : c_scale(static_cast<T>(1)),
        b(b),
        c(c),
        P(P),
        Kx(Kx),
        Ky(Ky),
        rho_cached(std::numeric_limits<T>::quiet_NaN()),
        factorized(false) {
    if (!this->P.empty()) {
      L.resize(this->P.size());
    }
  }

  T evaluate(const T *x, const T*) const {
    T val = std::inner_product(c.begin(), c.end(), x, static_cast<T>(0));
    if (!P.empty()) {
      const size_t n = c.size();
      T quad = static_cast<T>(0);
      for (size_t i = 0; i < n; ++i) {
        const T xi = x[i];
        const T *row = &P[i * n];
        for (size_t j = 0; j < n; ++j) {
          quad += xi * row[j] * x[j];
        }
      }
      val += static_cast<T>(0.5) * quad;
    }
    return val / c_scale;
  }

  void prox(const T *x_in, const T *y_in, T *x_out, T *y_out, T rho) const {
    if (P.empty()) {
      memcpy(x_out, x_in, c.size() * sizeof(T));
      auto x_updater = [rho](T ci, T xi) { return xi - ci / rho; };
      std::transform(c.begin(), c.end(), x_out, x_out, x_updater);
    } else {
      const size_t n = c.size();
      if (!factorized || rho != rho_cached) {
        memcpy(L.data(), P.data(), P.size() * sizeof(T));
        for (size_t i = 0; i < n; ++i) {
          L[i * n + i] += rho;
        }
        gsl::matrix<T, CblasRowMajor> L_mat =
            gsl::matrix_view_array<T, CblasRowMajor>(L.data(), n, n);
        gsl::linalg_cholesky_decomp(&L_mat);
        rho_cached = rho;
        factorized = true;
      }
      for (size_t i = 0; i < n; ++i) {
        x_out[i] = rho * x_in[i] - c[i];
      }
      gsl::matrix<T, CblasRowMajor> L_mat =
          gsl::matrix_view_array<T, CblasRowMajor>(L.data(), n, n);
      gsl::vector<T> x_vec = gsl::vector_view_array(x_out, n);
      gsl::linalg_cholesky_svx(&L_mat, &x_vec);
    }

    memcpy(y_out, y_in, b.size() * sizeof(T));
    std::transform(b.begin(), b.end(), y_out, y_out, std::minus<T>());

    ProxEvalConeCpu(Kx, c.size(), x_out, x_out);
    ProxEvalConeCpu(Ky, b.size(), y_out, y_out);

    std::transform(b.begin(), b.end(), y_out, y_out, std::minus<T>());
  }

  void scale(const T *d, const T *e) {
    std::transform(c.begin(), c.end(), e, c.begin(), std::multiplies<T>());
    std::transform(b.begin(), b.end(), d, b.begin(), std::multiplies<T>());
    if (!P.empty()) {
      const size_t n = c.size();
      for (size_t i = 0; i < n; ++i) {
        const T ei = e[i];
        T *row = &P[i * n];
        for (size_t j = 0; j < n; ++j) {
          row[j] *= ei * e[j];
        }
      }
      factorized = false;
    }

    T sum_sq = 0;
    for (T ci : c) {
      sum_sq += ci * ci;
    }
    if (sum_sq > static_cast<T>(0)) {
      c_scale = 1 / std::sqrt(sum_sq);
      for (T &ci : c) {
        ci *= c_scale;
      }
      if (!P.empty()) {
        for (T &pi : P) {
          pi *= c_scale;
        }
        factorized = false;
      }
    } else {
      c_scale = static_cast<T>(1);
    }
  }

  // Average the e_i in Kx
  void constrain_e(T *e) const {
    for (auto& cone : Kx) {
      if (IsSeparable(cone.cone))
        continue;
      T sum = static_cast<T>(0.);
      for (int i = 0; i < cone.size; ++i)
        sum += e[cone.idx[i]];
      for (int i = 0; i < cone.size; ++i)
        e[cone.idx[i]] = sum / cone.size;
    }
  }

  // Average the d_i in Ky
  void constrain_d(T *d) const {
    for (auto& cone : Ky) {
      if (IsSeparable(cone.cone))
        continue;
      T sum = static_cast<T>(0.);
      for (int i = 0; i < cone.size; ++i)
        sum += d[cone.idx[i]];
      for (int i = 0; i < cone.size; ++i)
        d[cone.idx[i]] = sum / cone.size;
    }
  }

  bool UseExactTol() const { return true; }
};

template <typename T>
class PogsObjectiveHsdeScale : public PogsObjective<T> {
 private:
  const std::vector<ConeConstraintRaw> &Kx, &Ky;
 public:
  PogsObjectiveHsdeScale(const std::vector<ConeConstraintRaw>& Kx,
                         const std::vector<ConeConstraintRaw>& Ky)
      : Kx(Kx), Ky(Ky) { }

  T evaluate(const T*, const T*) const { return static_cast<T>(0); }
  void prox(const T*, const T*, T*, T*, T) const { }
  void scale(const T*, const T*) { }

  void constrain_e(T *e) const {
    for (auto& cone : Kx) {
      if (IsSeparable(cone.cone))
        continue;
      T sum = static_cast<T>(0.);
      for (int i = 0; i < cone.size; ++i)
        sum += e[cone.idx[i]];
      for (int i = 0; i < cone.size; ++i)
        e[cone.idx[i]] = sum / cone.size;
    }
  }

  void constrain_d(T *d) const {
    for (auto& cone : Ky) {
      if (IsSeparable(cone.cone))
        continue;
      T sum = static_cast<T>(0.);
      for (int i = 0; i < cone.size; ++i)
        sum += d[cone.idx[i]];
      for (int i = 0; i < cone.size; ++i)
        d[cone.idx[i]] = sum / cone.size;
    }
  }
};

inline Cone DualCone(Cone cone) {
  switch (cone) {
    case kConeZero: return kConeZero;
    case kConeNonNeg: return kConeNonNeg;
    case kConeNonPos: return kConeNonPos;
    case kConeSoc: return kConeSoc;
    case kConeSdp: return kConeSdp;
    case kConeExpPrimal: return kConeExpDual;
    case kConeExpDual: return kConeExpPrimal;
    default: return kConeZero;
  }
}

inline void BuildDualCones(const std::vector<ConeConstraintRaw>& Ky,
                           std::vector<ConeConstraintRaw> *Ky_dual) {
  Ky_dual->clear();
  Ky_dual->reserve(Ky.size());
  for (const auto& cone : Ky) {
    if (cone.cone == kConeZero) {
      continue;  // Dual of zero cone is free.
    }
    ConeConstraintRaw raw;
    raw.cone = DualCone(cone.cone);
    raw.size = cone.size;
    raw.idx = cone.idx;  // Borrow indices; Ky owns the memory.
    Ky_dual->push_back(raw);
  }
}

template <typename T>
void ComputeColumnNormsSquared(const MatrixDense<T>& A, T* diag) {
  size_t m = A.Rows();
  size_t n = A.Cols();
  const T* data = A.Data();

  std::memset(diag, 0, n * sizeof(T));

  if (A.Order() == MatrixDense<T>::COL) {
    for (size_t j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      const T* col = data + j * m;
      for (size_t i = 0; i < m; ++i) {
        sum += col[i] * col[i];
      }
      diag[j] = sum;
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      const T* row = data + i * n;
      for (size_t j = 0; j < n; ++j) {
        diag[j] += row[j] * row[j];
      }
    }
  }
}

template <typename T>
void ComputeRowNormsSquared(const MatrixDense<T>& A, T* diag) {
  size_t m = A.Rows();
  size_t n = A.Cols();
  const T* data = A.Data();

  std::memset(diag, 0, m * sizeof(T));

  if (A.Order() == MatrixDense<T>::ROW) {
    for (size_t i = 0; i < m; ++i) {
      T sum = static_cast<T>(0);
      const T* row = data + i * n;
      for (size_t j = 0; j < n; ++j) {
        sum += row[j] * row[j];
      }
      diag[i] = sum;
    }
  } else {
    for (size_t j = 0; j < n; ++j) {
      const T* col = data + j * m;
      for (size_t i = 0; i < m; ++i) {
        diag[i] += col[i] * col[i];
      }
    }
  }
}

template <typename T>
void ComputeColumnNormsSquared(const MatrixSparse<T>& A, T* diag) {
  size_t n = A.Cols();
  const T* val = A.Data();
  const POGS_INT* ind = A.Ind();
  const POGS_INT* ptr = A.Ptr();

  std::memset(diag, 0, n * sizeof(T));

  if (A.Order() == MatrixSparse<T>::COL) {
    for (size_t j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      for (POGS_INT k = ptr[j]; k < ptr[j + 1]; ++k) {
        sum += val[k] * val[k];
      }
      diag[j] = sum;
    }
  } else {
    for (size_t i = 0; i < A.Rows(); ++i) {
      for (POGS_INT k = ptr[i]; k < ptr[i + 1]; ++k) {
        diag[ind[k]] += val[k] * val[k];
      }
    }
  }
}

template <typename T>
void ComputeRowNormsSquared(const MatrixSparse<T>& A, T* diag) {
  size_t m = A.Rows();
  const T* val = A.Data();
  const POGS_INT* ind = A.Ind();
  const POGS_INT* ptr = A.Ptr();

  std::memset(diag, 0, m * sizeof(T));

  if (A.Order() == MatrixSparse<T>::ROW) {
    for (size_t i = 0; i < m; ++i) {
      T sum = static_cast<T>(0);
      for (POGS_INT k = ptr[i]; k < ptr[i + 1]; ++k) {
        sum += val[k] * val[k];
      }
      diag[i] = sum;
    }
  } else {
    for (size_t j = 0; j < A.Cols(); ++j) {
      for (POGS_INT k = ptr[j]; k < ptr[j + 1]; ++k) {
        diag[ind[k]] += val[k] * val[k];
      }
    }
  }
}

template <typename T>
void ComputePDiagAndColNorms(const std::vector<T>& P, size_t n,
                             std::vector<T> *diag,
                             std::vector<T> *col_norms) {
  diag->assign(n, static_cast<T>(0));
  col_norms->assign(n, static_cast<T>(0));
  if (P.empty())
    return;
  for (size_t i = 0; i < n; ++i) {
    (*diag)[i] = P[i * n + i];
  }
  for (size_t j = 0; j < n; ++j) {
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < n; ++i) {
      const T v = P[i * n + j];
      sum += v * v;
    }
    (*col_norms)[j] = sum;
  }
}

template <typename T>
inline T Dot(const T *a, const T *b, size_t n) {
  T sum = static_cast<T>(0);
  for (size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template <typename T>
inline T Norm2(const T *a, size_t n) {
  return std::sqrt(Dot(a, a, n));
}

template <typename T, typename M>
struct HsdeOperator {
  const M& A;
  const std::vector<T>& b;
  const std::vector<T>& c;
  const std::vector<T>& P;
  size_t m;
  size_t n;
  std::vector<T> ax;
  std::vector<T> aty;

  HsdeOperator(const M& A,
               const std::vector<T>& b,
               const std::vector<T>& c,
               const std::vector<T>& P)
      : A(A), b(b), c(c), P(P), m(A.Rows()), n(A.Cols()),
        ax(m, static_cast<T>(0)), aty(n, static_cast<T>(0)) { }

  void ApplyQ(const T *in, T *out) {
    const T *x = in;
    const T *y = in + n;
    const T tau = in[n + m];

    A.Mul('n', static_cast<T>(1), x, static_cast<T>(0), ax.data());
    A.Mul('t', static_cast<T>(1), y, static_cast<T>(0), aty.data());

    if (!P.empty()) {
      for (size_t i = 0; i < n; ++i) {
        const T *row = &P[i * n];
        T sum = static_cast<T>(0);
        for (size_t j = 0; j < n; ++j) {
          sum += row[j] * x[j];
        }
        out[i] = sum;
      }
    } else {
      std::fill(out, out + n, static_cast<T>(0));
    }
    for (size_t i = 0; i < n; ++i) {
      out[i] += aty[i] + c[i] * tau;
    }
    for (size_t i = 0; i < m; ++i) {
      out[n + i] = -ax[i] + b[i] * tau;
    }
    out[n + m] = -Dot(c.data(), x, n) - Dot(b.data(), y, m);
  }

  void ApplyQT(const T *in, T *out) {
    const T *x = in;
    const T *y = in + n;
    const T tau = in[n + m];

    A.Mul('n', static_cast<T>(1), x, static_cast<T>(0), ax.data());
    A.Mul('t', static_cast<T>(1), y, static_cast<T>(0), aty.data());

    if (!P.empty()) {
      for (size_t i = 0; i < n; ++i) {
        const T *row = &P[i * n];
        T sum = static_cast<T>(0);
        for (size_t j = 0; j < n; ++j) {
          sum += row[j] * x[j];
        }
        out[i] = sum;
      }
    } else {
      std::fill(out, out + n, static_cast<T>(0));
    }
    for (size_t i = 0; i < n; ++i) {
      out[i] += -aty[i] - c[i] * tau;
    }
    for (size_t i = 0; i < m; ++i) {
      out[n + i] = ax[i] - b[i] * tau;
    }
    out[n + m] = Dot(c.data(), x, n) + Dot(b.data(), y, m);
  }

  void ApplyQScaled(const T *in, T *out, T rho) {
    ApplyQ(in, out);
    for (size_t i = 0; i < n + m + 1; ++i)
      out[i] *= rho;
  }

  void ApplyQTScaled(const T *in, T *out, T rho) {
    ApplyQT(in, out);
    for (size_t i = 0; i < n + m + 1; ++i)
      out[i] *= rho;
  }

  void ApplyNormal(const T *in, T *out, std::vector<T> *tmp) {
    ApplyQ(in, out);
    for (size_t i = 0; i < n + m + 1; ++i) {
      (*tmp)[i] = in[i] + out[i];
    }
    ApplyQT(tmp->data(), out);
    for (size_t i = 0; i < n + m + 1; ++i) {
      out[i] = (*tmp)[i] + out[i];
    }
  }

  void ApplyNormalScaled(const T *in, T *out, std::vector<T> *tmp, T rho) {
    ApplyQScaled(in, out, rho);
    for (size_t i = 0; i < n + m + 1; ++i) {
      (*tmp)[i] = in[i] + out[i];
    }
    ApplyQTScaled(tmp->data(), out, rho);
    for (size_t i = 0; i < n + m + 1; ++i) {
      out[i] = (*tmp)[i] + out[i];
    }
  }
};

// SMW-based HSDE linear solver using graph projector
// Solves (I + σQ) ũ = w where Q is the HSDE operator
// Uses Sherman-Morrison-Woodbury to reduce to graph projection: (I + σ²AᵀA)
template <typename T, typename M, typename P>
class HsdeLinearSolverSMW {
  const M& A;
  const std::vector<T>& b;
  const std::vector<T>& c;
  const std::vector<T>& P_mat;
  P& projector;
  const size_t n, m;
  const T sigma;  // scaling parameter (typically 1)

  // Precomputed SMW quantities: t = M⁻¹h, s = 1 + hᵀt
  std::vector<T> t_x;   // x-component of t = M⁻¹h
  std::vector<T> t_y;   // y-component of t = M⁻¹h
  T s;                  // SMW denominator: 1 + hᵀt

  // Work vectors
  std::vector<T> q_x, q_y;
  std::vector<T> p_x, p_y;
  std::vector<T> tmp_n, tmp_m;
  bool has_P;

 public:
  HsdeLinearSolverSMW(const M& A_, const std::vector<T>& b_,
                       const std::vector<T>& c_, const std::vector<T>& P_mat_,
                       P& proj, T sigma_ = static_cast<T>(1))
      : A(A_), b(b_), c(c_), P_mat(P_mat_), projector(proj),
        n(A_.Cols()), m(A_.Rows()), sigma(sigma_),
        t_x(n), t_y(m), s(static_cast<T>(1)),
        q_x(n), q_y(m), p_x(n), p_y(m), tmp_n(n), tmp_m(m),
        has_P(!P_mat_.empty()) {}

  // Setup: compute t = M⁻¹h and s = 1 + hᵀt
  void Setup() {
    // For QP: M = [I+σP, σAᵀ; -σA, I], h = [σc; σb]
    // For LP: M = [I, σAᵀ; -σA, I], h = [σc; σb]
    //
    // M-solve via elimination:
    // For QP: (I + σP + σ²AᵀA) t_x = σc - σ²Aᵀb, t_y = σb + σA t_x
    // For LP: (I + σ²AᵀA) t_x = σc - σ²Aᵀb, t_y = σb + σA t_x

    const T kOne = static_cast<T>(1);
    const T kZero = static_cast<T>(0);
    const T sigma_sq = sigma * sigma;

    // Compute rhs = σc - σ²Aᵀb
    A.Mul('t', -sigma_sq, b.data(), kZero, tmp_n.data());
    for (size_t i = 0; i < n; ++i)
      tmp_n[i] += sigma * c[i];

    // For QP with P matrix: solve (I + σP + σ²AᵀA) t_x = rhs
    // For LP (no P): solve (I + σ²AᵀA) t_x = rhs via projector
    if (has_P) {
      // QP case: need CG on (I + σP + σ²AᵀA)
      SolveMSystemQP(tmp_n.data(), t_x.data());
    } else {
      // LP case: use projector - solve (I + σ²AᵀA) t_x = tmp_n
      // Projector solves: min ||Ax - y0||² + s||x - x0||²
      // KKT: (AᵀA + sI)x = s*x0 + Aᵀy0
      // With s = σ², x0 = tmp_n/σ², y0 = 0:
      //   (AᵀA + σ²I)x = tmp_n
      // But we want (I + σ²AᵀA) - need to rescale
      // Let α = σ², then (I + α AᵀA)x = rhs
      // Divide: AᵀA x + (1/α)x = rhs/α
      // So use projector with s = 1/σ², x0 = rhs, y0 = 0, then scale

      // Actually simpler: Project(x0, y0, s, x, y, tol) solves
      // min ||Ax - y0||² + s||x - x0||² which has KKT:
      // (AᵀA + sI)x = AᵀAy0 + s*x0  ... wait that's not right either

      // Let me re-examine projector_direct_dense.cpp more carefully
      // Actually it does: x = (AᵀA + sI)⁻¹(s*x0 + Aᵀy0)
      // So with s=1, x0=tmp_n, y0=0: x = (AᵀA + I)⁻¹ tmp_n  ✓
      // This is exactly (I + AᵀA)⁻¹ tmp_n when σ=1

      // For general σ: (I + σ²AᵀA)⁻¹ = σ⁻²(σ⁻²I + AᵀA)⁻¹
      // Use projector with s = 1/σ², x0 = σ²*tmp_n, y0 = 0
      // Result: (AᵀA + s I)⁻¹(s*x0) = (AᵀA + σ⁻²I)⁻¹(σ⁻²*σ²*tmp_n)
      //       = (AᵀA + σ⁻²I)⁻¹ tmp_n
      // Hmm that gives (σ⁻²I + AᵀA)⁻¹, need to scale output by σ²

      // For σ=1: just use projector directly
      if (std::abs(sigma - kOne) < static_cast<T>(1e-10)) {
        // s=1: (I + AᵀA)t_x = tmp_n
        std::fill(tmp_m.begin(), tmp_m.end(), kZero);
        projector.Project(tmp_n.data(), tmp_m.data(), kOne,
                          t_x.data(), p_y.data(), static_cast<T>(1e-10));
      } else {
        // General σ: need to handle scaling
        // For now, just use σ=1 case
        std::fill(tmp_m.begin(), tmp_m.end(), kZero);
        projector.Project(tmp_n.data(), tmp_m.data(), sigma_sq,
                          t_x.data(), p_y.data(), static_cast<T>(1e-10));
      }
    }

    // t_y = σb + σA t_x
    A.Mul('n', sigma, t_x.data(), kZero, t_y.data());
    for (size_t i = 0; i < m; ++i)
      t_y[i] += sigma * b[i];

    // s = 1 + hᵀt = 1 + σcᵀt_x + σbᵀt_y
    s = kOne;
    for (size_t i = 0; i < n; ++i)
      s += sigma * c[i] * t_x[i];
    for (size_t i = 0; i < m; ++i)
      s += sigma * b[i] * t_y[i];
  }

  // Solve (I + σQ) ũ = w
  // Returns solution in w (in-place)
  //
  // System structure:
  //   [M    h ] [ũ_xy]   [w_xy]
  //   [-hᵀ  1 ] [ũ_τ ] = [w_τ ]
  //
  // where M = [I, σAᵀ; -σA, I] and h = [σc; σb]
  //
  // Solution via back-substitution:
  //   1. p = M⁻¹ w_xy
  //   2. ũ_τ = (w_τ + hᵀp) / s  where s = 1 + hᵀt, t = M⁻¹h
  //   3. ũ_xy = p - t * ũ_τ
  void Solve(T *w) {
    const T kOne = static_cast<T>(1);
    const T kZero = static_cast<T>(0);
    const T sigma_sq = sigma * sigma;

    T *w_x = w;
    T *w_y = w + n;
    T w_tau = w[n + m];

    // Step 1: p = M⁻¹ w_xy (solve M p = w_xy directly, NOT w_xy - w_τ h)
    // M = [I, σAᵀ; -σA, I]
    // Elimination: (I + σ²AᵀA) p_x = w_x - σAᵀw_y
    //              p_y = w_y + σA p_x
    A.Mul('t', -sigma, w_y, kZero, tmp_n.data());
    for (size_t i = 0; i < n; ++i)
      tmp_n[i] += w_x[i];

    if (has_P) {
      SolveMSystemQP(tmp_n.data(), p_x.data());
    } else {
      // LP case: use projector
      // Solve (I + σ²AᵀA) p_x = tmp_n
      if (std::abs(sigma - kOne) < static_cast<T>(1e-10)) {
        std::fill(tmp_m.begin(), tmp_m.end(), kZero);
        projector.Project(tmp_n.data(), tmp_m.data(), kOne,
                          p_x.data(), q_y.data(), static_cast<T>(1e-10));
      } else {
        std::fill(tmp_m.begin(), tmp_m.end(), kZero);
        projector.Project(tmp_n.data(), tmp_m.data(), sigma_sq,
                          p_x.data(), q_y.data(), static_cast<T>(1e-10));
      }
    }

    // p_y = w_y + σA p_x
    A.Mul('n', sigma, p_x.data(), kZero, p_y.data());
    for (size_t i = 0; i < m; ++i)
      p_y[i] += w_y[i];

    // Step 2: compute hᵀp = σcᵀp_x + σbᵀp_y
    T h_dot_p = kZero;
    for (size_t i = 0; i < n; ++i)
      h_dot_p += sigma * c[i] * p_x[i];
    for (size_t i = 0; i < m; ++i)
      h_dot_p += sigma * b[i] * p_y[i];

    // Step 3: ũ_τ = (w_τ + hᵀp) / s
    T u_tau = (w_tau + h_dot_p) / s;

    // Step 4: ũ_xy = p - t * ũ_τ
    for (size_t i = 0; i < n; ++i)
      w_x[i] = p_x[i] - t_x[i] * u_tau;
    for (size_t i = 0; i < m; ++i)
      w_y[i] = p_y[i] - t_y[i] * u_tau;
    w[n + m] = u_tau;
  }

 private:
  // CG solver for QP case: (I + σP + σ²AᵀA) x = rhs
  void SolveMSystemQP(const T *rhs, T *x) {
    const T kZero = static_cast<T>(0);
    const int max_iter = static_cast<int>(std::min<size_t>(n * 10, 5000u));
    const T tol = static_cast<T>(1e-12);

    std::vector<T> r(n), p(n), Ap(n);

    // x = 0 initially
    std::fill(x, x + n, kZero);

    // r = rhs - (I + σP + σ²AᵀA) x = rhs (since x=0)
    std::copy(rhs, rhs + n, r.data());

    // p = r
    std::copy(r.begin(), r.end(), p.data());

    T rr = Dot(r.data(), r.data(), n);
    T rhs_norm = std::sqrt(Dot(rhs, rhs, n));
    if (rhs_norm < tol) return;

    int k = 0;
    for (; k < max_iter; ++k) {
      // Ap = (I + σP + σ²AᵀA) p
      ApplyMOperatorQP(p.data(), Ap.data());

      T pAp = Dot(p.data(), Ap.data(), n);
      if (std::abs(pAp) < static_cast<T>(1e-30)) break;

      T alpha = rr / pAp;
      for (size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
      }

      T rr_new = Dot(r.data(), r.data(), n);
      if (std::sqrt(rr_new) < tol * rhs_norm) break;

      T beta = rr_new / rr;
      for (size_t i = 0; i < n; ++i)
        p[i] = r[i] + beta * p[i];

      rr = rr_new;
    }
  }

  // Apply (I + σP + σ²AᵀA) to vector
  void ApplyMOperatorQP(const T *in, T *out) {
    const T kOne = static_cast<T>(1);
    const T kZero = static_cast<T>(0);
    const T sigma_sq = sigma * sigma;

    // out = in + σ P in + σ² AᵀA in
    std::copy(in, in + n, out);

    // Add σ P in
    if (has_P) {
      for (size_t i = 0; i < n; ++i) {
        const T *row = &P_mat[i * n];
        T sum = kZero;
        for (size_t j = 0; j < n; ++j)
          sum += row[j] * in[j];
        out[i] += sigma * sum;
      }
    }

    // Add σ² AᵀA in
    A.Mul('n', kOne, in, kZero, tmp_m.data());
    A.Mul('t', sigma_sq, tmp_m.data(), kOne, out);
  }
};

template <typename T, typename M>
int CgSolveNormal(HsdeOperator<T, M>& op,
                  const std::vector<T>& rhs,
                  T rho,
                  const std::vector<T>& inv_diag,
                  std::vector<T> *x,
                  std::vector<T> *tmp,
                  std::vector<T> *r,
                  std::vector<T> *z,
                  std::vector<T> *p,
                  std::vector<T> *Ap,
                  T tol,
                  int max_iter) {
  const size_t n = rhs.size();
  op.ApplyNormalScaled(x->data(), Ap->data(), tmp, rho);
  for (size_t i = 0; i < n; ++i) {
    (*r)[i] = rhs[i] - (*Ap)[i];
    (*z)[i] = (*r)[i] * inv_diag[i];
    (*p)[i] = (*z)[i];
  }

  T rz_old = Dot(r->data(), z->data(), n);
  T rhs_norm = Norm2(rhs.data(), n);
  if (rhs_norm == static_cast<T>(0))
    return 0;

  int k = 0;
  for (; k < max_iter; ++k) {
    op.ApplyNormalScaled(p->data(), Ap->data(), tmp, rho);
    T pAp = Dot(p->data(), Ap->data(), n);
    if (std::abs(pAp) <= static_cast<T>(1e-20))
      break;
    T alpha = rz_old / pAp;
    for (size_t i = 0; i < n; ++i) {
      (*x)[i] += alpha * (*p)[i];
      (*r)[i] -= alpha * (*Ap)[i];
    }
    T r_norm = Norm2(r->data(), n);
    if (r_norm <= tol * rhs_norm)
      break;
    for (size_t i = 0; i < n; ++i) {
      (*z)[i] = (*r)[i] * inv_diag[i];
    }
    T rz_new = Dot(r->data(), z->data(), n);
    T beta = rz_new / rz_old;
    for (size_t i = 0; i < n; ++i) {
      (*p)[i] = (*z)[i] + beta * (*p)[i];
    }
    rz_old = rz_new;
  }
  return k + 1;
}

template <typename T, typename M>
PogsStatus SolveHsdeCone(const M& A,
                         const std::vector<T>& b_orig,
                         const std::vector<T>& c_orig,
                         const std::vector<T>& P_orig,
                         const std::vector<ConeConstraintRaw>& Ky,
                         const T *d,
                         const T *e,
                         T rho,
                         T abs_tol,
                         T rel_tol,
                         unsigned int max_iter,
                         unsigned int verbose,
                         std::vector<T> *x_out,
                         std::vector<T> *y_out,
                         std::vector<T> *lambda_out,
                         T *optval_out,
                         unsigned int *final_iter_out) {
  const size_t m = A.Rows();
  const size_t n = A.Cols();
  const size_t dim = n + m + 1;
  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);
  const T kAlphaMin = static_cast<T>(1.0);
  const T kAlphaMax = static_cast<T>(1.7);
  const T kAlphaGrow = static_cast<T>(1.02);
  const T kMinDiag = static_cast<T>(1e-8);
  const T kTauTol = static_cast<T>(1e-8);
  const T kKappaTol = static_cast<T>(1e-6);
  const T kLinTolMin = static_cast<T>(1e-10);
  const T kLinTolMax = static_cast<T>(1e-2);
  const T kLinTolScale = static_cast<T>(1e-1);
  const T kStaticReg = static_cast<T>(1e-8);
  const T kStaticRegProp =
      static_cast<T>(std::numeric_limits<T>::epsilon()
                     * std::numeric_limits<T>::epsilon());
  const T kDynamicRegDelta = static_cast<T>(2e-7);
  const bool kIterRefine = true;
  const int kIterRefineMax = 10;
  const T kIterRefineRelTol = static_cast<T>(1e-12);
  const T kIterRefineAbsTol = static_cast<T>(1e-12);
  const T kIterRefineStopRatio = static_cast<T>(5.0);
  const bool kUseAnderson = true;
  const size_t kAndersonMem = 5;
  const size_t kAndersonStart = 10;
  const T kAndersonSafety = static_cast<T>(2.0);
  const size_t kAndersonRejectReset = 5;
  const size_t kDirectLimit = 2000;
  (void)rho;
  const T kRho = kOne;
  const T kRhoSq = kOne;

  std::vector<T> b(b_orig);
  std::vector<T> c(c_orig);
  std::vector<T> P(P_orig);

  for (size_t i = 0; i < m; ++i)
    b[i] *= d[i];
  for (size_t i = 0; i < n; ++i)
    c[i] *= e[i];
  if (!P.empty()) {
    for (size_t i = 0; i < n; ++i) {
      const T ei = e[i];
      T *row = &P[i * n];
      for (size_t j = 0; j < n; ++j) {
        row[j] *= ei * e[j];
      }
    }
  }


  std::vector<ConeConstraintRaw> Ky_dual;
  BuildDualCones(Ky, &Ky_dual);

  HsdeOperator<T, M> op(A, b, c, P);

  std::vector<T> diag_x(n, kZero);
  std::vector<T> diag_y(m, kZero);
  std::vector<T> diag_p(n, kZero);
  std::vector<T> col_p(n, kZero);
  ComputeColumnNormsSquared(A, diag_x.data());
  ComputeRowNormsSquared(A, diag_y.data());
  ComputePDiagAndColNorms(P, n, &diag_p, &col_p);

  std::vector<T> inv_diag(dim, kOne);
  for (size_t i = 0; i < n; ++i) {
    T val = kOne + static_cast<T>(2) * kRho * diag_p[i]
        + kRhoSq * (col_p[i] + diag_x[i] + c[i] * c[i]);
    inv_diag[i] = kOne / std::max(val, kMinDiag);
  }
  for (size_t i = 0; i < m; ++i) {
    T val = kOne + kRhoSq * (diag_y[i] + b[i] * b[i]);
    inv_diag[n + i] = kOne / std::max(val, kMinDiag);
  }
  T tau_diag = kOne + kRhoSq * (Dot(c.data(), c.data(), n)
      + Dot(b.data(), b.data(), m));
  inv_diag[n + m] = kOne / std::max(tau_diag, kMinDiag);

  // SMW-based solver using graph projector (new approach - avoids normal equations)
  bool use_smw = false;
  std::unique_ptr<ProjectorDirect<T, M>> smw_projector;
  std::unique_ptr<HsdeLinearSolverSMW<T, M, ProjectorDirect<T, M>>> smw_solver;

  if constexpr (std::is_same<M, MatrixDense<T>>::value) {
    // Use SMW for LP only. QP with HSDE has a fundamental formulation issue
    // (the optimal solution is not a fixed point of the HSDE iteration).
    // For QP, users should use OSQP, SCS, or CLARABEL instead.
    use_smw = P.empty();
    if (use_smw) {
      smw_projector = std::make_unique<ProjectorDirect<T, M>>(A);
      smw_projector->Init();
      smw_solver = std::make_unique<HsdeLinearSolverSMW<T, M, ProjectorDirect<T, M>>>(
          A, b, c, P, *smw_projector, kRho);
      smw_solver->Setup();
      if (verbose > 0) {
        Printf("HSDE: Using SMW linear solver\n");
      }
    }
  }

  // Fallback: normal equations (for sparse matrices or if SMW fails)
  bool use_direct = false;
  std::vector<T> normal_mat;
  std::vector<T> normal_base;
  gsl::matrix<T, CblasRowMajor> normal_view;
  T normal_reg = kZero;
  if (!use_smw) {
    if constexpr (std::is_same<M, MatrixDense<T>>::value) {
      if (dim <= kDirectLimit) {
        use_direct = true;
        normal_mat.assign(dim * dim, kZero);

        std::vector<T> mat(dim * dim, kZero);
        const T *A_data = A.Data();
        const bool row_major = (A.Order() == MatrixDense<T>::ROW);

        for (size_t i = 0; i < n; ++i) {
          for (size_t j = 0; j < n; ++j) {
            const T pij = P.empty() ? kZero : P[i * n + j];
            mat[i * dim + j] = (i == j ? kOne : kZero) + kRho * pij;
          }
        }
        for (size_t i = 0; i < m; ++i) {
          const size_t row = n + i;
          mat[row * dim + row] = kOne;
          mat[row * dim + (n + m)] = kRho * b[i];
        }
        for (size_t i = 0; i < n; ++i) {
          mat[i * dim + (n + m)] = kRho * c[i];
          mat[(n + m) * dim + i] = -kRho * c[i];
        }
        for (size_t i = 0; i < m; ++i) {
          mat[(n + m) * dim + (n + i)] = -kRho * b[i];
        }
        mat[(n + m) * dim + (n + m)] = kOne;

        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < n; ++j) {
            const T aij = row_major ? A_data[i * n + j] : A_data[j * m + i];
            mat[j * dim + (n + i)] = kRho * aij;
            mat[(n + i) * dim + j] = -kRho * aij;
          }
        }

        gsl::matrix<T, CblasRowMajor> mat_view =
            gsl::matrix_view_array<T, CblasRowMajor>(mat.data(), dim, dim);
        normal_view =
            gsl::matrix_view_array<T, CblasRowMajor>(normal_mat.data(), dim, dim);
        gsl::blas_gemm(CblasTrans, CblasNoTrans, kOne, &mat_view, &mat_view,
                       kZero, &normal_view);

        normal_base = normal_mat;
        T max_diag = kZero;
        for (size_t i = 0; i < dim; ++i) {
          const T diag = std::abs(normal_base[i * dim + i]);
          if (diag > max_diag)
            max_diag = diag;
        }
        const T static_reg = kStaticReg + kStaticRegProp * max_diag;
        const T dynamic_delta = kDynamicRegDelta * std::max(kOne, max_diag);
        const size_t kMaxRegAttempts = 5;
        bool factored = false;
        for (size_t attempt = 0; attempt < kMaxRegAttempts; ++attempt) {
          normal_mat = normal_base;
          normal_reg = static_reg + static_cast<T>(attempt) * dynamic_delta;
          for (size_t i = 0; i < dim; ++i)
            normal_mat[i * dim + i] += normal_reg;
          normal_view = gsl::matrix_view_array<T, CblasRowMajor>(
              normal_mat.data(), dim, dim);
          gsl::linalg_cholesky_decomp(&normal_view);
          factored = true;
          for (size_t i = 0; i < dim; ++i) {
            const T diag = normal_mat[i * dim + i];
            if (!std::isfinite(diag) || diag <= kZero) {
              factored = false;
              break;
            }
          }
          if (factored)
            break;
        }
        if (!factored) {
          use_direct = false;
          normal_reg = kZero;
        }
      }
    }
  }

  std::vector<T> u(dim, kZero);
  std::vector<T> u_prev(dim, kZero);
  std::vector<T> u_acc;
  std::vector<T> w(dim, kZero);
  std::vector<T> z(dim, kZero);
  std::vector<T> rhs(dim, kZero);
  std::vector<T> rhs_t(dim, kZero);
  std::vector<T> tmp(dim, kZero);
  std::vector<T> tmp2(dim, kZero);
  std::vector<T> r(dim, kZero);
  std::vector<T> zcg(dim, kZero);
  std::vector<T> p(dim, kZero);
  std::vector<T> Ap(dim, kZero);
  std::vector<T> x_scaled(n, kZero);
  std::vector<T> y_scaled(m, kZero);
  std::vector<T> s_scaled(m, kZero);
  std::vector<T> s_proj(m, kZero);

  const T b_norm = Norm2(b.data(), m);
  const T c_norm = Norm2(c.data(), n);

  u[n + m] = kOne;

  T fp_resid = kOne;
  T prev_resid = std::numeric_limits<T>::max();
  T alpha = kAlphaMin;
  bool converged = false;
  PogsStatus status = POGS_MAX_ITER;
  std::unique_ptr<AndersonAccelerator<T>> anderson;
  size_t anderson_rejects = 0;
  if (kUseAnderson && kAndersonMem > 0) {
    anderson = std::make_unique<AndersonAccelerator<T>>(dim, kAndersonMem);
    u_acc.assign(dim, kZero);
    if (verbose > 0) {
      Printf("HSDE Anderson acceleration enabled: mem=%u, start=%u\n",
             static_cast<unsigned int>(kAndersonMem),
             static_cast<unsigned int>(kAndersonStart));
    }
  }

  for (unsigned int k = 0; k < max_iter; ++k) {
    if (anderson)
      std::copy(u.begin(), u.end(), u_prev.begin());

    // Linear solve: w = (I + ρQ)⁻¹ u
    if (use_smw) {
      // SMW approach: solve (I + ρQ)w = u directly (no normal equations!)
      std::copy(u.begin(), u.end(), w.begin());
      smw_solver->Solve(w.data());
    } else {
      // Fallback: normal equations approach
      rhs = u;
      op.ApplyQT(rhs.data(), tmp.data());
      for (size_t i = 0; i < dim; ++i)
        rhs_t[i] = rhs[i] + kRho * tmp[i];

      if (use_direct) {
        std::copy(rhs_t.begin(), rhs_t.end(), w.begin());
        gsl::vector<T> w_vec = gsl::vector_view_array(w.data(), dim);
        gsl::linalg_cholesky_svx(&normal_view, &w_vec);
        if (kIterRefine) {
          const T rhs_norm = Norm2(rhs_t.data(), dim);
          const T ref_tol = std::max(kIterRefineAbsTol,
              kIterRefineRelTol * rhs_norm);
          T prev_r_norm = std::numeric_limits<T>::max();
          for (int it = 0; it < kIterRefineMax; ++it) {
            op.ApplyNormalScaled(w.data(), tmp.data(), &tmp2, kRho);
            if (normal_reg != kZero) {
              for (size_t i = 0; i < dim; ++i)
                tmp[i] += normal_reg * w[i];
            }
            for (size_t i = 0; i < dim; ++i)
              r[i] = rhs_t[i] - tmp[i];
            const T r_norm = Norm2(r.data(), dim);
            if (r_norm <= ref_tol)
              break;
            if (prev_r_norm < std::numeric_limits<T>::max() &&
                r_norm > prev_r_norm / kIterRefineStopRatio) {
              break;
            }
            prev_r_norm = r_norm;
            std::copy(r.begin(), r.end(), tmp2.begin());
            gsl::vector<T> delta = gsl::vector_view_array(tmp2.data(), dim);
            gsl::linalg_cholesky_svx(&normal_view, &delta);
            for (size_t i = 0; i < dim; ++i)
              w[i] += tmp2[i];
          }
        }
      } else {
        const T lin_tol = std::max(kLinTolMin,
            std::min(kLinTolMax, kLinTolScale * fp_resid));
        int cg_max = static_cast<int>(std::min<size_t>(20000u, 20 * dim));
        CgSolveNormal(op, rhs_t, kRho, inv_diag, &w, &tmp2, &r, &zcg, &p, &Ap,
                      lin_tol, cg_max);
      }
    }

    for (size_t i = 0; i < dim; ++i)
      tmp[i] = static_cast<T>(2) * w[i] - u[i];

    std::copy(tmp.begin(), tmp.end(), z.begin());
    ProxEvalConeCpu(Ky_dual, m, z.data() + n, z.data() + n);
    z[n + m] = std::max(z[n + m], kZero);

    for (size_t i = 0; i < dim; ++i)
      u[i] += alpha * (z[i] - w[i]);

    for (size_t i = 0; i < dim; ++i)
      tmp[i] = z[i] - w[i];
    fp_resid = Norm2(tmp.data(), dim);

    if (anderson && k >= kAndersonStart) {
      if (anderson->Apply(u.data(), u_prev.data(), u_acc.data())) {
        T acc_step = kZero;
        for (size_t i = 0; i < dim; ++i) {
          const T diff = u_acc[i] - u[i];
          acc_step += diff * diff;
        }
        acc_step = std::sqrt(acc_step);

        T drs_step = kZero;
        for (size_t i = 0; i < dim; ++i) {
          const T diff = u[i] - u_prev[i];
          drs_step += diff * diff;
        }
        drs_step = std::sqrt(drs_step);

        if (acc_step < kAndersonSafety * drs_step + abs_tol) {
          std::copy(u_acc.begin(), u_acc.end(), u.begin());
          anderson_rejects = 0;
          if (verbose > 3) {
            Printf("HSDE Anderson: accepted (acc_step=%.2e, drs_step=%.2e)\n",
                   acc_step, drs_step);
          }
        } else {
          ++anderson_rejects;
          if (anderson_rejects >= kAndersonRejectReset) {
            anderson->Reset();
            anderson_rejects = 0;
          }
          if (verbose > 3) {
            Printf("HSDE Anderson: rejected (acc_step=%.2e > %.1f*drs_step=%.2e)\n",
                   acc_step, kAndersonSafety, kAndersonSafety * drs_step);
          }
        }
      }
    }

    if (k % 10 == 0 || k == max_iter - 1) {
      const T *state = w.data();
      const T tau = state[n + m];
      if (tau > kTauTol) {
        for (size_t i = 0; i < n; ++i)
          x_scaled[i] = state[i] / tau;
        for (size_t i = 0; i < m; ++i)
          y_scaled[i] = state[n + i] / tau;

        A.Mul('n', kOne, x_scaled.data(), kZero, tmp.data());
        for (size_t i = 0; i < m; ++i)
          s_scaled[i] = b[i] - tmp[i];

        ProxEvalConeCpu(Ky, m, s_scaled.data(), s_proj.data());
        for (size_t i = 0; i < m; ++i)
          tmp[i] = s_scaled[i] - s_proj[i];
        T r_pri = Norm2(tmp.data(), m);
        T s_norm = Norm2(s_scaled.data(), m);

        ProxEvalConeCpu(Ky_dual, m, y_scaled.data(), s_proj.data());
        for (size_t i = 0; i < m; ++i)
          tmp[i] = y_scaled[i] - s_proj[i];
        T r_dua_cone = Norm2(tmp.data(), m);

        A.Mul('t', kOne, y_scaled.data(), kZero, tmp2.data());
        T quad = kZero;
        if (!P.empty()) {
          for (size_t i = 0; i < n; ++i) {
            const T *row = &P[i * n];
            T sum = kZero;
            for (size_t j = 0; j < n; ++j)
              sum += row[j] * x_scaled[j];
            tmp[i] = sum;
          }
          quad = Dot(x_scaled.data(), tmp.data(), n);
          for (size_t i = 0; i < n; ++i)
            tmp2[i] += tmp[i];
        }
        T at_norm = Norm2(tmp2.data(), n);
        for (size_t i = 0; i < n; ++i)
          tmp2[i] += c[i];
        T r_dua = Norm2(tmp2.data(), n);

        T eps_pri = std::sqrt(static_cast<T>(m)) * abs_tol
            + rel_tol * std::max(b_norm, s_norm);

        T eps_dua = std::sqrt(static_cast<T>(n)) * abs_tol
            + rel_tol * std::max(at_norm, c_norm);
        T y_norm = Norm2(y_scaled.data(), m);
        T eps_cone = std::sqrt(static_cast<T>(m)) * abs_tol
            + rel_tol * std::max(kOne, y_norm);

        T gap = std::abs(Dot(c.data(), x_scaled.data(), n)
            + Dot(b.data(), y_scaled.data(), m) + quad);
        T eps_gap = abs_tol + rel_tol * std::max(kOne, gap);

        T curr_resid = r_pri + r_dua + r_dua_cone + gap;
        if (curr_resid <= prev_resid * static_cast<T>(0.99)) {
          alpha = std::min(kAlphaMax, alpha * kAlphaGrow);
        } else {
          alpha = kAlphaMin;
        }
        prev_resid = curr_resid;
        if (r_pri <= eps_pri && r_dua <= eps_dua && r_dua_cone <= eps_cone
            && gap <= eps_gap) {
          converged = true;
          *final_iter_out = k;
          break;
        }
      } else {
        const T *x_hat = state;
        const T *y_hat = state + n;
        const T kappa = -Dot(c.data(), x_hat, n) - Dot(b.data(), y_hat, m);
        const T fp_tol = abs_tol * std::sqrt(static_cast<T>(dim)) + rel_tol;
        if (kappa > kKappaTol && fp_resid <= fp_tol) {
          A.Mul('n', kOne, x_hat, kZero, tmp.data());
          T ax_norm = Norm2(tmp.data(), m);

          A.Mul('t', kOne, y_hat, kZero, tmp2.data());
          T aty_norm = Norm2(tmp2.data(), n);

          ProxEvalConeCpu(Ky_dual, m, y_hat, s_proj.data());
          for (size_t i = 0; i < m; ++i)
            tmp[i] = y_hat[i] - s_proj[i];
          T y_cone = Norm2(tmp.data(), m);

          T p_norm = kZero;
          if (!P.empty()) {
            for (size_t i = 0; i < n; ++i) {
              const T *row = &P[i * n];
              T sum = kZero;
              for (size_t j = 0; j < n; ++j)
                sum += row[j] * x_hat[j];
              tmp2[i] = sum;
            }
            p_norm = Norm2(tmp2.data(), n);
          }

          const T b_dot_y = Dot(b.data(), y_hat, m);
          const T c_dot_x = Dot(c.data(), x_hat, n);
          const T cert_tol = abs_tol + rel_tol;
          const T b_neg = -b_dot_y;

          if (b_neg > cert_tol && std::abs(c_dot_x) <= cert_tol * b_neg
              && aty_norm <= cert_tol * b_neg
              && y_cone <= cert_tol * b_neg) {
            status = POGS_INFEASIBLE;
            *final_iter_out = k;
            break;
          }

          const T c_neg = -c_dot_x;
          if (c_neg > cert_tol && std::abs(b_dot_y) <= cert_tol * c_neg
              && ax_norm <= cert_tol * c_neg
              && p_norm <= cert_tol * c_neg) {
            status = POGS_UNBOUNDED;
            *final_iter_out = k;
            break;
          }
        }
      }
    }
    *final_iter_out = k;
  }

  const T *state = w.data();
  T tau = state[n + m];

  if (tau > kTauTol) {
    x_out->assign(n, kZero);
    y_out->assign(m, kZero);
    lambda_out->assign(m, kZero);
    for (size_t i = 0; i < n; ++i)
      x_scaled[i] = state[i] / tau;
    for (size_t i = 0; i < m; ++i)
      y_scaled[i] = state[n + i] / tau;

    A.Mul('n', kOne, x_scaled.data(), kZero, tmp.data());
    for (size_t i = 0; i < m; ++i)
      s_scaled[i] = b[i] - tmp[i];

    for (size_t i = 0; i < n; ++i)
      (*x_out)[i] = x_scaled[i] * e[i];
    for (size_t i = 0; i < m; ++i) {
      const T s_orig = s_scaled[i] / d[i];
      (*y_out)[i] = b_orig[i] - s_orig;
      (*lambda_out)[i] = y_scaled[i] * d[i];
    }
  } else {
    x_out->assign(n, kZero);
    y_out->assign(m, kZero);
    lambda_out->assign(m, kZero);
  }

  T optval = kZero;
  if (!x_out->empty()) {
    optval = Dot(c_orig.data(), x_out->data(), n);
    if (!P_orig.empty()) {
      for (size_t i = 0; i < n; ++i) {
        const T xi = (*x_out)[i];
        const T *row = &P_orig[i * n];
        T sum = kZero;
        for (size_t j = 0; j < n; ++j)
          sum += row[j] * (*x_out)[j];
        optval += static_cast<T>(0.5) * xi * sum;
      }
    }
  }
  *optval_out = optval;

  if (status != POGS_MAX_ITER)
    return status;
  if (converged)
    return POGS_SUCCESS;
  return POGS_MAX_ITER;
}

void MakeRawCone(const std::vector<ConeConstraint> &K,
                 std::vector<ConeConstraintRaw> *K_raw) {
  for (const auto& cone_constraint : K) {
    ConeConstraintRaw raw;
    raw.size = cone_constraint.idx.size();
    raw.idx = new CONE_IDX[raw.size];
    memcpy(raw.idx, cone_constraint.idx.data(), raw.size * sizeof(CONE_IDX));
    raw.cone = cone_constraint.cone;
    K_raw->push_back(raw);
  }
}

}  // namespace

// Implementation of PogsCone
template <typename T, typename M, typename P>
PogsCone<T, M, P>::PogsCone(const M& A,
                            const std::vector<ConeConstraint>& Kx,
                            const std::vector<ConeConstraint>& Ky)
    : PogsImplementation<T, M, P>(A) {
  valid_cones = ValidCone(Kx, A.Cols()) && ValidCone(Ky, A.Rows());
  MakeRawCone(Kx, &this->Kx);
  MakeRawCone(Ky, &this->Ky);
}

template <typename T, typename M, typename P>
PogsCone<T, M, P>::~PogsCone() {
  for (const auto& cone_constraint : this->Kx)
    delete [] cone_constraint.idx;
  for (const auto& cone_constraint : this->Ky)
    delete [] cone_constraint.idx;
}

template <typename T, typename M, typename P>
PogsStatus PogsCone<T, M, P>::Solve(const std::vector<T>& b,
                                    const std::vector<T>& c) {
  return Solve(b, c, std::vector<T>());
}

template <typename T, typename M, typename P>
PogsStatus PogsCone<T, M, P>::Solve(const std::vector<T>& b,
                                    const std::vector<T>& c,
                                    const std::vector<T>& P_mat) {
  if (!valid_cones)
    return POGS_INVALID_CONE;
  if (!P_mat.empty()) {
    const size_t n = c.size();
    if (P_mat.size() != n * n) {
      Printf("ERROR: Quadratic objective matrix has wrong size.\n");
      return POGS_ERROR;
    }
    if (!this->Kx.empty()) {
      Printf("ERROR: Quadratic objectives with Kx constraints are not supported.\n");
      return POGS_ERROR;
    }
  }
  this->SetUseAnderson(false);
  const bool use_hsde = this->Kx.empty();
  if (use_hsde) {
    PogsObjectiveHsdeScale<T> scale_obj(this->Kx, this->Ky);
    if (!this->_done_init)
      this->_Init(&scale_obj);

    const size_t m = this->_A.Rows();
    const size_t n = this->_A.Cols();
    const T *d = this->_de;
    const T *e = this->_de + m;

    std::vector<T> x_out;
    std::vector<T> y_out;
    std::vector<T> lambda_out;
    PogsStatus status = SolveHsdeCone(this->_A, b, c, P_mat, this->Ky, d, e,
                                      this->_rho, this->_abs_tol,
                                      this->_rel_tol, this->_max_iter,
                                      this->_verbose, &x_out, &y_out,
                                      &lambda_out, &this->_optval,
                                      &this->_final_iter);

    std::copy(x_out.begin(), x_out.end(), this->_x);
    std::copy(y_out.begin(), y_out.end(), this->_y);
    std::copy(lambda_out.begin(), lambda_out.end(), this->_lambda);
    std::fill(this->_mu, this->_mu + n, static_cast<T>(0));
    return status;
  }

  PogsObjectiveCone<T> pogs_obj(b, c, P_mat, Kx, Ky);
  return this->PogsImplementation<T, M, P>::Solve(&pogs_obj);
}

// Explicit template instantiation.
#if !defined(POGS_DOUBLE) || POGS_DOUBLE==1
// Dense direct.
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsSeparable<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsSeparable<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;

template class PogsCone<double, MatrixDense<double>,
    ProjectorDirect<double, MatrixDense<double> > >;
template class PogsCone<double, MatrixDense<double>,
    ProjectorCgls<double, MatrixDense<double> > >;
template class PogsCone<double, MatrixSparse<double>,
    ProjectorCgls<double, MatrixSparse<double> > >;
#endif

#if !defined(POGS_SINGLE) || POGS_SINGLE==1
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;
template class PogsSeparable<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;
template class PogsSeparable<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;

template class PogsCone<float, MatrixDense<float>,
    ProjectorDirect<float, MatrixDense<float> > >;
template class PogsCone<float, MatrixDense<float>,
    ProjectorCgls<float, MatrixDense<float> > >;
template class PogsCone<float, MatrixSparse<float>,
    ProjectorCgls<float, MatrixSparse<float> > >;
#endif

}  // namespace pogs
