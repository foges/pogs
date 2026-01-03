#include "pogs.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>

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
  const T kProjTolMax     = static_cast<T>(1e-8);
  const T kProjTolMin     = static_cast<T>(1e-2);
  const T kProjTolPow     = static_cast<T>(2);
  const T kProjTolIni     = static_cast<T>(1e-5);
  const bool kUseExactTol = objective->UseExactTol();
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
    T proj_tol = kProjTolMin / std::pow(static_cast<T>(k + 1), kProjTolPow);
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

    // Rescale rho.
    if (_adaptive_rho) {
      if (nrm_s < xi * eps_dua && nrm_r > xi * eps_pri &&
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
