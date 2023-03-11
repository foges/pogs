
#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "pogs_c.h"

template <typename T, ORD O>
int Pogs(size_t m, size_t n, const T *A,
         const T *f_a, const T *f_b, const T *f_c, const T *f_d, const T *f_e,
         const FUNCTION *f_h,
         const T *g_a, const T *g_b, const T *g_c, const T *g_d, const T *g_e,
         const FUNCTION *g_h,
         T rho, T abs_tol, T rel_tol, unsigned int max_iter, unsigned int verbose,
         bool adaptive_rho, bool gap_stop, T *x, T *y, T *l, T *optval,
         unsigned int *final_iter) {
  // Create pogs struct.
  char ord = O == ROW_MAJ ? 'r' : 'c';
  pogs::MatrixDense<T> A_(ord, m, n, A);
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);

  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  // Set f and g.
  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(static_cast<Function>(f_h[i]), f_a[i], f_b[i], f_c[i],
        f_d[i], f_e[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(static_cast<Function>(g_h[i]), g_a[i], g_b[i], g_c[i],
         g_d[i], g_e[i]);

  // Set parameters.
  pogs_data.SetRho(rho);
  pogs_data.SetAbsTol(abs_tol);
  pogs_data.SetRelTol(rel_tol);
  pogs_data.SetMaxIter(max_iter);
  pogs_data.SetVerbose(verbose);
  pogs_data.SetAdaptiveRho(adaptive_rho);
  pogs_data.SetGapStop(gap_stop);

  // Solve.
  int err = pogs_data.Solve(f, g);
  *optval = pogs_data.GetOptval();
  *final_iter = pogs_data.GetFinalIter();

  memcpy(x, pogs_data.GetX(), n * sizeof(T));
  memcpy(y, pogs_data.GetY(), m * sizeof(T));
  memcpy(l, pogs_data.GetLambda(), m * sizeof(T));

  return err;
}

extern "C" {
int PogsD(enum ORD ord, size_t m, size_t n, const double *A,
          const double *f_a, const double *f_b, const double *f_c,
          const double *f_d, const double *f_e, const enum FUNCTION *f_h,
          const double *g_a, const double *g_b, const double *g_c,
          const double *g_d, const double *g_e, const enum FUNCTION *g_h,
          double rho, double abs_tol, double rel_tol, unsigned int max_iter,
          unsigned int verbose, int adaptive_rho, int gap_stop,
          double *x, double *y, double *l, double *optval,
          unsigned int *final_iter) {
  if (ord == COL_MAJ) {
    return Pogs<double, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval, final_iter);
  } else {
    return Pogs<double, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval, final_iter);
  }
}

int PogsS(enum ORD ord, size_t m, size_t n, const float *A,
          const float *f_a, const float *f_b, const float *f_c,
          const float *f_d, const float *f_e, const enum FUNCTION *f_h,
          const float *g_a, const float *g_b, const float *g_c,
          const float *g_d, const float *g_e, const enum FUNCTION *g_h,
          float rho, float abs_tol, float rel_tol, unsigned int max_iter,
          unsigned int verbose, int adaptive_rho, int gap_stop,
          float *x, float *y, float *l, float *optval,
          unsigned int *final_iter) {
  if (ord == COL_MAJ) {
    return Pogs<float, COL_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval, final_iter);
  } else {
    return Pogs<float, ROW_MAJ>(m, n, A, f_a, f_b, f_c, f_d, f_e, f_h,
        g_a, g_b, g_c, g_d, g_e, g_h, rho, abs_tol, rel_tol, max_iter,
        verbose, static_cast<bool>(adaptive_rho),
        static_cast<bool>(gap_stop), x, y, l, optval, final_iter);
  }
}

}

