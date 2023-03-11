#include <random>
#include <vector>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

// Quantile regression
//   minimize    (1/2) ||Ax - b||_1 + (tau - 1/2) \sum_i (a_i * x - b_i)
//
template <typename T>
double QuantileRegression(size_t m, size_t n) {
  std::vector<T> A(m * n);
  std::vector<T> b(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  std::vector<T> x_true(n );
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = u_dist(generator) < 0.8 ? 0 : n_dist(generator) / n;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j] + static_cast<T>(0.5) * n_dist(generator);

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);
  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  const T tau = static_cast<T>(0.2);

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(
      kAbs,
      static_cast<T>(1), 
      b[i], 
      static_cast<T>(0.5), 
      static_cast<T>(tau - static_cast<T>(0.5)));

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kZero);

  double t = timer<double>();
  pogs_data.Solve(f, g);

  return timer<double>() - t;
}

template double QuantileRegression<double>(size_t m, size_t n);
template double QuantileRegression<float>(size_t m, size_t n);

