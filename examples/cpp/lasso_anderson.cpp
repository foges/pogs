#include <random>
#include <vector>
#include <iostream>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

// Lasso with Anderson Acceleration
//   minimize    (1/2) ||Ax - b||_2^2 + \lambda ||x||_1
//
// This example demonstrates Anderson acceleration for ADMM.
// Compare with lasso.cpp to see the convergence improvement.
template <typename T>
double LassoAnderson(size_t m, size_t n, bool use_anderson = true) {
  std::vector<T> A(m * n);
  std::vector<T> b(m);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(static_cast<T>(0),
                                           static_cast<T>(1));
  std::normal_distribution<T> n_dist(static_cast<T>(0),
                                     static_cast<T>(1));

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  std::vector<T> x_true(n);
  for (unsigned int i = 0; i < n; ++i)
    x_true[i] = u_dist(generator) < static_cast<T>(0.8)
        ? static_cast<T>(0) : n_dist(generator) / static_cast<T>(std::sqrt(n));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * x_true[j];

  for (unsigned int i = 0; i < m; ++i)
    b[i] += static_cast<T>(0.5) * n_dist(generator);

  T lambda_max = static_cast<T>(0);
#ifdef _OPENMP
#pragma omp parallel for reduction(max : lambda_max)
#endif
  for (unsigned int j = 0; j < n; ++j) {
    T u = 0;
    for (unsigned int i = 0; i < m; ++i)
      u += A[i + j * m] * b[i];
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirect<T, pogs::MatrixDense<T> > pogs_data(A_);

  // Enable Anderson acceleration
  if (use_anderson) {
    pogs_data.SetUseAnderson(true);
    pogs_data.SetAndersonMem(5);       // Memory depth
    pogs_data.SetAndersonStart(10);    // Start after 10 iterations
    pogs_data.SetVerbose(2);           // Show convergence info
  }

  std::vector<FunctionObj<T> > f;
  std::vector<FunctionObj<T> > g;

  f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    f.emplace_back(kSquare, static_cast<T>(1), b[i]);

  g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    g.emplace_back(kAbs, static_cast<T>(0.2) * lambda_max);

  double t = timer<double>();
  pogs::PogsStatus status = pogs_data.Solve(f, g);

  double solve_time = timer<double>() - t;

  // Print results
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Lasso Test (m=" << m << ", n=" << n << ")" << std::endl;
  std::cout << "Anderson: " << (use_anderson ? "enabled" : "disabled") << std::endl;
  std::cout << "Status: " << pogs::PogsStatusString(status) << std::endl;
  std::cout << "Iterations: " << pogs_data.GetFinalIter() << std::endl;
  std::cout << "Solve time: " << solve_time << " seconds" << std::endl;
  std::cout << "Optimal value: " << pogs_data.GetOptval() << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  return solve_time;
}

template double LassoAnderson<double>(size_t m, size_t n, bool use_anderson);
template double LassoAnderson<float>(size_t m, size_t n, bool use_anderson);

int main() {
  std::cout << "\n=== Anderson Acceleration Test for POGS ===" << std::endl;
  std::cout << "\nRunning Lasso problem WITHOUT Anderson acceleration:" << std::endl;
  double time_no_anderson = LassoAnderson<double>(1000, 500, false);

  std::cout << "\nRunning Lasso problem WITH Anderson acceleration:" << std::endl;
  double time_with_anderson = LassoAnderson<double>(1000, 500, true);

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Speedup: " << (time_no_anderson / time_with_anderson) << "x" << std::endl;
  std::cout << "Time saved: " << (time_no_anderson - time_with_anderson) << " seconds" << std::endl;

  return 0;
}
