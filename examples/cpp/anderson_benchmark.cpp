#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "matrix/matrix_dense.h"
#include "pogs.h"
#include "timer.h"

// Benchmark result structure
struct BenchmarkResult {
  std::string problem_name;
  int iterations_no_anderson;
  int iterations_with_anderson;
  double time_no_anderson;
  double time_with_anderson;
  double optval_no_anderson;
  double optval_with_anderson;
  std::string status_no_anderson;
  std::string status_with_anderson;

  double speedup() const {
    return iterations_no_anderson / (double)iterations_with_anderson;
  }

  double time_ratio() const {
    return time_no_anderson / time_with_anderson;
  }
};

// Generate ill-conditioned matrix with specified condition number
template <typename T>
void GenerateIllConditioned(std::vector<T>& A, size_t m, size_t n, T condition_number) {
  std::default_random_engine generator(42);
  std::normal_distribution<T> n_dist(0, 1);

  // Generate random matrix
  for (size_t i = 0; i < m * n; ++i) {
    A[i] = n_dist(generator);
  }

  // Scale columns to create ill-conditioning
  // Singular values decay exponentially from 1 to 1/condition_number
  for (size_t j = 0; j < n; ++j) {
    T scale = std::pow(condition_number, -static_cast<T>(j) / static_cast<T>(n - 1));
    for (size_t i = 0; i < m; ++i) {
      A[i + j * m] *= scale;
    }
  }
}

// Test 1: Ill-conditioned Lasso
template <typename T>
BenchmarkResult BenchmarkIllConditionedLasso(size_t m, size_t n, T condition_number,
                                              T abs_tol = 1e-4, T rel_tol = 1e-3) {
  BenchmarkResult result;
  result.problem_name = "Ill-conditioned Lasso (kappa=" + std::to_string((int)condition_number) +
                        ", tol=" + std::to_string(abs_tol) + ")";

  std::vector<T> A(m * n);
  std::vector<T> b(m);

  GenerateIllConditioned(A, m, n, condition_number);

  // Generate sparse true solution
  std::default_random_engine generator(42);
  std::uniform_real_distribution<T> u_dist(0, 1);
  std::normal_distribution<T> n_dist(0, 1);

  std::vector<T> x_true(n);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = u_dist(generator) < 0.9 ? 0 : n_dist(generator);
  }

  // b = Ax_true + noise
  for (size_t i = 0; i < m; ++i) {
    b[i] = 0;
    for (size_t j = 0; j < n; ++j) {
      b[i] += A[i + j * m] * x_true[j];
    }
    b[i] += 0.1 * n_dist(generator);
  }

  T lambda_max = 0;
  for (size_t j = 0; j < n; ++j) {
    T u = 0;
    for (size_t i = 0; i < m; ++i) {
      u += A[i + j * m] * b[i];
    }
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  // Test WITHOUT Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(abs_tol);
    pogs_data.SetRelTol(rel_tol);
    pogs_data.SetVerbose(0);

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.1) * lambda_max);

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_no_anderson = timer<double>() - t;
    result.iterations_no_anderson = pogs_data.GetFinalIter();
    result.optval_no_anderson = pogs_data.GetOptval();
    result.status_no_anderson = pogs::PogsStatusString(status);
  }

  // Test WITH Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(abs_tol);
    pogs_data.SetRelTol(rel_tol);
    pogs_data.SetVerbose(0);
    pogs_data.SetUseAnderson(true);
    pogs_data.SetAndersonMem(10);   // Larger memory for ill-conditioned
    pogs_data.SetAndersonStart(5);  // Start earlier

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.1) * lambda_max);

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_with_anderson = timer<double>() - t;
    result.iterations_with_anderson = pogs_data.GetFinalIter();
    result.optval_with_anderson = pogs_data.GetOptval();
    result.status_with_anderson = pogs::PogsStatusString(status);
  }

  return result;
}

// Test 2: Basis Pursuit (underdetermined sparse recovery)
template <typename T>
BenchmarkResult BenchmarkBasisPursuit(size_t m, size_t n, T sparsity = 0.1) {
  BenchmarkResult result;
  result.problem_name = "Basis Pursuit (m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                        ", sparsity=" + std::to_string(sparsity) + ")";

  std::default_random_engine generator(42);
  std::normal_distribution<T> n_dist(0, 1);
  std::uniform_real_distribution<T> u_dist(0, 1);

  std::vector<T> A(m * n);
  std::vector<T> b(m);

  // Random Gaussian matrix
  for (size_t i = 0; i < m * n; ++i) {
    A[i] = n_dist(generator) / std::sqrt(static_cast<T>(m));
  }

  // Sparse true solution
  std::vector<T> x_true(n);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = u_dist(generator) < sparsity ? n_dist(generator) : 0;
  }

  // b = Ax_true (exact, no noise)
  for (size_t i = 0; i < m; ++i) {
    b[i] = 0;
    for (size_t j = 0; j < n; ++j) {
      b[i] += A[i + j * m] * x_true[j];
    }
  }

  // Test WITHOUT Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(1e-5);
    pogs_data.SetRelTol(1e-4);
    pogs_data.SetVerbose(0);

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kIndEq0, static_cast<T>(1), b[i]);  // Equality constraint
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(1));  // L1 norm

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_no_anderson = timer<double>() - t;
    result.iterations_no_anderson = pogs_data.GetFinalIter();
    result.optval_no_anderson = pogs_data.GetOptval();
    result.status_no_anderson = pogs::PogsStatusString(status);
  }

  // Test WITH Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(1e-5);
    pogs_data.SetRelTol(1e-4);
    pogs_data.SetVerbose(0);
    pogs_data.SetUseAnderson(true);
    pogs_data.SetAndersonMem(10);
    pogs_data.SetAndersonStart(10);

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kIndEq0, static_cast<T>(1), b[i]);
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(1));

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_with_anderson = timer<double>() - t;
    result.iterations_with_anderson = pogs_data.GetFinalIter();
    result.optval_with_anderson = pogs_data.GetOptval();
    result.status_with_anderson = pogs::PogsStatusString(status);
  }

  return result;
}

// Test 3: High-accuracy Lasso
template <typename T>
BenchmarkResult BenchmarkHighAccuracyLasso(size_t m, size_t n) {
  BenchmarkResult result;
  result.problem_name = "High-Accuracy Lasso (m=" + std::to_string(m) + ", n=" + std::to_string(n) + ")";

  std::default_random_engine generator(42);
  std::normal_distribution<T> n_dist(0, 1);
  std::uniform_real_distribution<T> u_dist(0, 1);

  std::vector<T> A(m * n);
  std::vector<T> b(m);

  for (size_t i = 0; i < m * n; ++i) {
    A[i] = n_dist(generator);
  }

  std::vector<T> x_true(n);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = u_dist(generator) < 0.9 ? 0 : n_dist(generator);
  }

  for (size_t i = 0; i < m; ++i) {
    b[i] = 0;
    for (size_t j = 0; j < n; ++j) {
      b[i] += A[i + j * m] * x_true[j];
    }
    b[i] += 0.1 * n_dist(generator);
  }

  T lambda_max = 0;
  for (size_t j = 0; j < n; ++j) {
    T u = 0;
    for (size_t i = 0; i < m; ++i) {
      u += A[i + j * m] * b[i];
    }
    lambda_max = std::max(lambda_max, std::abs(u));
  }

  // High accuracy tolerances
  T abs_tol = 1e-8;
  T rel_tol = 1e-6;

  // Test WITHOUT Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(abs_tol);
    pogs_data.SetRelTol(rel_tol);
    pogs_data.SetVerbose(0);
    pogs_data.SetMaxIter(5000);

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.1) * lambda_max);

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_no_anderson = timer<double>() - t;
    result.iterations_no_anderson = pogs_data.GetFinalIter();
    result.optval_no_anderson = pogs_data.GetOptval();
    result.status_no_anderson = pogs::PogsStatusString(status);
  }

  // Test WITH Anderson
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsDirect<T, pogs::MatrixDense<T>> pogs_data(A_);
    pogs_data.SetAbsTol(abs_tol);
    pogs_data.SetRelTol(rel_tol);
    pogs_data.SetVerbose(0);
    pogs_data.SetMaxIter(5000);
    pogs_data.SetUseAnderson(true);
    pogs_data.SetAndersonMem(10);
    pogs_data.SetAndersonStart(20);

    std::vector<FunctionObj<T>> f, g;
    f.reserve(m);
    for (size_t i = 0; i < m; ++i)
      f.emplace_back(kSquare, static_cast<T>(1), b[i]);
    g.reserve(n);
    for (size_t i = 0; i < n; ++i)
      g.emplace_back(kAbs, static_cast<T>(0.1) * lambda_max);

    double t = timer<double>();
    pogs::PogsStatus status = pogs_data.Solve(f, g);
    result.time_with_anderson = timer<double>() - t;
    result.iterations_with_anderson = pogs_data.GetFinalIter();
    result.optval_with_anderson = pogs_data.GetOptval();
    result.status_with_anderson = pogs::PogsStatusString(status);
  }

  return result;
}

void PrintResults(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n";
  std::cout << "==================================================================================\n";
  std::cout << "                    Anderson Acceleration Benchmark Results\n";
  std::cout << "==================================================================================\n\n";

  std::cout << std::left << std::setw(50) << "Problem"
            << std::right << std::setw(10) << "Iter (No)"
            << std::setw(12) << "Iter (AA)"
            << std::setw(12) << "Speedup"
            << std::setw(12) << "Time (No)"
            << std::setw(12) << "Time (AA)"
            << std::setw(12) << "Time Ratio" << "\n";
  std::cout << std::string(118, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::left << std::setw(50) << r.problem_name
              << std::right << std::setw(10) << r.iterations_no_anderson
              << std::setw(12) << r.iterations_with_anderson
              << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup()
              << std::setw(12) << std::scientific << std::setprecision(2) << r.time_no_anderson
              << std::setw(12) << r.time_with_anderson
              << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ratio() << "\n";
  }

  std::cout << "\n";
  std::cout << "Key: Iter (No) = Iterations without Anderson, Iter (AA) = Iterations with Anderson\n";
  std::cout << "     Speedup = Iter(No) / Iter(AA),  Time Ratio = Time(No) / Time(AA)\n";
  std::cout << "     Speedup > 1.0 means Anderson reduces iterations\n";
  std::cout << "     Time Ratio > 1.0 means Anderson reduces wall-clock time\n";
  std::cout << "\n==================================================================================\n";
}

int main() {
  std::vector<BenchmarkResult> results;

  std::cout << "\nRunning Anderson Acceleration Benchmark Suite...\n";
  std::cout << "This tests scenarios where Anderson acceleration is expected to help:\n";
  std::cout << "  1. Ill-conditioned problems\n";
  std::cout << "  2. Underdetermined problems (Basis Pursuit)\n";
  std::cout << "  3. High-accuracy requirements\n\n";

  // Test 1: Ill-conditioned Lasso with varying condition numbers
  std::cout << "Test 1: Ill-conditioned Lasso...\n";
  results.push_back(BenchmarkIllConditionedLasso<double>(200, 100, 10.0));
  results.push_back(BenchmarkIllConditionedLasso<double>(200, 100, 100.0));
  results.push_back(BenchmarkIllConditionedLasso<double>(200, 100, 1000.0));

  // Test 2: Basis Pursuit (underdetermined)
  std::cout << "Test 2: Basis Pursuit (underdetermined)...\n";
  results.push_back(BenchmarkBasisPursuit<double>(100, 500, 0.1));
  results.push_back(BenchmarkBasisPursuit<double>(200, 1000, 0.05));

  // Test 3: High-accuracy Lasso
  std::cout << "Test 3: High-accuracy Lasso...\n";
  results.push_back(BenchmarkHighAccuracyLasso<double>(300, 150));
  results.push_back(BenchmarkHighAccuracyLasso<double>(500, 250));

  // Print summary table
  PrintResults(results);

  return 0;
}
