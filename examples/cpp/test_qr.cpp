#include <iostream>
#include <iomanip>
#include <cmath>
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

using namespace gsl;

// Test QR decomposition
int main() {
  std::cout << "Testing QR Decomposition...\n\n";

  // Test problem: solve min ||Ax - b|| where A is 4x3
  const size_t m = 4, n = 3;
  double A_data[] = {
    1.0, 2.0, 3.0, 4.0,  // column 0
    2.0, 3.0, 4.0, 5.0,  // column 1
    3.0, 4.0, 5.0, 6.0   // column 2
  };
  double b_data[] = {1.0, 2.0, 3.0, 4.0};
  double x_data[] = {0.0, 0.0, 0.0};
  double tau_data[] = {0.0, 0.0, 0.0};

  matrix<double, CblasColMajor> A = matrix_view_array<double, CblasColMajor>(A_data, m, n);
  vector<double> b = vector_view_array(b_data, m);
  vector<double> x = vector_view_array(x_data, n);
  vector<double> tau = vector_view_array(tau_data, n);

  std::cout << "Original A:\n";
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      std::cout << std::setw(10) << matrix_get(&A, i, j) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nOriginal b: ";
  for (size_t i = 0; i < m; ++i) {
    std::cout << std::setw(10) << vector_get(&b, i) << " ";
  }
  std::cout << "\n\n";

  // Perform QR decomposition
  linalg_qr_decomp(&A, &tau);

  std::cout << "After QR decomposition, R (upper triangle of A):\n";
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i <= j) {
        std::cout << std::setw(10) << matrix_get(&A, i, j) << " ";
      } else {
        std::cout << std::setw(10) << "." << " ";
      }
    }
    std::cout << "\n";
  }

  std::cout << "\nTau values: ";
  for (size_t i = 0; i < n; ++i) {
    std::cout << std::setw(10) << vector_get(&tau, i) << " ";
  }
  std::cout << "\n\n";

  // Solve least squares
  linalg_qr_lssolve(&A, &tau, &b, &x);

  std::cout << "Least squares solution x:\n";
  for (size_t i = 0; i < n; ++i) {
    std::cout << "x[" << i << "] = " << vector_get(&x, i) << "\n";
  }

  // Compute residual with original A
  double A_orig[] = {
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    3.0, 4.0, 5.0, 6.0
  };

  std::cout << "\nVerification: Ax =\n";
  for (size_t i = 0; i < m; ++i) {
    double val = 0;
    for (size_t j = 0; j < n; ++j) {
      val += A_orig[j * m + i] * vector_get(&x, j);
    }
    std::cout << "  " << val << " (b = " << vector_get(&b, i) << ", diff = " << std::abs(val - vector_get(&b, i)) << ")\n";
  }

  std::cout << "\nTest completed!\n";
  return 0;
}
