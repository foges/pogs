#ifndef ANDERSON_H_
#define ANDERSON_H_

#include <cstring>
#include <algorithm>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

namespace pogs {

// Anderson Acceleration for fixed-point iterations.
//
// Implements the Anderson acceleration method (also known as Anderson mixing)
// to accelerate the convergence of fixed-point iterations x_{k+1} = g(x_k).
//
// Reference:
//   Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations"
//   SIAM J. Numer. Anal., 49(4), 2011.
//
// Algorithm:
//   Given m past iterates and residuals, solves a least-squares problem
//   to find the optimal linear combination that minimizes the residual.
//
template <typename T>
class AndersonAccelerator {
 private:
  size_t _n;           // Dimension of problem
  size_t _m;           // Memory depth (number of past iterates to store)
  size_t _k;           // Current iteration count
  size_t _stored;      // Number of iterates currently stored (min(_k, _m+1))

  // Circular buffer storage for iterates and residuals
  T** _X;              // History of iterates: _X[i] has dimension _n
  T** _R;              // History of residuals: _R[i] = _X[i+1] - _X[i]

  // Working memory for least-squares problem
  T* _F;               // Matrix F of residual differences (n x m_k)
  T* _r_k;             // Current residual (length n)
  T* _alpha;           // Combination weights (length m)
  T* _tau;             // Householder reflector scalings for QR (length m)

  bool _initialized;

 public:
  // Constructor: allocates storage for Anderson acceleration.
  // Parameters:
  //   n: dimension of the problem (size of iterate vectors)
  //   m: memory depth (number of past iterates to store, typical: 5-10)
  AndersonAccelerator(size_t n, size_t m);

  // Destructor: frees all allocated memory.
  ~AndersonAccelerator();

  // Apply Anderson acceleration to a new iterate.
  //
  // Parameters:
  //   x_new: new iterate from the fixed-point iteration (length n)
  //   x_prev: previous iterate (length n)
  //   x_acc: output accelerated iterate (length n)
  //
  // Returns:
  //   true if acceleration was applied, false otherwise (e.g., not enough history)
  //
  // Note: Requires at least 2 stored iterates before acceleration can be applied.
  bool Apply(const T* x_new, const T* x_prev, T* x_acc);

  // Reset the history (e.g., when algorithm parameters change significantly).
  // Clears all stored iterates and residuals, restarts from scratch.
  void Reset();

 private:
  // Solve the least-squares problem to find combination weights alpha.
  // Formulation: min ||F * alpha + r_k||_2
  // Uses QR decomposition for numerical stability.
  void SolveLeastSquares();

  // Update circular buffers with new iterate and residual.
  void UpdateHistory(const T* x_new, const T* x_prev);
};

// Constructor implementation
template <typename T>
AndersonAccelerator<T>::AndersonAccelerator(size_t n, size_t m)
    : _n(n), _m(m), _k(0), _stored(0), _initialized(true) {

  // Allocate storage for m+1 iterates (circular buffer)
  _X = new T*[_m + 1];
  _R = new T*[_m + 1];
  for (size_t i = 0; i <= _m; ++i) {
    _X[i] = new T[_n];
    _R[i] = new T[_n];
    std::memset(_X[i], 0, _n * sizeof(T));
    std::memset(_R[i], 0, _n * sizeof(T));
  }

  // Allocate working memory for least-squares
  _F = new T[_n * _m];        // Matrix F (column-major)
  _r_k = new T[_n];           // Current residual
  _alpha = new T[_m];         // Combination weights
  _tau = new T[_m];           // QR reflector scalings

  std::memset(_F, 0, _n * _m * sizeof(T));
  std::memset(_r_k, 0, _n * sizeof(T));
  std::memset(_alpha, 0, _m * sizeof(T));
  std::memset(_tau, 0, _m * sizeof(T));
}

// Destructor implementation
template <typename T>
AndersonAccelerator<T>::~AndersonAccelerator() {
  for (size_t i = 0; i <= _m; ++i) {
    delete[] _X[i];
    delete[] _R[i];
  }
  delete[] _X;
  delete[] _R;

  delete[] _F;
  delete[] _r_k;
  delete[] _alpha;
  delete[] _tau;
}

// Reset implementation
template <typename T>
void AndersonAccelerator<T>::Reset() {
  _k = 0;
  _stored = 0;

  // Clear all buffers
  for (size_t i = 0; i <= _m; ++i) {
    std::memset(_X[i], 0, _n * sizeof(T));
    std::memset(_R[i], 0, _n * sizeof(T));
  }
}

// Update history implementation
template <typename T>
void AndersonAccelerator<T>::UpdateHistory(const T* x_new, const T* x_prev) {
  // Compute current index in circular buffer
  size_t idx = _k % (_m + 1);

  // Store new iterate
  std::memcpy(_X[idx], x_new, _n * sizeof(T));

  // Compute and store residual: r_k = x_new - x_prev
  for (size_t i = 0; i < _n; ++i) {
    _R[idx][i] = x_new[i] - x_prev[i];
  }

  // Update storage counter
  _stored = std::min(_stored + 1, _m + 1);
}

// Solve least-squares implementation
template <typename T>
void AndersonAccelerator<T>::SolveLeastSquares() {
  // Determine effective memory depth (number of columns in F)
  size_t m_k = std::min(_stored - 1, _m);

  if (m_k == 0) return;

  // Build matrix F: F[:, j] = R[k-j] - R[k] for j = 1, ..., m_k
  // F is stored in column-major format for compatibility with GSL
  for (size_t j = 0; j < m_k; ++j) {
    size_t idx = (_k - j) % (_m + 1);
    for (size_t i = 0; i < _n; ++i) {
      _F[i + j * _n] = _R[idx][i] - _r_k[i];
    }
  }

  // Setup GSL views
  gsl::matrix<T, CblasColMajor> F_mat =
      gsl::matrix_view_array<T, CblasColMajor>(_F, _n, m_k);
  gsl::vector<T> tau_vec = gsl::vector_view_array(_tau, m_k);
  gsl::vector<T> rhs = gsl::vector_view_array(_r_k, _n);
  gsl::vector<T> alpha_vec = gsl::vector_view_array(_alpha, m_k);

  // QR decomposition of F
  gsl::linalg_qr_decomp(&F_mat, &tau_vec);

  // Solve least-squares: min ||F * alpha - (-r_k)||
  // We pass r_k and negate the result
  gsl::linalg_qr_lssolve(&F_mat, &tau_vec, &rhs, &alpha_vec);

  // Negate alpha (we solved F*alpha = r_k, but we want F*alpha = -r_k)
  for (size_t i = 0; i < m_k; ++i) {
    _alpha[i] = -_alpha[i];
  }
}

// Apply implementation
template <typename T>
bool AndersonAccelerator<T>::Apply(const T* x_new, const T* x_prev, T* x_acc) {
  // Compute current residual r_k = x_new - x_prev
  for (size_t i = 0; i < _n; ++i) {
    _r_k[i] = x_new[i] - x_prev[i];
  }

  // Update circular buffer with new iterate and residual
  UpdateHistory(x_new, x_prev);
  _k++;

  // Need at least 2 iterates to perform acceleration
  if (_stored < 2) {
    std::memcpy(x_acc, x_new, _n * sizeof(T));
    return false;
  }

  // Solve least-squares problem for combination weights
  SolveLeastSquares();

  // Compute accelerated iterate
  // x_acc = x_new + sum_{j=1}^{m_k} alpha[j] * (X[k-j] - x_new)
  std::memcpy(x_acc, x_new, _n * sizeof(T));

  size_t m_k = std::min(_stored - 1, _m);
  for (size_t j = 0; j < m_k; ++j) {
    size_t idx = (_k - 1 - j) % (_m + 1);
    T alpha_j = _alpha[j];
    for (size_t i = 0; i < _n; ++i) {
      x_acc[i] += alpha_j * (_X[idx][i] - x_new[i]);
    }
  }

  return true;
}

}  // namespace pogs

#endif  // ANDERSON_H_
