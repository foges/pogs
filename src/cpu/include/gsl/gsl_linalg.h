#ifndef GSL_LINALG_H_
#define GSL_LINALG_H_

#include <cmath>

#include "gsl_blas.h"
#include "gsl_matrix.h"
#include "gsl_vector.h"

namespace gsl {

// Non-Block Cholesky.
template <typename T, CBLAS_ORDER O>
void linalg_cholesky_decomp_noblk(matrix<T, O> *A) {
  size_t n = A->size1;
  for (size_t i = 0; i < n; ++i) {
    T l11 = std::sqrt(matrix_get(A, i, i));
    matrix_set(A, i, i, l11);
    if (i + 1 < n) {
      matrix<T, O> l21 = matrix_submatrix(A, i + 1, i, n - i - 1, 1);
      matrix_scale(&l21, 1 / l11);
      matrix<T, O> a22 = matrix_submatrix(A, i + 1, i + 1, n - i - 1,
          n - i - 1);
      blas_syrk(CblasLower, CblasNoTrans, static_cast<T>(-1), &l21,
          static_cast<T>(1), &a22);
    }
  }
}

// Block Cholesky.
//   l11 l11^T = a11
//   l21 = a21 l11^(-T)
//   a22 = a22 - l21 l21^T
//
// Stores result in Lower triangular part.
template <typename T, CBLAS_ORDER O>
void linalg_cholesky_decomp(matrix<T, O> *A) {
  size_t n = A->size1;
  // Block Dimension borrowed from Eigen.
  size_t blk_dim = std::max<size_t>(std::min<size_t>((n / 128) * 16, 8), 128);
  for (size_t i = 0; i < n; i += blk_dim) {
    size_t n11 = std::min<size_t>(blk_dim, n - i);
    matrix<T, O> l11 = matrix_submatrix(A, i, i, n11, n11);
    linalg_cholesky_decomp_noblk(&l11);
    if (i + blk_dim < n) {
      matrix<T, O> l21 = matrix_submatrix(A, i + n11, i, n - i - n11, n11);
      blas_trsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit,
          static_cast<T>(1), &l11, &l21);
      matrix<T, O> a22 = matrix_submatrix(A, i + blk_dim, i + blk_dim, 
          n - i - blk_dim, n - i - blk_dim);
      blas_syrk(CblasLower, CblasNoTrans, static_cast<T>(-1), &l21,
          static_cast<T>(1), &a22);
    }
  }
}

template <typename T, CBLAS_ORDER O>
void linalg_cholesky_svx(const matrix<T, O> *LLT, vector<T> *x) {
  blas_trsv(CblasLower, CblasNoTrans, CblasNonUnit, LLT, x);
  blas_trsv(CblasLower, CblasTrans, CblasNonUnit, LLT, x);
}

// Symmetric Eigenvalue Decomposition using LAPACK
// Computes eigenvalues and eigenvectors of a symmetric matrix A
// On input: A contains the symmetric matrix (only lower triangle is used)
// On output: A is overwritten with eigenvectors (column-major)
//            w contains eigenvalues in ascending order
extern "C" {
  void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
               const int* lda, double* w, double* work, const int* lwork,
               int* iwork, const int* liwork, int* info);
  void ssyevd_(const char* jobz, const char* uplo, const int* n, float* a,
               const int* lda, float* w, float* work, const int* lwork,
               int* iwork, const int* liwork, int* info);
}

template <typename T, CBLAS_ORDER O>
void linalg_syevd(matrix<T, O> *A, vector<T> *w);

template <>
inline void linalg_syevd<double, CblasColMajor>(matrix<double, CblasColMajor> *A,
                                                 vector<double> *w) {
  char jobz = 'V';  // Compute eigenvalues and eigenvectors
  char uplo = 'L';  // Lower triangle of A is stored
  int n = static_cast<int>(A->size1);
  int lda = n;
  int info;

  // Query optimal workspace size
  double wkopt;
  int lwork = -1;
  int iwork_opt;
  int liwork = -1;
  dsyevd_(&jobz, &uplo, &n, A->data, &lda, w->data, &wkopt, &lwork,
          &iwork_opt, &liwork, &info);

  lwork = static_cast<int>(wkopt);
  liwork = iwork_opt;
  double *work = new double[lwork];
  int *iwork = new int[liwork];

  // Compute eigenvalues and eigenvectors
  dsyevd_(&jobz, &uplo, &n, A->data, &lda, w->data, work, &lwork,
          iwork, &liwork, &info);

  delete[] work;
  delete[] iwork;

  if (info != 0) {
    // Eigenvalue decomposition failed
    // For now, just set all eigenvalues to zero (safeguard)
    for (size_t i = 0; i < w->size; ++i) {
      vector_set(w, i, 0.0);
    }
  }
}

template <>
inline void linalg_syevd<float, CblasColMajor>(matrix<float, CblasColMajor> *A,
                                                vector<float> *w) {
  char jobz = 'V';
  char uplo = 'L';
  int n = static_cast<int>(A->size1);
  int lda = n;
  int info;

  float wkopt;
  int lwork = -1;
  int iwork_opt;
  int liwork = -1;
  ssyevd_(&jobz, &uplo, &n, A->data, &lda, w->data, &wkopt, &lwork,
          &iwork_opt, &liwork, &info);

  lwork = static_cast<int>(wkopt);
  liwork = iwork_opt;
  float *work = new float[lwork];
  int *iwork = new int[liwork];

  ssyevd_(&jobz, &uplo, &n, A->data, &lda, w->data, work, &lwork,
          iwork, &liwork, &info);

  delete[] work;
  delete[] iwork;

  if (info != 0) {
    for (size_t i = 0; i < w->size; ++i) {
      vector_set(w, i, 0.0f);
    }
  }
}

// QR decomposition and least-squares solve using LAPACK
// For Anderson acceleration
extern "C" {
  // QR decomposition: A = Q*R
  void dgeqrf_(const int* m, const int* n, double* a, const int* lda,
               double* tau, double* work, const int* lwork, int* info);
  void sgeqrf_(const int* m, const int* n, float* a, const int* lda,
               float* tau, float* work, const int* lwork, int* info);

  // Apply Q from QR to vector: C = Q^T * C or C = Q * C
  void dormqr_(const char* side, const char* trans, const int* m, const int* n,
               const int* k, const double* a, const int* lda, const double* tau,
               double* c, const int* ldc, double* work, const int* lwork, int* info);
  void sormqr_(const char* side, const char* trans, const int* m, const int* n,
               const int* k, const float* a, const int* lda, const float* tau,
               float* c, const int* ldc, float* work, const int* lwork, int* info);

  // Triangular solve
  void dtrtrs_(const char* uplo, const char* trans, const char* diag,
               const int* n, const int* nrhs, const double* a, const int* lda,
               double* b, const int* ldb, int* info);
  void strtrs_(const char* uplo, const char* trans, const char* diag,
               const int* n, const int* nrhs, const float* a, const int* lda,
               float* b, const int* ldb, int* info);
}

// QR decomposition wrapper for double
template <typename T, CBLAS_ORDER O>
void linalg_qr_decomp(matrix<T, O> *A, vector<T> *tau);

template <>
inline void linalg_qr_decomp<double, CblasColMajor>(matrix<double, CblasColMajor> *A,
                                                     vector<double> *tau) {
  int m = static_cast<int>(A->size1);
  int n = static_cast<int>(A->size2);
  int lda = m;
  int info;

  // Query optimal workspace size
  double wkopt;
  int lwork = -1;
  dgeqrf_(&m, &n, A->data, &lda, tau->data, &wkopt, &lwork, &info);

  lwork = static_cast<int>(wkopt);
  double *work = new double[lwork];

  dgeqrf_(&m, &n, A->data, &lda, tau->data, work, &lwork, &info);

  delete[] work;
}

template <>
inline void linalg_qr_decomp<float, CblasColMajor>(matrix<float, CblasColMajor> *A,
                                                    vector<float> *tau) {
  int m = static_cast<int>(A->size1);
  int n = static_cast<int>(A->size2);
  int lda = m;
  int info;

  float wkopt;
  int lwork = -1;
  sgeqrf_(&m, &n, A->data, &lda, tau->data, &wkopt, &lwork, &info);

  lwork = static_cast<int>(wkopt);
  float *work = new float[lwork];

  sgeqrf_(&m, &n, A->data, &lda, tau->data, work, &lwork, &info);

  delete[] work;
}

// QR least-squares solve wrapper: solves min ||A*x - b||_2
// On input: A contains QR factorization from linalg_qr_decomp
//           b contains right-hand side
// On output: x contains the solution (stored in first n elements of b)
template <typename T, CBLAS_ORDER O>
void linalg_qr_lssolve(const matrix<T, O> *QR, const vector<T> *tau,
                       vector<T> *b, vector<T> *x);

template <>
inline void linalg_qr_lssolve<double, CblasColMajor>(
    const matrix<double, CblasColMajor> *QR, const vector<double> *tau,
    vector<double> *b, vector<double> *x) {
  int m = static_cast<int>(QR->size1);
  int n = static_cast<int>(QR->size2);
  int lda = m;
  int ldb = m;
  int nrhs = 1;
  int info;
  char side = 'L';
  char trans = 'T';
  char uplo = 'U';
  char notrans = 'N';
  char diag = 'N';

  // Apply Q^T to b: b = Q^T * b
  double wkopt;
  int lwork = -1;
  dormqr_(&side, &trans, &m, &nrhs, &n, QR->data, &lda, tau->data,
          b->data, &ldb, &wkopt, &lwork, &info);

  lwork = static_cast<int>(wkopt);
  double *work = new double[lwork];

  dormqr_(&side, &trans, &m, &nrhs, &n, QR->data, &lda, tau->data,
          b->data, &ldb, work, &lwork, &info);

  delete[] work;

  // Solve R*x = (Q^T*b) using triangular solve
  // Only first n elements of b are used
  dtrtrs_(&uplo, &notrans, &diag, &n, &nrhs, QR->data, &lda, b->data, &ldb, &info);

  // Copy solution to x
  for (int i = 0; i < n; ++i) {
    vector_set(x, i, vector_get(b, i));
  }
}

template <>
inline void linalg_qr_lssolve<float, CblasColMajor>(
    const matrix<float, CblasColMajor> *QR, const vector<float> *tau,
    vector<float> *b, vector<float> *x) {
  int m = static_cast<int>(QR->size1);
  int n = static_cast<int>(QR->size2);
  int lda = m;
  int ldb = m;
  int nrhs = 1;
  int info;
  char side = 'L';
  char trans = 'T';
  char uplo = 'U';
  char notrans = 'N';
  char diag = 'N';

  float wkopt;
  int lwork = -1;
  sormqr_(&side, &trans, &m, &nrhs, &n, QR->data, &lda, tau->data,
          b->data, &ldb, &wkopt, &lwork, &info);

  lwork = static_cast<int>(wkopt);
  float *work = new float[lwork];

  sormqr_(&side, &trans, &m, &nrhs, &n, QR->data, &lda, tau->data,
          b->data, &ldb, work, &lwork, &info);

  delete[] work;

  strtrs_(&uplo, &notrans, &diag, &n, &nrhs, QR->data, &lda, b->data, &ldb, &info);

  for (int i = 0; i < n; ++i) {
    vector_set(x, i, vector_get(b, i));
  }
}

}  // namespace gsl

#endif  // GSL_LINALG_H_

