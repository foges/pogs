/*
 * Symbol prefix mappings for scipy-openblas32 on Windows.
 *
 * scipy-openblas32 exports all BLAS/LAPACK symbols with a "scipy_" prefix
 * (e.g., scipy_cblas_sdot instead of cblas_sdot). This header remaps the
 * standard symbol names to the prefixed versions at compile time.
 *
 * This file is force-included on Windows builds via /FI compiler flag.
 */

#ifndef POGS_OPENBLAS_PREFIX_H_
#define POGS_OPENBLAS_PREFIX_H_

#ifdef _WIN32

/* CBLAS Level 1 */
#define cblas_saxpy scipy_cblas_saxpy
#define cblas_daxpy scipy_cblas_daxpy
#define cblas_sscal scipy_cblas_sscal
#define cblas_dscal scipy_cblas_dscal
#define cblas_sasum scipy_cblas_sasum
#define cblas_dasum scipy_cblas_dasum
#define cblas_sdot scipy_cblas_sdot
#define cblas_ddot scipy_cblas_ddot
#define cblas_snrm2 scipy_cblas_snrm2
#define cblas_dnrm2 scipy_cblas_dnrm2

/* CBLAS Level 2 */
#define cblas_sgemv scipy_cblas_sgemv
#define cblas_dgemv scipy_cblas_dgemv
#define cblas_strsv scipy_cblas_strsv
#define cblas_dtrsv scipy_cblas_dtrsv

/* CBLAS Level 3 */
#define cblas_ssyrk scipy_cblas_ssyrk
#define cblas_dsyrk scipy_cblas_dsyrk
#define cblas_sgemm scipy_cblas_sgemm
#define cblas_dgemm scipy_cblas_dgemm
#define cblas_strsm scipy_cblas_strsm
#define cblas_dtrsm scipy_cblas_dtrsm

/* LAPACK routines */
#define ssyevd_ scipy_ssyevd_
#define dsyevd_ scipy_dsyevd_
#define strtrs_ scipy_strtrs_
#define dtrtrs_ scipy_dtrtrs_

#endif /* _WIN32 */

#endif /* POGS_OPENBLAS_PREFIX_H_ */
