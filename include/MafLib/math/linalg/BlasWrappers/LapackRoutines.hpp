#ifndef BLAS_LAPACK_ROUTINES_H
#define BLAS_LAPACK_ROUTINES_H

#pragma once
#include "BlasWrapper.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/Matrix.hpp"

/**
 * @file LapackRoutines.hpp
 * @brief Object-level LAPACK routines built on top of the Matrix class.
 *
 * This header is included at the *bottom* of Matrix.hpp (before QR.hpp), so
 * the Matrix class is already complete here; including it anywhere earlier
 * would create a circular dependency. QRResult is only forward declared and
 * is defined in QR.hpp.
 *
 * Identical counterparts are provided by maf::math::acc (Accelerate, Apple)
 * and maf::math::oblas (OpenBLAS, Linux).
 */
namespace maf::math {
template <std::floating_point T>
struct QRResult;  // Defined in QR.hpp

namespace detail {
/** @brief Promotes an integral input type to double for LAPACK work. */
template <typename T>
using float_promote_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;
}  // namespace detail
}  // namespace maf::math

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
namespace maf::math::acc {

/**
 * @brief Computes the QR factorization of A using the LAPACK backend.
 * @details Follows the thin/full conventions of QR_decompostion():
 *  - R is (m x n) if full_R else (k x n), with k = min(m, n).
 *  - Q is (m x m) if full_Q else (m x k).
 * @param A The input matrix.
 * @param full_Q Whether to produce the full m x m orthogonal matrix Q.
 * @param full_R Whether to produce the full m x n upper triangular matrix R.
 * @return QRResult containing Q and R.
 * @throws std::invalid_argument if A is empty.
 * @throws std::runtime_error if the LAPACK backend reports a failure.
 */
template <Numeric T>
[[nodiscard]] auto qr_factorize(const Matrix<T> &A, bool full_Q = false,
                                bool full_R = false)
    -> QRResult<detail::float_promote_t<T>> {
  using DataType = detail::float_promote_t<T>;
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<DataType> A_work = A.template cast<DataType>();
  Matrix<DataType> A_lap = A_work.transposed();  // Column-major order for LAPACK
  auto *a = A_lap.data();

  auto mm = (__LAPACK_int)m;
  auto nn = (__LAPACK_int)n;
  auto kk = (__LAPACK_int)k;
  auto lda = (__LAPACK_int)m;

  std::vector<DataType> tau_lap(k);

  __LAPACK_int info = 0;
  __LAPACK_int lwork = -1;
  DataType wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, (float *)a, &lda, (float *)tau_lap.data(), (float *)&wq, &lwork,
            &info);
  } else {
    dgeqrf_(&mm, &nn, (double *)a, &lda, (double *)tau_lap.data(), (double *)&wq,
            &lwork, &info);
  }
  lwork = (__LAPACK_int)std::max<double>(1.0, std::ceil((double)wq));
  std::vector<DataType> work((size_t)lwork);

  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, (float *)a, &lda, (float *)tau_lap.data(), (float *)work.data(),
            &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("Accelerate/LAPACK geqrf failed");
  }

  // Get R
  Matrix<DataType> R(full_R ? m : k, n);
  R.fill(0.0);
  for (size_t i = 0; i < R.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {
      R[i][j] = A_lap[j][i];
    }
  }

  // Get Q
  if (full_Q && m > n) {
    Matrix<DataType> A_big(m, m);
    A_big.fill(0.0);
    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < m; ++c) {
        A_big[r][c] = A_lap[r][c];
      }
    }
    A_lap = std::move(A_big);
    a = A_lap.data();
    nn = (__LAPACK_int)m;
  }
  const size_t qcols_sz = full_Q ? m : k;
  auto qcols = (__LAPACK_int)qcols_sz;
  lwork = -1;
  wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (float *)tau_lap.data(), (float *)&wq,
            &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)&wq, &lwork, &info);
  }
  lwork = (__LAPACK_int)std::max<double>(1.0, std::ceil((double)wq));
  work.assign((size_t)lwork, DataType(0));

  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (float *)tau_lap.data(),
            (float *)work.data(), &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("Accelerate/LAPACK orgqr failed");
  }

  Matrix<DataType> Q(m, qcols_sz);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < qcols_sz; ++j) {
      Q[i][j] = A_lap[j][i];
    }
  }

  return {std::move(Q), std::move(R)};
}

}  // namespace maf::math::acc
#elifdef OPENBLAS_AVAILABLE
// OpenBLAS does not declare the (Fortran ABI) LAPACK routines through cblas.h.
extern "C" {
void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork,
             int *info);
void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork,
             int *info);
void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work,
             int *lwork, int *info);
void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work,
             int *lwork, int *info);
}

namespace maf::math::oblas {

/**
 * @brief Computes the QR factorization of A using the LAPACK backend.
 * @details Follows the thin/full conventions of QR_decompostion():
 *  - R is (m x n) if full_R else (k x n), with k = min(m, n).
 *  - Q is (m x m) if full_Q else (m x k).
 * @param A The input matrix.
 * @param full_Q Whether to produce the full m x m orthogonal matrix Q.
 * @param full_R Whether to produce the full m x n upper triangular matrix R.
 * @return QRResult containing Q and R.
 * @throws std::invalid_argument if A is empty.
 * @throws std::runtime_error if the LAPACK backend reports a failure.
 */
template <Numeric T>
[[nodiscard]] auto qr_factorize(const Matrix<T> &A, bool full_Q = false,
                                bool full_R = false)
    -> QRResult<detail::float_promote_t<T>> {
  using DataType = detail::float_promote_t<T>;
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<DataType> A_work = A.template cast<DataType>();
  Matrix<DataType> A_lap = A_work.transposed();  // Column-major order for LAPACK
  auto *a = A_lap.data();

  auto mm = static_cast<int>(m);
  auto nn = static_cast<int>(n);
  auto kk = static_cast<int>(k);
  auto lda = static_cast<int>(m);

  std::vector<DataType> tau_lap(k);

  int info = 0;
  int lwork = -1;
  DataType wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, (float *)a, &lda, (float *)tau_lap.data(), (float *)&wq, &lwork,
            &info);
  } else {
    dgeqrf_(&mm, &nn, (double *)a, &lda, (double *)tau_lap.data(), (double *)&wq,
            &lwork, &info);
  }
  lwork = static_cast<int>(std::max<double>(1.0, std::ceil((double)wq)));
  std::vector<DataType> work((size_t)lwork);

  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, (float *)a, &lda, (float *)tau_lap.data(), (float *)work.data(),
            &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("OpenBLAS/LAPACK geqrf failed");
  }

  // Get R
  Matrix<DataType> R(full_R ? m : k, n);
  R.fill(0.0);
  for (size_t i = 0; i < R.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {
      R[i][j] = A_lap[j][i];
    }
  }

  // Get Q
  if (full_Q && m > n) {
    Matrix<DataType> A_big(m, m);
    A_big.fill(0.0);
    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < m; ++c) {
        A_big[r][c] = A_lap[r][c];
      }
    }
    A_lap = std::move(A_big);
    a = A_lap.data();
    nn = static_cast<int>(m);
  }
  const size_t qcols_sz = full_Q ? m : k;
  auto qcols = static_cast<int>(qcols_sz);
  lwork = -1;
  wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (float *)tau_lap.data(), (float *)&wq,
            &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)&wq, &lwork, &info);
  }
  lwork = static_cast<int>(std::max<double>(1.0, std::ceil((double)wq)));
  work.assign((size_t)lwork, DataType(0));

  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (float *)tau_lap.data(),
            (float *)work.data(), &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (double *)tau_lap.data(),
            (double *)work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("OpenBLAS/LAPACK orgqr failed");
  }

  Matrix<DataType> Q(m, qcols_sz);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < qcols_sz; ++j) {
      Q[i][j] = A_lap[j][i];
    }
  }

  return {std::move(Q), std::move(R)};
}

}  // namespace maf::math::oblas
#endif

#endif  // ACCELERATE_LAPACK_ROUTINES_H
