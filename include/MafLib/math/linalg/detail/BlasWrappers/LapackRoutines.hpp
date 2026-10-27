#ifndef MAF_BLAS_LAPACK_ROUTINES_H
#define MAF_BLAS_LAPACK_ROUTINES_H

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/detail/BlasWrappers/BlasWrapper.hpp"
#include "MafLib/math/linalg/detail/Matrix.hpp"

/**
 * @file LapackRoutines.hpp
 * @brief Object-level LAPACK routines built on top of the Matrix class.
 *
 * It is pulled in by detail/linalg.hpp after the Matrix and Vector class
 * definitions, which it requires; it is never included earlier than them.
 * would create a circular dependency. QRResult is only forward declared and
 * is defined in QR.hpp.
 *
 * Identical counterparts are provided by maf::math::acc (Accelerate, Apple)
 * and maf::math::oblas (OpenBLAS, Linux).
 */
namespace maf::math {
template <std::floating_point T>
struct QRResult;  // Defined in QR.hpp
template <std::floating_point T>
struct PLUResult;  // Defined in PLU.hpp

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
[[nodiscard]] auto qr_factorize(const Matrix<T> &a_matrix, bool full_q = false,
                                bool full_r = false)
    -> QRResult<detail::float_promote_t<T>> {
  using DataType = detail::float_promote_t<T>;
  const size_t m = a_matrix.row_count();
  const size_t n = a_matrix.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<DataType> a_work = a_matrix.template cast<DataType>();
  Matrix<DataType> a_lap = a_work.transposed();  // Column-major order for LAPACK
  auto *a = a_lap.data();

  auto mm = static_cast<__LAPACK_int>(m);
  auto nn = static_cast<__LAPACK_int>(n);
  auto kk = static_cast<__LAPACK_int>(k);
  auto lda = static_cast<__LAPACK_int>(m);

  std::vector<DataType> tau_lap(k);

  __LAPACK_int info = 0;
  __LAPACK_int lwork = -1;
  DataType wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  }
  lwork = static_cast<__LAPACK_int>(
      std::max<double>(1.0, std::ceil(static_cast<double>(wq))));
  std::vector<DataType> work(static_cast<size_t>(lwork));

  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("Accelerate/LAPACK geqrf failed");
  }

  // Get R
  Matrix<DataType> r(full_r ? m : k, n);
  for (size_t i = 0; i < r.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {
      r[i, j] = a_lap[j, i];
    }
  }

  // Get Q
  if (full_q && m > n) {
    Matrix<DataType> a_big(m, m);
    a_big.fill(0.0);
    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < m; ++c) {
        a_big[r, c] = a_lap[r, c];
      }
    }
    a_lap = std::move(a_big);
    a = a_lap.data();
    nn = static_cast<__LAPACK_int>(m);
  }
  const size_t qcols_sz = full_q ? m : k;
  auto qcols = static_cast<__LAPACK_int>(qcols_sz);
  lwork = -1;
  wq = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  }
  lwork = static_cast<__LAPACK_int>(
      std::max<double>(1.0, std::ceil(static_cast<double>(wq))));
  work.assign(static_cast<size_t>(lwork), DataType(0));

  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  }
  if (info != 0) {
    throw std::runtime_error("Accelerate/LAPACK orgqr failed");
  }

  Matrix<DataType> q(m, qcols_sz);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < qcols_sz; ++j) {
      q[i, j] = a_lap[j, i];
    }
  }

  return {std::move(q), std::move(r)};
}

/**
 * @brief Computes the PLU factorization of A using the LAPACK backend (getrf).
 * @details Matches the conventions of plu():
 *  - P[i] is the index of the original row of A that ends up in row i, so
 *    (P * A) = L * U, i.e. row i of P*A is row P[i] of A.
 *  - L is unit lower triangular, U is upper triangular.
 * @param matrix The square input matrix.
 * @return PLUResult containing P, L, U and the sign of the permutation.
 * @throws std::invalid_argument if the matrix is not square.
 * @throws std::runtime_error if the matrix is singular or the backend fails.
 */
template <typename DataType, Numeric T>
[[nodiscard]] PLUResult<DataType> lu_factorize(const Matrix<T> &a_matrix) {
  if (a_matrix.row_count() != a_matrix.column_count()) {
    throw std::invalid_argument("Matrix must be square for PLU decomposition!");
  }
  const size_t n = a_matrix.row_count();
  if (n == 0) {
    throw std::invalid_argument("Matrix dimensions must be greater than zero.");
  }

  // Row-major -> column-major via a transposed copy (same trick as QR).
  Matrix<DataType> a_lap = a_matrix.template cast<DataType>().transposed();
  auto *a = a_lap.data();

  auto nn = static_cast<__LAPACK_int>(n);
  std::vector<__LAPACK_int> ipiv(n);
  __LAPACK_int info = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sgetrf_(&nn, &nn, a, &nn, ipiv.data(), &info);
  } else {
    dgetrf_(&nn, &nn, a, &nn, ipiv.data(), &info);
  }
  if (info > 0) {
    throw std::runtime_error("Matrix is singular; pivot is near zero.");
  }
  if (info < 0) {
    throw std::runtime_error("Accelerate/LAPACK getrf failed");
  }

  // Reconstruct the permutation vector: apply the same swaps LAPACK performed.
  std::vector<uint32> P(n);
  std::iota(P.begin(), P.end(), 0);
  int8 sign = 1;
  for (size_t k = 0; k < n; ++k) {
    const auto piv = static_cast<size_t>(ipiv[k]);
    if (piv - 1 != k) {  // ipiv entries are 1-based
      std::swap(P[k], P[piv - 1]);
      sign *= -1;
    }
  }

  // Extract L and U from the factored matrix (rows already in final order).
  Matrix<DataType> combined = a_lap.transposed();  // Row-major again
  Matrix<DataType> l = identity_matrix<DataType>(n);
  Matrix<DataType> u(n, n);
#pragma omp parallel for schedule(static) if (n > 256)
  for (size_t i = 0; i < n; ++i) {
    // L rows: the strict lower prefix is [0, i); unit diagonal stays 1.
    std::copy_n(&combined.at(i, 0), i, &l.at(i, 0));
    // U rows: the upper (and diagonal) part is [i, n).
    std::copy_n(&combined.at(i, i), n - i, &u.at(i, i));
  }

  return {std::move(P), std::move(l), std::move(u), sign};
}

/**
 * @brief Computes the Cholesky factorization A = L * L^T (potrf).
 * @details Matches the conventions of cholesky(): returns the lower
 *          triangular factor L as a row-major Matrix, upper half zeroed.
 * @param matrix The symmetric positive definite input matrix.
 * @return The lower triangular Matrix L.
 * @throws std::invalid_argument if the matrix is not symmetric or not
 *         positive definite.
 * @throws std::runtime_error if the backend reports a failure.
 */
template <typename DataType, Numeric T>
[[nodiscard]] Matrix<DataType> cholesky_factorize(const Matrix<T> &a_matrix) {
  if (!a_matrix.is_symmetric()) {
    throw std::invalid_argument(
        "Matrix must be symmetric to try Cholesky decomposition!");
  }
  const size_t n = a_matrix.row_count();
  if (n == 0) {
    throw std::invalid_argument("Matrix dimensions must be greater than zero.");
  }

  Matrix<DataType> a_lap = a_matrix.template cast<DataType>().transposed();
  auto *a = a_lap.data();

  auto nn = static_cast<__LAPACK_int>(n);
  const char uplo = 'L';
  __LAPACK_int info = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    spotrf_(&uplo, &nn, a, &nn, &info);
  } else {
    dpotrf_(&uplo, &nn, a, &nn, &info);
  }
  if (info > 0) {
    throw std::invalid_argument("Matrix is not positive definite!");
  }
  if (info < 0) {
    throw std::runtime_error("Accelerate/LAPACK potrf failed");
  }

  // Column-major lower factor -> row-major lower factor.
  Matrix<DataType> l = a_lap.transposed();
#pragma omp parallel for schedule(static) if (n > 256)
  for (size_t i = 0; i < n; ++i) {
    if (i + 1 >= n) {
      continue;  // Last row has no upper half to zero
    }
    std::fill_n(&l.at(i, i + 1), n - i - 1, DataType(0));
  }
  return l;
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
void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
void spotrf_(const char *uplo, int *n, float *a, int *lda, int *info);
void dpotrf_(const char *uplo, int *n, double *a, int *lda, int *info);
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
    sgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  }
  lwork = static_cast<int>(std::max<double>(1.0, std::ceil(static_cast<double>(wq))));
  std::vector<DataType> work(static_cast<size_t>(lwork));

  if constexpr (std::is_same_v<DataType, float>) {
    sgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  } else {
    dgeqrf_(&mm, &nn, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
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
    sorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), &wq, &lwork, &info);
  }
  lwork = static_cast<int>(std::max<double>(1.0, std::ceil(static_cast<double>(wq))));
  work.assign(static_cast<size_t>(lwork), DataType(0));

  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, a, &lda, tau_lap.data(), work.data(), &lwork, &info);
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

/**
 * @brief Computes the PLU factorization of A using the LAPACK backend (getrf).
 * @details Same P/L/U conventions as acc::lu_factorize() and plu().
 * @throws std::invalid_argument if the matrix is not square.
 * @throws std::runtime_error if the matrix is singular or the backend fails.
 */
template <typename DataType, Numeric T>
[[nodiscard]] PLUResult<DataType> lu_factorize(const Matrix<T> &A) {
  if (A.row_count() != A.column_count()) {
    throw std::invalid_argument("Matrix must be square for PLU decomposition!");
  }
  const size_t n = A.row_count();
  if (n == 0) {
    throw std::invalid_argument("Matrix dimensions must be greater than zero.");
  }

  Matrix<DataType> A_lap = A.template cast<DataType>().transposed();
  auto *a = A_lap.data();

  auto nn = static_cast<int>(n);
  std::vector<int> ipiv(n);
  int info = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    sgetrf_(&nn, &nn, a, &nn, ipiv.data(), &info);
  } else {
    dgetrf_(&nn, &nn, a, &nn, ipiv.data(), &info);
  }
  if (info > 0) {
    throw std::runtime_error("Matrix is singular; pivot is near zero.");
  }
  if (info < 0) {
    throw std::runtime_error("OpenBLAS/LAPACK getrf failed");
  }

  std::vector<uint32> P(n);
  std::iota(P.begin(), P.end(), 0);
  int8 sign = 1;
  for (size_t k = 0; k < n; ++k) {
    const auto piv = static_cast<size_t>(ipiv[k]);
    if (piv - 1 != k) {  // ipiv entries are 1-based
      std::swap(P[k], P[piv - 1]);
      sign *= -1;
    }
  }

  Matrix<DataType> combined = A_lap.transposed();
  Matrix<DataType> L = identity_matrix<DataType>(n);
  Matrix<DataType> U(n, n);
#pragma omp parallel for schedule(static) if (n > 256)
  for (size_t i = 0; i < n; ++i) {
    std::copy_n(&combined.at(i, 0), i, &L.at(i, 0));
    std::copy_n(&combined.at(i, i), n - i, &U.at(i, i));
  }

  return {std::move(P), std::move(L), std::move(U), sign};
}

/**
 * @brief Computes the Cholesky factorization A = L * L^T (potrf).
 * @details Same conventions as maf::math::cholesky(): lower triangular L,
 *          upper half zeroed.
 * @throws std::invalid_argument if the matrix is not symmetric or not
 *         positive definite.
 * @throws std::runtime_error if the backend reports a failure.
 */
template <typename DataType, Numeric T>
[[nodiscard]] Matrix<DataType> cholesky_factorize(const Matrix<T> &A) {
  if (!A.is_symmetric()) {
    throw std::invalid_argument(
        "Matrix must be symmetric to try Cholesky decomposition!");
  }
  const size_t n = A.row_count();
  if (n == 0) {
    throw std::invalid_argument("Matrix dimensions must be greater than zero.");
  }

  Matrix<DataType> A_lap = A.template cast<DataType>().transposed();
  auto *a = A_lap.data();

  auto nn = static_cast<int>(n);
  const char uplo = 'L';
  int info = 0;
  if constexpr (std::is_same_v<DataType, float>) {
    spotrf_(&uplo, &nn, a, &nn, &info);
  } else {
    dpotrf_(&uplo, &nn, a, &nn, &info);
  }
  if (info > 0) {
    throw std::invalid_argument("Matrix is not positive definite!");
  }
  if (info < 0) {
    throw std::runtime_error("OpenBLAS/LAPACK potrf failed");
  }

  Matrix<DataType> L = A_lap.transposed();
#pragma omp parallel for schedule(static) if (n > 256)
  for (size_t i = 0; i < n; ++i) {
    if (i + 1 >= n) {
      continue;  // Last row has no upper half to zero
    }
    std::fill_n(&L.at(i, i + 1), n - i - 1, DataType(0));
  }
  return L;
}

}  // namespace maf::math::oblas
#endif

#endif  // ACCELERATE_LAPACK_ROUTINES_H
