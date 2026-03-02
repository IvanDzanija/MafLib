#ifndef QR_HPP
#define QR_HPP

#pragma once
#include "AccelerateWrappers/AccelerateWrapper.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "Matrix.hpp"
#include "MatrixView.hpp"
#include "Vector.hpp"
#include "VectorView.hpp"

/**
 * @file QR.hpp
 * @brief High-performance QR decomposition using Householder reflections.
 *
 * This header defines the `QR_decomposition` function, which computes the QR
 * decomposition of a given matrix A using Householder reflections. The
 * implementation is designed for performance and numerical stability, and it
 * supports both single and double precision floating point types. The function
 * allows users to choose between returning full or thin versions of the Q and R
 * matrices based on their needs.
 *
 * More information:
 * https://en.wikipedia.org/wiki/QR_decomposition
 */
namespace maf::math {
/**
 * @brief Struct to hold the results of QR decomposition.
 * @tparam T The floating point type of the matrix elements (e.g., float, double).
 * @details This struct contains the orthogonal matrix Q and the upper triangular
 * matrix R resulting from the QR decomposition. The types of Q and R are both
 * Matrix<T>, where T is a floating point type. The user can choose to return
 * either the full or thin versions of Q and R based on their needs when calling
 * the QR_decomposition function.
 */
template <std::floating_point T>
struct QRResult {
  Matrix<T> Q;
  Matrix<T> R;
};

namespace detail {
/**
 * @brief Computes the Householder reflector for a column of A_work.
 * @param A_work The matrix containing the column to reflect. Modified in-place to store
 * the reflector.
 * @param j The index of the column to reflect (0-based).
 * @return The scalar tau for the Householder transformation.
 * @details This function computes the Householder reflector for column j of A_work,
 * which is used to zero out the sub-diagonal elements of that column. The reflector is
 * stored in a compact form: the first element (the "beta" value) is stored in
 * A_work(j,j), and the rest of the reflector vector (the "v" tail) is stored in
 * A_work(j+1..m-1, j). The function returns the scalar tau, which is used in the
 * Householder transformation H = I - tau * v * v^T.
 * @attention The input matrix A_work is modified in-place to store the reflector. The
 * caller should ensure that A_work has enough space to store the reflector and that j
 * is a valid column index.
 */
template <std::floating_point T>
[[nodiscard]] static inline T householder_column(MatrixView<T> A_work, size_t j) {
  const size_t m = A_work.row_count();
  if (j >= m) {
    throw std::out_of_range("Householder column index out of range!");
  }

  // sigma = x_tail^2
  T sigma = 0.0;
  for (size_t i = j + 1; i < m; ++i) {
    sigma += A_work[i][j] * A_work[i][j];
  }

  // Simple case, already in canonical form
  if (sigma == 0.0) {
    return 0.0;
  }

  const T alpha = A_work.at(j, j);
  const T normx = std::sqrt(alpha * alpha + sigma);
  const T beta = (alpha <= 0.0) ? normx : -normx;  // stability trick
  const T u0_inv = 1.0 / (alpha - beta);
  for (size_t i = j + 1; i < m; ++i) {
    A_work[i][j] *= u0_inv;
  }

  A_work[j][j] = beta;

  T vTv = 1.0;
  for (size_t i = j + 1; i < m; ++i) {
    vTv += A_work[i][j] * A_work[i][j];
  }
  return T(2) / vTv;
}

/**
 * @brief Loads the Householder reflector vector from A_work for column j.
 * @param A_work The matrix containing the reflector. The reflector is stored in a
 * compact form: the first element (the "beta" value) is stored in A_work(j,j), and the
 * rest of the reflector vector (the "v" tail) is stored in A_work(j+1..m-1, j).
 * @param j The index of the column for which to load the reflector (0-based).
 * @return A Vector<T> containing the full Householder reflector vector v, where v[0]
 * = 1.0 and v[1..len-1] are loaded from A_work(j+1..m-1, j).
 * @details This function reconstructs the full Householder reflector vector v from its
 * compact storage in A_work. The first element of v is set to 1.0, and the remaining
 * elements are copied from the corresponding entries in A_work. The length of the
 * returned vector is m - j, where m is the number of rows in A_work.
 * @attention The input matrix A_work should have been modified by householder_column()
 * to store the reflector. The caller should ensure that j is a valid column index and
 * that A_work has enough space to store the reflector.
 */
template <std::floating_point T>
[[nodiscard]] static inline Vector<T> load_reflector(const Matrix<T> &A_work,
                                                     size_t j) {
  const size_t m = A_work.row_count();
  const size_t len = m - j;
  Vector<T> v_out(len, COLUMN);
  v_out[0] = 1.0;
  for (size_t i = 1; i < len; ++i) {
    v_out[i] = A_work[i + j][j];
  }
  return v_out;
}

template <typename T>
using float_promote_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

}  // namespace detail

template <Numeric T>
using QRResultType = QRResult<detail::float_promote_t<T>>;

/**
 * @brief Computes the QR decomposition of a matrix A using Householder reflections.
 * @param A The input matrix to decompose.
 * @param full_Q If true, returns the full m x m orthogonal matrix Q. If false, returns
 * the thin m x k matrix Q (where k = min(m, n)).
 * @param full_R If true, returns the full m x n upper triangular matrix R. If false,
 * returns the thin k x n matrix R (where k = min(m, n)).
 * @return A QRResultType<T> containing the matrices Q and R of the decomposition.
 * @details This function performs QR decomposition using Householder reflections. It
 * iteratively computes Householder reflectors to zero out sub-diagonal elements of A,
 * storing the reflectors in-place. After processing all columns, it extracts the upper
 * triangular matrix R and constructs the orthogonal matrix Q from the stored
 * reflectors. The user can choose to return either the full or thin versions of Q and R
 * based on their needs.
 * @throws std::invalid_argument if A is empty or if dimensions are incompatible for
 * decomposition.
 */
template <Numeric T>
[[nodiscard]] QRResultType<T> QR_decompostion(const Matrix<T> &A, bool full_Q = false,
                                              bool full_R = false) {
  using DataType = detail::float_promote_t<T>;
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<DataType> A_work = A.template cast<DataType>();
  std::vector<DataType> tau(k, 0.0);

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
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
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (const float *)tau_lap.data(),
            (float *)&wq, &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (const double *)tau_lap.data(),
            (double *)&wq, &lwork, &info);
  }
  lwork = (__LAPACK_int)std::max<double>(1.0, std::ceil((double)wq));
  work.assign((size_t)lwork, DataType(0));

  if constexpr (std::is_same_v<DataType, float>) {
    sorgqr_(&mm, &qcols, &kk, (float *)a, &lda, (const float *)tau_lap.data(),
            (float *)work.data(), &lwork, &info);
  } else {
    dorgqr_(&mm, &qcols, &kk, (double *)a, &lda, (const double *)tau_lap.data(),
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
#else

  for (size_t j = 0; j < k; ++j) {
    auto Aw_view = A_work.view(0, 0, m, n);

    tau[j] = detail::householder_column(Aw_view, j);
    if (tau[j] == 0.0) {
      continue;
    }

    Vector<DataType> v = detail::load_reflector(A_work, j);
    auto v_view = v.view(0, v.size());

    if (j + 1 < n) {
      auto A_block = A_work.view(j, j + 1, m - j, n - (j + 1));

      // w = A_block^T * v
      Vector<DataType> w = kernels::gemv(kernels::OP::Trans, A_block, v_view);
      auto w_view = w.view(0, w.size());

      // A_block = A_block - (v_view * w_view) * tau[j];
      kernels::ger(A_block, v_view, w_view, -tau[j]);
    }
  }

  // Get R
  Matrix<DataType> R(full_R ? m : k, n);
  R.fill(0.0);
  for (size_t i = 0; i < R.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {
      R[i][j] = A_work[i][j];
    }
  }

  // Get Q
  auto Q_full = identity_matrix<DataType>(m);

  for (size_t t = 0; t < k; ++t) {
    const size_t j = (k - 1) - t;
    if (tau[j] == 0.0) {
      continue;
    }

    Vector<DataType> v = detail::load_reflector(A_work, j);
    auto v_view = v.view(0, v.size());

    auto Qblock = Q_full.view(j, j, m - j, m - j);

    Vector<DataType> w = kernels::gemv(kernels::OP::Trans, Qblock, v_view);

    kernels::ger(Qblock, v_view, w.view(0, w.size()), -tau[j]);
  }

  // Thin or full Q
  Matrix<DataType> Q = full_Q ? std::move(Q_full) : Matrix<DataType>(m, k);
  if (!full_Q) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        Q[i][j] = Q_full[i][j];
      }
    }
  }

  return {std::move(Q), std::move(R)};
#endif
}

}  // namespace maf::math
#endif
