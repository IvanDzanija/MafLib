#ifndef QR_HPP
#define QR_HPP
#include <stdexcept>

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "Matrix.hpp"
#include "MatrixView.hpp"
#include "Vector.hpp"
#include "VectorView.hpp"

#pragma once

namespace maf::math {
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
static inline T householder_column(MatrixView<T> A_work, size_t j) {
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

template <std::floating_point T>
static inline Vector<T> load_reflector(const Matrix<T> &A_work, size_t j) {
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

template <Numeric T>
[[nodiscard]] QRResultType<T> QR_decompostion(const Matrix<T> &A, bool full_R = false,
                                              bool full_Q = false) {
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<T> A_work = A;
  std::vector<T> tau(k, 0.0);

  for (size_t j = 0; j < k; ++j) {
    auto Aw_view = A_work.view(0, 0, m, n);

    tau[j] = detail::householder_column(Aw_view, j);
    if (tau[j] == 0.0) {
      continue;
    }

    Vector<double> v = detail::load_reflector(A_work, j);
    auto v_view = v.view(0, v.size());

    // DEBUG:
    std::cout << "Householder reflector for column " << j << ":\n";
    v.print();
    std::cout << "tau: " << tau[j] << "\n";
    std::cout << "A_work after forming reflector:\n";
    A_work.print();

    if (j + 1 < n) {
      auto A_block = A_work.view(j, j + 1, m - j, n - (j + 1));

      std::cout << "Next block :" << std::endl;
      A_block.print();

      // w = A_block^T * v
      Vector<T> w = kernels::gemv(kernels::OP::Trans, A_block, v_view);
      std::cout << "w: " << std::endl;
      w.print();

      auto w_view = w.view(0, w.size());

      // A_block = A_block - (v_view * w_view) * tau[j];
      kernels::ger(A_block, v_view, w_view, -tau[j]);

      std::cout << "A_block after applying reflector:\n";
      A_block.print();
      std::cout << std::endl;
    }
  }

  // Get R
  Matrix<T> R(full_R ? m : n, n);
  for (size_t i = 0; i < R.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {
      R[i][j] = A_work[i][j];
    }
  }

  // Get Q
  Matrix<T> Q_full = identity_matrix<T>(m);

  for (size_t t = 0; t < k; ++t) {
    const size_t j = (k - 1) - t;
    if (tau[j] == 0.0) {
      continue;
    }

    Vector<double> v = detail::load_reflector(A_work, j);
    auto v_view = v.view(0, v.size());

    auto Qblock = Q_full.view(j, 0, m - j, m);

    Vector<T> w = kernels::gemv(kernels::OP::Trans, Qblock, v_view);

    auto w_view = w.view(0, w.size());
    kernels::ger(Qblock, v_view, w_view, -tau[j]);
  }

  // Thin or full Q
  Matrix<T> Q = full_Q ? std::move(Q_full) : Matrix<double>(m, k);
  if (!full_Q) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        Q[i][j] = Q_full[i][j];
      }
    }
  }

  return {std::move(Q), std::move(R)};
}

}  // namespace maf::math
#endif
