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
// A -= alpha * v * w^T
// A: (m x n)
// v: length m (column)
// w: length n (row, but we store as a vector)
static inline void ger_minus(MatrixView<double> A, VectorView<double> &v,
                             VectorView<double> &w, double alpha) {
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (v.size() != m || w.size() != n) {
    throw std::invalid_argument("ger_minus: size mismatch");
  }

  for (size_t i = 0; i < m; ++i) {
    const double vi = alpha * v[i];
    double *row = A[i];  // row pointer
    for (size_t j = 0; j < n; ++j) {
      row[j] -= vi * w[j];
    }
  }
}

/**
 * @brief Computes the Householder reflector for a column of Awork.
 * @param Awork The matrix containing the column to reflect. Modified in-place to store
 * the reflector.
 * @param j The index of the column to reflect (0-based).
 * @return The scalar tau for the Householder transformation.
 * @details This function computes the Householder reflector for column j of Awork,
 * which is used to zero out the sub-diagonal elements of that column. The reflector is
 * stored in a compact form: the first element (the "beta" value) is stored in
 * Awork(j,j), and the rest of the reflector vector (the "v" tail) is stored in
 * Awork(j+1..m-1, j). The function returns the scalar tau, which is used in the
 * Householder transformation H = I - tau * v * v^T.
 * @attention The input matrix Awork is modified in-place to store the reflector. The
 * caller should ensure that Awork has enough space to store the reflector and that j is
 * a valid column index.
 */
template <std::floating_point T>
static inline T householder_column(MatrixView<T> Awork, size_t j) {
  // TODO: Delete this comment
  // Construct Householder reflector for column j in Awork, acting on rows j..m-1.
  // Packs v tail into Awork(j+1..m-1, j), sets Awork(j,j)=beta, returns tau.
  //
  // Math:
  //  x = A[j:,j]
  //  beta = -sign(x0)*||x||
  //  v = (x - beta e1) / (x0 - beta)  => v0=1, tail stored in Awork below diag
  //  tau = 2/(v^T v)
  //
  // After this, H = I - tau v v^T and H x = [beta, 0, ..., 0]^T
  const size_t m = Awork.row_count();
  if (j >= m) {
    throw std::out_of_range("Householder column index out of range!");
  }

  // sigma = x_tail^2
  T sigma = 0.0;
  for (size_t i = j + 1; i < m; ++i) {
    sigma += Awork[i][j] * Awork[i][j];
  }

  // Simple case, already in canonical form
  if (sigma == 0.0) {
    return 0.0;
  }

  const T alpha = Awork.at(j, j);
  const T normx = std::sqrt(alpha * alpha + sigma);
  const T beta = (alpha <= 0.0) ? normx : -normx;  // stability trick
  const T u0_inv = 1.0 / (alpha - beta);
  for (size_t i = j + 1; i < m; ++i) {
    Awork[i][j] *= u0_inv;
  }

  Awork[j][j] = beta;

  T vTv = 1.0;
  for (size_t i = j + 1; i < m; ++i) {
    vTv += Awork[i][j] * Awork[i][j];
  }
  return T(2) / vTv;
}

// Load explicit reflector vector v (length m-j) from packed storage in Awork.
//
// v = [1;
//      Awork(j+1,j);
//      Awork(j+2,j);
//      ...]
static inline void load_v(const Matrix<double> &Awork, size_t j,
                          Vector<double> &v_out) {
  const size_t m = Awork.row_count();
  const size_t len = m - j;
  v_out = Vector<double>(len, COLUMN);
  v_out[0] = 1.0;
  for (size_t i = 1; i < len; ++i) {
    v_out[i] = Awork.at(j + i, j);
  }
}

}  // namespace detail

template <typename T>
[[nodiscard]] QRResult<T> QR_decompostion(const Matrix<T> &A, bool full_R = false,
                                          bool full_Q = false) {
  const size_t m = A.row_count();
  const size_t n = A.column_count();
  if (m == 0 || n == 0) {
    throw std::invalid_argument("Cannot perform QR decomposition on empty matrix!");
  }

  // Smaller dimension determines number of reflectors
  const size_t k = std::min(m, n);

  Matrix<T> Awork = A;
  std::vector<T> tau(k, 0.0);
  Vector<double> v;  // reflector vector buffer
  Vector<double> w;  // gemv result buffer

  for (size_t j = 0; j < k; ++j) {
    auto Aw_view = Awork.view(0, 0, m, n);

    tau[j] = detail::householder_column(Aw_view, j);
    if (tau[j] == 0.0) {
      continue;
    }

    /// HEREEEE

    // Build explicit v from packed data for gemv and GER update
    load_v(Awork, j, v);
    auto v_view = v.view(0, v.size());  // VectorView<double>, orientation=COLUMN

    // Apply to trailing block Awork(j:m-1, j+1:n-1):
    // Ablock := (I - tau v v^T) Ablock = Ablock - tau v (v^T Ablock)
    if (j + 1 < n) {
      auto Ablock = Awork.view(j, j + 1, m - j, n - (j + 1));  // (m-j) x (n-j-1)

      // w = Ablock^T * v   (size n-j-1)
      // Requires v as COLUMN orientation, which it is.
      w = kernels::gemv(kernels::OP::Trans, Ablock, v_view);

      // Outer product update: Ablock -= tau * v * w^T
      // w from gemv will be a COLUMN vector; we treat it as a 1-D array of length
      // n-j-1.
      auto w_view = w.view(0, w.size());  // ok regardless of orientation for indexing
      detail::ger_minus(Ablock, v_view, w_view, tau[j]);
    }
  }

  // -------------------------
  // 2) Extract R from Awork (upper triangle)
  // -------------------------
  Matrix<double> R(full_R ? m : n, n);
  R.fill(0.0);
  for (size_t i = 0; i < R.row_count(); ++i) {
    for (size_t j = i; j < n; ++j) {  // only upper triangle
      R.at(i, j) = Awork.at(i, j);
    }
  }

  // -------------------------
  // 3) Form Q explicitly: Q = H0 H1 ... H{k-1}
  //    Start Qfull = I, apply H's in reverse
  // -------------------------
  Matrix<double> Qfull(m, m);
  Qfull.make_identity();

  for (size_t t = 0; t < k; ++t) {
    const size_t j = (k - 1) - t;
    if (tau[j] == 0.0) continue;

    load_v(Awork, j, v);
    auto v_view = v.view(0, v.size());  // COLUMN

    // Apply H_j to Qblock = Qfull(j:m-1, 0:m-1)
    auto Qblock = Qfull.view(j, 0, m - j, m);

    // w = Qblock^T * v   (size m)
    w = kernels::gemv(kernels::OP::Trans, Qblock, v_view);

    auto w_view = w.view(0, w.size());
    ger_minus(Qblock, v_view, w_view, tau[j]);
  }

  // Thin or full Q
  Matrix<double> Q = fullQ ? std::move(Qfull) : Matrix<double>(m, k);
  if (!fullQ) {
    for (size_t i = 0; i < m; ++i)
      for (size_t j = 0; j < k; ++j) Q.at(i, j) = Qfull.at(i, j);
  }

  return {std::move(Q), std::move(R)};
}

}  // namespace maf::math
#endif
