#ifndef KERNELS_H
#define KERNELS_H
#pragma once
#include "AccelerateWrappers/AccelerateWrapper.hpp"
#include "LinAlg.hpp"

namespace maf::math::kernels {
enum class OP : uint8 { NoTrans, Trans };

namespace detail {
template <Numeric T, Numeric U, Numeric R = std::common_type_t<T, U>>
void no_trans_gemv(const MatrixView<T> &A, const VectorView<U> &x, Vector<R> &y);

template <Numeric T, Numeric U, Numeric R = std::common_type_t<T, U>>
void trans_gemv(const VectorView<T> &x, const MatrixView<U> &A, Vector<R> &y);
}  // namespace detail

/** @brief General matrix-vector multiplication (GEMV).
 * @tparam T Numeric type of matrix A.
 * @tparam U Numeric type of vector x.
 *
 * @attention If types of VectorView and MatrixView are the same and are floating-point
 * types, the function will attempt to use optimized BLAS routines if available
 * Otherwise, it will fall back to a manual implementation.
 *
 * @param trans Specifies whether to transpose matrix A.
 * @param A The input matrix.
 * @param x The input vector.
 * @return The resulting vector y = A * x or y = A^T * x.
 */
template <Numeric T, Numeric U>
auto gemv(OP trans, const MatrixView<T> &A, const VectorView<U> &x) {
  using T_value_type = std::remove_cvref_t<T>;
  using U_value_type = std::remove_cvref_t<U>;

  using R = std::common_type_t<T, U>;

  size_t out_size = (trans == OP::NoTrans) ? A.row_count() : A.column_count();
  Vector<R> y(out_size, (trans == OP::NoTrans) ? COLUMN : ROW);

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
  if constexpr (std::is_same_v<T_value_type, float> &&
                std::is_same_v<U_value_type, float>) {
    // https://developer.apple.com/documentation/accelerate/cblas_sgemv(_:_:_:_:_:_:_:_:_:_:_:_:)?language=objc
    CBLAS_TRANSPOSE blas_trans = (trans == OP::NoTrans) ? CblasNoTrans : CblasTrans;
    cblas_sgemv(CblasRowMajor, blas_trans, (int)A.row_count(), (int)A.column_count(),
                1.0F, A.data(), (int)A.get_stride(), x.data(), (int)x.get_increment(),
                0.0F, y.data(), 1);
    return y;
  } else if constexpr (std::is_same_v<T_value_type, double> &&
                       std::is_same_v<U_value_type, double>) {
    // https://developer.apple.com/documentation/accelerate/cblas_dgemv(_:_:_:_:_:_:_:_:_:_:_:_:)?language=objc
    CBLAS_TRANSPOSE blas_trans = (trans == OP::NoTrans) ? CblasNoTrans : CblasTrans;
    cblas_dgemv(CblasRowMajor, blas_trans, (int)A.row_count(), (int)A.column_count(),
                1.0, A.data(), (int)A.get_stride(), x.data(), (int)x.get_increment(),
                0.0, y.data(), 1);
    return y;
  }
#endif

  if (trans == OP::NoTrans) {
    detail::no_trans_gemv(A, x, y);

  } else {
    detail::trans_gemv(x, A, y);
  }
  return y;
}

namespace detail {
template <Numeric T, Numeric U, Numeric R>
void no_trans_gemv(const MatrixView<T> &A, const VectorView<U> &x, Vector<R> &y) {
  // Dimensions are checked higher up the calls.
  if (A.row_count() * A.column_count() >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < A.row_count(); ++i) {
      auto row = A.row_span(i);
      R sum = 0;
#pragma omp simd
      for (size_t j = 0; j < A.column_count(); ++j) {
        sum += row[j] * x[j];
      }
      y[i] = sum;
    }
  } else {
    for (size_t i = 0; i < A.row_count(); ++i) {
      auto row = A.row_span(i);
      R sum = 0;
#pragma omp simd
      for (size_t j = 0; j < A.column_count(); ++j) {
        sum += row[j] * x[j];
      }
      y[i] = sum;
    }
  }
}

template <Numeric T, Numeric U, Numeric R>
void trans_gemv(const VectorView<T> &x, const MatrixView<U> &A, Vector<R> &y) {
  // Dimensions are checked higher up the calls.
  if (A.row_count() * A.column_count() >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
    for (size_t j = 0; j < A.column_count(); ++j) {
      R sum = 0;
      for (size_t i = 0; i < A.row_count(); ++i) {
        sum += x[i] * A[i][j];
      }
      y[j] = sum;
    }
  } else {
    for (size_t j = 0; j < A.column_count(); ++j) {
      R sum = 0;
      for (size_t i = 0; i < A.row_count(); ++i) {
        sum += x[i] * A[i][j];
      }
      y[j] = sum;
    }
  }
}

}  // namespace detail
}  // namespace maf::math::kernels
#endif
