#ifndef KERNELS_H
#define KERNELS_H
#pragma once
#include "LinAlg.hpp"

namespace maf::math::kernels {

enum class OP : uint8 { NoTrans, Trans };

/** @brief General matrix-vector multiplication (GEMV).
 * @tparam T Numeric type of matrix A.
 * @tparam U Numeric type of vector x.
 *
 * @param trans Specifies whether to transpose matrix A.
 * @param A The input matrix.
 * @param x The input vector.
 * @return The resulting vector y = A * x or y = A^T * x.
 */
template <Numeric T, Numeric U>
auto gemv(OP trans, MatrixView<T> A, VectorView<U> x) {
  using R = std::common_type_t<T, U>;

  size_t out_size = (trans == OP::NoTrans) ? A.row_count() : A.column_count();
  Vector<R> y(out_size, (trans == OP::NoTrans) ? COLUMN : ROW);

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
  if constexpr (std::is_same_v<R, float> || std::is_same_v<R, double>) {
    CBLAS_TRANSPOSE blas_trans = (trans == OP::NoTrans) ? CblasNoTrans : CblasTrans;

    if constexpr (std::is_same_v<R, float>) {
      cblas_sgemv(CblasRowMajor, blas_trans, (int)A.row_count(), (int)A.column_count(),
                  1.0F, A.data_ptr(), (int)A.stride(), x.data_ptr(), (int)x.increment(),
                  0.0F, y.data(), 1);
    } else {
      cblas_dgemv(CblasRowMajor, blas_trans, (int)A.row_count(), (int)A.column_count(),
                  1.0, A.data_ptr(), (int)A.stride(), x.data_ptr(), (int)x.increment(),
                  0.0, y.data(), 1);
    }
    return y;
  }
#endif

  if (trans == OP::NoTrans) {
    if (A.row_count() * A.column_count() >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < A.row_count(); ++i) {
        auto row = A.row_span(i);
        R sum = 0;
        for (size_t j = 0; j < A.column_count(); ++j) {
          sum += row[j] * x[j];
        }
        y[i] = sum;
      }
    } else {
      for (size_t i = 0; i < A.row_count(); ++i) {
        auto row = A.row_span(i);
        R sum = 0;
        for (size_t j = 0; j < A.column_count(); ++j) {
          sum += row[j] * x[j];
        }
        y[i] = sum;
      }
    }
  } else {
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
    return y;
  }
}

}  // namespace maf::math::kernels
#endif
