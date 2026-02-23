#ifndef MATRIX_VIEW_OPERATORS_H
#define MATRIX_VIEW_OPERATORS_H
#pragma once
#include "Kernels.hpp"

namespace maf::math {
/*
 * @brief Matrix-Vector multiplication operator.
 *
 * This operator overload allows for the multiplication of a Matrix-like
 * object with a Vector-like object. It supports both MatrixView and
 * Matrix types for the Matrix operand, and VectorView and Vector types
 * for the Vector operand.
 *
 * The multiplication is performed using the GEMV kernel, which handles
 * the actual computation efficiently, including any necessary type
 * conversions.
 * @attention This operator uses fast BLAS implementation if both types
 * are floating point type and are the same, otherwise uses naive algorithm.
 * It is usually worth casting a buffer for BLAS performance gains before hand.
 *
 * @tparam T The type of the Matrix-like object (MatrixView or Matrix).
 * @tparam V The type of the Vector-like object (VectorView or Vector).
 * @param A The Matrix-like operand.
 * @param x The Vector-like operand.
 * @return A new Vector resulting from the multiplication A * x.
 */
template <MatrixViewCompatible T, VectorViewCompatible V>
auto operator*(const T &A, const V &x) {
  if (A.column_count() != x.size() || x.orientation() != COLUMN) {
    throw std::invalid_argument(
        "Inner dimensions do not match for Matrix-Vector multiplication!");
  }

  using T_type = typename T::value_type;
  using V_type = typename V::value_type;

  MatrixView<T_type> A_view;
  if constexpr (std::is_same_v<T, MatrixView<T_type>>) {
    A_view = A;
  } else {
    A_view = A.view(0, 0, A.row_count(), A.column_count());
  }

  // GEMV does the conversion of types internally(naively)
  if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::gemv(kernels::OP::NoTrans, A_view, x);
  } else {
    return kernels::gemv(kernels::OP::NoTrans, A_view, x.view(0, x.size()));
  }
}

}  // namespace maf::math
#endif
