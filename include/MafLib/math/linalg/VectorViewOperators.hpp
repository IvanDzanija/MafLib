#ifndef VECTOR_VIEW_OPERATORS_H
#define VECTOR_VIEW_OPERATORS_H
#pragma once
#include "Kernels.hpp"

namespace maf::math {
/*
 * @brief Vector-Matrix multiplication operator.
 *
 * This operator overload allows for the multiplication of a Vector-like
 * object with a Matrix-like object. It supports both VectorView and
 * Vector types for the Vector operand, and MatrixView and Matrix types
 * for the Matrix operand.
 *
 * The multiplication is performed using the GEMV kernel, which handles
 * the actual computation efficiently, including any necessary type
 * conversions.
 * @attention This operator uses fast BLAS implementation if both types
 * are floating point type and are the same, otherwise uses naive algorithm.
 * It is usually worth casting a buffer for BLAS performance gains before hand.
 *
 * @tparam V The type of the Vector-like object (VectorView or Vector).
 * @tparam T The type of the Matrix-like object (MatrixView or Matrix).
 * @param x The Vector-like operand.
 * @param A The Matrix-like operand.
 * @return A new Vector resulting from the multiplication x * A.
 */
template <VectorViewCompatible V, MatrixViewCompatible T>
auto operator*(V &x, T &A) {
  if (A.row_count() != x.size() || x.orientation() != ROW) {
    throw std::invalid_argument(
        "Inner dimensions do not match for Vector-Matrix multiplication!");
  }

  using T_type = typename T::value_type;
  using V_type = typename V::value_type;

  MatrixView<T_type> A_view;
  if constexpr (std::is_same_v<T, MatrixView<T_type>>) {
    A_view = A;
  } else {
    A_view = A.view(0, 0, A.row_count(), A.column_count());
  }

  // GEMV does the conversion of types internally
  if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::gemv(kernels::OP::Trans, A_view, x);
  } else {
    return kernels::gemv(kernels::OP::Trans, A_view, x.view(0, x.size()));
  }
}
}  // namespace maf::math
#endif
