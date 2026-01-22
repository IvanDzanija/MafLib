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
 *
 * @tparam T The type of the Matrix-like object (MatrixView or Matrix).
 * @tparam V The type of the Vector-like object (VectorView or Vector).
 * @param A The Matrix-like operand.
 * @param x The Vector-like operand.
 * @return A new Vector resulting from the multiplication A * x.
 */
template <MatrixViewCompatible T, VectorViewCompatible V>
auto operator*(const T &A, const V &x) {
  using T_type = typename T::value_type;
  using V_type = typename V::value_type;

  MatrixView<T_type> A_view;
  if constexpr (std::is_same_v<T, MatrixView<T_type>>) {
    A_view = A;
  } else {
    A_view = A.view();
  }

  // GEMV does the conversion of types internally
  if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::gemv(kernels::OP::NoTrans, A_view, x);
  } else {
    return kernels::gemv(kernels::OP::NoTrans, A_view, x.view());
  }
}

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
 *
 * @tparam V The type of the Vector-like object (VectorView or Vector).
 * @tparam T The type of the Matrix-like object (MatrixView or Matrix).
 * @param x The Vector-like operand.
 * @param A The Matrix-like operand.
 * @return A new Vector resulting from the multiplication x * A.
 */
template <VectorViewCompatible V, MatrixViewCompatible T>
auto operator*(const V &x, const T &A) {
  using T_type = typename T::value_type;
  using V_type = typename V::value_type;
  MatrixView<T_type> A_view;
  if constexpr (std::is_same_v<T, MatrixView<T_type>>) {
    A_view = A;
  } else {
    A_view = A.view();
  }

  // GEMV does the conversion of types internally
  if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::gemv(kernels::OP::Trans, A_view, x);
  } else {
    return kernels::gemv(kernels::OP::Trans, A_view, x.view());
  }
}
}  // namespace maf::math
#endif
