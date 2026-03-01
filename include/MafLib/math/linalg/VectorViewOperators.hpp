#ifndef VECTOR_VIEW_OPERATORS_H
#define VECTOR_VIEW_OPERATORS_H
#pragma once
#include "MafLib/utility/Math.hpp"
#include "ViewKernels.hpp"

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

  if constexpr (std::is_same_v<V, VectorView<V_type>> &&
                std::is_same_v<T, MatrixView<T_type>>) {
    return kernels::gemv(kernels::OP::Trans, A, x);
  } else if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::gemv(kernels::OP::Trans,
                         A.view(0, 0, A.row_count(), A.column_count()), x);
  } else if constexpr (std::is_same_v<T, MatrixView<T_type>>) {
    return kernels::gemv(kernels::OP::Trans, A, x.view(0, x.size()));
  } else {
    return kernels::gemv(kernels::OP::Trans,
                         A.view(0, 0, A.row_count(), A.column_count()),
                         x.view(0, x.size()));
  }
}

/**
 * @brief Computes the dot product of two vectors.
 * @tparam V Numeric type of vector x.
 * @tparam T Numeric type of vector y.
 *
 * @param x The first input vector.
 * @param y The second input vector.
 * @return The resulting scalar which is the dot product of x and y.
 * @throws std::invalid_argument if dimensions do not match or if orientations
 * are not compatible for dot product (row x column).
 */
template <VectorViewCompatible V, VectorViewCompatible T>
auto operator*(const V &x, const T &y) {
  size_t n = x.size();
  size_t m = y.size();
  if (n != m) {
    throw std::invalid_argument("Dimensions do not match!");
  }

  using T_type = typename T::value_type;
  using V_type = typename V::value_type;
  if (x.orientation() == y.orientation() || x.orientation() == COLUMN) {
    throw std::invalid_argument(
        "Invalid multiplication: Vectors must be of different orientations (row x "
        "column) for dot product. If you are certain that you want the dot "
        "product, use the dot_product() method which does not require specific "
        "orientations.");
  }

  if constexpr (std::is_same_v<V, VectorView<V_type>> &&
                std::is_same_v<T, VectorView<T_type>>) {
    return kernels::dot(x, y);
  } else if constexpr (std::is_same_v<V, VectorView<V_type>>) {
    return kernels::dot(x, y.view(0, y.size()));
  } else if constexpr (std::is_same_v<T, VectorView<T_type>>) {
    return kernels::dot(x.view(0, x.size()), y);
  } else {
    return kernels::dot(x.view(0, x.size()), y.view(0, y.size()));
  }
}

/**
 * @brief Computes the outer product of two vectors.
 * @tparam V Numeric type of vector x.
 * @tparam T Numeric type of vector y.
 *
 * @param x The first input vector.
 * @param y The second input vector.
 * @return The resulting matrix which is the outer product of x and y.
 */
template <VectorViewCompatible V, VectorViewCompatible T>
auto outer_product(const V &x, const T &y) {
  using T_type = typename T::value_type;
  using V_type = typename V::value_type;
  using R = std::common_type_t<T_type, V_type>;

  size_t n = x.size();
  size_t m = y.size();

  if (x.orientation() == y.orientation()) {
    // 1x1 x 1x1 vector multiplication
    if (n == 1 && m == 1) {
      return Matrix<R>(1, 1, {x[0] * y[0]});
    }
    throw std::invalid_argument("Vector orientations do not match for outer product!");
  }

  switch (x.orientation()) {
    case COLUMN: {
      if (std::is_same_v<V, VectorView<V_type>> &&
          std::is_same_v<T, VectorView<T_type>>) {
        return kernels::outer(x, y);
      }
      if (std::is_same_v<V, VectorView<V_type>>) {
        return kernels::outer(x, y.view(0, y.size()));
      }
      if (std::is_same_v<T, VectorView<T_type>>) {
        return kernels::outer(x.view(0, x.size()), y);
      }
      return kernels::outer(x.view(0, x.size()), y.view(0, y.size()));
    }
    default:
      if (n != m) {
        throw std::invalid_argument(
            "Resulting operation is a vector dot product, but vector dimensions do "
            "not "
            "match!");
      }
      std::cout << "This results in a 1x1 matrix. Consider using vector dot "
                   "product."
                << std::endl;
      if (std::is_same_v<V, VectorView<V_type>> &&
          std::is_same_v<T, VectorView<T_type>>) {
        return Matrix<R>(1, 1, {kernels::dot(x, y)});
      }
      if (std::is_same_v<V, VectorView<V_type>>) {
        return Matrix<R>(1, 1, {kernels::dot(x, y.view(0, y.size()))});
      }
      if (std::is_same_v<T, VectorView<T_type>>) {
        return Matrix<R>(1, 1, {kernels::dot(x.view(0, x.size()), y)});
      }
      return Matrix<R>(1, 1, {kernels::dot(x.view(0, x.size()), y.view(0, y.size()))});
  }
}

}  // namespace maf::math
#endif
