#ifndef MATRIX_OPERATORS_H
#define MATRIX_OPERATORS_H

#pragma once
#include "MafLib/math/linalg/detail/Matrix.hpp"
#include "MafLib/math/linalg/detail/Vector.hpp"
#include "MafLib/utility/Conversions.hpp"

namespace maf::math {

/**
 * @brief Element-wise scalar division (scalar / Matrix).
 * @tparam U An arithmetic scalar type.
 * @return Matrix of the common, promoted type.
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator/(const U &scalar, const Matrix<T> &matrix) noexcept {
  using R =
      std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                         std::common_type_t<T, U>>;  // Forces double if both are ints

  Matrix<R> result(matrix.row_count(), matrix.column_count());
  R r_scalar = static_cast<R>(scalar);

  if (matrix.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < matrix.size(); ++i) {
      result.data()[i] = r_scalar / static_cast<R>(matrix.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < matrix.size(); ++i) {
      result.data()[i] = r_scalar / static_cast<R>(matrix.data()[i]);
    }
  }
  return result;
}

// Divide each element of matrix by a scalar
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator/=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  if constexpr (std::is_floating_point_v<R>) {
    R r_scalar_inv = R(1) / static_cast<R>(scalar);

    if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] *= r_scalar_inv;
      }
    } else {
#pragma omp simd
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] *= r_scalar_inv;
      }
    }
  } else {
    R r_scalar = static_cast<R>(scalar);

    if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] /= r_scalar;
      }
    } else {
#pragma omp simd
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] /= r_scalar;
      }
    }
  }
  return *this;
}

}  // namespace maf::math

#endif
