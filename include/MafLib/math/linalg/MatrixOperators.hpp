#ifndef MATRIX_OPERATORS_H
#define MATRIX_OPERATORS_H

#pragma once
#include "MafLib/utility/Conversions.hpp"
#include "Matrix.hpp"

namespace maf::math {
// Checks if elements are exactly equal
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::operator==(const Matrix &other) const noexcept {
  if (_rows != other._rows || _cols != other._cols) {
    return false;
  }
  return _data == other._data;
}

// Unary minus sign, creates a copy
template <Numeric T>
[[nodiscard]] auto Matrix<T>::operator-() const noexcept {
  Matrix<T> result(*this);
  result._invert_sign();
  return result;
}

// Add 2 matrices element-wise
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator+(const Matrix<U> &other) const {
  if (_rows != other.row_count() || _cols != other.column_count()) {
    throw std::invalid_argument("Matrices have to be of same dimensions for addition!");
  }
  using R = std::common_type_t<T, U>;

  Matrix<R> result(_rows, _cols);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) + static_cast<R>(other.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) + static_cast<R>(other.data()[i]);
    }
  }
  return result;
}

// Add a scalar to each element of matrix
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator+(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  Matrix<R> result(_rows, _cols);
  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) + r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) + r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar addition (scalar + Matrix).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator+(const U &scalar, const Matrix<T> &matrix) noexcept {
  return matrix + scalar;
}

// Add 2 matrices element-wise
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator+=(const Matrix<U> &other) {
  if (_rows != other.row_count() || _cols != other.column_count()) {
    throw std::invalid_argument("Matrices have to be of same dimensions for addition!");
  }

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] += static_cast<T>(other.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] += static_cast<T>(other.data()[i]);
    }
  }

  return *this;
}

// Add a scalar to each element of matrix
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator+=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] += r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] += r_scalar;
    }
  }
  return *this;
}

// Subtract 2 matrices element-wise
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator-(const Matrix<U> &other) const {
  if (_rows != other.row_count() || _cols != other.column_count()) {
    throw std::invalid_argument(
        "Matrices have to be of same dimensions for subtraction!");
  }
  using R = std::common_type_t<T, U>;

  Matrix<R> result(_rows, _cols);
  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) - static_cast<R>(other.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) - static_cast<R>(other.data()[i]);
    }
  }
  return result;
}

// Subtract a scalar from each element of matrix
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator-(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  Matrix<R> result(_rows, _cols);
  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) - r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) - r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar subtraction (scalar - Matrix).
 * @tparam U An arithmetic scalar type.
 * @return Matrix of the common, promoted type.
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator-(const U &scalar, const Matrix<T> &matrix) {
  using R = std::common_type_t<T, U>;

  Matrix<R> result(matrix.row_count(), matrix.column_count());
  R r_scalar = static_cast<R>(scalar);

  if (matrix.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < matrix.size(); ++i) {
      result.data()[i] = r_scalar - static_cast<R>(matrix.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < matrix.size(); ++i) {
      result.data()[i] = r_scalar - static_cast<R>(matrix.data()[i]);
    }
  }
  return result;
}

// Subtract 2 matrices element-wise
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator-=(const Matrix<U> &other) {
  if (_rows != other.row_count() || _cols != other.column_count()) {
    throw std::invalid_argument("Matrices have to be of same dimensions for addition!");
  }

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] -= static_cast<T>(other.data()[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] -= static_cast<T>(other.data()[i]);
    }
  }

  return *this;
}

// Subtract a scalar from each element of matrix
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator-=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] -= r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] -= r_scalar;
    }
  }
  return *this;
}

// Multiply each element of matrix by a scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator*(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  Matrix<R> result(_rows, _cols);
  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) * r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) * r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar multiplication (scalar * Matrix).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator*(const U &scalar, const Matrix<T> &matrix) noexcept {
  return matrix * scalar;
}

// Multiply each element of matrix by a scalar
template <Numeric T>
template <Numeric U>
Matrix<T> &Matrix<T>::operator*=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] *= r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] *= r_scalar;
    }
  }
  return *this;
}

// Matrix-Vector multiplication (Matrix * column_vector)
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator*(const Vector<U> &other) const {
  using R = std::common_type_t<T, U>;

  if (other.orientation() == Orientation::ROW) {
    throw std::invalid_argument(
        "Invalid multiplication: matrix * row vector.\n"
        "Did you mean Vector * Matrix?");
  }

  if (other.size() != _cols) {
    throw std::invalid_argument(
        "Dimension mismatch in Matrix * Vector multiplication.");
  }

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
  if constexpr (std::is_same_v<R, float>) {
    return Vector<R>(_rows,
                     acc::sgemvm(_rows, _cols, util::convert_if_needed<R>(_data),
                                 util::convert_if_needed<R>(other.data())),
                     COLUMN);

  } else if constexpr (std::is_same_v<R, double>) {
    return Vector<R>(_rows,
                     acc::dgemvm(_rows, _cols, util::convert_if_needed<R>(_data),
                                 util::convert_if_needed<R>(other.data())),
                     COLUMN);
  }
#endif

  Vector<R> result(_rows, std::vector<R>(_rows, R(0)), COLUMN);
  if (_rows * _cols >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _rows; ++i) {
      auto L_row_i = row_span(i);
#pragma omp simd
      for (size_t j = 0; j < _cols; ++j) {
        result[i] += static_cast<R>(L_row_i[j]) * static_cast<R>(other[j]);
      }
    }
  } else {
    for (size_t i = 0; i < _rows; ++i) {
      auto L_row_i = row_span(i);
#pragma omp simd
      for (size_t j = 0; j < _cols; ++j) {
        result[i] += static_cast<R>(L_row_i[j]) * static_cast<R>(other[j]);
      }
    }
  }
  return result;
}

// Divide each element of matrix by a scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Matrix<T>::operator/(const U &scalar) const noexcept {
  using R =
      std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                         std::common_type_t<T, U>>;  // Forces double if both are ints

  Matrix<R> result(_rows, _cols);
  R r_scalar_inv = R(1) / static_cast<R>(scalar);

  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) * r_scalar_inv;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < _data.size(); ++i) {
      result.data()[i] = static_cast<R>(_data[i]) * r_scalar_inv;
    }
  }
  return result;
}

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
