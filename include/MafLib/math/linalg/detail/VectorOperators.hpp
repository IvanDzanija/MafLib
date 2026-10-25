#ifndef VECTOR_OPERATORS_H
#define VECTOR_OPERATORS_H
#pragma once
#include "MafLib/utility/Conversions.hpp"
#include "MafLib/math/linalg/detail/Vector.hpp"
#include "MafLib/math/linalg/detail/Matrix.hpp"

namespace maf::math {

// Equality operator
template <Numeric T>
[[nodiscard]] constexpr bool Vector<T>::operator==(const Vector &other) const noexcept {
  if (_orientation != other._orientation || size() != other.size()) {
    return false;
  }
  // maybe loop over data and actually compare so different types give actually the
  // same result
  return _data == other._data;
}

// Unary minus sign, creates a copy
template <Numeric T>
[[nodiscard]] auto Vector<T>::operator-() const noexcept {
  Vector<T> result(*this);
  result._invert_sign();
  return result;
}

// Vector + Vector
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator+(const Vector<U> &other) const {
  using R = std::common_type_t<T, U>;

  if (_orientation != other.orientation() || _data.size() != other.size()) {
    throw std::invalid_argument("Vectors must be same orientation and size!");
  }

  size_t n = _data.size();
  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) + static_cast<R>(other[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) + static_cast<R>(other[i]);
    }
  }
  return result;
}

// Vector + Scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator+(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) + r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) + r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar addition (scalar + Vector).
 * @tparam U An arithmetic scalar type.
 * @param scalar The scalar value.
 * @param vec The vector.
 * @return A new Vector of the common, promoted type.
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator+(const U &scalar, const Vector<T> &vec) noexcept {
  return vec + scalar;
}

// Vector + Vector
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator+=(const Vector<U> &other) {
  if (_orientation != other.orientation() || _data.size() != other.size()) {
    throw std::invalid_argument("Vectors must be same orientation and size!");
  }

  size_t n = _data.size();
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      _data[i] += static_cast<T>(other[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      _data[i] += static_cast<T>(other[i]);
    }
  }
  return *this;
}

// Vector + Scalar
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator+=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      _data[i] += r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      _data[i] += r_scalar;
    }
  }
  return *this;
}

// Vector - Vector
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator-(const Vector<U> &other) const {
  using R = std::common_type_t<T, U>;

  if (_orientation != other.orientation() || _data.size() != other.size()) {
    throw std::invalid_argument("Vectors must be same orientation and size!");
  }

  size_t n = _data.size();
  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) - static_cast<R>(other[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) - static_cast<R>(other[i]);
    }
  }
  return result;
}

// Vector - Scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator-(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) - r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) - r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar subtraction (scalar - Vector).
 * @tparam U An arithmetic scalar type.
 * @param scalar The scalar value from which elements are subtracted.
 * @param vec The vector.
 * @return A new Vector of the common, promoted type.
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator-(const U &scalar, const Vector<T> &vec) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = vec.size();

  Vector<R> result(n, vec.orientation());
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = r_scalar - static_cast<R>(vec[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = r_scalar - static_cast<R>(vec[i]);
    }
  }
  return result;
}

// Vector - Vector
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator-=(const Vector<U> &other) {
  if (_orientation != other.orientation() || _data.size() != other.data().size()) {
    throw std::invalid_argument("Vectors must be same orientation and size!");
  }

  size_t n = _data.size();
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      _data[i] -= static_cast<T>(other[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      _data[i] -= static_cast<T>(other[i]);
    }
  }
  return *this;
}

// Vector - Scalar
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator-=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      _data[i] -= r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      _data[i] -= r_scalar;
    }
  }
  return *this;
}

// Vector * Scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator*(const U &scalar) const noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) * r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) * r_scalar;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar multiplication (scalar * Vector).
 * @tparam U An arithmetic scalar type.
 * @param scalar The scalar value.
 * @param vec The vector.
 * @return A new Vector of the common, promoted type.
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator*(const U &scalar, const Vector<T> &vec) noexcept {
  return vec * scalar;
}

// Vector * Scalar
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator*=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  R r_scalar = static_cast<R>(scalar);
  size_t n = _data.size();

  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      _data[i] *= r_scalar;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      _data[i] *= r_scalar;
    }
  }
  return *this;
}

// Vector * Vector -> Scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::dot_product(const Vector<U> &other) const {
  size_t n = _data.size();
  if (n != other.size()) {
    throw std::invalid_argument("Vectors must be of same size!");
  }

#ifdef BLAS_AVAILABLE
  if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>) {
    return blas::ddot(_data.size(), _data.data(), 1, other.data(), 1);
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, float>) {
    return blas::sdot(_data.size(), _data.data(), 1, other.data(), 1);
  }
#endif

  using R = std::common_type_t<T, U>;

  R result(0);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < n; ++i) {
      result += static_cast<R>(_data[i]) * static_cast<R>(other[i]);
    }
  } else {
#pragma omp simd reduction(+ : result)
    for (size_t i = 0; i < n; ++i) {
      result += static_cast<R>(_data[i]) * static_cast<R>(other[i]);
    }
  }
  return result;
}

// Vector * Vector -> Scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator*(const Vector<U> &other) const {
  if (_orientation == other.orientation() || _orientation == COLUMN) {
    throw std::invalid_argument(
        "Invalid multiplication: Vectors must be of different orientations (row x "
        "column) for dot product. If you are certain that you want the dot "
        "product, use the dot_product() method which does not require specific "
        "orientations.");
  }
  return dot_product(other);
}

// Vector * Vector -> Matrix
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::outer_product(const Vector<U> &other) const {
  using R = std::common_type_t<T, U>;

  size_t n = _data.size();
  size_t m = other.size();

  if (_orientation == other.orientation()) {
    // 1x1 x 1x1 vector multiplication
    if (n == 1 && m == 1) {
      return Matrix<R>(1, 1, {_data[0] * other[0]});
    }
    throw std::invalid_argument("Vector dimensions do not match!");
  }
  switch (_orientation) {
    case COLUMN: {
      Matrix<R> result(n, m);
      if (n * m > OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
          for (size_t j = 0; j < m; ++j) {
            result.data()[(i * m) + j] =
                static_cast<R>(_data[i]) * static_cast<R>(other[j]);
          }
        }
      } else {
        for (size_t i = 0; i < n; ++i) {
#pragma omp simd
          for (size_t j = 0; j < m; ++j) {
            result.data()[(i * m) + j] =
                static_cast<R>(_data[i]) * static_cast<R>(other[j]);
          }
        }
      }

      return result;
    }
    default:
      if (n != m) {
        throw std::invalid_argument("Vector dimensions do not match!");
      }

      std::cout << "This results in a 1x1 matrix. Consider using vector dot "
                   "product."
                << std::endl;

      return Matrix<R>(1, 1, {dot_product(other)});
  }
}

// Vector * Matrix -> Vector
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator*(const Matrix<U> &other) const {
  using R = std::common_type_t<T, U>;

  size_t n = size();
  size_t m = other.row_count();
  size_t r = other.column_count();

  if (n != m) {
    throw std::invalid_argument("Dimensions do not match!");
  }
  if (_orientation == COLUMN) {
    throw std::invalid_argument(
        "Invalid multiplication: column Vector * Matrix. "
        "Did you mean Matrix * Vector?");
  }

#ifdef BLAS_AVAILABLE
  if constexpr (std::is_same_v<R, float>) {
    std::vector<float> m_converted, v_converted;
    const float *m_ptr = util::to_pointer(m_converted, other.data_vector());
    const float *v_ptr = util::to_pointer(v_converted, _data);
    Vector<float> result(r, ROW);
    blas::sgevmm(m, r, m_ptr, v_ptr, 1, result.data(), r);
    return result;
  } else if constexpr (std::is_same_v<R, double>) {
    std::vector<double> m_converted, v_converted;
    const double *m_ptr = util::to_pointer(m_converted, other.data_vector());
    const double *v_ptr = util::to_pointer(v_converted, _data);
    Vector<double> result(r, ROW);
    blas::dgevmm(m, r, m_ptr, v_ptr, 1, result.data(), r);
    return result;
  }
#endif

  Vector<R> result(r, std::vector<R>(r), ROW);
  if (n * m >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result[i] += _data[j] * other.at(j, i);
      }
    }
  } else {
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < n; ++j) {
        result[i] += _data[j] * other.at(j, i);
      }
    }
  }
  return result;
}

// Vector / scalar
template <Numeric T>
template <Numeric U>
[[nodiscard]] auto Vector<T>::operator/(const U &scalar) const noexcept {
  using R =
      std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                         std::common_type_t<T, U>>;  // Forces double if both are ints

  R r_scalar_inv = R(1) / static_cast<R>(scalar);
  size_t n = _data.size();

  Vector<R> result(n, _orientation);
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) * r_scalar_inv;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<R>(_data[i]) * r_scalar_inv;
    }
  }
  return result;
}

/**
 * @brief Element-wise scalar division (scalar / Vector).
 * @tparam U An arithmetic scalar type.
 * @param scalar The scalar value.
 * @param vec The vector.
 * @return A new Vector of the common, promoted type (forced double if both are int).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator/(const U &scalar, const Vector<T> &vec) noexcept {
  using R =
      std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                         std::common_type_t<T, U>>;  // Forces double if both are ints

  R r_scalar = static_cast<R>(scalar);
  size_t n = vec.size();

  Vector<R> result(n, vec.orientation());
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      result[i] = r_scalar / static_cast<R>(vec[i]);
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
      result[i] = r_scalar / static_cast<R>(vec[i]);
    }
  }
  return result;
}

// Vector /= scalar
template <Numeric T>
template <Numeric U>
auto Vector<T>::operator/=(const U &scalar) noexcept {
  using R = std::common_type_t<T, U>;

  size_t n = _data.size();
  if constexpr (std::is_floating_point_v<R>) {
    R r_scalar_inv = R(1) / static_cast<R>(scalar);

    if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        _data[i] *= r_scalar_inv;
      }
    } else {
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        _data[i] *= r_scalar_inv;
      }
    }
  } else {
    R r_scalar = static_cast<R>(scalar);

    if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        _data[i] /= r_scalar;
      }
    } else {
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        _data[i] /= r_scalar;
      }
    }
  }
  return *this;
}

}  // namespace maf::math

#endif
