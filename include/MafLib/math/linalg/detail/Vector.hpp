#ifndef VECTOR_H
#define VECTOR_H
#pragma once
#include "MafLib/math/linalg/detail/BlasWrappers/BlasWrapper.hpp"
#include "MafLib/math/linalg/detail/Core.hpp"
#include "MafLib/math/linalg/detail/VectorView.hpp"
#include "MafLib/utility/Conversions.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {
using namespace maf::util;
/**
 * @brief A general-purpose mathematical vector class.
 *
 * This class implements a mathematical vector, wrapping a std::vector
 * for data storage. It supports both **ROW** and **COLUMN** orientations,
 * which is crucial for correct algebraic operations with the Matrix class.
 *
 * It is templated to support various numeric types (T) and provides
 * a range of constructors, element-wise operators, and common
 * vector operations like norm, normalization, and dot product.
 *
 * @tparam T The numeric type of the vector elements (e.g., float,
 * double, int).
 *
 * @version 1.0
 * @since 2025
 */
template <Numeric T>
class Vector {
 public:
  /** @brief The numeric type of the vector's elements. */
  using value_type = T;

  // --- Constructors ---

  /**
   * @brief Default constructor.
   * @details Creates an empty vector with default COLUMN orientation.
   */
  Vector() : _orientation(Orientation::COLUMN) {}

  /**
   * @brief Constructs an uninitialized vector of a given size.
   * @param size The number of elements in the vector.
   * @param orientation The vector's orientation (default: COLUMN).
   * @throws std::invalid_argument if size is zero.
   */
  Vector(size_t size, Orientation orientation = Orientation::COLUMN)
      : _orientation(orientation) {
    if (size == 0) {
      throw std::invalid_argument("Vector size must be greater than zero.");
    }
    _data.resize(size);
  }

  /**
   * @brief Constructs from a raw data pointer.
   * @details Elements are COPIED from the data pointer.
   * @param size The number of elements in the vector.
   * @param data Pointer to a contiguous memory.
   * @param orientation The vector's orientation (default: COLUMN).
   * @throws std::invalid_argument if size is zero or data is nullptr.
   */
  template <Numeric U>
  Vector(size_t size, const U *data, Orientation orientation = Orientation::COLUMN)
      : _orientation(orientation) {
    if (size == 0) {
      throw std::invalid_argument("Vector size must be greater than zero!");
    }
    if (data == nullptr) {
      throw std::invalid_argument("Data pointer cannot be null!");
    }

    _data.assign(data, data + size);
  }

  /**
   * @brief Constructs from a std::vector by copying its data.
   * @param size The number of elements. Must match data.size().
   * @param data The std::vector to copy from.
   * @param orientation The vector's orientation (default: COLUMN).
   * @throws std::invalid_argument if size is zero or data size
   * mismatches.
   */
  template <Numeric U>
  Vector(size_t size, const std::vector<U> &data,
         Orientation orientation = Orientation::COLUMN)
      : _orientation(orientation) {
    if (size == 0) {
      throw std::invalid_argument("Vector size must be greater than zero.");
    }

    if (data.size() != size) {
      throw std::invalid_argument("Data size does not match vector size.");
    }

    _data = data;
  }

  /**
   * @brief Constructs from a std::vector by moving its data.
   * @param size The number of elements. Must match data.size().
   * @param data The std::vector to move from (r-value).
   * @param orientation The vector's orientation (default: COLUMN).
   * @throws std::invalid_argument if size is zero or data size
   * mismatches.
   */
  Vector(size_t size, std::vector<T> &&data,
         Orientation orientation = Orientation::COLUMN)
      : _orientation(orientation) {
    if (size == 0) {
      throw std::invalid_argument("Vector size must be greater than zero.");
    }

    if (data.size() != size) {
      throw std::invalid_argument("Data size does not match vector size.");
    }

    _data = std::move(data);
  }

  /**
   * @brief Constructs from a std::array by copying its data.
   * @tparam U Type in the array (allows implicit conversion).
   * @tparam N Size of the array.
   * @param size The number of elements. Must match N.
   * @param data The std::array to copy from.
   * @param orientation The vector's orientation (default: COLUMN).
   * @throws std::invalid_argument if size is zero or array size
   * mismatches.
   */
  template <Numeric U, size_t N>
  Vector(size_t size, const std::array<U, N> &data,
         Orientation orientation = Orientation::COLUMN)
      : _orientation(orientation) {
    if (size == 0) {
      throw std::invalid_argument("Vector size must be greater than zero.");
    }
    if (N != size) {
      throw std::invalid_argument("Data size does not match vector size.");
    }

    _data.assign(data.begin(), data.end());
  }

  /**
   * @brief Converting constructor.
   */
  template <Numeric U>
  [[nodiscard]] Vector(const Vector<U> &other)
      : _orientation(other.orientation()),
        _data(other.data(), other.data() + other.size()) {}

  // --- Iterators ---
  /** @brief Returns an iterator to the beginning. */
  [[nodiscard]] auto begin() noexcept { return _data.begin(); }
  /** @brief Returns an iterator to the end. */
  [[nodiscard]] auto end() noexcept { return _data.end(); }

  /** @brief Returns a const iterator to the beginning. */
  [[nodiscard]] auto begin() const noexcept { return _data.begin(); }
  /** @brief Returns a const iterator to the end. */
  [[nodiscard]] auto end() const noexcept { return _data.end(); }

  /** @brief Returns a const iterator to the beginning. */
  [[nodiscard]] auto cbegin() const noexcept { return _data.cbegin(); }
  /** @brief Returns a const iterator to the end. */
  [[nodiscard]] auto cend() const noexcept { return _data.cend(); }

  /** @brief Returns a reverse iterator to the beginning. */
  [[nodiscard]] auto rbegin() noexcept { return _data.rbegin(); }
  /** @brief Returns a reverse iterator to the end. */
  [[nodiscard]] auto rend() noexcept { return _data.rend(); }

  /** @brief Returns a const reverse iterator to the beginning. */
  [[nodiscard]] auto rbegin() const noexcept { return _data.rbegin(); }
  /** @brief Returns a const reverse iterator to the end. */
  [[nodiscard]] auto rend() const noexcept { return _data.rend(); }

  /** @brief Returns a const reverse iterator to the beginning. */
  [[nodiscard]] auto crbegin() const noexcept { return _data.crbegin(); }
  /** @brief Returns a const reverse iterator to the end. */
  [[nodiscard]] auto crend() const noexcept { return _data.crend(); }

  // --- Getters ---

  /**
   * @brief Gets a const pointer to the underlying data.
   * store.
   * @return const T*
   */
  [[nodiscard]] const T *data() const noexcept { return _data.data(); }

  /**
   * @brief Gets a mutable pointer to the underlying data.
   * @return T*
   */
  [[nodiscard]] T *data() noexcept { return _data.data(); }

  /**
   * @brief Gets a const reference to the underlying std::vector data
   * store.
   * @return const std::vector<T>&
   */
  [[nodiscard]] const std::vector<T> &data_vector() const noexcept { return _data; }

  /**
   * @brief Gets a mutable reference to the underlying std::vector
   * data store.
   * @return std::vector<T>&
   */
  [[nodiscard]] std::vector<T> &data_vector() noexcept { return _data; }

  /** @brief Gets the number of elements in the vector. */
  [[nodiscard]] size_t size() const noexcept { return _data.size(); }

  /** @brief Gets the vector's orientation (ROW or COLUMN). */
  [[nodiscard]] Orientation orientation() const noexcept { return _orientation; }

  /**
   * @brief Gets a mutable reference to the element at a specific index with
   * bounds check.
   * @throws std::out_of_range if the index is invalid.
   */
  T &at(size_t index) { return _data.at(index); }

  /**
   * @brief Gets a const reference to the element at a specific index with
   * bounds check.
   * @throws std::out_of_range if the index is invalid.
   */
  const T &at(size_t index) const { return _data.at(index); }

  /**
   * @brief Accesses the element at a specific index with no bounds check.
   * @throws std::out_of_range if the index is invalid.
   */
  T &operator[](size_t index) { return _data[index]; }

  /**
   * @brief Accesses the element at a specific index with no bounds check.
   * @throws std::out_of_range if the index is invalid.
   */
  const T &operator[](size_t index) const { return _data[index]; }

  /**
   * @brief Creates a view (sub-Vector) into this Vector.
   * @param col Starting index of the view.
   * @param length Length of the view.
   * @return VectorView<T> representing the specified sub-Vector.
   * @throws std::invalid_argument if length is zero.
   * @throws std::out_of_range if the requested view exceeds Vector
   * dimensions.
   */
  [[nodiscard]] VectorView<T> view(size_t index, size_t length, size_t inc = 1) {
    if (length == 0) {
      throw std::invalid_argument("View dimensions must be greater than zero.");
    }
    if (index + length > size()) {
      throw std::out_of_range("Requested view exceeds Vector dimensions.");
    }
    return VectorView<T>(&_data[index], length, _orientation, inc);
  }

  /**
   * @brief Creates a const view (sub-Vector) into this Vector.
   * @param col Starting index of the view.
   * @param length Length of the view.
   * @return VectorView<T> representing the specified sub-Vector.
   * @throws std::invalid_argument if length is zero.
   * @throws std::out_of_range if the requested view exceeds Vector
   * dimensions.
   */
  [[nodiscard]] VectorView<const T> view(size_t index, size_t length,
                                         size_t inc = 1) const {
    if (length == 0) {
      throw std::invalid_argument("View dimensions must be greater than zero.");
    }
    if (index + length > size()) {
      throw std::out_of_range("Requested view exceeds Vector dimensions.");
    }
    return VectorView<const T>(&_data[index], length, _orientation, inc);
  }

  // --- Checkers ---

  /**
   * @brief Checks if the vector is a null vector.
   * @return true if all elements are close to zero, false otherwise.
   */
  [[nodiscard]] bool is_null() const noexcept {
    for (const T &val : _data) {
      if (!is_close(val, 0)) {
        return false;
      }
    }
    return true;
  }

  // --- Methods ---

  /** @brief Fills the entire vector with a single value. */
  void fill(T value) noexcept {
    omp_loop(_data.size(), [&](size_t i) { _data[i] = value; });
  }

  /** @brief Calculates the L2 norm (Euclidean length) of the vector. */
  [[nodiscard]] T norm() const noexcept {
    T sum = T(0);

    omp_loop_reduce(_data.size(), sum, [&](size_t i) { sum += _data[i] * _data[i]; });

    return std::sqrt(sum);
  }

  /** @brief Normalizes the vector in-place (divides by its L2 norm). */
  void normalize() {
    T norm = this->norm();
    if (is_close(norm, 0)) {
      throw std::invalid_argument("Vector norm is close to 0!");
    }

    T norm_inv = T(1) / norm;

    omp_loop(_data.size(), [&](size_t i) { _data[i] *= norm_inv; });
  }

  /** @brief Transposes the vector in-place (flips orientation). */
  void transpose() noexcept {
    _orientation =
        (_orientation == Orientation::COLUMN) ? Orientation::ROW : Orientation::COLUMN;
  }

  /** @brief Creates and returns a new vector that is the transpose of
   * this one. */
  [[nodiscard]] Vector<T> transposed() const noexcept {
    Orientation new_orientation =
        (_orientation == Orientation::COLUMN) ? Orientation::ROW : Orientation::COLUMN;
    return Vector<T>(this->size(), _data, new_orientation);
  }

  // --- Operators ---

  /**
   * @brief Checks for exact element-wise equality.
   * @details For floating-point, use `loosely_equal()`
   * @param other The vector to compare against.
   * @return true if size, orientation, and all elements are identical.
   */
  [[nodiscard]] constexpr bool operator==(const Vector &other) const noexcept {
    if (_orientation != other._orientation || size() != other.size()) {
      return false;
    }
    return _data == other._data;
  }

  /**
   * @brief Unary minus. Returns a new Vector with all elements negated.
   * @return `Vector<T>`
   */
  [[nodiscard]] auto operator-() const noexcept {
    Vector<T> result(*this);
    result._invert_sign();
    return result;
  }

  /**
   * @brief Element-wise vector addition (Vector + Vector).
   * @tparam U Numeric type of the other vector.
   * @param other The vector to add.
   * @return A new Vector of the common, promoted type.
   * @throws std::invalid_argument if dimensions or orientations do not
   * match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const Vector<U> &other) const {
    if (_orientation != other.orientation() || size() != other.size()) {
      throw std::invalid_argument("Vectors must be same orientation and size!");
    }
    using R = std::common_type_t<T, U>;

    const size_t n = size();
    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) {
      result[i] = static_cast<R>(_data[i]) + static_cast<R>(other[i]);
    });

    return result;
  }

  /**
   * @brief Element-wise scalar addition (Vector + scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to add to each element.
   * @return A new Vector of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);
    const size_t n = size();

    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) { result[i] = static_cast<R>(_data[i]) + r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise vector addition assignment (Vector + Vector).
   * @tparam U Numeric type of the other vector.
   * @param other The vector to add.
   * @attention This method doesn't cast the vector if U is a broader type.
   * @throws std::invalid_argument if dimensions or orientations do not
   * match.
   */
  template <Numeric U>
  auto operator+=(const Vector<U> &other) {
    if (_orientation != other.orientation() || size() != other.size()) {
      throw std::invalid_argument("Vectors must be same orientation and size!");
    }

    omp_loop(size(), [&](size_t i) { _data[i] += static_cast<T>(other[i]); });

    return *this;
  }

  /**
   * @brief Element-wise scalar addition assignment (Vector + scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to add to each element.
   * @attention This method doesn't cast the vector if U is a broader type.
   */
  template <Numeric U>
  auto operator+=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(size(), [&](size_t i) { _data[i] += r_scalar; });

    return *this;
  }

  /**
   * @brief Element-wise vector subtraction (Vector - Vector).
   * @tparam U Numeric type of the other vector.
   * @param other The vector to subtract.
   * @return A new Vector of the common, promoted type.
   * @throws std::invalid_argument if dimensions or orientations do not
   * match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const Vector<U> &other) const {
    if (_orientation != other.orientation() || size() != other.size()) {
      throw std::invalid_argument("Vectors must be same orientation and size!");
    }
    using R = std::common_type_t<T, U>;

    const size_t n = size();
    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) {
      result[i] = static_cast<R>(_data[i]) - static_cast<R>(other[i]);
    });

    return result;
  }

  /**
   * @brief Element-wise scalar subtraction (Vector - scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to subtract from each element.
   * @return A new Vector of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);
    const size_t n = size();

    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) { result[i] = static_cast<R>(_data[i]) - r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise vector subtraction assignment (Vector - Vector).
   * @tparam U Numeric type of the other vector.
   * @param other The vector to subtract.
   * @attention This method doesn't cast the vector if U is a broader type.
   * @throws std::invalid_argument if dimensions or orientations do not
   * match.
   */
  template <Numeric U>
  auto operator-=(const Vector<U> &other) {
    if (_orientation != other.orientation() || size() != other.size()) {
      throw std::invalid_argument("Vectors must be same orientation and size!");
    }

    omp_loop(size(), [&](size_t i) { _data[i] -= static_cast<T>(other[i]); });

    return *this;
  }

  /**
   * @brief Element-wise scalar subtraction assignment (Vector - scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to subtract from each element.
   * @attention This method doesn't cast the Vector if U is a broader type.
   */
  template <Numeric U>
  auto operator-=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(size(), [&](size_t i) { _data[i] -= r_scalar; });

    return *this;
  }

  /**
   * @brief Element-wise scalar multiplication (Vector * scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to multiply by.
   * @return A new Vector of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);
    const size_t n = size();

    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) { result[i] = static_cast<R>(_data[i]) * r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise scalar multiplication assignment (Vector * scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to multiply by.
   * @attention This method doesn't cast the Vector if U is a broader type.
   */
  template <Numeric U>
  auto operator*=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(size(), [&](size_t i) { _data[i] *= r_scalar; });

    return *this;
  }

  /**
   * @brief Calculates the dot product (inner product) of two Vectors.
   * @attention This method doesn't check for correct Vector orientations.
   * @details Uses the BLAS dot routines where available and a parallel
   * reduction loop otherwise.
   * @tparam U Numeric type of the other Vector.
   * @param other The other Vector.
   * @return A scalar value of the common, promoted type.
   * @throws std::invalid_argument if Vector sizes do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto dot_product(const Vector<U> &other) const {
    if (size() != other.size()) {
      throw std::invalid_argument("Vectors must be of same size!");
    }

#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<T, double> && std::is_same_v<U, double>) {
      return blas::ddot(size(), _data.data(), 1, other.data(), 1);
    } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, float>) {
      return blas::sdot(size(), _data.data(), 1, other.data(), 1);
    }
#endif

    using R = std::common_type_t<T, U>;

    R result(0);

    omp_loop_reduce(size(), result, [&](size_t i) {
      result += static_cast<R>(_data[i]) * static_cast<R>(other[i]);
    });

    return result;
  }

  /**
   * @brief Outer product (Column Vector * Row Vector).
   * @details This must be a (N x 1) * (1 x M) multiplication.
   * @tparam U Numeric type of the other vector.
   * @param other The row Vector (this Vector must be a column).
   * @return A new Matrix of size (this.size x other.size).
   * @throws std::invalid_argument if orientations are not COLUMN * ROW.
   */
  template <Numeric U>
  [[nodiscard]] auto outer_product(const Vector<U> &other) const {
    using R = std::common_type_t<T, U>;

    const size_t n = size();
    const size_t m = other.size();

    if (_orientation == other.orientation()) {
      // 1x1 x 1x1 vector multiplication
      if (n == 1 && m == 1) {
        return Matrix<R>(1, 1, {_data[0] * other[0]});
      }
      throw std::invalid_argument("Vector dimensions do not match!");
    }
    switch (_orientation) {
      case Orientation::COLUMN: {
        Matrix<R> result(n, m);

        omp_loop(n, m, [&](size_t i) {
          for (size_t j = 0; j < m; ++j) {
            result.data()[(i * m) + j] =
                static_cast<R>(_data[i]) * static_cast<R>(other[j]);
          }
        });

        return result;
      }
      default:
        if (n != m) {
          throw std::invalid_argument("Vector dimensions do not match!");
        }

        std::cout << "This results in a 1x1 matrix. Consider using vector dot "
                     "product.\n";

        return Matrix<R>(1, 1, {dot_product(other)});
    }
  }

  /**
   * @brief Dot product (Row Vector * Column Vector).
   * @attention This must be a (1 x N) * (N x 1) multiplication.
   * @details Checks for correct orientations then calls \ref dot_product().
   * @tparam U Numeric type of the other vector.
   * @param other The row Vector (this Vector must be a column).
   * @return A new Matrix of size (this.size x other.size).
   * @throws std::invalid_argument if orientations are not ROW * COLUMN.
   * @throws std::invalid_argument if Vector sizes do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const Vector<U> &other) const {
    if (_orientation == other.orientation() || _orientation == Orientation::COLUMN) {
      throw std::invalid_argument(
          "Invalid multiplication: Vectors must be of different orientations (row x "
          "column) for dot product. If you are certain that you want the dot "
          "product, use the dot_product() method which does not require specific "
          "orientations.");
    }
    return dot_product(other);
  }

  /**
   * @brief Matrix-Vector multiplication (Row Vector * Matrix).
   * @details This must be a (1 x N) * (N x M) multiplication.
   * @details Uses the BLAS matrix-vector routines where available.
   * @tparam U Numeric type of the matrix.
   * @param other The matrix to multiply by (this vector must be a row).
   * @return A new row Vector of size (1 x M).
   * @throws std::invalid_argument if this is not a row vector or
   * dimensions mismatch.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const Matrix<U> &other) const {
    using R = std::common_type_t<T, U>;

    const size_t n = size();
    const size_t m = other.row_count();
    const size_t r = other.column_count();

    if (n != m) {
      throw std::invalid_argument("Dimensions do not match!");
    }
    if (_orientation == Orientation::COLUMN) {
      throw std::invalid_argument(
          "Invalid multiplication: column Vector * Matrix. "
          "Did you mean Matrix * Vector?");
    }

#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<R, float>) {
      std::vector<float> m_converted;
      std::vector<float> v_converted;
      const float *m_ptr = util::to_pointer(m_converted, other.data_vector());
      const float *v_ptr = util::to_pointer(v_converted, _data);
      Vector<float> result(r, Orientation::ROW);
      blas::sgevmm(m, r, m_ptr, v_ptr, 1, result.data(), r);
      return result;
    } else if constexpr (std::is_same_v<R, double>) {
      std::vector<double> m_converted;
      std::vector<double> v_converted;
      const double *m_ptr = util::to_pointer(m_converted, other.data_vector());
      const double *v_ptr = util::to_pointer(v_converted, _data);
      Vector<double> result(r, Orientation::ROW);
      blas::dgevmm(m, r, m_ptr, v_ptr, 1, result.data(), r);
      return result;
    }
#endif

    Vector<R> result(r, std::vector<R>(r), Orientation::ROW);

    omp_loop(r, n, [&](size_t i) {
      for (size_t j = 0; j < n; ++j) {
        result[i] += _data[j] * other.at(j, i);
      }
    });

    return result;
  }

  /**
   * @brief Element-wise scalar division (Vector / scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to divide by.
   * @return A new Vector of the common, promoted type (forced double if both are
   * int).
   */
  template <Numeric U>
  [[nodiscard]] auto operator/(const U &scalar) const noexcept {
    using R =
        std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                           std::common_type_t<T, U>>;  // Forces double if both are ints

    R r_scalar_inv = R(1) / static_cast<R>(scalar);
    const size_t n = size();

    Vector<R> result(n, _orientation);

    omp_loop(n, [&](size_t i) { result[i] = static_cast<R>(_data[i]) * r_scalar_inv; });

    return result;
  }

  /**
   * @brief Element-wise scalar division assignment (Vector / scalar).
   * @tparam U An arithmetic scalar type.
   * @param scalar The scalar value to divide by.
   * @attention This method doesn't cast the Vector if U is a broader type.
   */
  template <Numeric U>
  auto operator/=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    if constexpr (std::is_floating_point_v<R>) {
      R r_scalar_inv = R(1) / static_cast<R>(scalar);

      omp_loop(size(), [&](size_t i) { _data[i] *= r_scalar_inv; });
    } else {
      R r_scalar = static_cast<R>(scalar);

      omp_loop(size(), [&](size_t i) { _data[i] /= r_scalar; });
    }

    return *this;
  }

  // --- Printing and debugging ---

  /**
   * @brief Prints the vector contents to std::cout.
   * @details Sets floating point precision for readability.
   */
  void print() const {
    if constexpr (std::is_floating_point_v<T>) {
      std::cout << std::setprecision(FLOAT_PRECISION);
    }
    if (_orientation == Orientation::COLUMN) {
      for (const T &val : _data) {
        std::cout << val << std::endl;
      }
    } else {
      for (const T &val : _data) {
        std::cout << val << ' ';
      }
      std::cout << "\n";
    }
    std::cout << std::fixed;
  }

 private:
  /** @brief Stores the vector's orientation (ROW or COLUMN). */
  Orientation _orientation;

  /** @brief Internal contiguous storage for the vector elements. */
  std::vector<T> _data;

  /**
   * @brief Internal helper to invert the sign of all elements in-place.
   */
  void _invert_sign() {
    omp_loop(_data.size(), [&](size_t i) { _data[i] = -_data[i]; });
  }
};

#pragma mark free_functions
// ----------------------------------
// FREE FUNCTIONS
// ----------------------------------

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
  const size_t n = vec.size();

  Vector<R> result(n, vec.orientation());

  omp_loop(n, [&](size_t i) { result[i] = r_scalar - static_cast<R>(vec[i]); });

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

/**
 * @brief Element-wise scalar division (scalar / Vector).
 * @tparam U An arithmetic scalar type.
 * @param scalar The scalar value.
 * @param vec The vector.
 * @return A new Vector of the common, promoted type (forced double if both are
 * int).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator/(const U &scalar, const Vector<T> &vec) noexcept {
  using R =
      std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                         std::common_type_t<T, U>>;  // Forces double if both are ints

  R r_scalar = static_cast<R>(scalar);
  const size_t n = vec.size();

  Vector<R> result(n, vec.orientation());

  omp_loop(n, [&](size_t i) { result[i] = r_scalar / static_cast<R>(vec[i]); });

  return result;
}

/**
 * @brief Checks if two Vectors are element-wise equal within a tolerance.
 * @tparam T Numeric type of the first Vector.
 * @tparam U Numeric type of the second Vector.
 * @param eps The absolute tolerance for equality.
 * @return true if sizes match and all elements are "close".
 */
template <Numeric T, Numeric U>
[[nodiscard]] constexpr bool loosely_equal(const Vector<T> &first,
                                           const Vector<U> &second,
                                           double eps = EPSILON) {
  size_t n = first.size();
  if (first.size() != second.size()) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!is_close(first[i], second[i], eps)) {
      return false;
    }
  }
  return true;
}

}  // namespace maf::math

#endif  // VECTOR_H
