#ifndef MATRIX_H
#define MATRIX_H
#pragma once
#include "MafLib/math/linalg/detail/BlasWrappers/BlasWrapper.hpp"
#include "MafLib/math/linalg/detail/Core.hpp"
#include "MafLib/math/linalg/detail/MatrixView.hpp"
#include "MafLib/math/linalg/detail/Vector.hpp"
#include "MafLib/utility/Conversions.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {

using namespace maf::util;
/**
 * @brief A general-purpose, row-major, dense matrix class.
 *
 * This class implements a matrix with contiguous row-major storage.
 * It supports a wide range of arithmetic operations, constructors, and
 * utility methods.
 *
 * The implementation is templated to support various numeric types (T).
 * Many operations (like multiplication and addition) are parallelized
 * using OpenMP and employ blocking strategies for cache efficiency.
 *
 * Additional optimizations in the form of LAPACK/BLAS subroutines
 * are automatically included on all **WORTHY** operating systems.
 *
 * @tparam T The numeric type of the matrix elements (e.g., float,
 * double, int).
 *
 * @version 1.0
 * @since 2025
 */
template <Numeric T>
class Matrix {
 public:
  /** @brief The numeric type of the matrix elements. */
  using value_type = T;

#pragma mark constructors
  // ----------------------------------
  // CONSTRUCTORS
  // ----------------------------------

  /**
   * @brief Default constructor. Creates an empty 0x0 matrix.
   */
  Matrix() : _rows(0), _cols(0) {}

  /**
   * @brief Constructs an uninitialized matrix of size rows x cols.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @throws std::invalid_argument if dimensions are zero.
   */
  Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    _data.resize(rows * cols);
  }

  /**
   * @brief Constructs a matrix from a raw data pointer.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data Pointer to a C-style array of size (rows * cols) in
   * row-major order. The data is COPIED into the matrix.
   * @throws std::invalid_argument if dimensions are zero or data is
   * nullptr.
   */
  Matrix(size_t rows, size_t cols, T *data) : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero!");
    }
    if (data == nullptr) {
      throw std::invalid_argument("Data pointer cannot be null!");
    }

    _data.assign(data, data + (rows * cols));
  }

  /**
   * @brief Constructs from a std::vector, filled by rows.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector of size (rows * cols) in row-major order.
   * @throws std::invalid_argument if dimensions are zero or data size
   * does not match.
   */
  Matrix(size_t rows, size_t cols, const std::vector<T> &data)
      : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    if (data.size() != rows * cols) {
      throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.assign(data.begin(), data.end());
  }

  /**
   * @brief Constructs from a std::vector, filled by rows (move
   * constructor).
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector of size (rows * cols) in row-major order.
   * @throws std::invalid_argument if dimensions are zero or data size
   * does not match.
   */
  Matrix(size_t rows, size_t cols, std::vector<T> &&data) : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (data.size() != rows * cols) {
      throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data = std::move(data);
  }

  /**
   * @brief Constructs from a nested std::vector (vector of vectors).
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector<std::vector<T>>. data.size() must equal
   * rows, and data[0].size() must equal cols.
   * @throws std::invalid_argument if dimensions are zero or data shape
   * does not match.
   */
  Matrix(size_t rows, size_t cols, const std::vector<std::vector<T>> &data)
      : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    if (data.size() != rows || data.at(0).size() != cols) {
      throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        _data.at(_get_index(i, j)) = static_cast<T>(data.at(i).at(j));
      }
    }
  }

  /**
   * @brief Constructs from a std::array, filled by rows.
   * @tparam U Type in the array (allows implicit conversion).
   * @tparam N Size of the array.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::array<U, N> where N must equal (rows * cols).
   * @throws std::invalid_argument if dimensions are zero or array size
   * does not match.
   */
  template <size_t N>
  Matrix(size_t rows, size_t cols, const std::array<T, N> &data)
      : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (N != rows * cols) {
      throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.assign(data.begin(), data.end());
  }

  /**
   * @brief Constructs from a std::initializer_list, filled by rows.
   * @tparam U Type in the list (allows implicit conversion).
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param list A std::initializer_list<U> of size (rows * cols).
   * @throws std::invalid_argument if dimensions are zero or list size
   * does not match.
   */
  template <Numeric U>
  Matrix(size_t rows, size_t cols, std::initializer_list<U> list)
      : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (static_cast<size_t>(list.size()) != rows * cols) {
      throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.assign(list.begin(), list.end());
  }

#pragma mark getters_setters
  // ----------------------------------
  // GETTERS & SETTERS
  // ----------------------------------

  /**
   * @brief Gets a mutable pointer to the underlying data.
   * @return T*
   */
  [[nodiscard]] T *data() noexcept { return _data.data(); }

  /**
   * @brief Gets a const pointer to the underlying data.
   * @return const std::vector<T>&
   */
  [[nodiscard]] const T *data() const noexcept { return _data.data(); }

  /**
   * @brief Gets a mutable reference to the underlying std::vector
   * data store.
   * @return std::vector<T>&
   */
  [[nodiscard]] std::vector<T> &data_vector() noexcept { return _data; }

  /**
   * @brief Gets a const reference to the underlying std::vector data
   * store.
   * @return const std::vector<T>&
   */
  [[nodiscard]] const std::vector<T> &data_vector() const noexcept { return _data; }

  /** @brief Gets the number of rows. */
  [[nodiscard]] size_t row_count() const noexcept { return _rows; }

  /** @brief Gets the number of columns. */
  [[nodiscard]] size_t column_count() const noexcept { return _cols; }

  /** @brief Gets the total number of elements (rows * cols). */
  [[nodiscard]] size_t size() const noexcept { return _data.size(); }

  /**
   * @brief Accesses the element at [row][col] with no bounds check.
   * @return pointer to the start of the specified row.
   */
  [[nodiscard]] T *operator[](size_t ind) noexcept { return &_data[ind * _cols]; }

  /**
   * @brief Accesses the element at [row][col] with no bounds check.
   * @return const pointer to the start of the specified row.
   */
  [[nodiscard]] const T *operator[](size_t ind) const noexcept {
    return &_data[ind * _cols];
  }

  /**
   * @brief Accesses the element at [row, col] with no bounds check.
   * @return reference to the element at (row, col).
   */
  [[nodiscard]] T &operator[](size_t row, size_t col) noexcept {
    return _data[(row * _cols) + col];
  }

  /**
   * @brief Accesses the element at [row, col] with no bounds check.
   * @return const reference to the element at (row, col).
   */
  [[nodiscard]] const T &operator[](size_t row, size_t col) const noexcept {
    return _data[(row * _cols) + col];
  }

  /**
   * @brief Gets a mutable reference to the element at (row, col).
   * @throws std::out_of_range if the index is invalid.
   */
  T &at(size_t row, size_t col) { return _data.at(_get_index(row, col)); }

  /**
   * @brief Gets a const reference to the element at (row, col).
   * @throws std::out_of_range if the index is invalid.
   */
  const T &at(size_t row, size_t col) const { return _data.at(_get_index(row, col)); }

  /**
   * @brief Gets a mutable std::span of a single row.
   * @throws std::out_of_range if the row is invalid.
   */
  [[nodiscard]] std::span<T> row_span(size_t row) {
    return std::span<T>(&_data.at(_get_index(row, 0)), _cols);
  }

  /**
   * @brief Gets a const std::span of a single row.
   * @throws std::out_of_range if the row is invalid.
   */
  [[nodiscard]] std::span<const T> row_span(size_t row) const {
    return std::span<const T>(&_data.at(_get_index(row, 0)), _cols);
  }

  /**
   * @brief Creates a view (sub-matrix) into this matrix.
   * @param row Starting row index of the view.
   * @param col Starting column index of the view.
   * @param height Height (number of rows) of the view.
   * @param width Width (number of columns) of the view.
   * @return MatrixView<T> representing the specified sub-matrix.
   * @throws std::invalid_argument if height or width is zero.
   * @throws std::out_of_range if the requested view exceeds matrix
   * dimensions.
   */
  [[nodiscard]] MatrixView<T> view(size_t row, size_t col, size_t height,
                                   size_t width) {
    if (height == 0 || width == 0) {
      throw std::invalid_argument("View dimensions must be greater than zero.");
    }
    if (row + height > _rows || col + width > _cols) {
      throw std::out_of_range("Requested view exceeds matrix dimensions.");
    }
    return MatrixView<T>(&_data[(row * _cols) + col], height, width, _cols);
  }

#pragma mark checkers
  // ----------------------------------
  // CHECKERS
  // ----------------------------------

  /** @brief Checks if the matrix is square (rows == cols). */
  [[nodiscard]] constexpr bool is_square() const { return _rows == _cols; }

  /** @brief Checks if the matrix is symmetric (A == A^T). */
  [[nodiscard]] constexpr bool is_symmetric() const {
    if (!is_square()) {
      return false;
    }
    for (size_t i = 0; i < _rows; ++i) {
      for (size_t j = i + 1; j < _cols; ++j) {
        if (!is_close(at(i, j), at(j, i))) {
          return false;
        }
      }
    }
    return true;
  }

  /** @brief Checks if the matrix is upper triangular */
  [[nodiscard]] constexpr bool is_upper_triangular() const {
    if (!is_square()) {
      return false;
    }

    static constexpr T ZERO = static_cast<T>(0);
    for (size_t i = 1; i < _rows; ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (!is_close(at(i, j), ZERO)) {
          return false;
        }
      }
    }
    return true;
  }

  /** @brief Checks if the matrix is lower triangular */
  [[nodiscard]] constexpr bool is_lower_triangular() const {
    if (!is_square()) {
      return false;
    }

    static constexpr T ZERO = static_cast<T>(0);

    for (size_t i = 0; i < _rows - 1; ++i) {
      for (size_t j = i + 1; j < _cols; ++j) {
        if (!is_close(at(i, j), ZERO)) {
          return false;
        }
      }
    }
    return true;
  }

  /** @brief Checks if the matrix is diagonal */
  [[nodiscard]] constexpr bool is_diagonal() const {
    if (!is_square()) {
      return false;
    }

    // Upper triangular
    static constexpr T ZERO = static_cast<T>(0);
    for (size_t i = 1; i < _rows; ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (!is_close(at(i, j), ZERO)) {
          return false;
        }
      }
    }

    // Lower triangular
    for (size_t i = 0; i < _rows - 1; ++i) {
      for (size_t j = i + 1; j < _cols; ++j) {
        if (!is_close(at(i, j), ZERO)) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * @brief Checks if the matrix is singular (non-invertible).
   * @details Equivalent to det(A) == 0.
   * @details Equivalent to rank(A) < max(dimensions(A)).
   */
  [[nodiscard]] constexpr bool is_singular() const {
    if (!is_square()) {
      return true;
    }
    try {
      plu(*this);
      return false;
    } catch (const std::runtime_error &e) {
      return true;
    }
  }

  /**
   * @brief Checks if the matrix is positive definite.
   * @details For symmetric matrices, attempts Cholesky decomposition.
   * TODO: Add Sylvester's criterion for non-symmetric.
   */
  [[nodiscard]] constexpr bool is_positive_definite() const {
    try {
      cholesky(*this);
      return true;
    } catch (std::exception &e) {
      return false;
    }
  }

#pragma mark methods
  // ----------------------------------
  // METHODS
  // ----------------------------------

  /** @brief Creates new matrix with same elements but different type.
   */
  template <Numeric U>
  [[nodiscard]] Matrix<U> cast() const {
    // Replace this with explicit constructor casting when implemented
    Matrix<U> result(_rows, _cols);
    omp_loop(_data.size(),
             [&](size_t i) { result.data()[i] = static_cast<U>(_data[i]); });
    return result;
  }

  /** @brief Fills the entire matrix with a single value.*/
  void fill(T value) {
    omp_loop(_data.size(), [&](size_t i) { _data[i] = value; });
  }

  /**
   * @brief Converts this matrix into an identity matrix.
   * @throws std::runtime_error if the matrix is not square.
   */
  void make_identity() {
    if (!is_square()) {
      throw std::invalid_argument("Only square matrices can be set to identity!");
    }

    fill(T(0));
    for (size_t i = 0; i < _rows; ++i) {
      (*this)[i, i] = T(1);
    }
  }

  /**
   * @brief Performs an in-place transpose of the matrix.
   * @details Calls `transposed()` and moves the result back into this matrix.
   * @throws std::invalid_argument if the matrix is not square.
   */
  void transpose() {
    if (!is_square()) {
      throw std::invalid_argument("Matrix must be square to transpose in-place.");
    }
#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      Matrix<T> temp = this->transposed();
      *this = std::move(temp);
      return;
    }
#endif
#pragma omp parallel for schedule(static) default(none)
    for (size_t i = 0; i < _rows; i += BLOCK_SIZE) {
      for (size_t j = i; j < _cols; j += BLOCK_SIZE) {
        const size_t n = std::min(i + BLOCK_SIZE, _rows);
        const size_t m = std::min(j + BLOCK_SIZE, _cols);
        for (size_t k = i; k < n; ++k) {
          for (size_t l = (i == j) ? k + 1 : j; l < m; ++l) {
            std::swap(this->at(k, l), this->at(l, k));
          }
        }
      }
    }
  }

  /**
   * @brief Creates and returns a new matrix that is the transpose of
   * this one.
   * @return A new Matrix<T> of size (cols x rows).
   */
  [[nodiscard]] Matrix<T> transposed() const {
    Matrix<T> result(_cols, _rows);
#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
      blas::smatcopy(_data.data(), result.data(), _rows, _cols);
      return result;
    } else if constexpr (std::is_same_v<T, double>) {
      blas::dmatcopy(_data.data(), result.data(), _rows, _cols);
      return result;
    }
#endif
    // Generic implementation
#pragma omp parallel for collapse(2) schedule(static) default(none) shared(result)
    for (size_t i = 0; i < _rows; i += BLOCK_SIZE) {
      for (size_t j = 0; j < _cols; j += BLOCK_SIZE) {
        size_t i_max = std::min(i + BLOCK_SIZE, _rows);
        size_t j_max = std::min(j + BLOCK_SIZE, _cols);

        for (size_t k = i; k < i_max; ++k) {
          for (size_t l = j; l < j_max; ++l) {
            result[l, k] = (*this)[k, l];
          }
        }
      }
    }
    return result;
  }

  template <bool is_spd = false>
  [[nodiscard]] auto determinant() const {
    if (!is_square()) {
      throw std::invalid_argument("Determinant is only defined for square matrices.");
    }

    if constexpr (is_spd) {
      // res = det(L)^2
      auto chol_res = cholesky(*this);
      // Get the promoted type
      using AccumType = std::decay_t<decltype(chol_res)>::value_type;
      AccumType det = 1;

      for (size_t i = 0; i < _rows; ++i) {
        det *= chol_res.at(i, i);
      }
      return det * det;
    } else {
      // res = det(U) * det(P), det(L) = 1
      auto plu_res = plu(*this);
      // Get the promoted type
      using AccumType = std::decay_t<decltype(plu_res.U)>::value_type;
      auto det = static_cast<AccumType>(plu_res.sign);

      for (size_t i = 0; i < _rows; ++i) {
        det *= plu_res.U.at(i, i);
      }
      return det;
    }
  }

  // TODO: Implement
  [[nodiscard]] Matrix<T> inverted() const;

#pragma mark operators
  // ----------------------------------
  // OPERATORS
  // ----------------------------------

  /**
   * @brief Checks for exact element-wise equality.
   * @details For floating-point, use `loosely_equal()`.
   */
  [[nodiscard]] constexpr bool operator==(const Matrix &other) const noexcept {
    if (_rows != other._rows || _cols != other._cols) {
      return false;
    }
    return _data == other._data;
  }

  /**
   * @brief Unary minus. Returns a new matrix with all elements negated.
   * @return `Matrix<T>`
   */
  [[nodiscard]] auto operator-() const noexcept {
    Matrix<T> result(*this);
    result._invert_sign();
    return result;
  }

  /**
   * @brief Element-wise matrix addition.
   * @tparam U Numeric type of the other matrix.
   * @return Matrix of the common, promoted type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const Matrix<U> &other) const {
    if (_rows != other.row_count() || _cols != other.column_count()) {
      throw std::invalid_argument(
          "Matrices have to be of same dimensions for addition!");
    }
    using R = std::common_type_t<T, U>;

    Matrix<R> result(_rows, _cols);

    omp_loop(_data.size(), [&](size_t i) {
      result.data()[i] = static_cast<R>(_data[i]) + static_cast<R>(other.data()[i]);
    });

    return result;
  }

  /**
   * @brief Element-wise scalar addition (Matrix + scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    Matrix<R> result(_rows, _cols);
    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(),
             [&](size_t i) { result.data()[i] = static_cast<R>(_data[i]) + r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise matrix addition assignment.
   * @tparam U Numeric type of the other matrix.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  Matrix<T> &operator+=(const Matrix<U> &other) {
    if (_rows != other.row_count() || _cols != other.column_count()) {
      throw std::invalid_argument(
          "Matrices have to be of same dimensions for addition!");
    }

    omp_loop(_data.size(),
             [&](size_t i) { _data[i] += static_cast<T>(other.data()[i]); });

    return *this;
  }

  /**
   * @brief Element-wise scalar addition assignment (Matrix + scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator+=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(), [&](size_t i) { _data[i] += r_scalar; });

    return *this;
  }

  /**
   * @brief Element-wise matrix subtraction.
   * @tparam U Numeric type of the other matrix.
   * @return Matrix of the common, promoted type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const Matrix<U> &other) const {
    if (_rows != other.row_count() || _cols != other.column_count()) {
      throw std::invalid_argument(
          "Matrices have to be of same dimensions for subtraction!");
    }
    using R = std::common_type_t<T, U>;

    Matrix<R> result(_rows, _cols);

    omp_loop(_data.size(), [&](size_t i) {
      result.data()[i] = static_cast<R>(_data[i]) - static_cast<R>(other.data()[i]);
    });
    return result;
  }

  /**
   * @brief Element-wise scalar subtraction (Matrix - scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    Matrix<R> result(_rows, _cols);
    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(),
             [&](size_t i) { result.data()[i] = static_cast<R>(_data[i]) - r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise matrix subtraction assignment.
   * @tparam U Numeric type of the other matrix.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  Matrix<T> &operator-=(const Matrix<U> &other) {
    if (_rows != other.row_count() || _cols != other.column_count()) {
      throw std::invalid_argument(
          "Matrices have to be of same dimensions for addition!");
    }

    omp_loop(_data.size(),
             [&](size_t i) { _data[i] -= static_cast<T>(other.data()[i]); });

    return *this;
  }

  /**
   * @brief Element-wise scalar subtraction assignment (Matrix - scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator-=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(), [&](size_t i) { _data[i] -= r_scalar; });

    return *this;
  }

  /**
   * @brief Standard algebraic matrix multiplication (A * B).
   * @details Implemented with a parallelized, cache-blocked algorithm.
   * @details Uses BLAS gemm where available.
   * @tparam U Numeric type of the other matrix.
   * @return Matrix of the common, promoted type.
   * @throws std::invalid_argument if inner dimensions do not match
   * (A.cols != B.rows).
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const Matrix<U> &other) const {
    if (_cols != other.row_count()) {
      throw std::invalid_argument(
          "Matrix inner dimensions do not match for multiplication!");
    }
    using R = std::common_type_t<T, U>;

    const size_t a_rows = _rows;
    const size_t b_cols = other.column_count();
    const size_t a_cols = _cols;
    Matrix<R> result(a_rows, b_cols);

    const T *a_data = this->_data.data();
    const U *b_data = other.data();
    R *c_data = result.data();

    result.fill(R(0));

#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<R, float>) {
      std::vector<float> a_converted;
      std::vector<float> b_converted;
      blas::sgemm(a_rows, b_cols, a_cols, util::to_pointer(a_converted, _data),
                  util::to_pointer(b_converted, other.data_vector()), c_data);
    } else if constexpr (std::is_same_v<R, double>) {
      std::vector<double> a_converted;
      std::vector<double> b_converted;
      blas::dgemm(a_rows, b_cols, a_cols, util::to_pointer(a_converted, _data),
                  util::to_pointer(b_converted, other.data_vector()), c_data);
    } else {
      // Integer types - use fallback
      _gemm(a_data, b_data, c_data, a_rows, a_cols, b_cols);
    }
#else
    // Default non-BLAS implementation
    _gemm(a_data, b_data, c_data, a_rows, a_cols, b_cols);
#endif
    return result;
  }

  /**
   * @brief Element-wise scalar multiplication (Matrix * scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const U &scalar) const noexcept {
    using R = std::common_type_t<T, U>;

    Matrix<R> result(_rows, _cols);
    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(),
             [&](size_t i) { result.data()[i] = static_cast<R>(_data[i]) * r_scalar; });

    return result;
  }

  /**
   * @brief Element-wise scalar multiplication assignment (Matrix * scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator*=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    R r_scalar = static_cast<R>(scalar);

    omp_loop(_data.size(), [&](size_t i) { _data[i] *= r_scalar; });
    return *this;
  }

  /**
   * @brief Matrix-Vector multiplication (Matrix * column_vector).
   * @tparam U Numeric type of the vector.
   * @return A new column Vector of the common, promoted type.
   * @throws std::invalid_argument if vector is not a column vector or
   * dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const Vector<U> &other) const {
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

    // Uses the BLAS gemv routines where available.
#ifdef BLAS_AVAILABLE
    if constexpr (std::is_same_v<R, float>) {
      std::vector<float> a_converted;
      std::vector<float> v_converted;
      const float *a_ptr = util::to_pointer(a_converted, _data);
      const float *v_ptr = util::to_pointer(v_converted, other.data_vector());
      Vector<float> result(_rows, Orientation::COLUMN);
      blas::sgemvm(_rows, _cols, a_ptr, v_ptr, 1, result.data(), _cols);
      return result;
    } else if constexpr (std::is_same_v<R, double>) {
      std::vector<double> a_converted;
      std::vector<double> v_converted;
      const double *a_ptr = util::to_pointer(a_converted, _data);
      const double *v_ptr = util::to_pointer(v_converted, other.data_vector());
      Vector<double> result(_rows, Orientation::COLUMN);
      blas::dgemvm(_rows, _cols, a_ptr, v_ptr, 1, result.data(), _cols);
      return result;
    }
#endif
    Vector<R> result(_rows, std::vector<R>(_rows, R(0)), Orientation::COLUMN);

    omp_loop(_rows, _cols, [&](size_t i) {
      auto row = row_span(i);
      for (size_t j = 0; j < _cols; ++j) {
        result[i] += static_cast<R>(row[j]) * static_cast<R>(other[j]);
      }
    });

    return result;
  }

  /**
   * @brief Element-wise scalar division (Matrix / scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator/(const U &scalar) const noexcept {
    using R =
        std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double,
                           std::common_type_t<T, U>>;  // Forces double if both are ints

    Matrix<R> result(_rows, _cols);
    R r_scalar_inv = R(1) / static_cast<R>(scalar);

    omp_loop(_data.size(), [&](size_t i) {
      result.data()[i] = static_cast<R>(_data[i]) * r_scalar_inv;
    });

    return result;
  }

  /**
   * @brief Element-wise scalar division assignment (Matrix / scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator/=(const U &scalar) noexcept {
    using R = std::common_type_t<T, U>;

    if constexpr (std::is_floating_point_v<R>) {
      R r_scalar_inv = R(1) / static_cast<R>(scalar);
      omp_loop(_data.size(), [&](size_t i) { _data[i] *= r_scalar_inv; });
    } else {
      R r_scalar = static_cast<R>(scalar);
      omp_loop(_data.size(), [&](size_t i) { _data[i] /= r_scalar; });
    }
    return *this;
  }

  // --- Debugging and printing ---

  /**
   * @brief Prints the matrix contents to std::cout.
   * @details Sets floating point precision for readability.
   */
  void print() const {
    if constexpr (std::is_floating_point_v<T>) {
      std::cout << std::fixed << std::setprecision(FLOAT_PRECISION);
    }
    for (size_t i = 0; i < _rows; ++i) {
      for (size_t j = 0; j < _cols; ++j) {
        std::cout << this->at(i, j) << ' ';
      }
      std::cout << "\n";
    }
  }

 private:
  size_t _rows;
  size_t _cols;
  std::vector<T> _data;

  /**
   * @brief Internal check if a row/column index is within bounds.
   * @return true if 0 <= row < _rows and 0 <= col < _cols.
   */
  [[nodiscard]] constexpr bool _is_valid_index(size_t row, size_t col) const {
    return row < _rows && col < _cols;
  }

  /**
   * @brief Converts a 2D (row, col) index to a 1D internal vector index.
   * @throws std::out_of_range if the index is invalid.
   * @return The 1D index into the _data vector.
   */
  [[nodiscard]] constexpr size_t _get_index(size_t row, size_t col) const {
    if (!_is_valid_index(row, col)) {
      throw std::out_of_range("Index out of bounds.");
    }
    return (row * _cols) + col;
  }

  /**
   * @brief Internal helper to invert the sign of all elements in-place.
   */
  void _invert_sign() {
    omp_loop(_data.size(), [&](size_t i) { _data[i] = -_data[i]; });
  }

  // Fallback matrix multiplication implementation
  template <typename A, typename B, typename C>
  void _gemm(const A *a_data, const B *b_data, C *c_data, size_t a_rows, size_t a_cols,
             size_t b_cols) const {
    static constexpr size_t OMP_DIM = 10000;
#pragma omp parallel for collapse(2) if (a_rows * b_cols > OMP_DIM) default(none) \
    shared(a_data, b_data, c_data, a_rows, a_cols, b_cols)
    for (size_t i = 0; i < a_rows; ++i) {
      for (size_t k = 0; k < a_cols; ++k) {
        for (size_t j = 0; j < b_cols; ++j) {
          c_data[i * b_cols + j] += static_cast<C>(a_data[i * a_cols + k]) *
                                    static_cast<C>(b_data[k * b_cols + j]);
        }
      }
    }
  }
};

#pragma mark friends
// ----------------------------------
// FRIEND OPERATORS
// ----------------------------------

/**
 * @brief Element-wise scalar addition (scalar + Matrix).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator+(const U &scalar, const Matrix<T> &matrix) noexcept {
  return matrix + scalar;
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

  omp_loop(matrix.size(), [&](size_t i) {
    result.data()[i] = r_scalar - static_cast<R>(matrix.data()[i]);
  });

  return result;
}

/**
 * @brief Element-wise scalar multiplication (scalar * Matrix).
 */
template <Numeric T, Numeric U>
[[nodiscard]] auto operator*(const U &scalar, const Matrix<T> &matrix) noexcept {
  return matrix * scalar;
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

  omp_loop(matrix.size(), [&](size_t i) {
    result.data()[i] = r_scalar / static_cast<R>(matrix.data()[i]);
  });

  return result;
}

/**
 * @brief Checks if two matrices are element-wise equal within a tolerance.
 * @tparam T Numeric type of the first matrix.
 * @tparam U Numeric type of the second matrix.
 * @param eps The absolute tolerance for equality.
 * @return true if dimensions match and all elements are "close".
 */
template <Numeric T, Numeric U>
[[nodiscard]] constexpr bool loosely_equal(const Matrix<T> &first,
                                           const Matrix<U> &second,
                                           double eps = EPSILON) {
  size_t n = first.row_count();
  size_t m = first.column_count();
  if (n != second.row_count() || m != second.column_count()) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (!is_close(first.at(i, j), second.at(i, j), eps)) {
        return false;
      }
    }
  }
  return true;
}

// ----------------------------------
// FRIEND FACTORIES
// ----------------------------------

/**
 * @brief Creates a new identity matrix of a given size.
 * @tparam T The numeric type of the matrix.
 * @param size The width and height of the square matrix.
 * @return An identity Matrix<T> of size (size x size).
 */
template <Numeric T>
[[nodiscard]] Matrix<T> inline identity_matrix(size_t size) {
  Matrix<T> result(size, size);  // This is slower then specialized constructor
  result.make_identity();
  return result;
}

/**
 * @brief Creates a new matrix filled with ones.
 * @tparam U The numeric type of the matrix.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return A Matrix<U> of size (rows x cols) filled with U(1).
 */
template <Numeric U>
[[nodiscard]] inline Matrix<U> ones(size_t rows, size_t cols) {
  Matrix<U> result(rows, cols);
  result.fill(U(1));
  return result;
}

/**
 * @brief Creates a new permutation matrix from a permutation vector.
 * @tparam T The numeric type of the matrix.
 * @param perm A vector where `perm[i] = j` means row `i` maps to
 * column `j`.
 * @return A sparse permutation Matrix<T>.
 */
template <Numeric T>
[[nodiscard]] Matrix<T> inline permutation_matrix(const std::vector<uint32> &perm) {
  size_t n = perm.size();
  Matrix<T> result(n, n);  // Initializes to zero
  omp_loop(n, [&](size_t i) { result[i][perm[i]] = static_cast<T>(1); });
  return result;
}

}  // namespace maf::math

#endif  // MATRIX_H
