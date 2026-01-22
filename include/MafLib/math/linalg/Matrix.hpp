#ifndef MATRIX_H
#define MATRIX_H
#pragma once
#include "AccelerateWrappers/AccelerateWrapper.hpp"
#include "LinAlg.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/utility/Math.hpp"
#include "MatrixView.hpp"

namespace maf::math {
template <typename T>
concept _MatrixViewCompatible = std::is_same_v<T, MatrixView<typename T::value_type>> ||
                                std::is_same_v<T, Matrix<typename T::value_type>>;

/** @brief Concept for types compatible with MatrixView. */
template <typename T>
concept MatrixViewCompatible = Numeric<T> && _MatrixViewCompatible<T>;

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

  // --- Constructors ---

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
  Matrix(size_t rows, size_t cols);

  /**
   * @brief Constructs a matrix from a raw data pointer.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data Pointer to a C-style array of size (rows * cols) in
   * row-major order. The data is COPIED into the matrix.
   * @throws std::invalid_argument if dimensions are zero or data is
   * nullptr.
   */
  Matrix(size_t rows, size_t cols, T *data);

  /**
   * @brief Constructs from a std::vector, filled by rows.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector of size (rows * cols) in row-major order.
   * @throws std::invalid_argument if dimensions are zero or data size
   * does not match.
   */
  Matrix(size_t rows, size_t cols, const std::vector<T> &data);

  /**
   * @brief Constructs from a std::vector, filled by rows (move
   * constructor).
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector of size (rows * cols) in row-major order.
   * @throws std::invalid_argument if dimensions are zero or data size
   * does not match.
   */
  Matrix(size_t rows, size_t cols, std::vector<T> &&data);

  /**
   * @brief Constructs from a nested std::vector (vector of vectors).
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param data A std::vector<std::vector<T>>. data.size() must equal
   * rows, and data[0].size() must equal cols.
   * @throws std::invalid_argument if dimensions are zero or data shape
   * does not match.
   */
  Matrix(size_t rows, size_t cols, const std::vector<std::vector<T>> &data);

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
  Matrix(size_t rows, size_t cols, const std::array<T, N> &data);

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
  Matrix(size_t rows, size_t cols, std::initializer_list<U> list);

  // --- Getters and setters ---

  /**
   * @brief Gets a mutable reference to the underlying std::vector
   * data store.
   * @return std::vector<T>&
   */
  [[nodiscard]] std::vector<T> &data() noexcept { return _data; }

  /**
   * @brief Gets a const reference to the underlying std::vector data
   * store.
   * @return const std::vector<T>&
   */
  [[nodiscard]] const std::vector<T> &data() const noexcept { return _data; }

  /** @brief Gets the number of rows. */
  [[nodiscard]] size_t row_count() const noexcept { return _rows; }

  /** @brief Gets the number of columns. */
  [[nodiscard]] size_t column_count() const noexcept { return _cols; }

  /** @brief Gets the total number of elements (rows * cols). */
  [[nodiscard]] size_t size() const noexcept { return _data.size(); }

  /**
   * @brief Accesses the element at (row, col) with no bounds check.
   */
  [[nodiscard]] T *operator[](size_t ind) noexcept {
    // TODO: add tests
    return &_data[ind * _cols];
  }

  /**
   * @brief Accesses the element at (row, col) with no bounds check.
   */
  [[nodiscard]] const T *operator[](size_t ind) const noexcept {
    // TODO: add tests
    return &_data[ind * _cols];
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

  // --- Checkers ---

  /** @brief Checks if the matrix is square (rows == cols). */
  [[nodiscard]] constexpr bool is_square() const;

  /** @brief Checks if the matrix is symmetric (A == A^T). */
  [[nodiscard]] constexpr bool is_symmetric() const;

  /** @brief Checks if the matrix is upper triangular */
  [[nodiscard]] constexpr bool is_upper_triangular() const;

  /** @brief Checks if the matrix is lower triangular */
  [[nodiscard]] constexpr bool is_lower_triangular() const;

  /** @brief Checks if the matrix is diagonal */
  [[nodiscard]] constexpr bool is_diagonal() const;

  /**
   * @brief Checks if the matrix is positive definite.
   * @details For symmetric matrices, attempts Cholesky decomposition.
   * TODO: Add Sylvester's criterion for non-symmetric.
   * Defined in MatrixCheckers.hpp
   */
  [[nodiscard]] constexpr bool is_positive_definite() const;

  /**
   * @brief Checks if the matrix is singular (non-invertible).
   * @details Equivalent to det(A) == 0.
   * @details Equivalent to rank(A) < max(dimensions(A)).
   */
  [[nodiscard]] constexpr bool is_singular() const;

  // --- Methods ---

  /** @brief Creates new matrix with same elements but different type.
   * @details Defined in MatrixMethods.hpp
   */
  template <Numeric U>
  [[nodiscard]] Matrix<U> cast() const;

  /** @brief Fills the entire matrix with a single value.*/
  void fill(T value);

  /**
   * @brief Converts this matrix into an identity matrix.
   * @throws std::runtime_error if the matrix is not square.
   * @details Defined in MatrixMethods.hpp
   */
  void make_identity();

  /**
   * @brief Performs an in-place transpose of the matrix.
   * @details Uses a parallelized, blocked algorithm.
   * @throws std::invalid_argument if the matrix is not square.
   */
  void transpose();

  /**
   * @brief Creates and returns a new matrix that is the transpose of
   * this one.
   * @return A new Matrix<T> of size (cols x rows).
   * @details Defined in MatrixMethods.hpp
   */
  [[nodiscard]] Matrix<T> transposed() const;

  // --- Operators ---

  /**
   * @brief Checks for exact element-wise equality.
   * @details For floating-point, use `loosely_equal()`.
   */
  [[nodiscard]] constexpr bool operator==(const Matrix &other) const noexcept;

  /**
   * @brief Unary minus. Returns a new matrix with all elements negated.
   * @return `Matrix<T>`
   */
  [[nodiscard]] auto operator-() const noexcept;

  /**
   * @brief Element-wise matrix addition.
   * @tparam U Numeric type of the other matrix.
   * @return Matrix of the common, promoted type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const Matrix<U> &other) const;

  /**
   * @brief Element-wise scalar addition (Matrix + scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator+(const U &scalar) const noexcept;

  /**
   * @brief Element-wise matrix addition assignment.
   * @tparam U Numeric type of the other matrix.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  Matrix<T> &operator+=(const Matrix<U> &other);

  /**
   * @brief Element-wise scalar addition assignment (Matrix + scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator+=(const U &scalar) noexcept;

  /**
   * @brief Element-wise matrix subtraction.
   * @tparam U Numeric type of the other matrix.
   * @return Matrix of the common, promoted type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const Matrix<U> &other) const;

  /**
   * @brief Element-wise scalar subtraction (Matrix - scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator-(const U &scalar) const noexcept;

  /**
   * @brief Element-wise matrix subtraction assignment.
   * @tparam U Numeric type of the other matrix.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @throws std::invalid_argument if dimensions do not match.
   */
  template <Numeric U>
  Matrix<T> &operator-=(const Matrix<U> &other);

  /**
   * @brief Element-wise scalar subtraction assignment (Matrix - scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator-=(const U &scalar) noexcept;

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
    const U *b_data = other.data().data();
    R *c_data = result.data().data();

    result.fill(R(0));

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
    // Handle all floating-point combinations with proper conversion
    if constexpr (std::is_floating_point_v<R>) {
      if constexpr (std::is_same_v<R, float>) {
        // Both matrices are float or can be safely converted to float
        std::vector<float> a_converted, b_converted;
        const float *a_ptr = a_data;
        const float *b_ptr = b_data;

        // Convert matrix A if needed
        if constexpr (!std::is_same_v<T, float>) {
          a_converted.resize(_data.size());
          std::transform(_data.begin(), _data.end(), a_converted.begin(),
                         [](T val) { return static_cast<float>(val); });
          a_ptr = a_converted.data();
        } else {
          a_ptr = a_data;
        }

        // Convert matrix B if needed
        if constexpr (!std::is_same_v<U, float>) {
          b_converted.resize(other.data().size());
          std::transform(other.data().begin(), other.data().end(), b_converted.begin(),
                         [](U val) { return static_cast<float>(val); });
          b_ptr = b_converted.data();
        } else {
          b_ptr = b_data;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, a_cols,
                    1.0F, a_ptr, a_cols, b_ptr, b_cols, 0.0F, c_data, b_cols);
      } else if constexpr (std::is_same_v<R, double>) {
        // Both matrices are double or can be safely converted to double
        std::vector<double> a_converted, b_converted;
        const double *a_ptr;
        const double *b_ptr;

        // Convert matrix A if needed
        if constexpr (!std::is_same_v<T, double>) {
          a_converted.resize(_data.size());
          std::transform(_data.begin(), _data.end(), a_converted.begin(),
                         [](T val) { return static_cast<double>(val); });
          a_ptr = a_converted.data();
        } else {
          a_ptr = a_data;
        }

        // Convert matrix B if needed
        if constexpr (!std::is_same_v<U, double>) {
          b_converted.resize(other.data().size());
          std::transform(other.data().begin(), other.data().end(), b_converted.begin(),
                         [](U val) { return static_cast<double>(val); });
          b_ptr = b_converted.data();
        } else {
          b_ptr = b_data;
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a_rows, b_cols, a_cols,
                    1.0, a_ptr, a_cols, b_ptr, b_cols, 0.0, c_data, b_cols);
      }
    } else {
      // Integer types - use fallback
      _fallback_matrix_multiply(a_data, b_data, c_data, a_rows, a_cols, b_cols);
    }
#else
    // Default non-Apple implementation
    _fallback_matrix_multiply(a_data, b_data, c_data, a_rows, a_cols, b_cols);
#endif
    return result;
  }

  /**
   * @brief Element-wise scalar multiplication (Matrix * scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const U &scalar) const noexcept;

  /**
   * @brief Element-wise scalar multiplication assignment (Matrix * scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator*=(const U &scalar) noexcept;

  /**
   * @brief Matrix-Vector multiplication (Matrix * column_vector).
   * @tparam U Numeric type of the vector.
   * @return A new column Vector of the common, promoted type.
   * @throws std::invalid_argument if vector is not a column vector or
   * dimensions do not match.
   */
  template <Numeric U>
  [[nodiscard]] auto operator*(const Vector<U> &other) const;

  /**
   * @brief Element-wise scalar division (Matrix / scalar).
   * @tparam U An arithmetic scalar type.
   * @return Matrix of the common, promoted type.
   */
  template <Numeric U>
  [[nodiscard]] auto operator/(const U &scalar) const noexcept;

  /**
   * @brief Element-wise scalar division assignment (Matrix / scalar).
   * @tparam U An arithmetic scalar type.
   * @attention This method doesn't cast the matrix if U is a broader type.
   * @return Matrix of the original matrix type.
   */
  template <Numeric U>
  Matrix<T> &operator/=(const U &scalar) noexcept;

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
      std::cout << std::endl;
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

  /** * @brief Internal-only accessor for high-performance kernels.
   * No bounds checking. Use only when indices are guaranteed to be valid.
   */
  [[nodiscard]] constexpr T &_unchecked_at(size_t row, size_t col) noexcept {
    return _data[(row * _cols) + col];
  }

  /** * @brief Internal-only accessor for high-performance kernels.
   * No bounds checking. Use only when indices are guaranteed to be valid.
   */
  [[nodiscard]] constexpr const T &_unchecked_at(size_t row,
                                                 size_t col) const noexcept {
    return _data[(row * _cols) + col];
  }

  /**
   * @brief Internal helper to invert the sign of all elements in-place.
   */
  void _invert_sign() {
    if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] = -_data[i];
      }
    } else {
      for (size_t i = 0; i < _data.size(); ++i) {
        _data[i] = -_data[i];
      }
    }
  }

  // Fallback matrix multiplication implementation
  template <typename A, typename B, typename C>
  void _fallback_matrix_multiply(const A *a_data, const B *b_data, C *c_data,
                                 size_t a_rows, size_t a_cols, size_t b_cols) const {
#pragma omp parallel for collapse(2) if (a_rows * b_cols > 10000)
    for (size_t ii = 0; ii < a_rows; ii += BLOCK_SIZE) {
      for (size_t jj = 0; jj < b_cols; jj += BLOCK_SIZE) {
        for (size_t kk = 0; kk < a_cols; kk += BLOCK_SIZE) {
          // Process block (your existing blocked implementation)
          const size_t i_end = std::min(ii + BLOCK_SIZE, a_rows);
          const size_t j_end = std::min(jj + BLOCK_SIZE, b_cols);
          const size_t k_end = std::min(kk + BLOCK_SIZE, a_cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t k = kk; k < k_end; ++k) {
              const C a_ik = static_cast<C>(a_data[(i * a_cols) + k]);
              const size_t b_offset = k * b_cols;
              const size_t c_offset = i * b_cols;

#pragma omp simd
              for (size_t j = jj; j < j_end; ++j) {
                c_data[c_offset + j] += a_ik * static_cast<C>(b_data[b_offset + j]);
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace maf::math

#include "Cholesky.hpp"
#include "MatrixCheckers.hpp"
#include "MatrixConstructors.hpp"
#include "MatrixFactories.hpp"
#include "MatrixMethods.hpp"
#include "MatrixOperators.hpp"
#include "PLU.hpp"

#endif
