#ifndef MATRIX_CHECKERS_H
#define MATRIX_CHECKERS_H
#pragma once
#include "Matrix.hpp"

/**
 * @file MatrixCheckers.hpp
 * @brief Contains implementations for the checker methods of the Matrix<T>
 * class.
 *
 * This file is intended to be included at the *end* of Matrix.hpp and
 * should not be included directly anywhere else.
 */
namespace maf::math {
// Checks if matrix is square.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_square() const {
    return _rows == _cols;
}

// Checks if matrix is symmetric.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_symmetric() const {
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

// Checks if matrix is upper triangular.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_upper_triangular() const {
    if (!is_square()) {
        return false;
    }

    static T ZERO = static_cast<T>(0);

    for (size_t i = 1; i < _rows; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (!is_close(at(i, j), ZERO)) {
                return false;
            }
        }
    }
    return true;
}

// Checks if matrix is lower triangular.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_lower_triangular() const {
    if (!is_square()) {
        return false;
    }

    static T ZERO = static_cast<T>(0);

    for (size_t i = 0; i < _rows - 1; ++i) {
        for (size_t j = i + 1; j < _cols; ++j) {
            if (!is_close(at(i, j), ZERO)) {
                return false;
            }
        }
    }
    return true;
}

// Checks if matrix is diagonal.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_diagonal() const {
    if (!is_square()) {
        return false;
    }
    // TODO: We can skip the checks for is square in both?
    return is_upper_triangular() && is_lower_triangular();
}

// Checks if matrix is singular.
template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_singular() const {
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

template <Numeric T>
[[nodiscard]] constexpr bool Matrix<T>::is_positive_definite() const {
    try {
        cholesky(*this);
        return true;
    } catch (std::exception e) {
        return false;
    }
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
                                           double eps = 1e-6) {
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

}  // namespace maf::math

#endif
