#ifndef MATRIX_FACTORIES_H
#define MATRIX_FACTORIES_H
#pragma once
#include "Matrix.hpp"

/**
 * @file MatrixFactories.hpp
 * @brief Contains various implementations for producing specific matrices.
 *
 * This file is intended to be included at the *end* of Matrix.hpp and
 * should not be included directly anywhere else.
 */
namespace maf::math {
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
#pragma omp parallel for if (n > 256)
  for (size_t i = 0; i < n; ++i) {
    const size_t j = perm.at(i);
    result[i][j] = static_cast<T>(1);
  }
  return result;
}

}  // namespace maf::math

#endif
