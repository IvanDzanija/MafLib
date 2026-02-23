#ifndef CHOLESKY_H
#define CHOLESKY_H
#pragma once
#include "Matrix.hpp"

namespace maf::math {
namespace detail {
/**
 * @brief Internal implementation of Cholesky decomposition.
 *
 * Computes L where A = LL^T for Hermitian symmetric positive definite matrix A.
 * Uses blocked algorithm with OpenMP parallelization.
 */
template <std::floating_point T>
[[nodiscard]] Matrix<T> _cholesky(const Matrix<T> &matrix) {
  if (!matrix.is_symmetric()) {
    throw std::invalid_argument(
        "Matrix must be symmetric to try Cholesky decomposition!");
  }

  size_t n = matrix.row_count();
  Matrix<T> L(n, n);

  for (size_t jj = 0; jj < n; jj += BLOCK_SIZE) {
    const size_t j_end = std::min(jj + BLOCK_SIZE, n);
    for (size_t j = jj; j < j_end; ++j) {
      T sum = 0;
      auto L_row_j = L.row_span(j);

#pragma omp simd
      for (size_t k = 0; k < j; ++k) {
        sum += L_row_j[k] * L_row_j[k];
      }

      T diag_val = matrix.at(j, j) - sum;
      if (diag_val <= 0) {
        throw std::invalid_argument("Matrix is not positive definite!");
      }
      L_row_j[j] = std::sqrt(diag_val);

      for (size_t i = j + 1; i < j_end; ++i) {
        T sum_i = 0;
        auto L_row_i = L.row_span(i);

#pragma omp simd
        for (size_t k = 0; k < j; ++k) {
          sum_i += L_row_i[k] * L_row_j[k];
        }
        L_row_i[j] = (matrix.at(i, j) - sum_i) / L_row_j[j];
      }
    }

#pragma omp parallel for if (n > 1000)
    for (size_t ii = j_end; ii < n; ii += BLOCK_SIZE) {
      const size_t i_end = std::min(ii + BLOCK_SIZE, n);

      for (size_t i = ii; i < i_end; ++i) {
        auto L_row_i = L.row_span(i);

        for (size_t j = jj; j < j_end; ++j) {
          T sum = 0;
          auto L_row_j = L.row_span(j);

#pragma omp simd
          for (size_t k = 0; k < j; ++k) {
            sum += L_row_i[k] * L_row_j[k];
          }
          L_row_i[j] = (matrix.at(i, j) - sum) / L_row_j[j];
        }
      }
    }
  }

  return L;
}

}  // namespace detail

/**
 * @brief Computes the Cholesky decomposition of a symmetric positive
 * definite matrix.
 *
 * This function computes the Cholesky factorization (A = LL^T) for a
 * given square, symmetric, positive definite matrix A, where:
 * - A is the input matrix.
 * - L is a lower triangular matrix with positive diagonal entries.
 * - L^T is the conjugate transpose of L (which is just the transpose for
 * real matrices).
 *
 * A matrix has a Cholesky decomposition if and only if it is symmetric
 * and positive definite. This function checks for symmetry first. It
 * then detects non-positive-definiteness during the computation.
 *
 * The decomposition is computed using a blocked Cholesky-Crout algorithm
 * for improved cache performance. The implementation is parallelized
 * using OpenMP (for both multi-threading and SIMD vectorization)
 * for further performance gains.
 *
 * More information:
 * https://en.wikipedia.org/wiki/Cholesky_decomposition
 *
 * @tparam T The floating point type of the matrix elements (e.g., float,
 * double).
 * @param matrix The const reference to the square, symmetric, positive
 * definite input matrix (A) to decompose.
 * @return (Matrix<T>) The lower triangular matrix (L).
 *
 * @throws std::invalid_argument if the input matrix is not symmetric,
 * or if it is not positive definite (detected during factorization).
 *
 * @version 1.0 (Blocked & Parallelized)
 * @since 2025
 */
template <typename ResultType = void, Numeric T>
[[nodiscard]] auto cholesky(const Matrix<T> &matrix) {
  using TargetType =
      std::conditional_t<std::is_same_v<ResultType, void>,
                         std::conditional_t<std::is_floating_point_v<T>, T, double>,
                         ResultType>;

  static_assert(std::is_floating_point_v<TargetType>,
                "Cholesky result type must be floating point!");

  if constexpr (std::is_same_v<TargetType, T>) {
    return detail::_cholesky(matrix);
  } else {
    auto converted = matrix.template cast<TargetType>();
    return detail::_cholesky(converted);
  }
}

}  // namespace maf::math
#endif
