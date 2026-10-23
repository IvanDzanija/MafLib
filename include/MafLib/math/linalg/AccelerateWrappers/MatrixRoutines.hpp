#ifndef ACCELERATE_MATRIX_ROUTINES_H
#define ACCELERATE_MATRIX_ROUTINES_H

#pragma once
#include "AccelerateWrapper.hpp"

/**
 * @file MatrixRoutines.hpp
 * @brief Pointer-level BLAS level 3 matrix routines.
 *
 * All routines take raw pointers and dimensions; they own no containers and
 * perform no type conversions. Identical counterparts are provided by
 * maf::math::acc (Accelerate, Apple) and maf::math::oblas (OpenBLAS, Linux).
 */

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
namespace maf::math::acc {

/** @brief C = A * B (row-major, CblasNoTrans, alpha = 1, beta = 0).
 *
 *  @param a Pointer to an (m x k) row-major matrix.
 *  @param b Pointer to a (k x n) row-major matrix.
 *  @param c Pointer to an (m x n) row-major result matrix.
 */
inline void sgemm(size_t m, size_t n, size_t k, const float *a, const float *b,
                  float *c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), 1.0F, a, static_cast<int>(k), b,
              static_cast<int>(n), 0.0F, c, static_cast<int>(n));
}

/** @brief C = A * B (row-major, CblasNoTrans, alpha = 1, beta = 0). */
inline void dgemm(size_t m, size_t n, size_t k, const double *a, const double *b,
                  double *c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k), b,
              static_cast<int>(n), 0.0, c, static_cast<int>(n));
}

/** @brief Transposes a row-major (rows x cols) matrix into @p transposed.
 *  @param a Pointer to a (rows x cols) row-major matrix.
 *  @param transposed Pointer to a (cols x rows) destination buffer.
 */
inline void smatcopy(const float *a, float *transposed, size_t rows, size_t cols) {
  vDSP_mtrans(a, 1, transposed, 1, cols, rows);
}

/** @brief Transposes a row-major (rows x cols) matrix into @p transposed. */
inline void dmatcopy(const double *a, double *transposed, size_t rows, size_t cols) {
  vDSP_mtransD(a, 1, transposed, 1, cols, rows);
}

}  // namespace maf::math::acc
#elifdef OPENBLAS_AVAILABLE
namespace maf::math::oblas {

/** @brief C = A * B (row-major, CblasNoTrans, alpha = 1, beta = 0).
 *
 *  @param a Pointer to an (m x k) row-major matrix.
 *  @param b Pointer to a (k x n) row-major matrix.
 *  @param c Pointer to an (m x n) row-major result matrix.
 */
inline void sgemm(size_t m, size_t n, size_t k, const float *a, const float *b,
                  float *c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), 1.0F, a, static_cast<int>(k), b,
              static_cast<int>(n), 0.0F, c, static_cast<int>(n));
}

/** @brief C = A * B (row-major, CblasNoTrans, alpha = 1, beta = 0). */
inline void dgemm(size_t m, size_t n, size_t k, const double *a, const double *b,
                  double *c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k), b,
              static_cast<int>(n), 0.0, c, static_cast<int>(n));
}

/** @brief Transposes a row-major (rows x cols) matrix into @p transposed.
 *  @param a Pointer to a (rows x cols) row-major matrix.
 *  @param transposed Pointer to a (cols x rows) destination buffer.
 */
inline void smatcopy(const float *a, float *transposed, size_t rows, size_t cols) {
  cblas_somatcopy(CblasRowMajor, CblasTrans, static_cast<int>(rows),
                  static_cast<int>(cols), 1.0F, a, static_cast<int>(cols), transposed,
                  static_cast<int>(rows));
}

/** @brief Transposes a row-major (rows x cols) matrix into @p transposed. */
inline void dmatcopy(const double *a, double *transposed, size_t rows, size_t cols) {
  cblas_domatcopy(CblasRowMajor, CblasTrans, static_cast<int>(rows),
                  static_cast<int>(cols), 1.0, a, static_cast<int>(cols), transposed,
                  static_cast<int>(rows));
}

}  // namespace maf::math::oblas
#endif

#endif  // ACCELERATE_MATRIX_ROUTINES_H