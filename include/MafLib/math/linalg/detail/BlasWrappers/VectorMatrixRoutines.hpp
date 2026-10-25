#ifndef VECTOR_MATRIX_ROUTINES_H
#define VECTOR_MATRIX_ROUTINES_H

#pragma once
#include "MafLib/math/linalg/detail/BlasWrappers/BlasWrapper.hpp"

/**
 * @file VectorMatrixRoutines.hpp
 * @brief Pointer-level BLAS level 2 matrix-vector routines.
 *
 * All routines take raw pointers, dimensions and strides; they own no
 * containers and perform no type conversions. Identical counterparts are
 * provided by maf::math::acc (Accelerate, Apple) and maf::math::oblas
 * (OpenBLAS, Linux).
 */

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
namespace maf::math::acc {
//=============================================================================
// BLAS LEVEL 2 ROUTINES
//=============================================================================

/** @brief y = A * x (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param a Pointer to a (rows x cols) row-major matrix (dense part of
 *  possibly larger buffer).
 *  @param x Pointer to a vector of cols elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of rows elements (overwritten).
 *  @param lda Leading dimension (row stride) of A; equals cols for a dense
 *  matrix.
 */
inline void sgemvm(size_t rows, size_t cols, const float *a, const float *x,
                   size_t incx, float *y, size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows),
              static_cast<int>(cols), 1.0F, a, static_cast<int>(lda), x,
              static_cast<int>(incx), 0.0F, y, 1);
}

/** @brief y = A * x (double). */
inline void dgemvm(size_t rows, size_t cols, const double *a, const double *x,
                   size_t incx, double *y, size_t lda) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows),
              static_cast<int>(cols), 1.0, a, static_cast<int>(lda), x,
              static_cast<int>(incx), 0.0, y, 1);
}

/** @brief y = x^T * A (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param a Pointer to a (rows x cols) row-major matrix (dense part of
 *  possibly larger buffer).
 *  @param x Pointer to a vector of rows elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of cols elements (overwritten).
 *  @param lda Leading dimension (row stride) of A; equals cols for a dense
 *  matrix.
 */
inline void sgevmm(size_t rows, size_t cols, const float *a, const float *x,
                   size_t incx, float *y, size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasTrans, static_cast<int>(rows), static_cast<int>(cols),
              1.0F, a, static_cast<int>(lda), x, static_cast<int>(incx), 0.0F, y, 1);
}

/** @brief y = x^T * A (double). */
inline void dgevmm(size_t rows, size_t cols, const double *a, const double *x,
                   size_t incx, double *y, size_t lda) {
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(rows), static_cast<int>(cols),
              1.0, a, static_cast<int>(lda), x, static_cast<int>(incx), 0.0, y, 1);
}

/** @brief A += alpha * x y^T (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param alpha Scalar multiplier.
 *  @param x Pointer to a vector of rows elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of cols elements.
 *  @param incy Stride between elements of y.
 *  @param a Pointer to a (rows x cols) row-major matrix (dense part of
 *  possibly larger buffer, updated in-place).
 *  @param ldc Leading dimension (row stride) of A; equals cols for a dense
 *  matrix.
 */
inline void sger(size_t rows, size_t cols, float alpha, const float *x, size_t incx,
                 const float *y, size_t incy, float *a, size_t ldc) {
  cblas_sger(CblasRowMajor, static_cast<int>(rows), static_cast<int>(cols), alpha, x,
             static_cast<int>(incx), y, static_cast<int>(incy), a,
             static_cast<int>(ldc));
}

/** @brief A += alpha * x y^T (double). */
inline void dger(size_t rows, size_t cols, double alpha, const double *x, size_t incx,
                 const double *y, size_t incy, double *a, size_t ldc) {
  cblas_dger(CblasRowMajor, static_cast<int>(rows), static_cast<int>(cols), alpha, x,
             static_cast<int>(incx), y, static_cast<int>(incy), a,
             static_cast<int>(ldc));
}

}  // namespace maf::math::acc
#elifdef OPENBLAS_AVAILABLE
namespace maf::math::oblas {
//=============================================================================
// BLAS LEVEL 2 ROUTINES
//=============================================================================

/** @brief y = A * x (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param a Pointer to a (rows x cols) row-major matrix.
 *  @param x Pointer to a vector of cols elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of rows elements (overwritten).
 */
inline void sgemvm(size_t rows, size_t cols, const float *a, const float *x,
                   size_t incx, float *y, size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows),
              static_cast<int>(cols), 1.0F, a, static_cast<int>(lda), x,
              static_cast<int>(incx), 0.0F, y, 1);
}

/** @brief y = A * x (double). */
inline void dgemvm(size_t rows, size_t cols, const double *a, const double *x,
                   size_t incx, double *y, size_t lda) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows),
              static_cast<int>(cols), 1.0, a, static_cast<int>(lda), x,
              static_cast<int>(incx), 0.0, y, 1);
}

/** @brief y = x^T * A (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param a Pointer to a (rows x cols) row-major matrix.
 *  @param x Pointer to a vector of rows elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of cols elements (overwritten).
 */
inline void sgevmm(size_t rows, size_t cols, const float *a, const float *x,
                   size_t incx, float *y, size_t lda) {
  cblas_sgemv(CblasRowMajor, CblasTrans, static_cast<int>(rows), static_cast<int>(cols),
              1.0F, a, static_cast<int>(lda), x, static_cast<int>(incx), 0.0F, y, 1);
}

/** @brief y = x^T * A (double). */
inline void dgevmm(size_t rows, size_t cols, const double *a, const double *x,
                   size_t incx, double *y, size_t lda) {
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(rows), static_cast<int>(cols),
              1.0, a, static_cast<int>(lda), x, static_cast<int>(incx), 0.0, y, 1);
}

/** @brief A += alpha * x y^T (float).
 *
 *  @param rows Number of rows of A.
 *  @param cols Number of columns of A.
 *  @param alpha Scalar multiplier.
 *  @param x Pointer to a vector of rows elements.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to a vector of cols elements.
 *  @param incy Stride between elements of y.
 *  @param a Pointer to a (rows x cols) row-major matrix (updated in-place).
 */
inline void sger(size_t rows, size_t cols, float alpha, const float *x, size_t incx,
                 const float *y, size_t incy, float *a, size_t ldc) {
  cblas_sger(CblasRowMajor, static_cast<int>(rows), static_cast<int>(cols), alpha, x,
             static_cast<int>(incx), y, static_cast<int>(incy), a,
             static_cast<int>(ldc));
}

/** @brief A += alpha * x y^T (double). */
inline void dger(size_t rows, size_t cols, double alpha, const double *x, size_t incx,
                 const double *y, size_t incy, double *a, size_t ldc) {
  cblas_dger(CblasRowMajor, static_cast<int>(rows), static_cast<int>(cols), alpha, x,
             static_cast<int>(incx), y, static_cast<int>(incy), a,
             static_cast<int>(ldc));
}

}  // namespace maf::math::oblas
#endif

#endif  // ACCELERATE_VECTOR_MATRIX_ROUTINES_H
