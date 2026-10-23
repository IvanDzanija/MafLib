#ifndef ACCELERATE_VECTOR_ROUTINES_H
#define ACCELERATE_VECTOR_ROUTINES_H

#pragma once
#include "AccelerateWrapper.hpp"

/**
 * @file VectorRoutines.hpp
 * @brief BLAS level 1 vector routines.
 *
 * Identical counterparts are provided by maf::math::acc (Accelerate, Apple)
 * and maf::math::oblas (OpenBLAS, Linux). The dot products are pointer-level;
 * they own no containers and perform no type conversions.
 */

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
namespace maf::math::acc {
//=============================================================================
// BLAS LEVEL 1 ROUTINES
//=============================================================================

// R = Alpha * X + Y (double)
[[nodiscard]] inline std::vector<double> daxpy(const std::vector<double> &x,
                                               const std::vector<double> &y,
                                               const double alpha = 1) {
  std::vector<double> result(y);
  cblas_daxpy(x.size(), alpha, x.data(), 1, result.data(), 1);
  return result;
}

// Y = Alpha * X + Y (double)
inline void inplace_daxpy(const std::vector<double> &x, std::vector<double> &y,
                          const double alpha = 1) {
  cblas_daxpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

// R = Alpha * X + Y (float)
[[nodiscard]] inline std::vector<float> saxpy(const std::vector<float> &x,
                                              const std::vector<float> &y,
                                              const float alpha = 1) {
  std::vector<float> result(y);
  cblas_saxpy(x.size(), alpha, x.data(), 1, result.data(), 1);
  return result;
}

// Y = Alpha * X + Y (float)
inline void inplace_saxpy(const std::vector<float> &x, std::vector<float> &y,
                          const float alpha = 1) {
  cblas_saxpy(x.size(), alpha, x.data(), 1, y.data(), 1);
}

// R = X + Y (int32)
[[nodiscard]] inline std::vector<int> vaddi(const std::vector<int> &x,
                                            const std::vector<int> &y) {
  std::vector<int> result(y);
  vDSP_vaddi(x.data(), 1, y.data(), 1, result.data(), 1, x.size());
  return result;
}

// Y = X + Y (int32)
inline void inplace_vaddi(const std::vector<int> &x, std::vector<int> &y) {
  vDSP_vaddi(x.data(), 1, y.data(), 1, y.data(), 1, x.size());
}

// R = X + Alpha (float)
[[nodiscard]] inline std::vector<float> vsadds(const std::vector<float> &x,
                                               const float alpha = 1) {
  std::vector<float> result(x.size());
  vDSP_vsadd(x.data(), 1, &alpha, result.data(), 1, x.size());
  return result;
}

// X = Alpha + X (float)
inline void inplace_vsadds(std::vector<float> &x, const float alpha = 1) {
  vDSP_vsadd(x.data(), 1, &alpha, x.data(), 1, x.size());
}

// R = X + Alpha (double)
[[nodiscard]] inline std::vector<double> vsaddd(const std::vector<double> &x,
                                                const double alpha = 1) {
  std::vector<double> result(x.size());
  vDSP_vsaddD(x.data(), 1, &alpha, result.data(), 1, x.size());
  return result;
}

// X = Alpha + X (double)
inline void inplace_vsaddd(std::vector<double> &x, const double alpha = 1) {
  vDSP_vsaddD(x.data(), 1, &alpha, x.data(), 1, x.size());
}

// R = X + Alpha (int32)
[[nodiscard]] inline std::vector<int32> vsaddi(const std::vector<int32> &x,
                                               const int32 alpha = 1) {
  std::vector<int32> result(x.size());
  vDSP_vsaddi(x.data(), 1, &alpha, result.data(), 1, x.size());
  return result;
}

// X = Alpha + X (int32)
inline void inplace_vsaddi(std::vector<int32> &x, const int32 alpha = 1) {
  vDSP_vsaddi(x.data(), 1, &alpha, x.data(), 1, x.size());
}

/** @brief Dot product of x and y (float).
 *
 *  @param x Pointer to the first vector.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to the second vector.
 *  @param incy Stride between elements of y.
 */
[[nodiscard]] inline float sdot(size_t n, const float *x, size_t incx, const float *y,
                                size_t incy) {
  return cblas_sdot(static_cast<int>(n), x, static_cast<int>(incx), y,
                    static_cast<int>(incy));
}

/** @brief Dot product of x and y (double). */
[[nodiscard]] inline double ddot(size_t n, const double *x, size_t incx,
                                 const double *y, size_t incy) {
  return cblas_ddot(static_cast<int>(n), x, static_cast<int>(incx), y,
                    static_cast<int>(incy));
}

}  // namespace maf::math::acc
#elifdef OPENBLAS_AVAILABLE
namespace maf::math::oblas {

/** @brief Dot product of x and y (float).
 *
 *  @param x Pointer to the first vector.
 *  @param incx Stride between elements of x.
 *  @param y Pointer to the second vector.
 *  @param incy Stride between elements of y.
 */
[[nodiscard]] inline float sdot(size_t n, const float *x, size_t incx, const float *y,
                                size_t incy) {
  return cblas_sdot(static_cast<int>(n), x, static_cast<int>(incx), y,
                    static_cast<int>(incy));
}

/** @brief Dot product of x and y (double). */
[[nodiscard]] inline double ddot(size_t n, const double *x, size_t incx,
                                 const double *y, size_t incy) {
  return cblas_ddot(static_cast<int>(n), x, static_cast<int>(incx), y,
                    static_cast<int>(incy));
}

}  // namespace maf::math::oblas
#endif

#endif  // ACCELERATE_VECTOR_ROUTINES_H