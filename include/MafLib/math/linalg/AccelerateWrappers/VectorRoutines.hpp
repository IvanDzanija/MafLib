#ifndef ACCELERATE_VECTOR_ROUTINES_H
#define ACCELERATE_VECTOR_ROUTINES_H

#pragma once
#include "AccelerateWrapper.hpp"

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

// Dot product (double)
[[nodiscard]] inline double ddot(const std::vector<double> &x,
                                 const std::vector<double> &y) {
  return cblas_ddot(x.size(), x.data(), 1, y.data(), 1);
}

// Dot product (float)
[[nodiscard]] inline float sdot(const std::vector<float> &x,
                                const std::vector<float> &y) {
  return cblas_sdot(x.size(), x.data(), 1, y.data(), 1);
}

}  // namespace maf::math::acc
#endif
