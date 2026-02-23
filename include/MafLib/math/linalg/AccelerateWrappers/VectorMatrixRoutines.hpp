#ifndef ACCELERATE_VECTOR_MATRIX_ROUTINES_H
#define ACCELERATE_VECTOR_MATRIX_ROUTINES_H

#pragma once
#include "AccelerateWrapper.hpp"

namespace maf::math::acc {
//=============================================================================
// BLAS LEVEL 2 ROUTINES
//=============================================================================

// y = A * x (float)
[[nodiscard]] inline std::vector<float> sgemvm(size_t rows, size_t cols,
                                               const std::vector<float> &matrix,
                                               const std::vector<float> &vec) {
  std::vector<float> result(rows, 0.0F);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0F, matrix.data(), cols,
              vec.data(), 1, 0.0F, result.data(), 1);
  return result;
}

// y = A * x (double)
[[nodiscard]] inline std::vector<double> dgemvm(size_t rows, size_t cols,
                                                const std::vector<double> &matrix,
                                                const std::vector<double> &vec) {
  std::vector<double> result(rows, 0.0);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0F, matrix.data(), cols,
              vec.data(), 1, 0.0, result.data(), 1);
  return result;
}

// y = x * A (float)
[[nodiscard]] inline std::vector<float> sgevmm(size_t rows, size_t cols,
                                               const std::vector<float> &matrix,
                                               const std::vector<float> &vec) {
  std::vector<float> result(cols, 0.0F);
  cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols, 1.0F, matrix.data(), cols,
              vec.data(), 1, 0.0F, result.data(), 1);
  return result;
}

// y = x * A (double)
[[nodiscard]] inline std::vector<double> dgevmm(size_t rows, size_t cols,
                                                const std::vector<double> &matrix,
                                                const std::vector<double> &vec) {
  std::vector<double> result(cols, 0.0F);
  cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, 1.0F, matrix.data(), cols,
              vec.data(), 1, 0.0F, result.data(), 1);
  return result;
}

}  // namespace maf::math::acc
#endif
