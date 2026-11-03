#ifndef MAF_EIGEN_H
#define MAF_EIGEN_H

#include "MafLib/math/linalg/detail/BlasWrappers/LapackRoutines.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

#pragma once

namespace maf::math {
template <std::floating_point T>
struct EigenPair {
  Vector<T> eigen_vector;
  T eigen_value;
};

template <std::floating_point T>
using EigenResult = std::vector<EigenPair<T>>;

namespace detail {}

template <typename ResultType = void, Numeric T>
[[nodiscard]] auto eigen_decomposition(const Matrix<T> &matrix) {
  using TargetType =
      std::conditional_t<std::is_same_v<ResultType, void>,
                         std::conditional_t<std::is_floating_point_v<T>, T, double>,
                         ResultType>;

  static_assert(std::is_floating_point_v<TargetType>,
                "Eigen result type must be floating point!");

#ifdef BLAS_AVAILABLE
  if constexpr (std::is_same_v<TargetType, float> ||
                std::is_same_v<TargetType, double>) {
    return blas::eigen_decompose<TargetType>(matrix);
  } else
#endif
  {
    if constexpr (std::is_same_v<TargetType, T>) {
      //      return detail::_eigen_decomposition(matrix);
    } else {
      auto converted = matrix.template cast<TargetType>();
      //     return detail::_eigen_decomposition(converted);
    }
  }
}

}  // namespace maf::math

#endif
