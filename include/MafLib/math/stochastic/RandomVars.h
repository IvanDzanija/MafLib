#ifndef RANDOMVARS_H
#define RANDOMVARS_H
#pragma once
#include "MafLib/math/linalg/LinAlg.hpp"

namespace maf::math {

/// Creates a vector of correlated random variables from a vector of independant
/// random variables based on Cholesky decomposition of covariance matrix.
/// https://en.wikipedia.org/wiki/Cholesky_decomposition#Monte_Carlo_simulation
/// @return Vector of common promoted type.
template <typename T, typename U>
auto correlated_shocks(const Matrix<U> &L, const Vector<T> &Z) {
  using R = std::common_type_t<T, U>;

  size_t n = Z.size();
  Vector<R> Y(n, Vector<R>::COLUMN);

  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k <= i; ++k) {
      Y.at(i) += L.at(i, k) * Z.at(k);
    }
  }
  return Y;
}

}  // namespace maf::math
#endif
