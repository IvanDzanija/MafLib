#ifndef STATISTICS_H
#define STATISTICS_H
#pragma once
#include "MafLib/math/linalg/Vector.hpp"

namespace maf::math {

/// Calculates the unbiased estimator of expected value aka mean.
template <typename T>
double mean(const Vector<T> &data) {
    double sum = 0.;
    for (const T &value : data) {
        sum += value;
    }
    return sum / data.size();
}

/// Calculates the unbiased estimator of sample covariance.
template <typename T>
double covariance(const Vector<T> &x, const Vector<T> &y) {
    size_t n = x.size();
    if (n != y.size()) {
        throw std::invalid_argument("Dimension mismatch.");
    }

    double mean_x = mean(x);
    double mean_y = mean(y);
    double cov = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cov += (x.at(i) - mean_x) * (y.at(i) - mean_y);
    }
    return cov / (n - 1);
}

/// Calculates the unbiased estimar of sample variance.
/// Use if means are already cached.
template <typename T>
double covariance(const std::vector<T> &x,
                  T mean_x,
                  const std::vector<T> &y,
                  T mean_y) {
    size_t n = x.size();
    if (n != y.size()) {
        throw std::invalid_argument("Dimension mismatch.");
    }
    double cov = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cov += (x.at(i) - mean_x) * (y.at(i) - mean_y);
    }
    return cov / (n - 1);
}

}  // namespace maf::math

#endif
