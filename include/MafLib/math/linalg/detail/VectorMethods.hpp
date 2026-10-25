#ifndef VECTOR_METHODS_H
#define VECTOR_METHODS_H
#pragma once
#include "MafLib/math/linalg/detail/Vector.hpp"

namespace maf::math {
// Inplace fill
template <Numeric T>
void Vector<T>::fill(T value) noexcept {
    if (_data.size() > OMP_LINEAR_LIMIT) {
        #pragma omp parallel for
        for (size_t i = 0; i < _data.size(); ++i) {
            _data[i] = value;
        }
    } else {
        for (size_t i = 0; i < _data.size(); ++i) {
            _data[i] = value;
        }
    }
}

// L2 Norm
template <Numeric T>
[[nodiscard]] T Vector<T>::norm() const noexcept {
    T sum = 0.0;
    size_t n = _data.size();
    if (n > OMP_LINEAR_LIMIT) {
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < n; ++i) {
            sum += _data[i] * _data[i];
        }
    } else {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            sum += _data[i] * _data[i];
        }
    }
    return std::sqrt(sum);
}

// Inplace normalize with L2 norm
template <Numeric T>
void Vector<T>::normalize() {
    T norm = this->norm();
    if (is_close(norm, 0)) {
        throw std::invalid_argument("Vector norm is close to 0!");
    }

    T norm_inv = T(1) / norm;
    size_t n = _data.size();

    if (n > OMP_LINEAR_LIMIT) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            _data[i] *= norm_inv;
        }
    } else {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            _data[i] *= norm_inv;
        }
    }
}

// Inplace transpose
template <Numeric T>
void Vector<T>::transpose() noexcept {
    _orientation = (_orientation == COLUMN) ? ROW : COLUMN;
}

// Creates new transposed vector
template <Numeric T>
[[nodiscard]] Vector<T> Vector<T>::transposed() const noexcept {
    Orientation new_orientation = (_orientation == COLUMN) ? ROW : COLUMN;
    return Vector<T>(this->size(), _data, new_orientation);
}

}  // namespace maf::math

#endif
