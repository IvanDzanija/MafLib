#ifndef VECTOR_CHECKERS_H
#define VECTOR_CHECKERS_H
#pragma once
#include "Vector.hpp"

/**
 * @file VectorCheckers.hpp
 * @brief Contains implementations for the checker methods of the Vector<T>
 * class.
 *
 * This file is intended to be included at the *end* of Vector.hpp and
 * should not be included directly anywhere else.
 */
namespace maf::math {
// Checks if vector is null vector
template <Numeric T>
[[nodiscard]] bool Vector<T>::is_null() const noexcept {
    for (const T& val : _data) {
        if (!is_close(val, 0)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Checks if two Vectors are element-wise equal within a tolerance.
 * @tparam T Numeric type of the first Vector.
 * @tparam U Numeric type of the second Vector.
 * @param eps The absolute tolerance for equality.
 * @return true if sizes match and all elements are "close".
 */
template <Numeric T, Numeric U>
[[nodiscard]] constexpr bool loosely_equal(const Vector<T>& first,
                                           const Vector<U>& second,
                                           double eps = 1e-6) {
    size_t n = first.size();
    if (first.size() != second.size()) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!is_close(first[i], second[i], eps)) {
            return false;
        }
    }
    return true;
}

}  // namespace maf::math

#endif
