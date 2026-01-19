#ifndef UTIL_MATH_H
#define UTIL_MATH_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::util {
inline constexpr double EPSILON = 1e-6;
template <typename T>
[[nodiscard]] bool is_close(T v1, T v2, double epsilon = EPSILON) {
    return std::abs(v1 - v2) < epsilon;
}

template <typename T, typename U>
[[nodiscard]] bool is_close(T v1, U v2, double epsilon = EPSILON) {
    using R = std::common_type_t<T, U>;
    return std::abs(static_cast<R>(v1) - static_cast<R>(v2)) < epsilon;
}

}  // namespace maf::util

#endif
