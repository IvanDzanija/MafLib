#ifndef UTIL_MATH_H
#define UTIL_MATH_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::util {
//=============================================================================
// CONCEPTS
//=============================================================================
/** @brief Concept for numeric types. */
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

//=============================================================================
// ENUMS
//=============================================================================
/** @brief Specifies if the vector behaves as a row or column vector. */
enum Orientation : uint8 { ROW, COLUMN };

//=============================================================================
// CONSTANTS
//=============================================================================
/*** @brief Constant used as OMP lower bound for linear algorithms. */
constexpr static size_t OMP_LINEAR_LIMIT = 500000;
/*** @brief Constant used as OMP lower bound for quadratic algorithms. */
constexpr static size_t OMP_QUADRATIC_LIMIT = 500 * 500;
/*** @brief Constant used as OMP lower bound for cubic algorithms. */
constexpr static size_t OMP_CUBIC_LIMIT = 50 * 50;
/** @brief Block size used in block algorithms. */
inline static constexpr uint8 BLOCK_SIZE = 64;
/** @brief Precision for floating point number string conversion. */
inline static constexpr uint8 FLOAT_PRECISION = 5;
/** @brief Epsilon value for floating point comparisons. */
inline static constexpr double EPSILON = 1e-6;

//=============================================================================
// Methods
//=============================================================================
/** @brief Check if two numeric values of potentially different types are close to each
 * other within a given epsilon.
 *  @tparam T Numeric type of the first value.
 *  @tparam U Numeric type of the second value.
 *  @param v1 First value.
 *  @param v2 Second value.
 *  @param epsilon Tolerance for closeness check (default is EPSILON).
 *  @return True if the values are close, false otherwise.
 */
template <typename T, typename U>
[[nodiscard]] bool is_close(T v1, U v2, double epsilon = EPSILON) {
  using R = std::common_type_t<T, U>;
  return std::abs(static_cast<R>(v1) - static_cast<R>(v2)) < epsilon;
}

}  // namespace maf::util

#endif
