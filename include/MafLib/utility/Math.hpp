#ifndef UTIL_MATH_H
#define UTIL_MATH_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::util {
#pragma mark concepts
//=============================================================================
// CONCEPTS
//=============================================================================
/** @brief Concept for numeric types. */
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

#pragma mark enums
//=============================================================================
// ENUMS
//=============================================================================
/** @brief Specifies if the vector behaves as a row or column vector. */
enum class Orientation : uint8 { ROW, COLUMN };

#pragma mark constants
//=============================================================================
// CONSTANTS
//=============================================================================
/*** @brief Constant used as OMP lower bound for linear algorithms. */
inline static constexpr size_t OMP_LINEAR_LIMIT = 500000UL;
/*** @brief Constant used as OMP lower bound for quadratic algorithms. */
inline static constexpr size_t OMP_QUADRATIC_LIMIT = 500UL * 500UL;
/*** @brief Constant used as OMP lower bound for cubic algorithms. */
inline static constexpr size_t OMP_CUBIC_LIMIT = 50UL * 50UL;
/** @brief Block size used in block algorithms. */
inline static constexpr uint8 BLOCK_SIZE = 16;
/** @brief Precision for floating point number string conversion. */
inline static constexpr uint8 FLOAT_PRECISION = 5;
/** @brief Epsilon value for floating point comparisons. */
inline static constexpr double EPSILON = 1e-6;

#pragma mark methods
//=============================================================================
// METHODS
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

/** @brief Parallel loop helper that uses OpenMP for large iterations and falls back to
 * a regular loop for smaller sizes.
 *  @tparam F A callable type that takes a size_t index as an argument.
 *  @param n The number of iterations to perform.
 *  @param fn The function to call for each iteration, which should accept a size_t
 * index.
 */
template <std::invocable<size_t> F>
inline void omp_loop(size_t n, const F &fn) {
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for default(none) shared(fn, n)
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  }
}

/** @brief Parallel loop helper that uses OpenMP for large iterations and falls back to
 * a regular loop for smaller sizes.
 *  @tparam F A callable type that takes a size_t index as an argument.
 *  @param n The number of iterations to perform.
 *  @param m The number of iterations to perform in the inner loop (used to determine if
 *  parallelization is needed).
 *  @param fn The function to call for each iteration, which should accept a size_t
 * index.
 */
template <std::invocable<size_t> F>
inline void omp_loop(size_t n, size_t m, const F &fn) {
  if (n * m >= OMP_QUADRATIC_LIMIT) {
#pragma omp parallel for default(none) shared(fn, n, m)
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  }
}

/** @brief Parallel reduction loop helper that uses OpenMP for large iterations and
 * falls back to a regular loop for smaller sizes.
 *  @details The callable accumulates into `acc` with `operator+`; OpenMP reduces
 * the per-thread partial results at the end of the parallel region.
 *  @tparam R Type of the reduction accumulator (must support `operator+`).
 *  @tparam F A callable type that takes a size_t index as an argument.
 *  @param acc Reference to the reduction accumulator; it holds the final sum after
 * the call.
 *  @param n The number of iterations to perform.
 *  @param fn The function to call for each iteration, which should accept a size_t
 * index and accumulate into `acc`.
 */
template <typename R, std::invocable<size_t> F>
inline void omp_loop_reduce(size_t n, R &acc, const F &fn) {
  if (n > OMP_LINEAR_LIMIT) {
#pragma omp parallel for default(none) shared(fn, n) reduction(+ : acc)
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
  }
}

}  // namespace maf::util

#endif
