#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "Optimizer.hpp"

namespace maf::math {
/**
 * @brief Class for finding fixed points of a function using the Fixed Point Iteration
 * method.
 */
template <std::floating_point T>
class FixedPoint : public Optimizer<T> {
 public:
  FixedPoint() = delete;  // Make sure function is provided.
  /**
   * @brief Constructor for FixedPoint class.
   * @param function The function for which to find the fixed point.
   * @param start The initial guess for the fixed point.
   */
  explicit FixedPoint(const std::function<T(T)> &function, T start)
      : Optimizer<T>(function), _start(start) {}

  /**
   * @brief Get the initial guess for the fixed point.
   * @return The initial guess.
   */
  [[nodiscard]] T get_start() const { return _start; }

  /**
   * @brief Set the initial guess for the fixed point.
   * @param start The initial guess to set.
   */
  void set_start(T start) { _start = start; }

  /**
   * @brief Find the fixed point using the Fixed Point Iteration method.
   * @param tolerance The tolerance for convergence.
   * @param max_iterations The maximum number of iterations to perform.
   * @return An OptimizerResult containing the solution, error, and optionally an error
   * message.
   */
  [[nodiscard]] OptimizerResult<T> solve(T tolerance = static_cast<T>(1e-7),
                                         int32 max_iterations = 1000) override {
    T x = _start;
    T error = _get_error(x);

    while (error > tolerance && max_iterations-- > 0) {
      x = this->_function(x);
      error = _get_error(x);

      if (std::isinf(x) || std::isnan(x)) {
        return {.solution = x,
                .error = error,
                .error_message = "Method diverged (NaN or Infinity)."};
      }
    }

    if (error > tolerance) {
      return {.solution = x,
              .error = error,
              .error_message = "Method did not converge within the iteration limit."};
    }

    return {.solution = x, .error = error, .error_message = std::nullopt};
  }

 private:
  /** @brief The initial guess for the fixed point. */
  T _start;

  /**
   * @brief Calculate the error at the given point.
   * @param current_point The current point at which to evaluate the error.
   * @return The absolute difference between f(current_point) and current_point.
   */
  T _get_error(T current_point) {
    return std::abs(this->_function(current_point) - current_point);
  }
};
}  // namespace maf::math
#endif
