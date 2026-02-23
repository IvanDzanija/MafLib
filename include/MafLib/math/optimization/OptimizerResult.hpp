#ifndef OPTIMIZER_RESULT_HPP
#define OPTIMIZER_RESULT_HPP

#pragma once
#include <MafLib/main/GlobalHeader.hpp>
namespace maf::math {
/**
 * @brief Struct to hold the result of a solver operation.
 */
template <typename T>
struct OptimizerResult {
  /** @brief The solution found by the solver. */
  T solution;

  /** @brief The error associated with the solution.
   * @attention This is meaningful only if the solver succeeded.
   */
  double error;

  /** @brief An optional error message if the solver failed. */
  std::optional<std::string_view> error_message;

  /** @brief Check if the solver was successful.
   * @return True if the solver succeeded, false otherwise.
   */
  explicit operator bool() const { return !error_message.has_value(); }
};
}  // namespace maf::math
#endif
