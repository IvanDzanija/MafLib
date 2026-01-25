#ifndef SOLVER_RESULT_HPP
#define SOLVER_RESULT_HPP

#pragma once
#include <MafLib/main/GlobalHeader.hpp>
namespace maf::math {
/**
 * @brief Struct to hold the result of a solver operation.
 */
template <typename T>
struct SolverResult {
    /** @brief The solution found by the solver. */
    T solution;
    /** @brief The error associated with the solution.
     * @attention This is meaningful only if the solver succeeded.
     */
    double error;
    /** @brief An optional error message if the solver failed. */
    std::optional<std::string_view> error_message;

    explicit operator bool() const {
        return !error_message.has_value();
    }
};
}  // namespace maf::math
#endif
