#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "OptimizerResult.hpp"

namespace maf::math {
/**
 * @brief Class for optimization algorithms.
 * @tparam T The floating-point type to use (e.g., float, double).
 */
template <std::floating_point T>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /** @brief Get the function being optimized/solved.
     * @return The function.
     */
    [[nodiscard]] std::function<T(T)> get_function() const {
        return _function;
    }

    /** @brief Set the function to optimize/solve.
     * @param function The function to set.
     */
    void set_function(const std::function<T(T)> &function) {
        _function = function;
    }

    /** @brief Solve for the root/fixed point or local/global minima.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations.
     * @return A OptimizerResult containing the solution and metadata.
     */
    virtual OptimizerResult<T> solve(
        T tolerance = static_cast<T>(1e-7),
        uint32_t max_iterations = 100) = 0;  // Pure virtual

protected:
    /** @brief The function to optimize/solve. */
    std::function<T(T)> _function;

    /** @brief Protected default constructor - only derived classes can construct. */
    Optimizer() = default;

    /** @brief Protected constructor with function.
     * @param function The function to optimize/solve.
     */
    explicit Optimizer(const std::function<T(T)> &function) : _function(function) {}
};
}  // namespace maf::math

#endif
