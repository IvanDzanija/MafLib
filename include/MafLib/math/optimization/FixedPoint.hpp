#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "SolverResult.hpp"

namespace maf::math {
/**
 * @brief Class for finding fixed points of a function using the Fixed Point Iteration
 * method.
 */
class FixedPoint {
public:
    FixedPoint() = default;
    /**
     * @brief Constructor for FixedPoint class.
     * @param function The function for which to find the fixed point.
     * @param start The initial guess for the fixed point.
     */
    FixedPoint(const std::function<double(double)> &function, double start)
        : _function(function), _start(start) {}

    /** @brief Get the function for which to find the fixed point.
     * @return The function.
     */
    [[nodiscard]] std::function<double(double)> get_function() const {
        return _function;
    }

    /**
     * @brief Get the initial guess for the fixed point.
     * @return The initial guess.
     */
    [[nodiscard]] double get_start() const {
        return _start;
    }

    /**
     * @brief Set the function for which to find the fixed point.
     * @param function The function to set.
     */
    void set_function(const std::function<double(double)> &function) {
        _function = function;
    }

    /**
     * @brief Set the initial guess for the fixed point.
     * @param start The initial guess to set.
     */
    void set_start(double start) {
        _start = start;
    }

    /**
     * @brief Find the fixed point using the Fixed Point Iteration method.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return A SolverResult containing the solution, error, and optionally an error message.
     */
    SolverResult<double> solve(double tolerance = 1e-7,
                               uint32_t max_iterations = 1000) {
        if (!_function) {
            return {.solution = NAN,
                    .error = NAN,
                    .error_message = "Function is not defined."};
        }

        double x = _start;
        double error = _get_error(x);
        uint32_t iterations = 0;

        while (error > tolerance && iterations < max_iterations) {
            x = _function(x);
            error = _get_error(x);
            iterations++;

            if (std::isinf(x) || std::isnan(x)) {
                return {.solution = x,
                        .error = error,
                        .error_message = "Method diverged (NaN or Infinity)."};
            }
        }

        if (error > tolerance) {
            return {
                .solution = x,
                .error = error,
                .error_message = "Method did not converge within the iteration limit."};
        }

        return {.solution = x, .error = error, .error_message = std::nullopt};
    }

private:
    /** @brief The function for which to find the fixed point. */
    std::function<double(double)> _function;
    /** @brief The initial guess for the fixed point. */
    double _start;

    /**
     * @brief Calculate the error at the given point.
     * @param current_point The current point at which to evaluate the error.
     * @return The absolute difference between f(current_point) and current_point.
     */
    double _get_error(double current_point) {
        return std::abs(_function(current_point) - current_point);
    }
};
}  // namespace maf::math
#endif
