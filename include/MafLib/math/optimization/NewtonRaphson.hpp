#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "OptimizerResult.hpp"

namespace maf::math {
/**
 * @brief Class for finding fixed points of a function using the Newton-Raphson method.
 * @tparam T The floating-point type to use (e.g., float, double).
 */
template <std::floating_point T>
class NewtonRaphson {
public:
    NewtonRaphson() = default;

    /** @brief Constructor for NewtonRaphson class.
     * @param function The function for which to find the fixed point.
     * @param derivative The derivative of the function.
     * @param start The initial guess for the fixed point.
     */
    NewtonRaphson(const std::function<T(T)> &function,
                  const std::function<T(T)> &derivative,
                  T start)
        : _function(function), _derivative(derivative), _start(start) {}

    /** @brief Get the function for which to find the fixed point.
     * @return The function.
     */
    [[nodiscard]] std::function<T(T)> get_function() const {
        return _function;
    }

    /** @brief Get the derivative of the function.
     * @return The derivative function.
     */
    [[nodiscard]] std::function<T(T)> get_derivative() const {
        return _derivative;
    }

    /** @brief Get the initial guess for the fixed point.
     * @return The initial guess.
     */
    [[nodiscard]] T get_start() const {
        return _start;
    }

    /** @brief Set the function for which to find the fixed point.
     * @param function The function to set.
     */
    void set_function(const std::function<T(T)> &function) {
        _function = function;
    }

    /** @brief Set the derivative of the function.
     * @param derivative The derivative function to set.
     */
    void set_derivative(const std::function<T(T)> &derivative) {
        _derivative = derivative;
    }

    /** @brief Set the initial guess for the fixed point.
     * @param start The initial guess to set.
     */
    void set_start(T start) {
        _start = start;
    }

    /** @brief Find the fixed point using the Newton-Raphson method.
     * @detail If the derivative is not provided, the secant method will be used.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return A OptimizerResult containing the solution, error, and optionally an error
     * message.
     */
    OptimizerResult<T> solve(T tolerance = static_cast<T>(1e-7),
                             uint32_t max_iterations = 100) {
        if (!_function) {
            return {.solution = NAN,
                    .error = NAN,
                    .error_message = "Function is not defined."};
        }
        if (!_derivative) {
            return _secant_solve(tolerance, max_iterations);
        }
        // Pure method implementation of Newton-Raphson should be provided as well for
        // pure speed and constexpr contexts
        return _newton_raphson_solve(tolerance, max_iterations);
    }

private:
    /** @brief The function for which to find the fixed point. */
    std::function<T(T)> _function;
    /** @brief The derivative of the function. */
    std::optional<std::function<T(T)>> _derivative;
    /** @brief The initial guess for the fixed point. */
    T _start;

    /** @brief Find the fixed point using the Newton-Raphson method.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return A OptimizerResult containing the solution, error, and optionally an error
     * message.
     */
    OptimizerResult<T> _newton_raphson_solve(T tolerance, uint32_t max_iterations) {
        T x = _start;
        while (max_iterations-- > 0) {
            T f_x = _function(x);
            T f_prime_x = _derivative.value()(x);
            if (std::abs(f_prime_x) <= std::numeric_limits<T>::epsilon()) {
                return {.solution = x,
                        .error = std::abs(f_x),
                        .error_message =
                            "Derivative is too small; potential division by zero."};
            }
            T x_new = x - (f_x / f_prime_x);
            T rel_error = std::abs(x_new - x) / std::abs(x);
            if (rel_error < tolerance) {
                return {.solution = x_new,
                        .error = rel_error,
                        .error_message = std::nullopt};
            }
            x = x_new;
        }
        return {.solution = x,
                .error = std::abs(_function(x)),
                .error_message = "Maximum iterations reached without convergence."};
    }

    /** @brief Find the fixed point using the secant method.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return A OptimizerResult containing the solution, error, and optionally an error
     * message.
     */
    OptimizerResult<T> _secant_solve(T tolerance, uint32_t max_iterations) {
        return OptimizerResult<T>{
            .solution = NAN,
            .error = NAN,
            .error_message = "Secant method not implemented yet."};
    }
};
}  // namespace maf::math
#endif
