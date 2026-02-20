#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

#include <limits>
#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "Optimizer.hpp"

namespace maf::math {
/**
 * @brief Class for finding roots of a function using the Newton-Raphson method.
 * @tparam T The floating-point type to use (e.g., float, double).
 */
template <std::floating_point T>
class NewtonRaphson : public Optimizer<T> {
public:
    NewtonRaphson() = delete;  // Make sure the function is provided.

    /** @brief Constructor for NewtonRaphson class.
     * @param function The function for which to find the root.
     * @param derivative The derivative of the function.
     * @param start The initial guess for the root.
     */
    NewtonRaphson(const std::function<T(T)> &function,
                  const std::function<T(T)> &derivative,
                  T start)
        : Optimizer<T>(function), _derivative(derivative), _start(start) {}

    /** @brief Get the derivative of the function.
     * @return The derivative function.
     */
    [[nodiscard]] const std::function<T(T)> &get_derivative() const {
        return _derivative;
    }

    /** @brief Get the initial guess for the root.
     * @return The initial guess.
     */
    [[nodiscard]] T get_start() const {
        return _start;
    }

    /** @brief Set the derivative of the function.
     * @param derivative The derivative function to set.
     */
    void set_derivative(const std::function<T(T)> &derivative) {
        if (!derivative) {
            _derivative.reset();
        } else {
            _derivative = derivative;
        }
    }

    /** @brief Set the initial guess for the root.
     * @param start The initial guess to set.
     */
    void set_start(T start) {
        _start = start;
    }

    /** @brief Find the root using the Newton-Raphson method.
     * @details If the derivative is not provided, the secant method will be used.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return An OptimizerResult containing the solution, error, and optionally an
     * error message.
     */
    [[nodiscard]] OptimizerResult<T> solve(T tolerance = static_cast<T>(1e-7),
                                           int32 max_iterations = 100) override {
        if (!_derivative) {
            return _secant_solve(tolerance, max_iterations);
        }
        // NOTE:
        // Pure method implementation of Newton-Raphson should be provided as well for
        // pure speed and constexpr contexts
        return _newton_raphson_solve(tolerance, max_iterations);
    }

private:
    /** @brief The derivative of the function. */
    std::optional<std::function<T(T)>> _derivative;
    /** @brief The initial guess for the root. */
    T _start;

    /** @brief Find the root using the Newton-Raphson method.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return An OptimizerResult containing the solution, error, and optionally an
     * error message.
     */
    OptimizerResult<T> _newton_raphson_solve(T tolerance, int32 max_iterations) {
        // This gets called only if the derivative is provided.
        T x = _start;
        while (max_iterations-- > 0) {
            T f_x = this->_function(x);
            T f_prime_x = _derivative.value()(x);
            if (std::abs(f_prime_x) <= std::numeric_limits<T>::epsilon()) {
                return {.solution = x,
                        .error = std::abs(f_x),
                        .error_message =
                            "Derivative is too small; potential division by zero."};
            }
            T x_new = x - (f_x / f_prime_x);
            T denom = std::abs(x);
            if (denom < std::numeric_limits<T>::epsilon()) {
                denom = std::numeric_limits<T>::epsilon();
            }
            T rel_error = std::abs(x_new - x) / denom;
            if (rel_error < tolerance) {
                return {.solution = x_new,
                        .error = rel_error,
                        .error_message = std::nullopt};
            }
            x = x_new;
        }
        return {.solution = x,
                .error = std::abs(this->_function(x)),
                .error_message = "Maximum iterations reached without convergence."};
    }

    /** @brief Find the root using the secant method.
     * @param tolerance The tolerance for convergence.
     * @param max_iterations The maximum number of iterations to perform.
     * @return An OptimizerResult containing the solution, error, and optionally an
     * error message.
     */
    OptimizerResult<T> _secant_solve(T tolerance, int32 max_iterations) {
        return OptimizerResult<T>{
            .solution = std::numeric_limits<T>::quiet_NaN(),
            .error = std::numeric_limits<T>::quiet_NaN(),
            .error_message = "Secant method not implemented yet."};
    }
};
}  // namespace maf::math
#endif
