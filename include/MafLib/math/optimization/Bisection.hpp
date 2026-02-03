#ifndef BISECTION_HPP
#define BISECTION_HPP

#pragma once
#include "Optimizer.hpp"
#include "OptimizerResult.hpp"
namespace maf::math {
/**
 * @brief Class for finding roots of a function using the Bisection method.
 * @tparam T The floating-point type to use (e.g., float, double).
 */
template <std::floating_point T>
class Bisection : public Optimizer<T> {
public:
    Bisection() = delete;  // Make sure function is provided.
    /**
     * @brief Constructor for Bisection class.
     * @param function The function for which to find the root.
     *   @param lower_bound The lower bound of the interval.
     *   @param upper_bound The upper bound of the interval.
     */
    explicit Bisection(const std::function<T(T)> &function,
                       T lower_bound,
                       T upper_bound)
        : Optimizer<T>(function),
          _lower_bound(lower_bound),
          _upper_bound(upper_bound) {}

    /** @brief Get the lower bound of the interval.
     * @return The lower bound.
     */
    [[nodiscard]] T get_lower_bound() const {
        return _lower_bound;
    }

    /** @brief Get the upper bound of the interval.
     * @return The upper bound.
     */
    [[nodiscard]] T get_upper_bound() const {
        return _upper_bound;
    }

    /**
     * @brief Set the lower bound of the interval.
     * @param lower_bound The lower bound to set.
     */
    void set_lower_bound(T lower_bound) {
        _lower_bound = lower_bound;
    }

    /**
     * @brief Set the upper bound of the interval.
     * @param upper_bound The upper bound to set.
     */
    void set_upper_bound(T upper_bound) {
        _upper_bound = upper_bound;
    }

    OptimizerResult<T> solve(T tolerance = static_cast<T>(1e-7),
                             uint32_t max_iterations = 100) override {
        T a = _lower_bound;
        T b = _upper_bound;
        T fa = this->_function(a);
        T fb = this->_function(b);

        if (fa * fb > 0) {
            return {.solution = NAN,
                    .error = NAN,
                    .error_message =
                        "Function has the same sign at the interval endpoints."};
        }

        T c = a;
        T fc = fa;

        while (max_iterations-- > 0) {
            c = (a + b) / 2;
            fc = this->_function(c);

            if (std::abs(fc) < std::numeric_limits<T>::epsilon() ||
                (b - a) / 2 < tolerance) {
                return {.solution = c,
                        .error = std::abs(fc),
                        .error_message = std::nullopt};
            }

            if (fa * fc < 0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }

        return {.solution = c,
                .error = std::abs(fc),
                .error_message = "Maximum iterations reached without convergence."};
    }

private:
    /** @brief The lower bound of the interval. */
    T _lower_bound;
    /** @brief The upper bound of the interval. */
    T _upper_bound;
};
}  // namespace maf::math
#endif
