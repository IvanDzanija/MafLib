#ifndef GOLDEN_SECTION_HPP
#define GOLDEN_SECTION_HPP

#pragma once
#include "Optimizer.hpp"

namespace maf::math {
/**
 * @brief Class for finding the minimum of a unimodal function using the Golden Section
 * Search method.
 * @tparam T The floating-point type to use (e.g., float, double).
 */
template <std::floating_point T>
class GoldenSection : public Optimizer<T> {
public:
    GoldenSection() = delete;  // Make sure function is provided.
    /**
     * @brief Constructor for GoldenSection class.
     * @param function The function for which to find the minimum.
     * @param lower_bound The lower bound of the interval.
     * @param upper_bound The upper bound of the interval.
     */
    GoldenSection(const std::function<T(T)> &function, T lower_bound, T upper_bound)
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

    [[nodiscard]] OptimizerResult<T> solve(T tolerance = static_cast<T>(1e-7),
                                           int32 max_iterations = 100) override {
        T a = _lower_bound;
        T b = _upper_bound;
        T h = b - a;

        // Initial check for trivial case
        if (h <= tolerance) {
            T solution = (a + b) / static_cast<T>(2);
            return {.solution = solution,
                    .error = h / static_cast<T>(2),
                    .error_message = std::nullopt};
        }
        T x1 = b - ratio * h;
        T x2 = a + ratio * h;
        T f1 = this->_function(x1);
        T f2 = this->_function(x2);
        while (h > tolerance && max_iterations-- > 0) {
            h = ratio * h;
            if (f1 < f2) {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - (ratio * h);
                f1 = this->_function(x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + (ratio * h);
                f2 = this->_function(x2);
            }
        }
        // This should probably never happen since this method will find the local
        // minima at least.
        if (max_iterations == 0 && h > tolerance) {
            return {.solution = (a + b) / static_cast<T>(2),
                    .error = h / static_cast<T>(2),
                    .error_message =
                        "Maximum number of iterations reached without convergence."};
        }
        T solution = (a + b) / static_cast<T>(2);
        T error = h / static_cast<T>(2);
        return {.solution = solution, .error = error, .error_message = std::nullopt};
    }

private:
    /** @brief The lower bound of the interval. */
    T _lower_bound;

    /** @brief The upper bound of the interval. */
    T _upper_bound;

    // Internal constant
    static constexpr T ratio = T(1) / std::numbers::phi_v<T>;
};

}  // namespace maf::math
#endif
