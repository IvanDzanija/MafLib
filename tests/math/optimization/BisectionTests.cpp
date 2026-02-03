#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/Bisection.hpp"
#include "MafLib/math/optimization/Optimizer.hpp"

namespace maf::test {
using namespace maf;

class BisectionTests : public ITest {
private:
    void should_find_root_with_bisection() {
        auto func = [](double x) { return x * x - 2; };
        double lower_bound = 1.0;
        double upper_bound = 2.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 100;

        maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result =
            bisection.solve(tolerance, max_iterations);

        double expected = std::numbers::sqrt2;
        ASSERT_TRUE(std::abs(result.solution - expected) < tolerance);
    }

    void should_handle_wrong_initial_interval() {
        auto func = [](double x) { return x * x - 2; };
        double lower_bound = 1.5;
        double upper_bound = 2.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 100;

        maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result =
            bisection.solve(tolerance, max_iterations);

        ASSERT_TRUE(result.error_message.has_value() &&
                    result.error_message.value() ==
                        "Function has the same sign at the interval endpoints.");
    }

    void should_handle_non_converging_bisection() {
        auto func = [](double x) { return x * x - 2; };
        double lower_bound = 0.0;
        double upper_bound = 2.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 2;

        maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result =
            bisection.solve(tolerance, max_iterations);

        ASSERT_TRUE(result.error_message.has_value() &&
                    result.error_message.value() ==
                        "Maximum iterations reached without convergence.");
    }

    void check_inheritance_and_methods() {
        maf::math::Bisection<double> bisection([](double x) { return x; }, 0.0, 1.0);
        ASSERT_TRUE(dynamic_cast<maf::math::Optimizer<double> *>(&bisection) !=
                    nullptr);

        bisection.set_lower_bound(1.0);
        bisection.set_upper_bound(2.0);
        ASSERT_TRUE(bisection.get_lower_bound() == 1.0);
        ASSERT_TRUE(bisection.get_upper_bound() == 2.0);
    }

public:
    int run_all_tests() override {
        should_find_root_with_bisection();
        should_handle_wrong_initial_interval();
        should_handle_non_converging_bisection();
        check_inheritance_and_methods();
        return 0;
    }
};

}  // namespace maf::test
