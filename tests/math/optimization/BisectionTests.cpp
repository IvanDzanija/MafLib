#include <numbers>

#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/Bisection.hpp"
#include "MafLib/math/optimization/Optimizer.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::test {
using namespace maf;

class BisectionTests : public ITest {
 private:
  void should_find_root_with_bisection() {
    constexpr auto func = [](double x) { return x * x - 2; };
    constexpr double lower_bound = 1.0;
    constexpr double upper_bound = 2.0;
    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 100;

    maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
    math::OptimizerResult<double> result = bisection.solve(tolerance, max_iterations);

    constexpr double expected = std::numbers::sqrt2;
    ASSERT_TRUE(util::is_close(expected, result.solution, tolerance));
  }

  void should_handle_wrong_initial_interval() {
    constexpr auto func = [](double x) { return x * x - 2; };
    constexpr double lower_bound = 1.5;
    constexpr double upper_bound = 2.0;
    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 100;

    maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
    math::OptimizerResult<double> result = bisection.solve(tolerance, max_iterations);

    ASSERT_TRUE(result.error_message.has_value() &&
                result.error_message.value() ==
                    "Function has the same sign at the interval endpoints.");
  }

  void should_handle_non_converging_bisection() {
    constexpr auto func = [](double x) { return x * x - 2; };
    constexpr double lower_bound = 0.0;
    constexpr double upper_bound = 2.0;
    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 2;

    maf::math::Bisection<double> bisection(func, lower_bound, upper_bound);
    math::OptimizerResult<double> result = bisection.solve(tolerance, max_iterations);

    ASSERT_TRUE(result.error_message.has_value() &&
                result.error_message.value() ==
                    "Maximum iterations reached without convergence.");
  }

  void check_inheritance_and_methods() {
    std::vector<std::shared_ptr<maf::math::Optimizer<double>>> optimizers;
    maf::math::Bisection<double> bisection([](double x) { return x * x - 2; }, 1.0,
                                           2.0);
    optimizers.push_back(std::make_shared<maf::math::Bisection<double>>(bisection));

    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 100;
    auto result = optimizers[0]->solve(tolerance, max_iterations);

    constexpr double expected = std::numbers::sqrt2;
    ASSERT_TRUE(util::is_close(expected, result.solution, tolerance));
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
