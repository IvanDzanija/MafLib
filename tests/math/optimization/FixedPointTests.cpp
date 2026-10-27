#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/FixedPoint.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::test {
using namespace maf;
using namespace std::chrono;

class FixedPointTests : public ITest {
 private:
  void should_perform_fixed_point_iteration() {
    constexpr auto func = [](double x) { return std::cos(x); };
    constexpr double initial_guess = 0.5;
    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 1000;

    maf::math::FixedPoint<double> fp(func, initial_guess);
    math::OptimizerResult<double> result = fp.solve(tolerance, max_iterations);

    constexpr double expected = 0.739085;
    assert_true(std::abs(result.solution - expected) < tolerance);
  }

  void should_handle_non_converging_fixed_point() {
    constexpr auto func = [](double x) { return 2 * x; };
    constexpr double initial_guess = 1.0;
    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 100;

    maf::math::FixedPoint<double> fp(func, initial_guess);
    math::OptimizerResult<double> result = fp.solve(tolerance, max_iterations);
    assert_true(!result);
  }

  void check_inheritance_and_methods() {
    std::vector<std::shared_ptr<maf::math::Optimizer<double>>> optimizers;
    maf::math::FixedPoint<double> fp([](double x) { return std::cos(x); }, 0.5);
    optimizers.push_back(std::make_shared<maf::math::FixedPoint<double>>(fp));

    constexpr double tolerance = 1e-6;
    constexpr int32 max_iterations = 1000;
    auto result = optimizers[0]->solve(tolerance, max_iterations);

    constexpr double expected = 0.739085;
    assert_true(util::is_close(expected, result.solution, tolerance));
  }

 public:
  std::vector<TestTask> get_tests() override {
    return {
        [this] { should_perform_fixed_point_iteration(); },
        [this] { should_handle_non_converging_fixed_point(); },
        [this] { check_inheritance_and_methods(); },
    };
  }
};

}  // namespace maf::test
