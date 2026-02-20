#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/NewtonRaphson.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::test {
using namespace maf;
using namespace std::chrono;

class NewtonRaphsonTests : public ITest {
private:
    void should_perform_newton_raphson() {
        constexpr auto func = [](double x) { return x * x - 2; };
        constexpr auto derivative = [](double x) { return 2 * x; };
        constexpr double initial_guess = 1.0;
        constexpr double tolerance = 1e-6;
        constexpr int32 max_iterations = 100;

        maf::math::NewtonRaphson<double> nr(func, derivative, initial_guess);
        math::OptimizerResult<double> result = nr.solve(tolerance, max_iterations);

        constexpr double expected = std::numbers::sqrt2;
        ASSERT_TRUE(util::is_close(expected, result.solution, tolerance));
    }

    void should_handle_non_converging_newton_raphson() {
        constexpr auto func = [](double x) { return std::atan(x); };
        constexpr auto derivative = [](double x) { return 1 / (1 + (x * x)); };

        constexpr double initial_guess = 1.5;
        constexpr double tolerance = 1e-6;
        constexpr int32 max_iterations = 100;

        maf::math::NewtonRaphson<double> nr(func, derivative, initial_guess);
        math::OptimizerResult<double> result = nr.solve(tolerance, max_iterations);

        ASSERT_TRUE(!result);
    }

    // TODO: Add secant method tests when added

    void check_inheritance_and_methods() {
        std::vector<std::shared_ptr<maf::math::Optimizer<double>>> optimizers;
        maf::math::NewtonRaphson<double> nr(
            [](double x) { return x * x - 2; }, [](double x) { return 2 * x; }, 1.0);

        optimizers.push_back(std::make_shared<maf::math::NewtonRaphson<double>>(nr));

        constexpr double tolerance = 1e-6;
        constexpr int32 max_iterations = 100;
        auto result = optimizers[0]->solve(tolerance, max_iterations);

        constexpr double expected = std::numbers::sqrt2;
        ASSERT_TRUE(util::is_close(expected, result.solution, tolerance));
    }

public:
    int run_all_tests() override {
        should_perform_newton_raphson();
        should_handle_non_converging_newton_raphson();
        check_inheritance_and_methods();
        return 0;
    }
};

}  // namespace maf::test
