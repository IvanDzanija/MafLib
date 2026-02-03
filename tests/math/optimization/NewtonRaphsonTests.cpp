#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/NewtonRaphson.hpp"

namespace maf::test {
using namespace maf;
using namespace std::chrono;

class NewtonRaphsonTests : public ITest {
private:
    void should_perform_newton_raphson() {
        auto func = [](double x) { return x * x - 2; };
        auto derivative = [](double x) { return 2 * x; };
        double initial_guess = 1.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 100;

        maf::math::NewtonRaphson<double> nr(func, derivative, initial_guess);
        math::OptimizerResult<double> result = nr.solve(tolerance, max_iterations);

        double expected = std::numbers::sqrt2;
        ASSERT_TRUE(std::abs(result.solution - expected) < tolerance);
    }

public:
    int run_all_tests() override {
        should_perform_newton_raphson();
        return 0;
    }
};

}  // namespace maf::test
