#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/FixedPoint.hpp"
#include "MafLib/math/optimization/SolverResult.hpp"

namespace maf::test {
using namespace maf;
using namespace std::chrono;

class OptimizationTests : public ITest {
private:
    void should_perform_fixed_point_iteration() {
        auto func = [](double x) { return std::cos(x); };
        double initial_guess = 0.5;
        double tolerance = 1e-6;
        uint32 max_iterations = 1000;

        maf::math::FixedPoint fp(func, initial_guess);
        math::SolverResult result = fp.solve(tolerance, max_iterations);

        double expected = 0.739085;
        ASSERT_TRUE(std::abs(result.solution - expected) < tolerance);
    }

    void should_handle_non_converging_fixed_point() {
        auto func = [](double x) { return 2 * x; };
        double initial_guess = 1.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 100;

        maf::math::FixedPoint fp(func, initial_guess);
        math::SolverResult result = fp.solve(tolerance, max_iterations);
        ASSERT_TRUE(!result);
    }

public:
    int run_all_tests() override {
        should_perform_fixed_point_iteration();
        should_handle_non_converging_fixed_point();
        return 0;
    }
};

}  // namespace maf::test
