#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/GoldenSection.hpp"
#include "MafLib/math/optimization/Optimizer.hpp"

namespace maf::test {
using namespace maf;

class GoldenSectionTests : public ITest {
private:
    void should_find_minimum_with_golden_section() {
        auto func = [](double x) { return (x - 2) * (x - 2); };
        double lower_bound = 1.0;
        double upper_bound = 3.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 100;

        maf::math::GoldenSection<double> gs(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result = gs.solve(tolerance, max_iterations);

        double expected = 2.0;
        ASSERT_TRUE(std::abs(result.solution - expected) < tolerance);
    }

    void should_handle_non_converging_golden_section() {
        auto func = [](double x) { return (x - 2) * (x - 2); };
        double lower_bound = 1.0;
        double upper_bound = 3.0;
        double tolerance = 1e-6;
        uint32 max_iterations = 2;

        maf::math::GoldenSection<double> gs(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result = gs.solve(tolerance, max_iterations);
        ASSERT_TRUE(result.error_message.has_value());

        //    ASSERT_TRUE(result.error_message.has_value() &&
        //                result.error_message.value() ==
        //                    "Maximum iterations reached without convergence.");
    }

    void check_inheritance_and_methods() {
        maf::math::GoldenSection<double> gs([](double x) { return x; }, 0.0, 1.0);
        ASSERT_TRUE(dynamic_cast<maf::math::Optimizer<double> *>(&gs) != nullptr);

        gs.set_lower_bound(1.0);
        gs.set_upper_bound(2.0);
        ASSERT_TRUE(gs.get_lower_bound() == 1.0);
        ASSERT_TRUE(gs.get_upper_bound() == 2.0);
    }

public:
    int run_all_tests() override {
        should_find_minimum_with_golden_section();
        should_handle_non_converging_golden_section();
        check_inheritance_and_methods();
        return 0;
    }
};

}  // namespace maf::test
