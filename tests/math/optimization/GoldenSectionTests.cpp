#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/optimization/GoldenSection.hpp"
#include "MafLib/math/optimization/Optimizer.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::test {
using namespace maf;

class GoldenSectionTests : public ITest {
private:
    void should_find_minimum_with_golden_section() {
        auto func = [](double x) { return (x - 2) * (x - 2); };
        constexpr double lower_bound = 1.0;
        constexpr double upper_bound = 3.0;
        constexpr double tolerance = 1e-6;
        constexpr uint32 max_iterations = 100;

        maf::math::GoldenSection<double> gs(func, lower_bound, upper_bound);
        math::OptimizerResult<double> result = gs.solve(tolerance, max_iterations);

        constexpr double expected = 2.0;
        ASSERT_TRUE(std::abs(result.solution - expected) < tolerance);
    }

    void check_inheritance_and_methods() {
        std::vector<std::shared_ptr<maf::math::Optimizer<double>>> optimizers;
        maf::math::GoldenSection<double> gs(
            [](double x) { return (x - 2) * (x - 2); }, 1.0, 3.0);
        optimizers.push_back(std::make_shared<maf::math::GoldenSection<double>>(gs));

        constexpr double tolerance = 1e-6;
        constexpr uint32 max_iterations = 100;
        auto result = optimizers[0]->solve(tolerance, max_iterations);

        constexpr double expected = 2.0;
        ASSERT_TRUE(util::is_close(expected, result.solution, tolerance));
    }

public:
    int run_all_tests() override {
        should_find_minimum_with_golden_section();
        // This method is deterministic and always finds at least a local minimum.
        check_inheritance_and_methods();
        return 0;
    }
};

}  // namespace maf::test
