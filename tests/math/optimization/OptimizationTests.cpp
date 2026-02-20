#include "BisectionTests.cpp"
#include "FixedPointTests.cpp"
#include "GoldenSectionTests.cpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "NewtonRaphsonTests.cpp"

int main() {
    std::cout << "=== Running Fixed Point tests ===" << std::endl;
    auto fixed_point_tests = maf::test::FixedPointTests();
    fixed_point_tests.run_all_tests();
    fixed_point_tests.print_summary();
    std::cout << "=== Running Newton Raphson tests ===" << std::endl;
    auto newton_raphson_tests = maf::test::NewtonRaphsonTests();
    newton_raphson_tests.run_all_tests();
    newton_raphson_tests.print_summary();
    std::cout << "=== Running Bisection tests ===" << std::endl;
    auto bisection_tests = maf::test::BisectionTests();
    bisection_tests.run_all_tests();
    bisection_tests.print_summary();
    std::cout << "=== Running Golden Section tests ===" << std::endl;
    auto golden_section_tests = maf::test::GoldenSectionTests();
    golden_section_tests.run_all_tests();
    golden_section_tests.print_summary();

    return 0;
}
