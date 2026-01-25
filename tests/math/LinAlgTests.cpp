#include "MafLib/main/GlobalHeader.hpp"
#include "MatrixTests.cpp"
#include "OptimizationTests.cpp"
#include "VectorTests.cpp"

int main() {
    std::cout << "=== Running Matrix tests ===" << std::endl;
    auto matrix_tests = maf::test::MatrixTests();
    matrix_tests.run_all_tests();
    matrix_tests.print_summary();

    auto vector_tests = maf::test::VectorTests();
    vector_tests.run_all_tests();

    std::cout << "=== Running Optimization tests ===" << std::endl;
    auto optimization_tests = maf::test::OptimizationTests();
    optimization_tests.run_all_tests();
    optimization_tests.print_summary();

    return 0;
}
