#include "BisectionTests.cpp"
#include "FixedPointTests.cpp"
#include "GoldenSectionTests.cpp"
#include "NewtonRaphsonTests.cpp"
#include "TestRunner.hpp"

int main() {
  maf::test::TestRunner runner;
  std::cout << "=== Running optimization tests! ===\n";
  runner.add_suite(std::make_unique<maf::test::FixedPointTests>());
  runner.add_suite(std::make_unique<maf::test::NewtonRaphsonTests>());
  runner.add_suite(std::make_unique<maf::test::BisectionTests>());
  runner.add_suite(std::make_unique<maf::test::GoldenSectionTests>());

  runner.run_all_parallel();

  return 0;
}
