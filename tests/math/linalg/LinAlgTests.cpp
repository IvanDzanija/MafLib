#include "MatrixTests.cpp"
#include "TestRunner.hpp"
#include "VectorTests.cpp"
#include "ViewTests.cpp"

int main() {
  maf::test::TestRunner runner;
  std::cout << "=== Running linear algebra tests! ===\n";
  runner.add_suite(std::make_unique<maf::test::MatrixTests>());
  runner.add_suite(std::make_unique<maf::test::VectorTests>());
  runner.add_suite(std::make_unique<maf::test::ViewTests>());

  runner.run_all_parallel();

  return 0;
}
