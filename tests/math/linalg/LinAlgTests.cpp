#include "MafLib/main/GlobalHeader.hpp"
#include "MatrixTests.cpp"
#include "VectorTests.cpp"
#include "ViewTests.cpp"

int main() {
  // TODO: Uncomment this when parallel testing is implemented
  std::cout << "=== Running Matrix tests ===" << std::endl;
  auto matrix_tests = maf::test::MatrixTests();
  matrix_tests.run_all_tests();
  matrix_tests.print_summary();

  std::cout << "\n=== Running Vector tests ===" << std::endl;
  auto vector_tests = maf::test::VectorTests();
  vector_tests.run_all_tests();
  vector_tests.print_summary();

  std::cout << "\n=== Running View tests ===" << std::endl;
  auto view_tests = maf::test::ViewTests();
  view_tests.run_all_tests();
  view_tests.print_summary();

  return 0;
}
