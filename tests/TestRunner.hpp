#ifdef MAFLIB_OPENMP_AVAILABLE
#include <omp.h>
#endif

#include <memory>
#include <vector>

#include "ITest.hpp"

namespace maf::test {

class TestRunner {
 private:
  std::vector<std::unique_ptr<ITest>> _suites;

 public:
  void add_suite(std::unique_ptr<ITest> suite) { _suites.push_back(std::move(suite)); }

  void run_all_parallel() {
#pragma omp parallel default(none)
    {
#pragma omp single
      {
        for (auto &suite : _suites) {
          auto tasks = suite->get_tests();
          for (auto task : tasks) {
#pragma omp task default(none) firstprivate(task)
            {
              task();
            }
          }
        }
      }
    }

    for (auto &suite : _suites) {
      suite->print_summary();
    }
  }
};

}  // namespace maf::test
