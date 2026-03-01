#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::test {
class ITest {
 private:
  std::atomic<uint32> _passed = 0;
  std::atomic<uint32> _failed = 0;

 public:
  virtual ~ITest() = default;

  virtual int run_all_tests() = 0;

  void print_summary() const {
    std::cout << "\n=== Checks Summary ===\n"
              << "Passed: " << _passed << "\n"
              << "Failed: " << _failed << "\n"
              << "====================\n";
  }

 protected:
  void assert_true(bool condition, const std::string &msg, const char *file, int line) {
    if (!condition) {
      assert(condition);
      std::cerr << "[FAIL] " << file << ":" << line << " → " << msg << std::endl;
      ++_failed;
    } else {
      std::cout << "[PASS] " << msg << std::endl;
      ++_passed;
    }
  }

  template <typename Ex, typename Fn>
  void assert_throw(Fn &&fn, const std::string &msg, const char *file, int line) {
    bool thrown = false;
    try {
      fn();
    } catch (const Ex &) {
      thrown = true;
      std::cout << "[PASS] " << msg << std::endl;
    } catch (...) {
      // Wrong exception type (or non-std exception)
      thrown = false;
      std::cout << "[FAIL] " << file << ":" << line << " → Expected exception of type "
                << typeid(Ex).name() << " but got a different exception." << std::endl;
    }
    assert_true(thrown, msg, file, line);
  }
};

#define ASSERT_SAME_TYPE(expr, Type)                                       \
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(expr)>, Type>, \
                "Type mismatch: " #expr " is not of type " #Type)

#define ASSERT_TRUE(cond) assert_true((cond), __func__, __FILE__, __LINE__)

#define ASSERT_THROW(expr, ExType) \
  assert_throw<ExType>([&]() { (void)(expr); }, __func__, __FILE__, __LINE__)
}  // namespace maf::test
