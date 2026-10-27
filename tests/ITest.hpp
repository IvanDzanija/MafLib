#pragma once
#include <mutex>
#include <source_location>

#include "MafLib/main/GlobalHeader.hpp"

namespace maf::test {
class ITest {
 private:
  std::atomic<uint32> _passed = 0;
  std::atomic<uint32> _failed = 0;
  inline static std::mutex _display_mutex;

 public:
  virtual ~ITest() = default;
  ITest() = default;
  ITest(ITest &other) = delete;
  ITest(ITest &&other) = delete;
  ITest operator=(ITest &other) = delete;
  ITest operator=(ITest &&other) = delete;

  using TestTask = std::function<void()>;
  virtual std::vector<TestTask> get_tests() = 0;

  uint32 passed() { return _passed.load(); }
  uint32 failed() { return _failed.load(); }

  void print_summary() const {
    std::scoped_lock lock(_display_mutex);
    std::cout << "\n=== Checks Summary ===\n"
              << "Passed: " << _passed << "\n"
              << "Failed: " << _failed << "\n"
              << "====================\n";
  }

 protected:
  constexpr void assert_true(bool condition, const std::source_location &loc =
                                                 std::source_location::current()) {
    if (!condition) {
      std::scoped_lock lock(_display_mutex);
      std::cerr << "[FAIL] " << loc.file_name() << ":" << loc.line() << " → "
                << loc.function_name() << "\n";
      ++_failed;
      assert(condition);
    } else {
      std::scoped_lock lock(_display_mutex);
      std::cout << "[PASS] " << loc.function_name() << "\n";
      ++_passed;
    }
  }

  template <typename Ex, typename Fn>
  void assert_throw(Fn &&fn,
                    const std::source_location &loc = std::source_location::current()) {
    bool thrown = false;
    bool wrong_type = false;
    try {
      std::forward<Fn>(fn)();
    } catch (const Ex &) {
      thrown = true;
    } catch (...) {
      // Wrong exception type (or non-std exception)
      wrong_type = true;
    }
    if (thrown) {
      std::scoped_lock lock(_display_mutex);
      std::cout << "[PASS] " << loc.function_name() << "\n";
      ++_passed;
    } else {
      std::scoped_lock lock(_display_mutex);
      std::cerr << "[FAIL] " << loc.file_name() << ":" << loc.line()
                << (wrong_type ? " → expected exception of type "
                               : " → no exception thrown, expected ")
                << typeid(Ex).name() << "\n";
      ++_failed;
    }
    assert(thrown);
  }
};

#define ASSERT_SAME_TYPE(expr, Type)                                       \
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(expr)>, Type>, \
                "Type mismatch: " #expr " is not of type " #Type)

}  // namespace maf::test
