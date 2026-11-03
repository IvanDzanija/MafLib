#include <algorithm>
#include <cmath>
#include <vector>

#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/LinAlg.hpp"

namespace maf::test {
using namespace maf;
using namespace util;

class EigenTests : public ITest {
 private:
  template <typename T>
  static std::vector<T> sorted_values(const math::EigenResult<T> &pairs) {
    std::vector<T> values;
    values.reserve(pairs.size());
    for (const auto &pair : pairs) {
      values.push_back(pair.eigen_value);
    }
    std::sort(values.begin(), values.end());
    return values;
  }

  // A correct decomposition must satisfy A * v = lambda * v for every pair.
  template <typename T>
  static bool _eigen_pairs_are_correct(const math::Matrix<T> &a,
                                       const math::EigenResult<T> &pairs) {
    for (const auto &pair : pairs) {
      math::Vector av = a * pair.eigen_vector;
      av.print();
      auto lambda_v = pair.eigen_value * pair.eigen_vector;
      for (size_t i = 0; i < av.size(); ++i) {
        T residual = av.at(i) - lambda_v.at(i);
        std::println("{}", residual);
        if (!is_close(residual, T{0})) {
          return false;
        }
      }
    }
    return true;
  }

 public:
  //=============================================================================
  // REAL / SYMMETRIC MATRIX TESTS
  //=============================================================================
  void should_decompose_diagonal_matrix() {
    // Eigenvalues of a diagonal matrix are its diagonal entries.
    math::Matrix<double> a(3, 3, {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
    auto result = math::eigen_decomposition(a);

    assert_true(result.size() == 3);
    auto values = sorted_values(result);
    for (size_t i = 0; i < 3; ++i) {
      assert_true(util::is_close(values[i], static_cast<double>(i + 1)));
    }
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_symmetric_2x2_matrix() {
    math::Matrix<double> a(2, 2, {{2, 1}, {1, 2}});
    auto result = math::eigen_decomposition(a);

    assert_true(result.size() == 2);
    auto values = sorted_values(result);
    assert_true(util::is_close(values[0], 1.0));
    assert_true(util::is_close(values[1], 3.0));
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_symmetric_3x3_matrix() {
    math::Matrix<double> a(3, 3, {{2, 1, 0}, {1, 2, 1}, {0, 1, 2}});
    auto result = math::eigen_decomposition(a);
    ASSERT_SAME_TYPE(result, maf::math::EigenResult<double>);

    assert_true(result.size() == 3);
    const double sqrt2 = std::numbers::sqrt2;
    auto values = sorted_values(result);
    assert_true(util::is_close(values[0], 2.0 - sqrt2));
    assert_true(util::is_close(values[1], 2.0));
    assert_true(util::is_close(values[2], 2.0 + sqrt2));
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_identity_matrix() {
    math::Matrix<double> a(4, 4,
                           {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    auto result = math::eigen_decomposition(a);
    ASSERT_SAME_TYPE(result, maf::math::EigenResult<double>);

    assert_true(result.size() == 4);
    for (const auto &pair : result) {
      assert_true(util::is_close(pair.eigen_value, 1.0));
    }
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_check_repeated_eigenvalues() {
    math::Matrix<double> a(4, 4,
                           {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}});
    auto result = math::eigen_decomposition(a);

    assert_true(result.size() == 4);
    for (const auto &pair : result) {
      assert_true(util::is_close(pair.eigen_value, 2.0));
    }
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_symmetric_2x2_float_matrix() {
    math::Matrix<float> a(2, 2, {{2.0f, 1.0f}, {1.0f, 2.0f}});
    auto result = math::eigen_decomposition(a);

    ASSERT_SAME_TYPE(result, maf::math::EigenResult<float>);
    assert_true(result.size() == 2);
    auto values = sorted_values(result);
    std::println("{}", values[0] - 1.0F);
    assert_true(util::is_close(values[0], 1.0F, 1e-5f));
    assert_true(util::is_close(values[1], 3.0F, 1e-5f));
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_symmetric_3x3_float_matrix() {
    math::Matrix<float> a(3, 3, {{2, 1, 0}, {1, 2, 1}, {0, 1, 2}});
    auto result = math::eigen_decomposition(a);
    ASSERT_SAME_TYPE(result, maf::math::EigenResult<float>);

    assert_true(result.size() == 3);
    const float sqrt2 = std::numbers::sqrt2;
    auto values = sorted_values(result);
    assert_true(util::is_close(values[0], 2.0F - sqrt2));
    assert_true(util::is_close(values[1], 2.0F));
    assert_true(util::is_close(values[2], 2.0F + sqrt2));
    assert_true(_eigen_pairs_are_correct(a, result));
  }

  void should_decompose_integer_matrix_as_double() {
    // Non-floating input is promoted; result type must be EigenResult<double>.
    math::Matrix<int> a(2, 2, {{2, 1}, {1, 2}});
    auto result = math::eigen_decomposition(a);

    ASSERT_SAME_TYPE(result, maf::math::EigenResult<double>);
    assert_true(result.size() == 2);
    auto values = sorted_values(result);
    assert_true(util::is_close(values[0], 1.0));
    assert_true(util::is_close(values[1], 3.0));
    assert_true(_eigen_pairs_are_correct(math::Matrix<double>(a), result));
  }

  void should_verify_trace_and_determinant_for_symmetric_matrix() {
    // Sum of eigenvalues == trace, product of eigenvalues == determinant.
    math::Matrix<double> a(3, 3, {{6, 2, 1}, {2, 3, 1}, {1, 1, 5}});
    auto result = math::eigen_decomposition(a);

    double trace = a.at(0, 0) + a.at(1, 1) + a.at(2, 2);
    double values_sum = 0.0;
    for (const auto &pair : result) {
      values_sum += pair.eigen_value;
    }
    assert_true(util::is_close(values_sum, trace));

    // det for this matrix: 6*(15-1) - 2*(10-1) + 1*(2-3) = 84 - 18 - 1 = 65
    double determinant = 65.0;
    double values_product = 1.0;
    for (const auto &pair : result) {
      values_product *= pair.eigen_value;
    }
    assert_true(util::is_close(values_product, determinant));
    assert_true(_eigen_pairs_are_correct(a, result));
  }

 public:
  std::vector<TestTask> get_tests() override {
    return {
        //[this] { should_decompose_diagonal_matrix(); },
        //[this] { should_decompose_symmetric_2x2_matrix(); },
        //[this] { should_decompose_symmetric_3x3_matrix(); },
        //[this] { should_decompose_identity_matrix(); },
        //[this] { should_check_repeated_eigenvalues(); },
        [this] { should_decompose_float_symmetric_matrix(); },
        // [this] { should_decompose_integer_matrix_as_double(); },
        // [this] { should_verify_trace_and_determinant_for_symmetric_matrix(); },
    };
  }
};

}  // namespace maf::test
