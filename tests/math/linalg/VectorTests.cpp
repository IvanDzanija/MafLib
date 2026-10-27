#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/LinAlg.hpp"

namespace maf::test {

using namespace maf;
using namespace util;
using namespace std::chrono;
class VectorTests : public ITest {
 private:
  //=============================================================================
  // VECTOR CONSTRUCTORS TESTS
  //=============================================================================
  void should_construct_empty_vector_with_zero_size() {
    math::Vector<int> v;
    assert_true(v.size() == 0);
    assert_true(v.orientation() == math::Orientation::COLUMN);
  }

  void should_construct_vector_of_given_size() {
    math::Vector<int> v_col(5);
    ASSERT_SAME_TYPE(v_col, math::Vector<int>);
    assert_true(v_col.size() == 5);
    assert_true(v_col.orientation() == math::Orientation::COLUMN);

    math::Vector<double> v_row(3, math::Orientation::ROW);
    ASSERT_SAME_TYPE(v_row, math::Vector<double>);
    assert_true(v_row.size() == 3);
    assert_true(v_row.orientation() == math::Orientation::ROW);
  }

  void should_throw_if_constructed_with_zero_size() {
    bool thrown = false;
    try {
      math::Vector<double> v(0);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_raw_data() {
    int data[3] = {10, 20, 30};
    math::Vector<int> v(3, data, math::Orientation::ROW);
    ASSERT_SAME_TYPE(v, math::Vector<int>);
    assert_true(v.size() == 3);
    assert_true(v.orientation() == math::Orientation::ROW);
    assert_true(v[0] == 10);
    assert_true(v[1] == 20);
    assert_true(v[2] == 30);
  }

  void should_throw_if_raw_data_constructor_has_zero_size() {
    bool thrown = false;
    int data[3] = {1, 2, 3};
    try {
      math::Vector<int> v(0, data);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_std_vector_copy() {
    std::vector<int> data = {5, 10, 15};
    math::Vector<int> v(3, data);
    ASSERT_SAME_TYPE(v, math::Vector<int>);
    assert_true(v.size() == 3);
    assert_true(v[1] == 10);
    data[1] = 99;
    assert_true(v[1] == 10);
  }

  void should_throw_if_std_vector_copy_size_mismatch() {
    std::vector<int> data = {1, 2};
    bool thrown = false;
    try {
      math::Vector<int> v(3, data);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_std_vector_move() {
    std::vector<int> data = {5, 10, 15};
    math::Vector<int> v(3, std::move(data));
    ASSERT_SAME_TYPE(v, math::Vector<int>);
    assert_true(v.size() == 3);
    assert_true(v[1] == 10);
  }

  void should_throw_if_std_vector_move_size_mismatch() {
    std::vector<int> data = {1, 2};
    bool thrown = false;
    try {
      math::Vector<int> v(3, std::move(data));
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_std_array() {
    std::array<float, 3> data = {1.1f, 2.2f, 3.3f};
    math::Vector<float> v(3, data, math::Orientation::ROW);
    ASSERT_SAME_TYPE(v, math::Vector<float>);
    assert_true(v.size() == 3);
    assert_true(v.orientation() == math::Orientation::ROW);
    assert_true(is_close(v[1], 2.2f));
  }

  void should_throw_if_std_array_size_mismatch() {
    std::array<int, 2> data = {1, 2};
    bool thrown = false;
    try {
      math::Vector<int> v(3, data);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  //=============================================================================
  // VECTOR ITERATORS TESTS
  //=============================================================================
  void should_access_elements_with_at_and_operator() {
    math::Vector<int> v(3);
    v[0] = 1;
    v[1] = 2;
    v.at(2) = 3;

    assert_true(v[0] == 1);
    assert_true(v.at(1) == 2);
    assert_true(v[2] == 3);

    const auto &cv = v;
    assert_true(cv[0] == 1);
    assert_true(cv.at(1) == 2);
  }

  void should_throw_on_out_of_bounds_access() {
    math::Vector<int> v(3);
    bool thrown_at = false;

    try {
      v.at(3) = 10;
    } catch (const std::out_of_range &e) {
      thrown_at = true;
    }

    assert_true(thrown_at);
  }

  void should_iterate_over_elements() {
    math::Vector<int> v(3);
    v[0] = 10;
    v[1] = 20;
    v[2] = 30;

    int sum = 0;
    for (int val : v) {
      sum += val;
    }
    assert_true(sum == 60);

    const auto &cv = v;
    int const_sum = std::accumulate(cv.cbegin(), cv.cend(), 0);
    assert_true(const_sum == 60);
  }

  //=============================================================================
  // VECTOR CHECKERS TESTS
  //=============================================================================
  void should_check_if_vector_is_null() {
    math::Vector<int> v(3);
    v.fill(0);
    assert_true(v.is_null());

    v[1] = 1;
    assert_true(!v.is_null());

    math::Vector<double> vf(3);
    vf.fill(0.0);
    assert_true(vf.is_null());

    vf[2] = 1.0;
    assert_true(!vf.is_null());
  }

  //=============================================================================
  // VECTOR METHODS TESTS
  //=============================================================================
  void should_fill_vector_with_value() {
    math::Vector<int> v(10);
    v.fill(77);
    assert_true(v[0] == 77);
    assert_true(v[5] == 77);
    assert_true(v[9] == 77);
  }

  void should_calculate_l2_norm() {
    math::Vector<double> v(2);
    v[0] = 3.0;
    v[1] = 4.0;
    assert_true(is_close(v.norm(), 5.0));

    math::Vector<double> v2(3);
    v2[0] = 1.0;
    v2[1] = 2.0;
    v2[2] = 2.0;
    assert_true(is_close(v2.norm(), 3.0));
  }

  void should_normalize_vector_in_place() {
    math::Vector<double> v(2);
    v[0] = 3.0;
    v[1] = 4.0;
    v.normalize();

    assert_true(is_close(v[0], 0.6));
    assert_true(is_close(v[1], 0.8));
    assert_true(is_close(v.norm(), 1.0));
  }

  void should_transpose_vector_in_place() {
    math::Vector<int> v(3, math::Orientation::COLUMN);
    assert_true(v.orientation() == math::Orientation::COLUMN);
    v.transpose();
    assert_true(v.orientation() == math::Orientation::ROW);
    v.transpose();
    assert_true(v.orientation() == math::Orientation::COLUMN);
  }

  void should_return_transposed_copy() {
    math::Vector<int> v_col(3, math::Orientation::COLUMN);
    auto v_row = v_col.transposed();
    ASSERT_SAME_TYPE(v_row, math::Vector<int>);

    assert_true(v_col.orientation() == math::Orientation::COLUMN);
    assert_true(v_row.orientation() == math::Orientation::ROW);
    assert_true(v_row.size() == 3);
  }

  //=============================================================================
  // VECTOR OPERATORS TESTS
  //=============================================================================
  void should_check_equality() {
    math::Vector<int> v1(2);
    v1[0] = 1;
    v1[1] = 2;
    math::Vector<int> v2(2);
    v2[0] = 1;
    v2[1] = 2;
    math::Vector<int> v3(2);
    v3[0] = 1;
    v3[1] = 9;
    math::Vector<int> v4(3);
    math::Vector<int> v5(2, math::Orientation::ROW);
    v5[0] = 1;
    v5[1] = 2;
    math::Vector<float> v6(2);
    v6[0] = 1.F;
    v6[1] = 2.F;

    assert_true(v1 == v2);
    assert_true(!(v1 == v3));
    assert_true(!(v1 == v4));
    assert_true(!(v1 == v5));
  }

  void should_perform_unary_minus() {
    math::Vector<int> v(2);
    v[0] = 5;
    v[1] = -10;
    auto v_neg = -v;
    ASSERT_SAME_TYPE(v_neg, math::Vector<int>);
    assert_true(v[0] == 5);
    assert_true(v_neg[0] == -5);
    assert_true(v_neg[1] == 10);
  }

  void should_add_two_vectors() {
    math::Vector<int> v1(2);
    v1[0] = 1;
    v1[1] = 2;
    math::Vector<float> v2(2);
    v2[0] = 10;
    v2[1] = 20;
    auto v_sum = v1 + v2;
    ASSERT_SAME_TYPE(v_sum, math::Vector<float>);
    assert_true(v_sum.size() == 2);
    assert_true(is_close(v_sum[0], 11.0F));
    assert_true(is_close(v_sum[1], 22.0F));
  }

  void should_add_scalar_and_vector() {
    math::Vector<int> v(2);
    v[0] = 1;
    v[1] = 2;
    auto v_sum = v + 10;
    ASSERT_SAME_TYPE(v_sum, math::Vector<int>);
    assert_true(v_sum[0] == 11);
    assert_true(v_sum[1] == 12);

    auto v_sum2 = 10 + v;
    ASSERT_SAME_TYPE(v_sum2, math::Vector<int>);
    assert_true(v_sum2[0] == 11);
    assert_true(v_sum2[1] == 12);

    auto v_sum3 = 10.F + v;
    ASSERT_SAME_TYPE(v_sum3, math::Vector<float>);
    assert_true(is_close(v_sum3[0], 11.0F));
    assert_true(is_close(v_sum3[1], 12.0F));

    auto v_sum4 = v + 10.F;
    ASSERT_SAME_TYPE(v_sum4, math::Vector<float>);
    assert_true(is_close(v_sum4[0], 11.0F));
    assert_true(is_close(v_sum4[1], 12.0F));
  }

  void should_subtract_two_vectors() {
    math::Vector<int> v1(2);
    v1[0] = 10;
    v1[1] = 20;
    math::Vector<double> v2(2);
    v2[0] = 1;
    v2[1] = 2;
    auto v_diff = v1 - v2;

    assert_true(v_diff.size() == 2);
    assert_true(v_diff[0] == 9);
    assert_true(v_diff[1] == 18);
    ASSERT_SAME_TYPE(v_diff, math::Vector<double>);
  }

  void should_subtract_scalar_and_vector() {
    math::Vector<int> v(2);
    v[0] = 11;
    v[1] = 12;
    auto v_diff = v - 1;
    assert_true(v_diff[0] == 10);
    assert_true(v_diff[1] == 11);

    auto v_diff2 = 100.F - v;
    assert_true(is_close(v_diff2[0], 89.0F));
    assert_true(is_close(v_diff2[1], 88.0F));
    ASSERT_SAME_TYPE(v_diff, math::Vector<int>);
    ASSERT_SAME_TYPE(v_diff2, math::Vector<float>);
  }

  void should_multiply_vector_and_scalar() {
    math::Vector<int> v(2);
    v[0] = 2;
    v[1] = 3;
    auto v_prod = v * 5;
    assert_true(v_prod[0] == 10);
    assert_true(v_prod[1] == 15);

    auto v_prod2 = 5.F * v;
    assert_true(is_close(v_prod2[0], 10.0F));
    assert_true(is_close(v_prod2[1], 15.0F));
    ASSERT_SAME_TYPE(v_prod, math::Vector<int>);
    ASSERT_SAME_TYPE(v_prod2, math::Vector<float>);
  }

  void should_multiply_vector_by_scalar_and_assign() {
    math::Vector<int> v(2);
    v[0] = 2;
    v[1] = 3;
    v *= 5;
    assert_true(v[0] == 10);
    assert_true(v[1] == 15);

    math::Vector<float> v2(2);
    v2[0] = 2.0f;
    v2[1] = 3.0f;
    v2 *= 5.0f;

    assert_true(is_close(v2[0], 10.0f));
    assert_true(is_close(v2[1], 15.0f));
  }

  void should_divide_vector_by_scalar() {
    math::Vector<int> v(2);
    v[0] = 10;
    v[1] = 15;
    auto v_div = v / 5;
    ASSERT_SAME_TYPE(v_div, math::Vector<double>);
    assert_true(v_div[0] == 2);
    assert_true(v_div[1] == 3);

    math::Vector<float> v2(2);
    v2[0] = 10.0f;
    v2[1] = 15.0f;
    auto v_div2 = v2 / 5.0f;
    ASSERT_SAME_TYPE(v_div2, math::Vector<float>);
    assert_true(is_close(v_div2[0], 2.0f));
    assert_true(is_close(v_div2[1], 3.0f));
  }

  void should_divide_vector_by_scalar_and_assign() {
    math::Vector<int> v(2);
    v[0] = 10;
    v[1] = 15;
    v /= 5;
    assert_true(v[0] == 2);
    assert_true(v[1] == 3);

    math::Vector<float> v2(2);
    v2[0] = 10.0f;
    v2[1] = 15.0f;
    v2 /= 5.0f;
    assert_true(is_close(v2[0], 2.0f));
    assert_true(is_close(v2[1], 3.0f));
  }

  void should_calculate_dot_product() {
    math::Vector<int> v1(3);
    v1[0] = 1;
    v1[1] = 2;
    v1[2] = 3;
    math::Vector<int> v2(3);
    v2[0] = 4;
    v2[1] = 5;
    v2[2] = 6;
    assert_true(v1.dot_product(v2) == 32);
    assert_true(v1.transposed() * v2 == 32);

    bool thrown = false;
    try {
      auto v3 = v1 * v2;
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_calculate_outer_product() {
    math::Vector<int> v_col(2, math::Orientation::COLUMN);
    v_col[0] = 1;
    v_col[1] = 2;
    math::Vector<double> v_row(3, math::Orientation::ROW);
    v_row[0] = 3;
    v_row[1] = 4;
    v_row[2] = 5;

    // (2x1) * (1x3) -> (2x3) Matrix
    // [1*3, 1*4, 1*5] = [3, 4, 5]
    // [2*3, 2*4, 2*5] = [6, 8, 10]
    auto m = v_col.outer_product(v_row);
    assert_true(m.row_count() == 2);
    assert_true(m.column_count() == 3);
    assert_true(m.at(0, 0) == 3);
    assert_true(m.at(0, 1) == 4);
    assert_true(m.at(0, 2) == 5);
    assert_true(m.at(1, 0) == 6);
    assert_true(m.at(1, 1) == 8);
    assert_true(m.at(1, 2) == 10);
    ASSERT_SAME_TYPE(m, math::Matrix<double>);
  }
  //
  void should_multiply_row_vector_and_matrix() {
    // (1x2) * (2x2) -> (1x2)
    math::Vector<int> v_row(2, math::Orientation::ROW);
    v_row[0] = 1;
    v_row[1] = 2;

    math::Matrix<int> m(2, 2);
    m.at(0, 0) = 10;
    m.at(0, 1) = 20;
    m.at(1, 0) = 30;
    m.at(1, 1) = 40;

    auto res = v_row * m;
    // res[0] = (1*10) + (2*30) = 70
    // res[1] = (1*20) + (2*40) = 100
    assert_true(res.size() == 2);
    assert_true(res.orientation() == math::Orientation::ROW);
    assert_true(res[0] == 70);
    assert_true(res[1] == 100);

    math::Matrix<double> m1(2, 2);
    m1.at(0, 0) = 10.5;
    m1.at(0, 1) = 20.5;
    m1.at(1, 0) = 30.0;
    m1.at(1, 1) = 40.0;

    auto res1 = v_row * m1;
    // res1[0] = (1*10.5) + (2*30.0) = 70.5
    // res1[1] = (1*20.5) + (2*40.0) = 100.5
    ASSERT_SAME_TYPE(res1, math::Vector<double>);
    assert_true(res1.size() == 2);
    assert_true(is_close(res1[0], 70.5));
    assert_true(is_close(res1[1], 100.5));
  }

 public:
  std::vector<TestTask> get_tests() override {
    return {
        // Constructors
        [this] { should_construct_empty_vector_with_zero_size(); },
        [this] { should_construct_vector_of_given_size(); },
        [this] { should_throw_if_constructed_with_zero_size(); },
        [this] { should_construct_from_raw_data(); },
        [this] { should_throw_if_raw_data_constructor_has_zero_size(); },
        [this] { should_construct_from_std_vector_copy(); },
        [this] { should_throw_if_std_vector_copy_size_mismatch(); },
        [this] { should_construct_from_std_vector_move(); },
        [this] { should_throw_if_std_vector_move_size_mismatch(); },
        [this] { should_construct_from_std_array(); },
        [this] { should_throw_if_std_array_size_mismatch(); },
        // Access & iterators
        [this] { should_access_elements_with_at_and_operator(); },
        [this] { should_throw_on_out_of_bounds_access(); },
        [this] { should_iterate_over_elements(); },
        // Checkers
        [this] { should_check_if_vector_is_null(); },
        // Methods
        [this] { should_fill_vector_with_value(); },
        [this] { should_calculate_l2_norm(); },
        [this] { should_normalize_vector_in_place(); },
        [this] { should_transpose_vector_in_place(); },
        [this] { should_return_transposed_copy(); },
        // Operators
        [this] { should_check_equality(); },
        [this] { should_perform_unary_minus(); },
        [this] { should_add_two_vectors(); },
        [this] { should_add_scalar_and_vector(); },
        [this] { should_subtract_two_vectors(); },
        [this] { should_subtract_scalar_and_vector(); },
        [this] { should_multiply_vector_and_scalar(); },
        [this] { should_multiply_vector_by_scalar_and_assign(); },
        [this] { should_divide_vector_by_scalar(); },
        [this] { should_divide_vector_by_scalar_and_assign(); },
        [this] { should_calculate_dot_product(); },
        [this] { should_calculate_outer_product(); },
        [this] { should_multiply_row_vector_and_matrix(); },
    };
  }
};

}  // namespace maf::test
