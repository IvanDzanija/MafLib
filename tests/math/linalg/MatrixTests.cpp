#include <stdexcept>

#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/LinAlg.hpp"

namespace maf::test {
using namespace maf;
using namespace util;

class MatrixTests : public ITest {
 private:
  //=============================================================================
  // MATRIX CONSTRUCTORS TESTS
  //=============================================================================
  void should_construct_empty_matrix_with_zero_rows_and_columns() {
    math::Matrix<int> m;
    assert_true(m.row_count() == 0);
    assert_true(m.column_count() == 0);
    assert_true(m.size() == 0);
  }

  void should_construct_empty_matrix_of_given_size() {
    math::Matrix<int> m(2, 2);
    assert_true(m.size() == 4);
  }

  void should_throw_if_constructed_with_zero_dimensions() {
    bool thrown = false;
    try {
      math::Matrix<double> m(0, 3);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_raw_data() {
    int data[4] = {1, 2, 3, 4};
    math::Matrix<int> m(2, 2, data);
    assert_true(m.row_count() == 2);
    assert_true(m.column_count() == 2);
    assert_true(m.at(0, 0) == 1);
    assert_true(m.at(0, 1) == 2);
    assert_true(m.at(1, 0) == 3);
    assert_true(m.at(1, 1) == 4);
  }

  void should_throw_if_raw_data_is_null() {
    bool thrown = false;
    try {
      math::Matrix<int> m(2, 2, nullptr);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_std_vector() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    math::Matrix<int> m(2, 3, data);
    assert_true(m.at(0, 0) == 1);
    assert_true(m.at(0, 1) == 2);
    assert_true(m.at(0, 2) == 3);
    assert_true(m.at(1, 0) == 4);
    assert_true(m.at(1, 1) == 5);
    assert_true(m.at(1, 2) == 6);
  }

  void should_throw_if_vector_size_mismatch() {
    std::vector<int> data = {1, 2, 3};
    bool thrown = false;
    try {
      math::Matrix<int> m(2, 2, data);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_nested_vector() {
    std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
    math::Matrix<int> m(2, 2, data);
    assert_true(m.at(0, 0) == 1);
    assert_true(m.at(0, 1) == 2);
    assert_true(m.at(1, 0) == 3);
    assert_true(m.at(1, 1) == 4);
  }

  void should_throw_if_nested_vector_dimensions_mismatch() {
    std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
    bool thrown = false;
    try {
      math::Matrix<int> m(2, 2, data);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_construct_from_std_array() {
    std::array<float, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    math::Matrix<float> m(2, 3, data);
    assert_true(is_close(m.at(1, 2) - 6.0, 0));
  }

  void should_construct_from_initializer_list() {
    math::Matrix<int> m(2, 2, {1, 2, 3, 4});
    assert_true(m.at(0, 0) == 1);
    assert_true(m.at(0, 1) == 2);
    assert_true(m.at(1, 0) == 3);
    assert_true(m.at(1, 1) == 4);
  }

  void should_throw_if_initializer_list_size_mismatch() {
    bool thrown = false;
    try {
      math::Matrix<int> m(2, 2, {1, 2, 3});
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  //=============================================================================
  // MATRIX ELEMENT ACCESS TESTS
  //=============================================================================
  void should_access_row_with_operator() {
    math::Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    assert_true(m[0][2] == 3);
    assert_true(m[1][0] == 4);

    m[1][0] = 10;
    assert_true(m.at(1, 0) == 10);
    assert_true(m[1][0] == 10);
  }

  void should_access_element_with_row_col_operator() {
    math::Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    assert_true((m[0, 0] == 1));
    assert_true((m[0, 2] == 3));
    assert_true((m[1, 1] == 5));

    m[1, 2] = 42;
    assert_true(m.at(1, 2) == 42);
    assert_true((m[1, 2] == 42));
  }

  void should_access_with_const_operator() {
    const math::Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    assert_true(m[0][2] == 3);
    assert_true((m[1, 1] == 5));
    assert_true((m[1, 2] == 6));
  }

  //=============================================================================
  // MATRIX CHECKERS TESTS
  //=============================================================================
  void should_return_true_for_square_matrix() {
    std::array<float, 9> data1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    std::array<float, 6> data2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    math::Matrix<float> m1(3, 3, data1);
    math::Matrix<float> m2(2, 3, data2);
    assert_true(m1.is_square());
    assert_true(!m2.is_square());
  }

  void should_return_true_for_symmetric_matrix() {
    std::array<float, 9> data1 = {1.1, 1.2, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3};
    std::array<float, 6> data2 = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0};
    std::array<float, 9> data3 = {1.1, 0.0, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3};
    math::Matrix<float> m1(3, 3, data1);
    math::Matrix<float> m2(2, 3, data2);
    math::Matrix<float> m3(3, 3, data3);
    assert_true(m1.is_symmetric());
    assert_true(!m2.is_symmetric());
    assert_true(!m3.is_symmetric());
  }

  void should_return_true_for_triangular_matrix() {
    std::array<float, 9> data1 = {1.1, 1.2, 1.3, 0.0, 2.2, 2.3, 0.0, 0, 3.3};
    std::array<float, 9> data2 = {1.1, 0.0, 0.0, 1.5, 2.2, 0.0, 0.0, 0, 3.3};
    std::array<float, 9> data3 = {1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3};
    math::Matrix<float> m1(3, 3, data1);
    math::Matrix<float> m2(3, 3, data2);
    math::Matrix<float> m3(3, 3, data3);
    math::Matrix<float> m4 = math::identity_matrix<float>(3);
    assert_true(m1.is_upper_triangular());
    assert_true(m2.is_lower_triangular());
    assert_true(m3.is_lower_triangular() && m3.is_upper_triangular());
    assert_true(m4.is_lower_triangular() && m4.is_upper_triangular());
  }

  void should_return_true_for_diagonal_matrix() {
    std::array<float, 9> data1 = {1.1, 0, 0, 0, 2.2, 0, 0.0, 0, 3.3};
    std::array<float, 9> data2 = {1.1, 0.0, 0.0, 1, 2.2, 0.0, 0.0, 0, 3.3};
    std::array<float, 9> data3 = {1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3};
    math::Matrix<float> m1(3, 3, data1);
    math::Matrix<float> m2(3, 3, data2);
    math::Matrix<float> m3(3, 3, data3);
    math::Matrix<float> m4 = math::identity_matrix<float>(3);
    assert_true(m1.is_diagonal());
    assert_true(!m2.is_diagonal());
    assert_true(m3.is_diagonal());
    assert_true(m4.is_diagonal());
  }

  void should_return_true_for_positive_definite_matrix() {
    math::Matrix<int> m1(3, 3, {1, 2, 1, 2, 5, 2, 1, 2, 10});
    math::Matrix<int> m2(3, 3, {1, 2, 1, 2, -5, 2, 1, 2, 10});
    assert_true(m1.is_positive_definite());
    assert_true(!m2.is_positive_definite());
  }

  void should_return_true_for_non_square_matrix() {
    math::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    assert_true(m.is_singular() == true);
  }

  void should_return_false_for_non_singular_matrix() {
    math::Matrix<double> m = math::identity_matrix<double>(3);
    math::Matrix<double> m2(2, 2, {1, 2, 3, 4});
    assert_true(m.is_singular() == false);
    assert_true(m2.is_singular() == false);
  }

  void should_return_true_for_singular_matrix() {
    math::Matrix<double> m(2, 2, {1, 2, 2, 4});
    math::Matrix<double> m2(3, 3, {0, 1, 2, 0, 3, 4, 0, 5, 6});
    assert_true(m.is_singular() == true);
    assert_true(m2.is_singular() == true);
  }

  void should_return_false_for_big_non_singular_matrix() {
    math::Matrix<double> A(
        8, 8, {1, 0, 0, 0, 0, 0, 0, 0, 1,  1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 0, -1,
               0, 0, 0, 0, 1, 3, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3});
    assert_true(!A.is_singular());
  }

  //=============================================================================
  // MATRIX METHODS TESTS
  //=============================================================================
  void should_cast_int_matrix_to_float() {
    math::Matrix<int> m_int(2, 2, {1, 2, 3, 4});
    auto m_float = m_int.cast<float>();
    ASSERT_SAME_TYPE(m_float, math::Matrix<float>);

    assert_true(m_float.row_count() == 2);
    assert_true(m_float.column_count() == 2);
    assert_true(is_close(m_float.at(0, 0), 1.0f));
    assert_true(is_close(m_float.at(0, 1), 2.0f));
    assert_true(is_close(m_float.at(1, 0), 3.0f));
    assert_true(is_close(m_float.at(1, 1), 4.0f));
  }

  void should_cast_float_matrix_to_int() {
    math::Matrix<float> m_float(2, 3, {1.7f, 2.3f, 3.9f, 4.1f, 5.5f, 6.8f});
    auto m_int = m_float.cast<int>();
    ASSERT_SAME_TYPE(m_int, math::Matrix<int>);

    assert_true(m_int.row_count() == 2);
    assert_true(m_int.column_count() == 3);
    assert_true(m_int.at(0, 0) == 1);
    assert_true(m_int.at(0, 1) == 2);
    assert_true(m_int.at(0, 2) == 3);
    assert_true(m_int.at(1, 0) == 4);
    assert_true(m_int.at(1, 1) == 5);
    assert_true(m_int.at(1, 2) == 6);
  }

  void should_cast_int_matrix_to_double() {
    math::Matrix<int> m_int(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto m_double = m_int.cast<double>();
    ASSERT_SAME_TYPE(m_double, math::Matrix<double>);

    assert_true(m_double.row_count() == 3);
    assert_true(m_double.column_count() == 3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        assert_true(is_close(m_double.at(i, j), static_cast<double>(m_int.at(i, j))));
      }
    }
  }

  void should_preserve_matrix_properties_after_cast() {
    auto m_int = math::identity_matrix<int>(4);
    auto m_double = m_int.cast<double>();
    ASSERT_SAME_TYPE(m_double, math::Matrix<double>);

    assert_true(m_double.is_square());
    assert_true(m_double.is_diagonal());
    assert_true(m_double.is_symmetric());
    assert_true(m_double.is_upper_triangular());
    assert_true(m_double.is_lower_triangular());
  }

  void should_cast_negative_values_correctly() {
    math::Matrix<int> m_int(2, 2, {-1, -2, -3, -4});
    auto m_float = m_int.cast<float>();
    ASSERT_SAME_TYPE(m_float, math::Matrix<float>);

    assert_true(is_close(m_float.at(0, 0), -1.0f));
    assert_true(is_close(m_float.at(0, 1), -2.0f));
    assert_true(is_close(m_float.at(1, 0), -3.0f));
    assert_true(is_close(m_float.at(1, 1), -4.0f));
  }

  void should_cast_large_matrix_efficiently() {
    const size_t n = 100;
    math::Matrix<int> m_int(n, n);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        m_int.at(i, j) = static_cast<int>(i * n + j);
      }
    }

    auto m_double = m_int.cast<double>();
    ASSERT_SAME_TYPE(m_double, math::Matrix<double>);

    assert_true(is_close(m_double.at(0, 0), 0.0));
    assert_true(is_close(m_double.at(10, 10), 1010.0));
    assert_true(is_close(m_double.at(n - 1, n - 1),
                         static_cast<double>((n - 1) * n + (n - 1))));
  }

  void should_allow_chaining_cast_with_operations() {
    math::Matrix<int> m_int(2, 2, {1, 2, 3, 4});

    auto result = m_int.cast<double>() * 2.5;
    ASSERT_SAME_TYPE(result, math::Matrix<double>);

    assert_true(is_close(result.at(0, 0), 2.5));
    assert_true(is_close(result.at(0, 1), 5.0));
    assert_true(is_close(result.at(1, 0), 7.5));
    assert_true(is_close(result.at(1, 1), 10.0));
  }

  void should_cast_after_matrix_operations() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    math::Matrix<int> b(2, 2, {5, 6, 7, 8});

    auto c = (a + b).cast<float>();
    ASSERT_SAME_TYPE(c, math::Matrix<float>);

    assert_true(is_close(c.at(0, 0), 6.0f));
    assert_true(is_close(c.at(0, 1), 8.0f));
    assert_true(is_close(c.at(1, 0), 10.0f));
    assert_true(is_close(c.at(1, 1), 12.0f));
  }

  void should_fill_matrix_with_value() {
    math::Matrix<int> m(2, 3);
    m.fill(9);
    for (size_t i = 0; i < m.row_count(); ++i) {
      for (size_t j = 0; j < m.column_count(); ++j) {
        assert_true(m.at(i, j) == 9);
      }
    }
  }

  void should_make_identity_matrix() {
    math::Matrix<int> m1(3, 3);
    m1.make_identity();
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        if (i == j) {
          assert_true(m1.at(i, j) == 1);
        } else {
          assert_true(m1.at(i, j) == 0);
        }
      }
    }
  }

  void should_transpose_square_matrix_in_place() {
    math::Matrix<int> m(2, 2, {1, 2, 3, 4});
    m.transpose();
    assert_true(m.at(0, 1) == 3);
    assert_true(m.at(1, 0) == 2);
  }

  void should_return_transposed_copy_for_non_square_matrix() {
    math::Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    math::Matrix<int> t = m.transposed();
    assert_true(m.row_count() == 2);
    assert_true(m.column_count() == 3);
    assert_true(t.row_count() == 3);
    assert_true(t.column_count() == 2);
    assert_true(t.at(1, 0) == 2);
    assert_true(t.at(0, 1) == 4);
  }

  //=============================================================================
  // MATRIX OPERATORS TESTS
  //=============================================================================
  void should_check_equality_between_identical_matrices() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    math::Matrix<int> b(2, 2, {1, 2, 3, 4});
    assert_true(a == b);
  }

  void should_not_be_equal_if_any_element_differs() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    math::Matrix<int> b(2, 2, {1, 9, 3, 4});
    assert_true(!(a == b));
  }

  void should_correctly_perform_unary_minus() {
    math::Matrix<int> m1(2, 3, {1, 2, 3, 4, 5, 6});
    math::Matrix<int> m2(2, 3, {-1, -2, -3, -4, -5, -6});
    assert_true(-m1 == m2);
  }

  void should_add_two_matrices_of_same_size() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    math::Matrix<int> b(2, 2, {10, 20, 30, 40});
    math::Matrix<float> c(2, 2, {1.5, 2.5, 3.5, 4.5});
    auto d = a + b;
    auto e = d + c;
    ASSERT_SAME_TYPE(d, math::Matrix<int>);
    ASSERT_SAME_TYPE(e, math::Matrix<float>);
    assert_true(d.at(0, 0) == 11);
    assert_true(d.at(1, 1) == 44);
    assert_true(e.at(0, 0) == 12.5);
    assert_true(e.at(0, 1) == 24.5);
    assert_true(e.at(1, 0) == 36.5);
    assert_true(e.at(1, 1) == 48.5);
  }

  void should_add_scalar_and_matrix() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    auto c = a + 10;
    auto d = a + 4.5;
    auto e = 4.5 + a;
    ASSERT_SAME_TYPE(c, math::Matrix<int>);
    ASSERT_SAME_TYPE(d, math::Matrix<double>);
    ASSERT_SAME_TYPE(e, math::Matrix<double>);
    assert_true(c.at(0, 0) == 11);
    assert_true(c.at(0, 1) == 12);
    assert_true(c.at(1, 0) == 13);
    assert_true(c.at(1, 1) == 14);
    assert_true(d.at(0, 0) == 5.5);
    assert_true(d.at(0, 1) == 6.5);
    assert_true(d.at(1, 0) == 7.5);
    assert_true(d.at(1, 1) == 8.5);
    assert_true(e == d);
  }

  void should_add_assign_matrix() {
    math::Matrix<float> a(2, 2, {1.5f, 2.5f, 3.5f, 4.5f});
    math::Matrix<int> b(2, 2, {10, 20, 30, 40});
    a += b;
    ASSERT_SAME_TYPE(a, math::Matrix<float>);
    assert_true(is_close(a.at(0, 0), 11.5f));
    assert_true(is_close(a.at(0, 1), 22.5f));
    assert_true(is_close(a.at(1, 0), 33.5f));
    assert_true(is_close(a.at(1, 1), 44.5f));
  }

  void should_add_assign_scalar() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    a += 10;
    ASSERT_SAME_TYPE(a, math::Matrix<int>);
    assert_true(a.at(0, 0) == 11);
    assert_true(a.at(0, 1) == 12);
    assert_true(a.at(1, 0) == 13);
    assert_true(a.at(1, 1) == 14);

    math::Matrix<double> b(2, 2, {1.5, 2.5, 3.5, 4.5});
    b += 0.5;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 2.0));
    assert_true(is_close(b.at(0, 1), 3.0));
    assert_true(is_close(b.at(1, 0), 4.0));
    assert_true(is_close(b.at(1, 1), 5.0));

    b += 10;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 12.0));
    assert_true(is_close(b.at(1, 1), 15.0));
  }

  void should_subtract_two_matrices_of_same_size() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    math::Matrix<int> b(2, 2, {10, 20, 30, 40});
    math::Matrix<float> c(2, 2, {1.5, 2.5, 3.5, 4.5});
    auto d = b - a;
    auto e = b - c;
    ASSERT_SAME_TYPE(d, math::Matrix<int>);
    ASSERT_SAME_TYPE(e, math::Matrix<float>);
    assert_true(d.at(0, 0) == 9);
    assert_true(d.at(1, 1) == 36);
    assert_true(e.at(0, 0) == 8.5);
    assert_true(e.at(0, 1) == 17.5);
    assert_true(e.at(1, 0) == 26.5);
    assert_true(e.at(1, 1) == 35.5);
  }

  void should_subtract_scalar_and_matrix() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    auto c = a - 10;
    auto d = a - 4.5;
    auto e = 4.5 - a;
    ASSERT_SAME_TYPE(c, math::Matrix<int>);
    ASSERT_SAME_TYPE(d, math::Matrix<double>);
    ASSERT_SAME_TYPE(e, math::Matrix<double>);
    assert_true(c.at(0, 0) == -9);
    assert_true(c.at(0, 1) == -8);
    assert_true(c.at(1, 0) == -7);
    assert_true(c.at(1, 1) == -6);
    assert_true(d.at(0, 0) == -3.5);
    assert_true(d.at(0, 1) == -2.5);
    assert_true(d.at(1, 0) == -1.5);
    assert_true(d.at(1, 1) == -0.5);
    assert_true(e.at(0, 0) == 3.5);
    assert_true(e.at(1, 1) == 0.5);
  }

  void should_subtract_assign_matrix() {
    math::Matrix<float> a(2, 2, {11.5f, 22.5f, 33.5f, 44.5f});
    math::Matrix<int> b(2, 2, {10, 20, 30, 40});

    a -= b;
    ASSERT_SAME_TYPE(a, math::Matrix<float>);
    assert_true(is_close(a.at(0, 0), 1.5f));
    assert_true(is_close(a.at(0, 1), 2.5f));
    assert_true(is_close(a.at(1, 0), 3.5f));
    assert_true(is_close(a.at(1, 1), 4.5f));
  }

  void should_subtract_assign_scalar() {
    math::Matrix<int> a(2, 2, {11, 12, 13, 14});

    a -= 10;
    ASSERT_SAME_TYPE(a, math::Matrix<int>);
    assert_true(a.at(0, 0) == 1);
    assert_true(a.at(0, 1) == 2);
    assert_true(a.at(1, 0) == 3);
    assert_true(a.at(1, 1) == 4);

    math::Matrix<double> b(2, 2, {12.0, 13.0, 14.0, 15.0});
    b -= 0.5;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 11.5));
    assert_true(is_close(b.at(0, 1), 12.5));
    assert_true(is_close(b.at(1, 0), 13.5));
    assert_true(is_close(b.at(1, 1), 14.5));

    b -= 10;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 1.5));
    assert_true(is_close(b.at(1, 1), 4.5));
  }

  void should_multiply_matrix_and_scalar() {
    math::Matrix<int> a(2, 2, {1, 2, -3, 4});
    auto b = a * 2.0;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 2.0));
    assert_true(is_close(b.at(0, 1), 4.0));
    assert_true(is_close(b.at(1, 0), -6.0));
    assert_true(is_close(b.at(1, 1), 8.0));

    auto c = 2 * a;
    ASSERT_SAME_TYPE(c, math::Matrix<int>);
    assert_true(is_close(c.at(0, 0), 2.0));
    assert_true(is_close(c.at(0, 1), 4.0));
    assert_true(is_close(c.at(1, 0), -6.0));
    assert_true(is_close(c.at(1, 1), 8.0));

    auto d = a * 2.5f;
    ASSERT_SAME_TYPE(d, math::Matrix<float>);
    assert_true(is_close(d.at(0, 0), 2.5f));
    assert_true(is_close(d.at(1, 1), 10.0f));
  }

  void should_multiply_assign_scalar() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    a *= 2.5f;
    ASSERT_SAME_TYPE(a, math::Matrix<int>);
    assert_true(a.at(0, 0) == 2);
    assert_true(a.at(1, 1) == 10);

    math::Matrix<double> b(2, 2, {1.0, 2.0, 3.0, 4.0});
    b *= 0.5;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 0.5));
    assert_true(is_close(b.at(1, 1), 2.0));
  }

  void should_divide_matrix_and_scalar() {
    math::Matrix<int> a(2, 2, {1, 2, 3, 4});
    auto b = a / 2;
    ASSERT_SAME_TYPE(b, math::Matrix<double>);
    assert_true(is_close(b.at(0, 0), 0.5));
    assert_true(is_close(b.at(1, 1), 2.0));

    math::Matrix<float> c(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    auto d = c / 0.5;
    ASSERT_SAME_TYPE(d, math::Matrix<double>);
    assert_true(is_close(d.at(0, 0), 2.0));
    assert_true(is_close(d.at(1, 1), 8.0));

    math::Matrix<int> e(2, 2, {1, 2, 4, 8});
    auto f = 10.0 / e;
    ASSERT_SAME_TYPE(f, math::Matrix<double>);
    assert_true(is_close(f.at(0, 0), 10.0));
    assert_true(is_close(f.at(0, 1), 5.0));
    assert_true(is_close(f.at(1, 0), 2.5));
    assert_true(is_close(f.at(1, 1), 1.25));
  }

  void should_divide_assign_scalar() {
    math::Matrix<int> a(2, 2, {10, 20, 30, 40});
    a /= 3;
    ASSERT_SAME_TYPE(a, math::Matrix<int>);
    assert_true(a.at(0, 0) == 3);
    assert_true(a.at(1, 1) == 13);

    math::Matrix<float> b(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    b /= 2.0f;
    ASSERT_SAME_TYPE(b, math::Matrix<float>);
    assert_true(is_close(b.at(0, 0), 0.5f));
    assert_true(is_close(b.at(1, 1), 2.0f));

    math::Matrix<float> c(2, 2, {5.0f, 10.0f, 15.0f, 20.0f});
    c /= 2.0;
    ASSERT_SAME_TYPE(c, math::Matrix<float>);
    assert_true(is_close(c.at(0, 0), 2.5f));
  }

  void should_multiply_matrices() {
    math::Matrix<int> a(2, 3, {1, 2, 3, 4, 5, 6});
    math::Matrix<double> b(3, 2, {0.5, 1.5, -1.0, 2.0, 0.0, 1.0});

    math::Matrix<double> expected(
        2, 2,
        {(1 * 0.5F) + (2 * (-1.0)) + (3 * 0.0F), (1 * 1.5F) + (2 * 2.0F) + (3 * 1.0),
         4 * 0.5 + 5 * (-1.0) + 6 * 0.0, 4 * 1.5 + 5 * 2.0 + 6 * 1.0});

    auto result = a * b;
    ASSERT_SAME_TYPE(result, math::Matrix<double>);
    assert_true(result.row_count() == 2 && result.column_count() == 2);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        assert_true(is_close(result.at(i, j), expected.at(i, j)));
      }
    }
  }

  void should_multiply_matrix_and_vector() {
    math::Matrix<float> m(2, 3, {1.0, 0.5, -2.0, 4.0, 1.0, 3.0});
    math::Vector<int> v(3, std::vector<int>{2, 4, 6}, math::Orientation::COLUMN);

    math::Vector<float> expected(
        2,
        std::vector<float>{1.0 * 2 + 0.5 * 4 + (-2.0) * 6, 4.0 * 2 + 1.0 * 4 + 3.0 * 6},
        math::Orientation::COLUMN);

    auto result = m * v;
    ASSERT_SAME_TYPE(result, math::Vector<float>);
    assert_true(result.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
      assert_true(is_close(result.at(i), expected.at(i)));
    }
  }

  void matmul_large_matrix_test() {
    constexpr size_t N = 512;
    math::Matrix<float> A(N, N);
    math::Matrix<float> B(N, N);

    std::mt19937 gen(42);  // Fixed seed: deterministic test
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        A.at(i, j) = dis(gen);
        B.at(i, j) = dis(gen);
      }
    }
    auto C = A * B;

    // Reference accumulated in double: a float reference would drift with
    // K=N terms per element and make the comparison backend- (and run-)
    // dependent.
    math::Matrix<double> D(N, N);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
          D.at(i, j) += A.at(i, k) * B.at(k, j);
        }
      }
    }
    // C is a float gemm result: float summation over 512 terms with
    // cancellation allows ~1% relative / small absolute deviation.
    size_t mismatches = 0;
    double worst_rel = 0.0;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (!is_close(C.at(i, j), D.at(i, j), 1e-2, 0.05)) {
          ++mismatches;
          double rel =
              std::abs(C.at(i, j) - D.at(i, j)) / std::max(std::abs(D.at(i, j)), 1.0);
          worst_rel = std::max(worst_rel, rel);
        }
      }
    }
    assert_true(mismatches == 0);
    if (mismatches > 0) {
      std::cerr << "matmul_large_matrix_test: " << mismatches
                << " mismatches, worst relative error " << worst_rel << "\n";
    }
  }

  //=============================================================================
  // MATRIX PLU TESTS
  //=============================================================================
  void should_throw_if_plu_called_on_non_square_matrix() {
    math::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    bool thrown = false;
    try {
      (void)plu(m);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_throw_for_singular_matrix() {
    math::Matrix<double> m(3, 3, {1, 2, 3, 2, 4, 6, 1, 2, 3});
    bool thrown = false;
    try {
      auto [p, L, U, s] = plu(m);
    } catch (const std::runtime_error &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_correctly_perform_plu_decomposition_on_small_matrix() {
    math::Matrix<double> A(3, 3, {2, 1, 1, 4, -6, 0, -2, 7, 2});

    auto [p, L, U, s] = plu(A);

    assert_true(L.is_square() && U.is_square());
    assert_true(L.row_count() == 3 && U.row_count() == 3);
    assert_true(p.size() == 3);

    for (size_t i = 0; i < 3; ++i) {
      assert_true(is_close(L.at(i, i), 1.0));
      for (size_t j = i + 1; j < 3; ++j) {
        assert_true(is_close(L.at(i, j), 0.0));
      }
    }

    for (size_t i = 1; i < 3; ++i) {
      for (size_t j = 0; j < i; ++j) {
        assert_true(is_close(U.at(i, j), 0.0));
      }
    }
    math::Matrix P_ = math::identity_matrix<double>(3);
    math::Matrix<double> P(3, 3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        P.at(i, j) = P_.at(p.at(i), j);
      }
    }
    math::Matrix<double> PA = P * A;
    math::Matrix<double> LU = L * U;
    assert_true(loosely_equal(PA, LU));
  }

  void should_correctly_handle_identity_matrix_in_plu() {
    math::Matrix<double> I = math::identity_matrix<double>(3);
    auto [P, L, U, s] = plu(I);
    assert_true(L == I);
    assert_true(U == I);
    for (size_t i = 0; i < 3; ++i) {
      assert_true(P.at(i) == i);
    }
  }

  void should_correctly_decompose_upper_triangular_matrix() {
    math::Matrix<double> U_true(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    auto [P, L, U, s] = plu(U_true);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        assert_true(is_close(L.at(i, j), (i == j ? 1.0 : 0.0)));
      }
    }

    math::Matrix<double> Pm(3, 3);

    for (size_t i = 0; i < 3; ++i) {
      Pm.at(i, P.at(i)) = 1.0;
    }
    math::Matrix<double> PA = Pm * U_true;
    math::Matrix<double> LU = L * U;
    assert_true(loosely_equal(PA, LU));
  }

  void should_correctly_handle_negative_pivots_in_plu() {
    math::Matrix<double> A(2, 2, {-4, -5, -2, -1});
    auto [P, L, U, s] = plu(A);
    math::Matrix<double> Pm(2, 2);

    for (size_t i = 0; i < 2; ++i) {
      Pm.at(i, P[i]) = 1.0;
    }
    math::Matrix<double> PA = Pm * A;
    math::Matrix<double> LU = L * U;
    assert_true(loosely_equal(PA, LU));
  }

  //=============================================================================
  // MATRIX CHOLESKY TESTS
  //=============================================================================
  void should_decompose_identity_matrix() {
    math::Matrix<double> I = math::identity_matrix<double>(4);
    math::Matrix<double> L = cholesky(I);
    assert_true(loosely_equal(L, I));
  }

  void should_decompose_known_small_matrix() {
    math::Matrix<double> A(3, 3,
                           {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0});

    math::Matrix<double> expectedL(3, 3,
                                   {2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0});

    math::Matrix<double> L = cholesky(A);
    assert_true(loosely_equal(L, expectedL));
    math::Matrix<double> LLt = L * L.transposed();
    assert_true(loosely_equal(LLt, A));
  }

  void should_decompose_diagonal_matrix() {
    math::Matrix<double> D(3, 3, {9.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 25.0});
    math::Matrix<double> L = cholesky(D);
    math::Matrix<double> expectedL(3, 3, {3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0});
    assert_true(loosely_equal(L, expectedL));
  }

  void should_reconstruct_from_random_b_times_b_t() {
    math::Matrix<double> B(3, 3, {1.0, 2.0, 3.0, 0.5, -1.0, 2.0, 4.0, 0.0, 1.0});
    math::Matrix<double> A = B * B.transposed();
    assert_true(A.is_symmetric());
    assert_true(A.is_positive_definite());

    math::Matrix<double> L = cholesky(A);
    math::Matrix<double> LLt = L * L.transposed();
    assert_true(loosely_equal(LLt, A));
  }

  void should_correctly_decompose_for_known_example() {
    math::Matrix<double> B(3, 3, {1, 2, 1, 2, 5, 2, 1, 2, 10});
    math::Matrix<double> L = cholesky(B);
    math::Matrix<double> LLt = L * L.transposed();
    assert_true(math::loosely_equal(LLt, B));
  }

  void should_throw_if_non_symmetric() {
    math::Matrix<double> A(2, 2, {1.0, 2.0, 3.0, 4.0});
    bool thrown = false;
    try {
      auto L = cholesky(A);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_throw_if_not_positive_definite() {
    math::Matrix<double> A(2, 2, {1.0, 2.0, 2.0, 4.0});
    assert_true(A.is_symmetric());
    bool thrown = false;
    try {
      auto L = cholesky(A);
    } catch (const std::invalid_argument &e) {
      thrown = true;
    }
    assert_true(thrown);
  }

  void should_auto_convert_int_matrix_to_double_in_cholesky() {
    math::Matrix<int> m_int(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});

    auto L = cholesky(m_int);

    ASSERT_SAME_TYPE(L, math::Matrix<double>);

    auto LLt = L * L.transposed();
    auto m_double = m_int.cast<double>();
    assert_true(math::loosely_equal(LLt, m_double));
  }

  void should_preserve_float_type_in_cholesky() {
    math::Matrix<float> m_float(
        3, 3, {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f});
    auto L = cholesky(m_float);

    ASSERT_SAME_TYPE(L, math::Matrix<float>);

    auto LLt = L * L.transposed();
    assert_true(math::loosely_equal(LLt, m_float));
  }

  void should_preserve_double_type_in_cholesky() {
    math::Matrix<double> m_double(
        3, 3, {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0});

    auto L = cholesky(m_double);

    ASSERT_SAME_TYPE(L, math::Matrix<double>);

    auto LLt = L * L.transposed();
    assert_true(math::loosely_equal(LLt, m_double));
  }

  void should_explicitly_convert_int_to_float_in_cholesky() {
    math::Matrix<int> m_int(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});

    auto L = cholesky<float>(m_int);

    ASSERT_SAME_TYPE(L, math::Matrix<float>);

    auto LLt = L * L.transposed();
    auto m_float = m_int.cast<float>();
    assert_true(math::loosely_equal(LLt, m_float));
  }

  void should_explicitly_convert_float_to_double_in_cholesky() {
    math::Matrix<float> m_float(
        3, 3, {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f});

    auto L = cholesky<double>(m_float);

    ASSERT_SAME_TYPE(L, math::Matrix<double>);

    auto LLt = L * L.transposed();
    auto m_double = m_float.cast<double>();
    assert_true(math::loosely_equal(LLt, m_double));
  }

  void should_handle_int_identity_matrix_in_cholesky() {
    auto I_int = math::identity_matrix<int>(4);

    auto L = cholesky(I_int);

    ASSERT_SAME_TYPE(L, math::Matrix<double>);

    auto I_double = math::identity_matrix<double>(4);
    assert_true(math::loosely_equal(L, I_double));
  }

  void should_handle_diagonal_int_matrix_in_cholesky() {
    math::Matrix<int> D(3, 3, {9, 0, 0, 0, 16, 0, 0, 0, 25});

    auto L = cholesky(D);

    math::Matrix<double> expectedL(3, 3, {3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0});
    assert_true(math::loosely_equal(L, expectedL));
  }

  //=============================================================================
  // MATRIX DETERMINANT TESTS
  //=============================================================================
  void should_compute_determinant_of_1x1_matrix() {
    math::Matrix<double> m1(1, 1, {-7.25});
    math::Matrix<double> m2(1, 1, {3.0});
    assert_true(is_close(m1.determinant(), -7.25));
    assert_true(is_close(m2.determinant(), 3.0));
  }

  void should_compute_determinant_of_2x2_matrix() {
    math::Matrix<double> m1(2, 2, {1, 2, 3, 4});
    math::Matrix<double> m2(2, 2, {-4, -5, -2, -1});
    assert_true(is_close(m1.determinant(), -2.0));
    assert_true(is_close(m2.determinant(), -6.0));
  }

  void should_compute_determinant_of_3x3_matrix() {
    math::Matrix<double> m(3, 3, {2, 1, 1, 4, -6, 0, -2, 7, 2});
    assert_true(is_close(m.determinant(), -16.0));
  }

  void should_compute_determinant_of_4x4_matrix() {
    math::Matrix<double> m(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    static constexpr double EXPECTED = -120.0;
    assert_true(is_close(m.determinant(), EXPECTED));
  }

  void should_compute_determinant_of_identity_matrix() {
    math::Matrix<double> I3 = math::identity_matrix<double>(3);
    math::Matrix<double> I4 = math::identity_matrix<double>(4);
    assert_true(is_close(I3.determinant(), 1.0));
    assert_true(is_close(I4.determinant(), 1.0));
  }

  void should_compute_determinant_of_diagonal_matrix() {
    math::Matrix<double> m(3, 3, {9, 0, 0, 0, 16, 0, 0, 0, 25});
    static constexpr double EXPECTED = 3600.0;
    assert_true(is_close(m.determinant(), EXPECTED));
  }

  void should_compute_determinant_of_triangular_matrix() {
    math::Matrix<double> m(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    static constexpr double EXPECTED = 24.0;
    assert_true(is_close(m.determinant(), EXPECTED));
  }

  void should_compute_determinant_of_permutation_matrix() {
    math::Matrix<double> m1(4, 4, {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    math::Matrix<double> m2(3, 3, {1, 0, 0, 0, 0, 1, 0, 1, 0});
    math::Matrix<double> m3(3, 3, {0, 0, 1, 1, 0, 0, 0, 1, 0});
    assert_true(is_close(m1.determinant(), -1.0));
    assert_true(is_close(m2.determinant(), -1.0));
    assert_true(is_close(m3.determinant(), 1.0));
  }

  void should_throw_if_determinant_called_on_non_square_matrix() {
    math::Matrix<double> m1(2, 3, {1, 2, 3, 4, 5, 6});
    math::Matrix<double> m2(3, 2, {1, 2, 3, 4, 5, 6});
    assert_throw<std::invalid_argument>([&] { (void)m1.determinant(); });
    assert_throw<std::invalid_argument>([&] { (void)m2.determinant(); });
  }

  void should_throw_for_singular_matrix_when_computing_determinant() {
    math::Matrix<double> m(2, 2, {1, 2, 2, 4});
    math::Matrix<double> z(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    assert_throw<std::runtime_error>([&] { (void)m.determinant(); });
    assert_throw<std::runtime_error>([&] { (void)z.determinant(); });
  }

  void should_throw_for_empty_matrix_when_computing_determinant() {
    math::Matrix<int> m0;
    assert_throw<std::invalid_argument>([&] { (void)m0.determinant(); });
  }

  void should_compute_determinant_of_int_matrix() {
    math::Matrix<int> m1(2, 2, {1, 2, 3, 4});
    math::Matrix<int> m2(2, 2, {3, 1, 1, 1});
    math::Matrix<int> m3(3, 3, {2, 1, 1, 4, -6, 0, -2, 7, 2});
    math::Matrix<int> m4(3, 3, {9, 0, 0, 0, 16, 0, 0, 0, 25});
    math::Matrix<int> I = math::identity_matrix<int>(3);

    ASSERT_SAME_TYPE(m1.determinant(), double);
    assert_true(m1.determinant() == -2.0);
    assert_true(m2.determinant() == 2.0);
    assert_true(m3.determinant() == -16.0);
    assert_true(m4.determinant() == 3600.0);
    assert_true(I.determinant() == 1.0);
  }

  void should_promote_int_matrix_to_double_when_computing_determinant() {
    math::Matrix<int> m1(3, 3, {2, 3, 1, 4, 7, 5, 6, 8, 9});
    math::Matrix<int> m2(3, 3, {-7, -4, -8, -5, 0, -4, -7, -8, 8});
    math::Matrix<int> m3(2, 2, {3, 1, 1, 1});

    assert_true(is_close(m1.determinant(), 18));
    assert_true(is_close(m2.determinant(), -368));
    assert_true(is_close(m3.determinant(), 2));
  }
  void should_compute_determinant_of_spd_int_matrix() {
    math::Matrix<int> spd(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});
    static constexpr double EXPECTED = 36.0;
    assert_true(is_close(spd.determinant<true>(), EXPECTED));
  }

  void should_compute_determinant_of_float_matrix() {
    math::Matrix<float> m(
        3, 3, {4.0F, 12.0F, -16.0F, 12.0F, 37.0F, -43.0F, -16.0F, -43.0F, 98.0F});
    static constexpr double EXPECTED = 36.0F;
    ASSERT_SAME_TYPE(m.determinant(), float);
    assert_true(is_close(m.determinant(), EXPECTED, 1e-4));
  }

  void should_equal_determinant_of_transpose() {
    math::Matrix<double> m(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    double det = m.determinant();
    double det_t = m.transposed().determinant();
    static constexpr double EXPECTED = -120.0;
    assert_true(is_close(det, EXPECTED));
    assert_true(is_close(det, det_t));
  }

  void should_satisfy_determinant_multiplicativity() {
    math::Matrix<double> A(3, 3, {2, 1, 1, 4, -6, 0, -2, 7, 2});
    math::Matrix<double> B(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    double det_A = A.determinant();
    double det_B = B.determinant();
    double det_AB = (A * B).determinant();
    assert_true(is_close(det_A, -16.0));
    assert_true(is_close(det_B, 24.0));
    assert_true(is_close(det_AB, det_A * det_B));
    assert_true(is_close(det_AB, -384.0));
  }

  void should_compute_determinant_of_spd_matrix() {
    math::Matrix<double> m(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});
    math::Matrix<double> m2(3, 3, {1, 2, 1, 2, 5, 2, 1, 2, 10});
    assert_true(is_close(m.determinant<true>(), 36.0));
    assert_true(is_close(m2.determinant<true>(), 9.0));
  }

  void should_match_spd_and_plu_determinant_paths() {
    math::Matrix<double> m(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});
    math::Matrix<double> d(3, 3, {9, 0, 0, 0, 16, 0, 0, 0, 25});
    assert_true(is_close(m.determinant(), m.determinant<true>()));
    assert_true(is_close(m.determinant(), 36.0));
    assert_true(is_close(d.determinant(), d.determinant<true>()));
    assert_true(is_close(d.determinant(), 3600.0));
  }

  void should_throw_if_spd_determinant_on_non_symmetric_matrix() {
    math::Matrix<double> m(2, 2, {1.0, 2.0, 3.0, 4.0});
    assert_throw<std::invalid_argument>([&] { (void)m.determinant<true>(); });
  }

  void should_throw_if_spd_determinant_on_non_positive_definite_matrix() {
    math::Matrix<double> m(2, 2, {1.0, 2.0, 2.0, 4.0});
    assert_throw<std::invalid_argument>([&] { (void)m.determinant<true>(); });
  }

  //=============================================================================
  // MATRIX QR TESTS
  //=============================================================================

  void should_decompose_identity_matrix_qr() {
    math::Matrix<double> I = math::identity_matrix<double>(4);
    auto [Q, R] = math::qr_decompostion(I);
    assert_true(loosely_equal(Q, I));
    assert_true(loosely_equal(R, I));
  }

  void should_decompose_known_small_matrix_qr() {
    math::Matrix<double> A(3, 3,
                           {10.0, 9.0, 18.0, 20.0, -15.0, -15.0, 20.0, -12.0, 51.0});
    math::Matrix<double> expected_Q(
        3, 3,
        {-1.0 / 3.0, 14.0 / 15.0, -2.0 / 15.0, -2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0,
         -2.0 / 3.0, -2.0 / 15.0, 11.0 / 15.0});
    math::Matrix<double> expected_R(
        3, 3, {-30.0, 15.0, -30.0, 0.0, 15.0, 15.0, 0.0, 0.0, 45.0});
    auto [Q, R] = math::qr_decompostion(A);

    assert_true(loosely_equal(Q, expected_Q));
    assert_true(loosely_equal(R, expected_R));
    assert_true(loosely_equal(Q * R, A));
  }

  void should_throw_on_empty_matrix() {
    math::Matrix<double> A0;
    assert_throw<std::invalid_argument>([&] { (void)math::qr_decompostion(A0); });
  }

  void should_return_thin_q_and_square_r_by_default_square_case() {
    math::Matrix<double> A(5, 5);
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        A.at(i, j) = (i == j) ? 2.0 : (double)((int)i - (int)j);
      }
    }

    auto [Q, R] = math::qr_decompostion(A);
    assert_true(Q.row_count() == 5u);
    assert_true(Q.column_count() == 5u);
    assert_true(R.row_count() == 5u);
    assert_true(R.column_count() == 5u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_return_thin_q_and_square_r_by_default_tall_case() {
    math::Matrix<double> A(7, 3);
    for (size_t i = 0; i < 7; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        A.at(i, j) = ((double)(i + 1) * (double)(j + 2)) - (0.25 * (double)(i));
      }
    }
    auto [Q, R] = math::qr_decompostion(A);
    assert_true(Q.row_count() == 7u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 3u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_return_thin_q_and_square_r_by_default_wide_case() {
    math::Matrix<double> A(3, 7);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 7; ++j) {
        A.at(i, j) = ((double)(i - 1) * 1.5) + ((double)j * 0.2);
      }
    }
    auto [Q, R] = math::qr_decompostion(A);
    assert_true(Q.row_count() == 3u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 7u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_square_fullq_false_fullr_false() {
    math::Matrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    auto [Q, R] = math::qr_decompostion(A, false, false);
    (Q * R).print();
    A.print();
    assert_true(Q.row_count() == 4u);
    assert_true(Q.column_count() == 4u);
    assert_true(R.row_count() == 4u);
    assert_true(R.column_count() == 4u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_square_fullq_true_fullr_false() {
    math::Matrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    auto [Q, R] = math::qr_decompostion(A, true, false);
    assert_true(Q.row_count() == 4u);
    assert_true(Q.column_count() == 4u);
    assert_true(R.row_count() == 4u);
    assert_true(R.column_count() == 4u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_square_fullq_false_fullr_true() {
    math::Matrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    auto [Q, R] = math::qr_decompostion(A, false, true);
    assert_true(Q.row_count() == 4u);
    assert_true(Q.column_count() == 4u);
    assert_true(R.row_count() == 4u);
    assert_true(R.column_count() == 4u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_square_fullq_true_fullr_true() {
    math::Matrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(Q.row_count() == 4u);
    assert_true(Q.column_count() == 4u);
    assert_true(R.row_count() == 4u);
    assert_true(R.column_count() == 4u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_tall_fullq_false_fullr_false() {
    math::Matrix<double> A(6, 3,
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1, 2, -2, 2, 3, 1, 0});
    auto [Q, R] = math::qr_decompostion(A, false, false);
    assert_true(Q.row_count() == 6u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 3u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_tall_fullq_true_fullr_false() {
    math::Matrix<double> A(6, 3,
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1, 2, -2, 2, 3, 1, 0});
    auto [Q, R] = math::qr_decompostion(A, true, false);
    assert_true(Q.row_count() == 6u);
    assert_true(Q.column_count() == 6u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 3u);
    // Matrix multiplication does not work here because of economy output
    // assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_tall_fullq_false_fullr_true() {
    math::Matrix<double> A(6, 3,
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1, 2, -2, 2, 3, 1, 0});
    auto [Q, R] = math::qr_decompostion(A, false, true);
    assert_true(Q.row_count() == 6u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 6u);
    assert_true(R.column_count() == 3u);
    // Matrix multiplication does not work here because of economy output
    // assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_tall_fullq_true_fullr_true() {
    math::Matrix<double> A(6, 3,
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1, 2, -2, 2, 3, 1, 0});
    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(Q.row_count() == 6u);
    assert_true(Q.column_count() == 6u);
    assert_true(R.row_count() == 6u);
    assert_true(R.column_count() == 3u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_wide_fullq_false_fullr_false() {
    math::Matrix<double> A(3, 6,
                           {1, 2, 3, 4, 5, 6, 0, -1, 2, -3, 4, -5, 2, 2, 1, 0, -1, -2});
    auto [Q, R] = math::qr_decompostion(A, false, false);

    assert_true(Q.row_count() == 3u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 6u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_wide_fullq_true_fullr_false() {
    math::Matrix<double> A(3, 6,
                           {1, 2, 3, 4, 5, 6, 0, -1, 2, -3, 4, -5, 2, 2, 1, 0, -1, -2});
    auto [Q, R] = math::qr_decompostion(A, true, false);
    assert_true(Q.row_count() == 3u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 6u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_wide_fullq_false_fullr_true() {
    math::Matrix<double> A(3, 6,
                           {1, 2, 3, 4, 5, 6, 0, -1, 2, -3, 4, -5, 2, 2, 1, 0, -1, -2});
    auto [Q, R] = math::qr_decompostion(A, false, true);
    assert_true(Q.row_count() == 3u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 6u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_wide_fullq_true_fullr_true() {
    math::Matrix<double> A(3, 6,
                           {1, 2, 3, 4, 5, 6, 0, -1, 2, -3, 4, -5, 2, 2, 1, 0, -1, -2});
    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(Q.row_count() == 3u);
    assert_true(Q.column_count() == 3u);
    assert_true(R.row_count() == 3u);
    assert_true(R.column_count() == 6u);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_qr_promote_int_matrix_to_double_result_and_reconstruct() {
    math::Matrix<int> A(4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, -1, 2});
    auto qr = math::qr_decompostion(A);

    math::Matrix<double> Ad(4, 3);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        Ad.at(i, j) = (double)A.at(i, j);
      }
    }
    ASSERT_SAME_TYPE(qr.Q, math::Matrix<double>);
    ASSERT_SAME_TYPE(qr.R, math::Matrix<double>);
    assert_true(loosely_equal(qr.Q * qr.R, Ad));
  }

  void should_qr_work_with_float_input() {
    math::Matrix<float> A(5, 2);
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        A.at(i, j) = ((float)((int)i - 2) * 0.75f) + ((float)j * 0.1f);
      }
    }

    auto qr = math::qr_decompostion(A);

    assert_true(loosely_equal(qr.Q * qr.R, A));
  }

  void should_decompose_1x1_matrix() {
    math::Matrix<double> A(1, 1, {-7.25});
    auto [Q, R] = math::qr_decompostion(A);

    assert_true(Q.row_count() == 1u);
    assert_true(Q.column_count() == 1u);
    assert_true(R.row_count() == 1u);
    assert_true(R.column_count() == 1u);

    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_nx1_column_vector() {
    math::Matrix<double> A(6, 1, {3, -1, 0, 5, 2, -4});
    auto [Q, R] = math::qr_decompostion(A);

    assert_true(Q.row_count() == 6u);
    assert_true(Q.column_count() == 1u);
    assert_true(R.row_count() == 1u);
    assert_true(R.column_count() == 1u);

    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_1xn_row_vector() {
    math::Matrix<double> A(1, 6, {3, -1, 0, 5, 2, -4});
    auto [Q, R] = math::qr_decompostion(A);

    assert_true(Q.row_count() == 1u);
    assert_true(Q.column_count() == 1u);
    assert_true(R.row_count() == 1u);
    assert_true(R.column_count() == 6u);

    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_upper_triangular_matrix() {
    math::Matrix<double> A(4, 4, {5, 2, -1, 3, 0, -4, 7, 1, 0, 0, 2, -6, 0, 0, 0, 9});
    auto [Q, R] = math::qr_decompostion(A);
    assert_true(loosely_equal(Q * R, A));

    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < i; ++j) {
        assert_true(std::abs(R.at(i, j)) < 1e-10);
      }
    }
  }

  void should_decompose_matrix_with_first_column_already_canonical() {
    math::Matrix<double> A(4, 3, {3, 1, 2, 0, -4, 5, 0, 6, -1, 0, 2, 7});
    auto [Q, R] = math::qr_decompostion(A);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_zero_matrix() {
    math::Matrix<double> A(5, 4);
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        A.at(i, j) = 0.0;
      }
    }

    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_rank_deficient_duplicate_columns() {
    math::Matrix<double> A(4, 3, {1, 2, 4, 2, 3, 6, 3, 4, 8, 4, 5, 10});
    auto [Q, R] = math::qr_decompostion(A);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_decompose_matrix_with_zero_column() {
    math::Matrix<double> A(4, 3, {1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8});
    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(loosely_equal(Q * R, A));
  }

  void should_produce_orthonormal_columns_thin_q_tall() {
    math::Matrix<double> A(6, 3,
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1, 2, -2, 2, 3, 1, 0});
    auto [Q, R] = math::qr_decompostion(A, false, false);
    assert_true(loosely_equal(Q * R, A));

    math::Matrix<double> QtQ(3, 3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        double s = 0.0;
        for (size_t r = 0; r < 6; ++r) {
          s += Q.at(r, i) * Q.at(r, j);
        }
        QtQ.at(i, j) = s;
      }
    }
    auto I = math::identity_matrix<double>(3);
    assert_true(loosely_equal(QtQ, I));
  }

  void should_produce_orthonormal_columns_full_q_square() {
    math::Matrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 2, -1, 0, 3, 9, 1, -2, 5});
    auto [Q, R] = math::qr_decompostion(A, false, true);
    assert_true(loosely_equal(Q * R, A));

    math::Matrix<double> QtQ(4, 4);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        double s = 0.0;
        for (size_t r = 0; r < 4; ++r) {
          s += Q.at(r, i) * Q.at(r, j);
        }
        QtQ.at(i, j) = s;
      }
    }
    auto I = math::identity_matrix<double>(4);
    assert_true(loosely_equal(QtQ, I));
  }

  void should_return_upper_triangular_r_in_square_case() {
    math::Matrix<double> A(5, 5);
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        A.at(i, j) = (double)((int)(i + 1) * (int)(j + 2)) - 0.5 * (double)j;
      }
    }

    auto [Q, R] = math::qr_decompostion(A);
    assert_true(loosely_equal(Q * R, A));

    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < i; ++j) {
        assert_true(std::abs(R.at(i, j)) < 1e-10);
      }
    }
  }

  void should_return_upper_triangular_r_in_tall_case_default_r_is_nxn() {
    math::Matrix<double> A(8, 3);
    for (size_t i = 0; i < 8; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        A.at(i, j) = (double)(i + 1) - (2.0 * (double)j) + (0.1 * (double)(i * j));
      }
    }

    auto [Q, R] = math::qr_decompostion(A);
    assert_true(loosely_equal(Q * R, A));

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < i; ++j) {
        assert_true(std::abs(R.at(i, j)) < 1e-10);
      }
    }
  }

  void should_return_upper_triangular_r_in_wide_case_default_r_is_nxn() {
    math::Matrix<double> A(3, 8);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 8; ++j) {
        A.at(i, j) = ((double)(i + 2) * 0.3) + (((double)j - 3) * 0.7);
      }
    }
    A.print();
    std::cout << std::endl;

    auto [Q, R] = math::qr_decompostion(A, true, true);
    assert_true(loosely_equal(Q * R, A));

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < i; ++j) {
        assert_true(std::abs(R.at(i, j)) < 1e-10);
      }
    }
  }

 public:
  std::vector<TestTask> get_tests() override {
    return {
        // Constructors
        [this] { should_construct_empty_matrix_with_zero_rows_and_columns(); },
        [this] { should_construct_empty_matrix_of_given_size(); },
        [this] { should_throw_if_constructed_with_zero_dimensions(); },
        [this] { should_construct_from_raw_data(); },
        [this] { should_throw_if_raw_data_is_null(); },
        [this] { should_construct_from_std_vector(); },
        [this] { should_throw_if_vector_size_mismatch(); },
        [this] { should_construct_from_nested_vector(); },
        [this] { should_throw_if_nested_vector_dimensions_mismatch(); },
        [this] { should_construct_from_std_array(); },
        [this] { should_construct_from_initializer_list(); },
        [this] { should_throw_if_initializer_list_size_mismatch(); },
        // Element access
        [this] { should_access_row_with_operator(); },
        [this] { should_access_element_with_row_col_operator(); },
        [this] { should_access_with_const_operator(); },
        // Checkers
        [this] { should_return_true_for_square_matrix(); },
        [this] { should_return_true_for_symmetric_matrix(); },
        [this] { should_return_true_for_triangular_matrix(); },
        [this] { should_return_true_for_diagonal_matrix(); },
        [this] { should_return_true_for_positive_definite_matrix(); },
        [this] { should_return_true_for_non_square_matrix(); },
        [this] { should_return_false_for_non_singular_matrix(); },
        [this] { should_return_true_for_singular_matrix(); },
        // Cast
        [this] { should_cast_int_matrix_to_float(); },
        [this] { should_cast_float_matrix_to_int(); },
        [this] { should_cast_int_matrix_to_double(); },
        [this] { should_preserve_matrix_properties_after_cast(); },
        [this] { should_cast_negative_values_correctly(); },
        [this] { should_cast_large_matrix_efficiently(); },
        [this] { should_allow_chaining_cast_with_operations(); },
        [this] { should_cast_after_matrix_operations(); },
        // Methods
        [this] { should_fill_matrix_with_value(); },
        [this] { should_make_identity_matrix(); },
        [this] { should_transpose_square_matrix_in_place(); },
        [this] { should_return_transposed_copy_for_non_square_matrix(); },
        // Operators
        [this] { should_check_equality_between_identical_matrices(); },
        [this] { should_not_be_equal_if_any_element_differs(); },
        [this] { should_correctly_perform_unary_minus(); },
        [this] { should_add_two_matrices_of_same_size(); },
        [this] { should_add_scalar_and_matrix(); },
        [this] { should_add_assign_matrix(); },
        [this] { should_add_assign_scalar(); },
        [this] { should_subtract_two_matrices_of_same_size(); },
        [this] { should_subtract_scalar_and_matrix(); },
        [this] { should_subtract_assign_matrix(); },
        [this] { should_subtract_assign_scalar(); },
        [this] { should_multiply_matrix_and_scalar(); },
        [this] { should_multiply_assign_scalar(); },
        [this] { should_divide_matrix_and_scalar(); },
        [this] { should_divide_assign_scalar(); },
        [this] { should_multiply_matrices(); },
        [this] { should_multiply_matrix_and_vector(); },
        [this] { matmul_large_matrix_test(); },
        // PLU
        [this] { should_throw_if_plu_called_on_non_square_matrix(); },
        [this] { should_throw_for_singular_matrix(); },
        [this] { should_correctly_perform_plu_decomposition_on_small_matrix(); },
        [this] { should_correctly_handle_identity_matrix_in_plu(); },
        [this] { should_correctly_decompose_upper_triangular_matrix(); },
        [this] { should_correctly_handle_negative_pivots_in_plu(); },
        // Cholesky
        [this] { should_decompose_identity_matrix(); },
        [this] { should_decompose_known_small_matrix(); },
        [this] { should_correctly_decompose_for_known_example(); },
        [this] { should_decompose_diagonal_matrix(); },
        [this] { should_reconstruct_from_random_b_times_b_t(); },
        [this] { should_throw_if_non_symmetric(); },
        [this] { should_throw_if_not_positive_definite(); },
        [this] { should_auto_convert_int_matrix_to_double_in_cholesky(); },
        [this] { should_preserve_float_type_in_cholesky(); },
        [this] { should_preserve_double_type_in_cholesky(); },
        [this] { should_explicitly_convert_int_to_float_in_cholesky(); },
        [this] { should_explicitly_convert_float_to_double_in_cholesky(); },
        [this] { should_handle_int_identity_matrix_in_cholesky(); },
        [this] { should_handle_diagonal_int_matrix_in_cholesky(); },
        // Determinant
        [this] { should_compute_determinant_of_1x1_matrix(); },
        [this] { should_compute_determinant_of_2x2_matrix(); },
        [this] { should_compute_determinant_of_3x3_matrix(); },
        [this] { should_compute_determinant_of_4x4_matrix(); },
        [this] { should_compute_determinant_of_identity_matrix(); },
        [this] { should_compute_determinant_of_diagonal_matrix(); },
        [this] { should_compute_determinant_of_triangular_matrix(); },
        [this] { should_compute_determinant_of_permutation_matrix(); },
        [this] { should_throw_if_determinant_called_on_non_square_matrix(); },
        [this] { should_throw_for_singular_matrix_when_computing_determinant(); },
        [this] { should_throw_for_empty_matrix_when_computing_determinant(); },
        [this] { should_compute_determinant_of_int_matrix(); },
        [this] { should_promote_int_matrix_to_double_when_computing_determinant(); },
        [this] { should_compute_determinant_of_spd_int_matrix(); },
        [this] { should_compute_determinant_of_float_matrix(); },
        [this] { should_equal_determinant_of_transpose(); },
        [this] { should_satisfy_determinant_multiplicativity(); },
        [this] { should_compute_determinant_of_spd_matrix(); },
        [this] { should_match_spd_and_plu_determinant_paths(); },
        [this] { should_throw_if_spd_determinant_on_non_symmetric_matrix(); },
        [this] { should_throw_if_spd_determinant_on_non_positive_definite_matrix(); },
        // QR
        [this] { should_decompose_identity_matrix_qr(); },
        [this] { should_decompose_known_small_matrix_qr(); },
        [this] { should_throw_on_empty_matrix(); },
        [this] { should_return_thin_q_and_square_r_by_default_square_case(); },
        [this] { should_return_thin_q_and_square_r_by_default_tall_case(); },
        [this] { should_return_thin_q_and_square_r_by_default_wide_case(); },
        [this] { should_decompose_square_fullq_false_fullr_false(); },
        [this] { should_decompose_square_fullq_true_fullr_false(); },
        [this] { should_decompose_square_fullq_false_fullr_true(); },
        [this] { should_decompose_square_fullq_true_fullr_true(); },
        [this] { should_decompose_tall_fullq_false_fullr_false(); },
        [this] { should_decompose_tall_fullq_true_fullr_false(); },
        [this] { should_decompose_tall_fullq_false_fullr_true(); },
        [this] { should_decompose_tall_fullq_true_fullr_true(); },
        [this] { should_decompose_wide_fullq_false_fullr_false(); },
        [this] { should_decompose_wide_fullq_true_fullr_false(); },
        [this] { should_decompose_wide_fullq_false_fullr_true(); },
        [this] { should_decompose_wide_fullq_true_fullr_true(); },
        [this] { should_qr_promote_int_matrix_to_double_result_and_reconstruct(); },
        [this] { should_qr_work_with_float_input(); },
        [this] { should_decompose_1x1_matrix(); },
        [this] { should_decompose_nx1_column_vector(); },
        [this] { should_decompose_1xn_row_vector(); },
        [this] { should_decompose_upper_triangular_matrix(); },
        [this] { should_decompose_matrix_with_first_column_already_canonical(); },
        [this] { should_decompose_zero_matrix(); },
        [this] { should_decompose_rank_deficient_duplicate_columns(); },
        [this] { should_decompose_matrix_with_zero_column(); },
        [this] { should_produce_orthonormal_columns_thin_q_tall(); },
        [this] { should_produce_orthonormal_columns_full_q_square(); },
        [this] { should_return_upper_triangular_r_in_square_case(); },
        [this] { should_return_upper_triangular_r_in_tall_case_default_r_is_nxn(); },
        [this] { should_return_upper_triangular_r_in_wide_case_default_r_is_nxn(); },
    };
  }
};
}  // namespace maf::test
