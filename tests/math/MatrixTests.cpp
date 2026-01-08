#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/Matrix.hpp"
#include "MafLib/math/linalg/MatrixCheckers.hpp"
#include "MafLib/math/linalg/Vector.hpp"

namespace maf::test {
using namespace maf;
using namespace util;
using namespace std::chrono;

class MatrixTests : public ITest {
private:
    //=============================================================================
    // MATRIX CONSTRUCTORS TESTS
    //=============================================================================
    void should_construct_empty_matrix_with_zero_rows_and_columns() {
        math::Matrix<int> m;
        ASSERT_TRUE(m.row_count() == 0);
        ASSERT_TRUE(m.column_count() == 0);
        ASSERT_TRUE(m.size() == 0);
    }

    void should_construct_empty_matrix_of_given_size() {
        math::Matrix<int> m(2, 2);
        ASSERT_TRUE(m.size() == 4);
    }

    void should_throw_if_constructed_with_zero_dimensions() {
        bool thrown = false;
        try {
            math::Matrix<double> m(0, 3);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_raw_data() {
        int data[4] = {1, 2, 3, 4};
        math::Matrix<int> m(2, 2, data);
        ASSERT_TRUE(m.row_count() == 2);
        ASSERT_TRUE(m.column_count() == 2);
        ASSERT_TRUE(m.at(0, 0) == 1);
        ASSERT_TRUE(m.at(0, 1) == 2);
        ASSERT_TRUE(m.at(1, 0) == 3);
        ASSERT_TRUE(m.at(1, 1) == 4);
    }

    void should_throw_if_raw_data_is_null() {
        bool thrown = false;
        try {
            math::Matrix<int> m(2, 2, nullptr);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_std_vector() {
        std::vector<int> data = {1, 2, 3, 4, 5, 6};
        math::Matrix<int> m(2, 3, data);
        ASSERT_TRUE(m.at(0, 0) == 1);
        ASSERT_TRUE(m.at(0, 1) == 2);
        ASSERT_TRUE(m.at(0, 2) == 3);
        ASSERT_TRUE(m.at(1, 0) == 4);
        ASSERT_TRUE(m.at(1, 1) == 5);
        ASSERT_TRUE(m.at(1, 2) == 6);
    }

    void should_throw_if_vector_size_mismatch() {
        std::vector<int> data = {1, 2, 3};
        bool thrown = false;
        try {
            math::Matrix<int> m(2, 2, data);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_nested_vector() {
        std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
        math::Matrix<int> m(2, 2, data);
        ASSERT_TRUE(m.at(0, 0) == 1);
        ASSERT_TRUE(m.at(0, 1) == 2);
        ASSERT_TRUE(m.at(1, 0) == 3);
        ASSERT_TRUE(m.at(1, 1) == 4);
    }

    void should_throw_if_nested_vector_dimensions_mismatch() {
        std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
        bool thrown = false;
        try {
            math::Matrix<int> m(2, 2, data);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_std_array() {
        std::array<float, 6> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        math::Matrix<float> m(2, 3, data);
        ASSERT_TRUE(is_close(m.at(1, 2) - 6.0, 0));
    }

    void should_construct_from_initializer_list() {
        math::Matrix<int> m(2, 2, {1, 2, 3, 4});
        ASSERT_TRUE(m.at(0, 0) == 1);
        ASSERT_TRUE(m.at(0, 1) == 2);
        ASSERT_TRUE(m.at(1, 0) == 3);
        ASSERT_TRUE(m.at(1, 1) == 4);
    }

    void should_throw_if_initializer_list_size_mismatch() {
        bool thrown = false;
        try {
            math::Matrix<int> m(2, 2, {1, 2, 3});
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    //=============================================================================
    // MATRIX CHECKERS TESTS
    //=============================================================================
    void should_return_true_for_square_matrix() {
        std::array<float, 9> data1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        std::array<float, 6> data2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        math::Matrix<float> m1(3, 3, data1);
        math::Matrix<float> m2(2, 3, data2);
        ASSERT_TRUE(m1.is_square());
        ASSERT_TRUE(!m2.is_square());
    }

    void should_return_true_for_symmetric_matrix() {
        std::array<float, 9> data1 = {1.1, 1.2, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3};
        std::array<float, 6> data2 = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0};
        std::array<float, 9> data3 = {1.1, 0.0, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3};
        math::Matrix<float> m1(3, 3, data1);
        math::Matrix<float> m2(2, 3, data2);
        math::Matrix<float> m3(3, 3, data3);
        ASSERT_TRUE(m1.is_symmetric());
        ASSERT_TRUE(!m2.is_symmetric());
        ASSERT_TRUE(!m3.is_symmetric());
    }

    void should_return_true_for_triangular_matrix() {
        std::array<float, 9> data1 = {1.1, 1.2, 1.3, 0.0, 2.2, 2.3, 0.0, 1e-9, 3.3};
        std::array<float, 9> data2 = {1.1, 0.0, 0.0, 1.5, 2.2, 0.0, 0.0, 1e-9, 3.3};
        std::array<float, 9> data3 = {1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3};
        math::Matrix<float> m1(3, 3, data1);
        math::Matrix<float> m2(3, 3, data2);
        math::Matrix<float> m3(3, 3, data3);
        math::Matrix<float> m4 = math::identity_matrix<float>(3);
        ASSERT_TRUE(m1.is_upper_triangular());
        ASSERT_TRUE(m2.is_lower_triangular());
        ASSERT_TRUE(m3.is_lower_triangular() && m3.is_upper_triangular());
        ASSERT_TRUE(m4.is_lower_triangular() && m4.is_upper_triangular());
    }

    void should_return_true_for_diagonal_matrix() {
        std::array<float, 9> data1 = {1.1, 0, 0, 0, 2.2, 0, 0.0, 1e-9, 3.3};
        std::array<float, 9> data2 = {1.1, 0.0, 0.0, 1, 2.2, 0.0, 0.0, 0, 3.3};
        std::array<float, 9> data3 = {1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3};
        math::Matrix<float> m1(3, 3, data1);
        math::Matrix<float> m2(3, 3, data2);
        math::Matrix<float> m3(3, 3, data3);
        math::Matrix<float> m4 = math::identity_matrix<float>(3);
        ASSERT_TRUE(m1.is_diagonal());
        ASSERT_TRUE(!m2.is_diagonal());
        ASSERT_TRUE(m3.is_diagonal());
        ASSERT_TRUE(m4.is_diagonal());
    }

    void should_return_true_for_positive_definite_matrix() {
        math::Matrix<int> m1(3, 3, {1, 2, 1, 2, 5, 2, 1, 2, 10});
        math::Matrix<int> m2(3, 3, {1, 2, 1, 2, -5, 2, 1, 2, 10});
        ASSERT_TRUE(m1.is_positive_definite());
        ASSERT_TRUE(!m2.is_positive_definite());
    }

    void should_return_true_for_non_square_matrix() {
        math::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
        ASSERT_TRUE(m.is_singular() == true);
    }

    void should_return_false_for_non_singular_matrix() {
        math::Matrix<double> m = math::identity_matrix<double>(3);
        math::Matrix<double> m2(2, 2, {1, 2, 3, 4});
        ASSERT_TRUE(m.is_singular() == false);
        ASSERT_TRUE(m2.is_singular() == false);
    }

    void should_return_true_for_singular_matrix() {
        math::Matrix<double> m(2, 2, {1, 2, 2, 4});
        math::Matrix<double> m2(3, 3, {0, 1, 2, 0, 3, 4, 0, 5, 6});
        ASSERT_TRUE(m.is_singular() == true);
        ASSERT_TRUE(m2.is_singular() == true);
    }

    void should_return_false_for_big_non_singular_matrix() {
        math::Matrix<double> A(
            8, 8, {1, 0, 0, 0, 0, 0, 0, 0, 1,  1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 0, -1,
                   0, 0, 0, 0, 1, 3, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3});
        ASSERT_TRUE(!A.is_singular());
    }

    //=============================================================================
    // MATRIX METHODS TESTS
    //=============================================================================
    void should_cast_int_matrix_to_float() {
        math::Matrix<int> m_int(2, 2, {1, 2, 3, 4});
        auto m_float = m_int.cast<float>();

        ASSERT_TRUE(m_float.row_count() == 2);
        ASSERT_TRUE(m_float.column_count() == 2);
        ASSERT_TRUE(is_close(m_float.at(0, 0) + 0.5f, 1.5f));
    }

    void should_cast_float_matrix_to_int() {
        math::Matrix<float> m_float(2, 3, {1.7f, 2.3f, 3.9f, 4.1f, 5.5f, 6.8f});
        auto m_int = m_float.cast<int>();

        ASSERT_TRUE(m_int.row_count() == 2);
        ASSERT_TRUE(m_int.column_count() == 3);
        ASSERT_TRUE(m_int.at(0, 0) == 1);
        ASSERT_TRUE(m_int.at(0, 1) == 2);
        ASSERT_TRUE(m_int.at(0, 2) == 3);
        ASSERT_TRUE(m_int.at(1, 0) == 4);
        ASSERT_TRUE(m_int.at(1, 1) == 5);
        ASSERT_TRUE(m_int.at(1, 2) == 6);
    }

    void should_cast_int_matrix_to_double() {
        math::Matrix<int> m_int(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto m_double = m_int.cast<double>();

        ASSERT_TRUE(m_double.row_count() == 3);
        ASSERT_TRUE(m_double.column_count() == 3);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                ASSERT_TRUE(is_close(m_double.at(i, j) + 0.5f,
                                     static_cast<double>(m_int.at(i, j)) + 0.5f));
            }
        }
    }

    void should_preserve_matrix_properties_after_cast() {
        auto m_int = math::identity_matrix<int>(4);
        auto m_double = m_int.cast<double>();

        ASSERT_TRUE(m_double.is_square());
        ASSERT_TRUE(m_double.is_diagonal());
        ASSERT_TRUE(m_double.is_symmetric());
        ASSERT_TRUE(m_double.is_upper_triangular());
        ASSERT_TRUE(m_double.is_lower_triangular());
    }

    void should_cast_negative_values_correctly() {
        math::Matrix<int> m_int(2, 2, {-1, -2, -3, -4});
        auto m_float = m_int.cast<float>();

        ASSERT_TRUE(is_close(m_float.at(0, 0), -1.0f));
        ASSERT_TRUE(is_close(m_float.at(0, 1), -2.0f));
        ASSERT_TRUE(is_close(m_float.at(1, 0), -3.0f));
        ASSERT_TRUE(is_close(m_float.at(1, 1), -4.0f));
    }

    void should_cast_large_matrix_efficiently() {
        const size_t n = 100;
        math::Matrix<int> m_int(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                m_int.at(i, j) = static_cast<int>(i * n + j);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto m_double = m_int.cast<double>();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        ASSERT_TRUE(is_close(m_double.at(0, 0), 0.0));
        ASSERT_TRUE(is_close(m_double.at(10, 10), 1010.0));
        ASSERT_TRUE(is_close(m_double.at(n - 1, n - 1),
                             static_cast<double>((n - 1) * n + (n - 1))));

        std::cout << "Cast (" << n << "x" << n << ") elapsed time: " << elapsed.count()
                  << " seconds\n";
    }

    void should_allow_chaining_cast_with_operations() {
        math::Matrix<int> m_int(2, 2, {1, 2, 3, 4});

        auto result = m_int.cast<double>() * 2.5;

        ASSERT_TRUE(is_close(result.at(0, 0), 2.5));
        ASSERT_TRUE(is_close(result.at(0, 1), 5.0));
        ASSERT_TRUE(is_close(result.at(1, 0), 7.5));
        ASSERT_TRUE(is_close(result.at(1, 1), 10.0));
    }

    void should_cast_after_matrix_operations() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        math::Matrix<int> b(2, 2, {5, 6, 7, 8});

        auto c = (a + b).cast<float>();

        ASSERT_TRUE(is_close(c.at(0, 0), 6.0f));
        ASSERT_TRUE(is_close(c.at(0, 1), 8.0f));
        ASSERT_TRUE(is_close(c.at(1, 0), 10.0f));
        ASSERT_TRUE(is_close(c.at(1, 1), 12.0f));
    }

    void should_fill_matrix_with_value() {
        math::Matrix<int> m(2, 3);
        m.fill(9);
        for (size_t i = 0; i < m.row_count(); ++i) {
            for (size_t j = 0; j < m.column_count(); ++j) {
                ASSERT_TRUE(m.at(i, j) == 9);
            }
        }
    }

    void should_make_identity_matrix() {
        math::Matrix<int> m1(3, 3);
        m1.make_identity();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (i == j) {
                    ASSERT_TRUE(m1.at(i, j) == 1);
                } else {
                    ASSERT_TRUE(m1.at(i, j) == 0);
                }
            }
        }
    }

    void should_transpose_square_matrix_in_place() {
        math::Matrix<int> m(2, 2, {1, 2, 3, 4});
        m.transpose();
        ASSERT_TRUE(m.at(0, 1) == 3);
        ASSERT_TRUE(m.at(1, 0) == 2);
    }

    void should_return_transposed_copy_for_non_square_matrix() {
        math::Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
        math::Matrix<int> t = m.transposed();
        ASSERT_TRUE(m.row_count() == 2);
        ASSERT_TRUE(m.column_count() == 3);
        ASSERT_TRUE(t.row_count() == 3);
        ASSERT_TRUE(t.column_count() == 2);
        ASSERT_TRUE(t.at(1, 0) == 2);
        ASSERT_TRUE(t.at(0, 1) == 4);
    }

    //=============================================================================
    // MATRIX OPERATORS TESTS
    //=============================================================================
    void should_check_equality_between_identical_matrices() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        math::Matrix<int> b(2, 2, {1, 2, 3, 4});
        ASSERT_TRUE(a == b);
    }

    void should_not_be_equal_if_any_element_differs() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        math::Matrix<int> b(2, 2, {1, 9, 3, 4});
        ASSERT_TRUE(!(a == b));
    }

    void should_correctly_perform_unary_minus() {
        math::Matrix<int> m1(2, 3, {1, 2, 3, 4, 5, 6});
        math::Matrix<int> m2(2, 3, {-1, -2, -3, -4, -5, -6});
        ASSERT_TRUE(-m1 == m2);
    }

    void should_add_two_matrices_of_same_size() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        math::Matrix<int> b(2, 2, {10, 20, 30, 40});
        math::Matrix<float> c(2, 2, {1.5, 2.5, 3.5, 4.5});
        auto d = a + b;
        auto e = d + c;
        ASSERT_TRUE(d.at(0, 0) == 11);
        ASSERT_TRUE(d.at(1, 1) == 44);
        ASSERT_TRUE(e.at(0, 0) == 12.5);
        ASSERT_TRUE(e.at(0, 1) == 24.5);
        ASSERT_TRUE(e.at(1, 0) == 36.5);
        ASSERT_TRUE(e.at(1, 1) == 48.5);
    }

    void should_add_scalar_and_matrix() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        auto c = a + 10;
        auto d = a + 4.5;
        auto e = 4.5 + a;
        ASSERT_TRUE(c.at(0, 0) == 11);
        ASSERT_TRUE(c.at(0, 1) == 12);
        ASSERT_TRUE(c.at(1, 0) == 13);
        ASSERT_TRUE(c.at(1, 1) == 14);
        ASSERT_TRUE(d.at(0, 0) == 5.5);
        ASSERT_TRUE(d.at(0, 1) == 6.5);
        ASSERT_TRUE(d.at(1, 0) == 7.5);
        ASSERT_TRUE(d.at(1, 1) == 8.5);
        ASSERT_TRUE(e == d);
    }

    void should_add_assign_matrix() {
        math::Matrix<float> a(2, 2, {1.5f, 2.5f, 3.5f, 4.5f});
        math::Matrix<int> b(2, 2, {10, 20, 30, 40});

        a += b;

        ASSERT_TRUE(is_close(a.at(0, 0), 11.5f));
        ASSERT_TRUE(is_close(a.at(0, 1), 22.5f));
        ASSERT_TRUE(is_close(a.at(1, 0), 33.5f));
        ASSERT_TRUE(is_close(a.at(1, 1), 44.5f));
    }

    void should_add_assign_scalar() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});

        a += 10;

        ASSERT_TRUE(a.at(0, 0) == 11);
        ASSERT_TRUE(a.at(0, 1) == 12);
        ASSERT_TRUE(a.at(1, 0) == 13);
        ASSERT_TRUE(a.at(1, 1) == 14);

        math::Matrix<double> b(2, 2, {1.5, 2.5, 3.5, 4.5});
        b += 0.5;
        ASSERT_TRUE(is_close(b.at(0, 0), 2.0));
        ASSERT_TRUE(is_close(b.at(0, 1), 3.0));
        ASSERT_TRUE(is_close(b.at(1, 0), 4.0));
        ASSERT_TRUE(is_close(b.at(1, 1), 5.0));

        b += 10;
        ASSERT_TRUE(is_close(b.at(0, 0), 12.0));
        ASSERT_TRUE(is_close(b.at(1, 1), 15.0));
    }

    void should_subtract_two_matrices_of_same_size() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        math::Matrix<int> b(2, 2, {10, 20, 30, 40});
        math::Matrix<float> c(2, 2, {1.5, 2.5, 3.5, 4.5});
        auto d = b - a;
        auto e = b - c;
        ASSERT_TRUE(d.at(0, 0) == 9);
        ASSERT_TRUE(d.at(1, 1) == 36);
        ASSERT_TRUE(e.at(0, 0) == 8.5);
        ASSERT_TRUE(e.at(0, 1) == 17.5);
        ASSERT_TRUE(e.at(1, 0) == 26.5);
        ASSERT_TRUE(e.at(1, 1) == 35.5);
    }

    void should_subtract_scalar_and_matrix() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        auto c = a - 10;
        auto d = a - 4.5;
        auto e = 4.5 - a;
        ASSERT_TRUE(c.at(0, 0) == -9);
        ASSERT_TRUE(c.at(0, 1) == -8);
        ASSERT_TRUE(c.at(1, 0) == -7);
        ASSERT_TRUE(c.at(1, 1) == -6);
        ASSERT_TRUE(d.at(0, 0) == -3.5);
        ASSERT_TRUE(d.at(0, 1) == -2.5);
        ASSERT_TRUE(d.at(1, 0) == -1.5);
        ASSERT_TRUE(d.at(1, 1) == -0.5);
        ASSERT_TRUE(e.at(0, 0) == 3.5);
        ASSERT_TRUE(e.at(1, 1) == 0.5);
    }

    void should_subtract_assign_matrix() {
        math::Matrix<float> a(2, 2, {11.5f, 22.5f, 33.5f, 44.5f});
        math::Matrix<int> b(2, 2, {10, 20, 30, 40});

        a -= b;

        ASSERT_TRUE(is_close(a.at(0, 0), 1.5f));
        ASSERT_TRUE(is_close(a.at(0, 1), 2.5f));
        ASSERT_TRUE(is_close(a.at(1, 0), 3.5f));
        ASSERT_TRUE(is_close(a.at(1, 1), 4.5f));
    }

    void should_subtract_assign_scalar() {
        math::Matrix<int> a(2, 2, {11, 12, 13, 14});

        a -= 10;

        ASSERT_TRUE(a.at(0, 0) == 1);
        ASSERT_TRUE(a.at(0, 1) == 2);
        ASSERT_TRUE(a.at(1, 0) == 3);
        ASSERT_TRUE(a.at(1, 1) == 4);

        math::Matrix<double> b(2, 2, {12.0, 13.0, 14.0, 15.0});
        b -= 0.5;
        ASSERT_TRUE(is_close(b.at(0, 0), 11.5));
        ASSERT_TRUE(is_close(b.at(0, 1), 12.5));
        ASSERT_TRUE(is_close(b.at(1, 0), 13.5));
        ASSERT_TRUE(is_close(b.at(1, 1), 14.5));

        b -= 10;
        ASSERT_TRUE(is_close(b.at(0, 0), 1.5));
        ASSERT_TRUE(is_close(b.at(1, 1), 4.5));
    }

    void should_multiply_matrix_and_scalar() {
        math::Matrix<int> a(2, 2, {1, 2, -3, 4});
        auto b = a * 2.0;
        ASSERT_TRUE(is_close(b.at(0, 0), 2.0));
        ASSERT_TRUE(is_close(b.at(0, 1), 4.0));
        ASSERT_TRUE(is_close(b.at(1, 0), -6.0));
        ASSERT_TRUE(is_close(b.at(1, 1), 8.0));

        auto c = 2 * a;
        ASSERT_TRUE(is_close(c.at(0, 0), 2.0));
        ASSERT_TRUE(is_close(c.at(0, 1), 4.0));
        ASSERT_TRUE(is_close(c.at(1, 0), -6.0));
        ASSERT_TRUE(is_close(c.at(1, 1), 8.0));

        auto d = a * 2.5f;
        ASSERT_TRUE(is_close(d.at(0, 0), 2.5f));
        ASSERT_TRUE(is_close(d.at(1, 1), 10.0f));
    }

    void should_multiply_assign_scalar() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        a *= 2.5f;

        ASSERT_TRUE(a.at(0, 0) == 2);
        ASSERT_TRUE(a.at(1, 1) == 10);

        math::Matrix<double> b(2, 2, {1.0, 2.0, 3.0, 4.0});
        b *= 0.5;
        ASSERT_TRUE(is_close(b.at(0, 0), 0.5));
        ASSERT_TRUE(is_close(b.at(1, 1), 2.0));
    }

    void should_divide_matrix_and_scalar() {
        math::Matrix<int> a(2, 2, {1, 2, 3, 4});
        auto b = a / 2;

        ASSERT_TRUE(is_close(b.at(0, 0), 0.5));
        ASSERT_TRUE(is_close(b.at(1, 1), 2.0));

        math::Matrix<float> c(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        auto d = c / 0.5;

        ASSERT_TRUE(is_close(d.at(0, 0), 2.0));
        ASSERT_TRUE(is_close(d.at(1, 1), 8.0));

        math::Matrix<int> e(2, 2, {1, 2, 4, 8});
        auto f = 10.0 / e;

        ASSERT_TRUE(is_close(f.at(0, 0), 10.0));
        ASSERT_TRUE(is_close(f.at(0, 1), 5.0));
        ASSERT_TRUE(is_close(f.at(1, 0), 2.5));
        ASSERT_TRUE(is_close(f.at(1, 1), 1.25));
    }

    void should_divide_assign_scalar() {
        math::Matrix<int> a(2, 2, {10, 20, 30, 40});
        a /= 3;
        ASSERT_TRUE(a.at(0, 0) == 3);
        ASSERT_TRUE(a.at(1, 1) == 13);

        math::Matrix<float> b(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        b /= 2.0f;
        ASSERT_TRUE(is_close(b.at(0, 0), 0.5f));
        ASSERT_TRUE(is_close(b.at(1, 1), 2.0f));

        math::Matrix<float> c(2, 2, {5.0f, 10.0f, 15.0f, 20.0f});
        c /= 2.0;
        ASSERT_TRUE(is_close(c.at(0, 0), 2.5f));
        static_assert(std::is_same_v<typename decltype(c)::value_type, float>);
    }

    void should_multiply_matrices() {
        math::Matrix<int> a(2, 3, {1, 2, 3, 4, 5, 6});
        math::Matrix<double> b(3, 2, {0.5, 1.5, -1.0, 2.0, 0.0, 1.0});

        math::Matrix<double> expected(2,
                                      2,
                                      {1 * 0.5 + 2 * (-1.0) + 3 * 0.0,
                                       1 * 1.5 + 2 * 2.0 + 3 * 1.0,
                                       4 * 0.5 + 5 * (-1.0) + 6 * 0.0,
                                       4 * 1.5 + 5 * 2.0 + 6 * 1.0});

        auto result = a * b;
        ASSERT_TRUE(result.row_count() == 2 && result.column_count() == 2);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                ASSERT_TRUE(is_close(result.at(i, j), expected.at(i, j)));
            }
        }
    }

    void should_multiply_matrix_and_vector() {
        math::Matrix<float> m(2, 3, {1.0, 0.5, -2.0, 4.0, 1.0, 3.0});
        math::Vector<int> v(3, std::vector<int>{2, 4, 6}, math::COLUMN);

        math::Vector<float> expected(2,
                                     std::vector<float>{1.0 * 2 + 0.5 * 4 + (-2.0) * 6,
                                                        4.0 * 2 + 1.0 * 4 + 3.0 * 6},
                                     math::COLUMN);

        auto result = m * v;
        ASSERT_TRUE(result.size() == 2);
        for (size_t i = 0; i < 2; ++i) {
            ASSERT_TRUE(is_close(result.at(i), expected.at(i)));
        }
    }

    void matmul_time_test() {
        const size_t n = 1024;
        math::Matrix<double> A(n, n);
        math::Matrix<double> B(n, n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A.at(i, j) = dis(gen);
                B.at(i, j) = dis(gen);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto C = A * B;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "MATMUL elapsed time:" << elapsed.count() << " seconds.\n";

        math::Matrix<double> D(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    D.at(i, j) += A.at(i, k) * B.at(k, j);
                }
            }
        }
        ASSERT_TRUE(math::loosely_equal(C, D));
    }

    //=============================================================================
    // MATRIX PLU TESTS
    //=============================================================================
    void should_throw_if_plu_called_on_non_square_matrix() {
        math::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
        bool thrown = false;
        try {
            auto [P, L, U] = plu(m);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_throw_for_singular_matrix() {
        math::Matrix<double> m(3, 3, {1, 2, 3, 2, 4, 6, 1, 2, 3});
        bool thrown = false;
        try {
            auto [p, L, U] = math::plu(m);
        } catch (const std::runtime_error& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_correctly_perform_plu_decomposition_on_small_matrix() {
        math::Matrix<double> A(3, 3, {2, 1, 1, 4, -6, 0, -2, 7, 2});

        auto [p, L, U] = plu(A);

        ASSERT_TRUE(L.is_square() && U.is_square());
        ASSERT_TRUE(L.row_count() == 3 && U.row_count() == 3);
        ASSERT_TRUE(p.size() == 3);

        for (size_t i = 0; i < 3; ++i) {
            ASSERT_TRUE(is_close(L.at(i, i), 1.0));
            for (size_t j = i + 1; j < 3; ++j) {
                ASSERT_TRUE(is_close(L.at(i, j), 0.0));
            }
        }

        for (size_t i = 1; i < 3; ++i) {
            for (size_t j = 0; j < i; ++j) {
                ASSERT_TRUE(is_close(U.at(i, j), 0.0));
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
        ASSERT_TRUE(loosely_equal(PA, LU));
    }

    void should_correctly_handle_identity_matrix_in_plu() {
        math::Matrix<double> I = math::identity_matrix<double>(3);
        auto [P, L, U] = plu(I);
        ASSERT_TRUE(L == I);
        ASSERT_TRUE(U == I);
        for (size_t i = 0; i < 3; ++i) {
            ASSERT_TRUE(P.at(i) == i);
        }
    }

    void should_correctly_decompose_upper_triangular_matrix() {
        math::Matrix<double> U_true(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
        auto [P, L, U] = plu(U_true);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                ASSERT_TRUE(is_close(L.at(i, j), (i == j ? 1.0 : 0.0)));
            }
        }

        math::Matrix<double> Pm(3, 3);

        for (size_t i = 0; i < 3; ++i) {
            Pm.at(i, P.at(i)) = 1.0;
        }
        math::Matrix<double> PA = Pm * U_true;
        math::Matrix<double> LU = L * U;
        ASSERT_TRUE(loosely_equal(PA, LU));
    }

    void should_correctly_handle_negative_pivots_in_plu() {
        math::Matrix<double> A(2, 2, {-4, -5, -2, -1});
        auto [P, L, U] = plu(A);
        math::Matrix<double> Pm(2, 2);

        for (size_t i = 0; i < 2; ++i) {
            Pm.at(i, P[i]) = 1.0;
        }
        math::Matrix<double> PA = Pm * A;
        math::Matrix<double> LU = L * U;
        ASSERT_TRUE(loosely_equal(PA, LU));
    }

    void plu_time_test() {
        const size_t n = 1000;
        math::Matrix<double> A(n, n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A.at(i, j) = dis(gen);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto [p, L, U] = plu(A);

        auto P = math::permutation_matrix<double>(p);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "PLU elapsed time:" << elapsed.count() << " seconds.\n";
        auto PA = P * A;
        auto LU = L * U;
        ASSERT_TRUE(math::loosely_equal(PA, LU));
    }

    //=============================================================================
    // MATRIX CHOLESKY TESTS
    //=============================================================================
    void should_decompose_identity_matrix() {
        math::Matrix<double> I = math::identity_matrix<double>(4);
        math::Matrix<double> L = cholesky(I);
        ASSERT_TRUE(loosely_equal(L, I));
    }

    void should_decompose_known_small_matrix() {
        math::Matrix<double> A(
            3, 3, {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0});

        math::Matrix<double> expectedL(
            3, 3, {2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0});

        math::Matrix<double> L = cholesky(A);
        ASSERT_TRUE(loosely_equal(L, expectedL));
        math::Matrix<double> LLt = L * L.transposed();
        ASSERT_TRUE(loosely_equal(LLt, A));
    }

    void should_decompose_diagonal_matrix() {
        math::Matrix<double> D(3, 3, {9.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 25.0});
        math::Matrix<double> L = cholesky(D);
        math::Matrix<double> expectedL(
            3, 3, {3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0});
        ASSERT_TRUE(loosely_equal(L, expectedL));
    }

    void should_reconstruct_from_random_b_times_b_t() {
        math::Matrix<double> B(3, 3, {1.0, 2.0, 3.0, 0.5, -1.0, 2.0, 4.0, 0.0, 1.0});
        math::Matrix<double> A = B * B.transposed();
        ASSERT_TRUE(A.is_symmetric());
        ASSERT_TRUE(A.is_positive_definite());

        math::Matrix<double> L = cholesky(A);
        math::Matrix<double> LLt = L * L.transposed();
        ASSERT_TRUE(loosely_equal(LLt, A));
    }

    void should_correctly_decompose_for_known_example() {
        math::Matrix<double> B(3, 3, {1, 2, 1, 2, 5, 2, 1, 2, 10});
        math::Matrix<double> L = cholesky(B);
        math::Matrix<double> LLt = L * L.transposed();
        ASSERT_TRUE(math::loosely_equal(LLt, B));
    }

    void should_throw_if_non_symmetric() {
        math::Matrix<double> A(2, 2, {1.0, 2.0, 3.0, 4.0});
        bool thrown = false;
        try {
            auto L = cholesky(A);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_throw_if_not_positive_definite() {
        math::Matrix<double> A(2, 2, {1.0, 2.0, 2.0, 4.0});
        ASSERT_TRUE(A.is_symmetric());
        bool thrown = false;
        try {
            auto L = cholesky(A);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_auto_convert_int_matrix_to_double_in_cholesky() {
        math::Matrix<int> m_int(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});

        auto L = cholesky(m_int);

        static_assert(std::is_same_v<decltype(L), math::Matrix<double>>,
                      "Integer matrix should auto-promote to double");

        auto LLt = L * L.transposed();
        auto m_double = m_int.cast<double>();
        ASSERT_TRUE(math::loosely_equal(LLt, m_double));
    }

    void should_preserve_float_type_in_cholesky() {
        math::Matrix<float> m_float(
            3, 3, {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f});
        auto L = cholesky(m_float);

        static_assert(std::is_same_v<decltype(L), math::Matrix<float>>,
                      "Float matrix should return float");

        auto LLt = L * L.transposed();
        ASSERT_TRUE(math::loosely_equal(LLt, m_float));
    }

    void should_preserve_double_type_in_cholesky() {
        math::Matrix<double> m_double(
            3, 3, {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0});

        auto L = cholesky(m_double);

        static_assert(std::is_same_v<decltype(L), math::Matrix<double>>,
                      "Double matrix should return double");

        auto LLt = L * L.transposed();
        ASSERT_TRUE(math::loosely_equal(LLt, m_double));
    }

    void should_explicitly_convert_int_to_float_in_cholesky() {
        math::Matrix<int> m_int(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});

        auto L = cholesky<float>(m_int);

        static_assert(std::is_same_v<decltype(L), math::Matrix<float>>,
                      "Explicit float template should return float");

        auto LLt = L * L.transposed();
        auto m_float = m_int.cast<float>();
        ASSERT_TRUE(math::loosely_equal(LLt, m_float));
    }

    void should_explicitly_convert_float_to_double_in_cholesky() {
        math::Matrix<float> m_float(
            3, 3, {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f});

        auto L = cholesky<double>(m_float);

        static_assert(std::is_same_v<decltype(L), math::Matrix<double>>,
                      "Explicit double template should return double");

        auto LLt = L * L.transposed();
        auto m_double = m_float.cast<double>();
        ASSERT_TRUE(math::loosely_equal(LLt, m_double));
    }

    void should_handle_int_identity_matrix_in_cholesky() {
        auto I_int = math::identity_matrix<int>(4);

        auto L = cholesky(I_int);

        static_assert(std::is_same_v<decltype(L), math::Matrix<double>>);

        auto I_double = math::identity_matrix<double>(4);
        ASSERT_TRUE(math::loosely_equal(L, I_double));
    }

    void should_handle_diagonal_int_matrix_in_cholesky() {
        math::Matrix<int> D(3, 3, {9, 0, 0, 0, 16, 0, 0, 0, 25});

        auto L = cholesky(D);

        math::Matrix<double> expectedL(
            3, 3, {3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0});
        ASSERT_TRUE(math::loosely_equal(L, expectedL));
    }

    void cholesky_time_test() {
        const size_t n = 1000;
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0);
        math::Matrix<double> X(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                X.at(i, j) = dist(gen);
            }
        }

        double eps = 1e-7;
        auto A = X.transposed() * X + eps;

        auto start = high_resolution_clock::now();
        auto L = math::cholesky(A);
        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;
        std::cout << "Cholesky elapsed time: " << elapsed.count() << " seconds\n";
        auto C = L * L.transposed();
        ASSERT_TRUE(math::loosely_equal(C, A));
    }

public:
    int run_all_tests() override {
        should_construct_empty_matrix_with_zero_rows_and_columns();
        should_construct_empty_matrix_of_given_size();
        should_throw_if_constructed_with_zero_dimensions();
        should_construct_from_raw_data();
        should_throw_if_raw_data_is_null();
        should_construct_from_std_vector();
        should_throw_if_vector_size_mismatch();
        should_construct_from_nested_vector();
        should_throw_if_nested_vector_dimensions_mismatch();
        should_construct_from_std_array();
        should_construct_from_initializer_list();
        should_throw_if_initializer_list_size_mismatch();
        should_return_true_for_square_matrix();
        should_return_true_for_symmetric_matrix();
        should_return_true_for_triangular_matrix();
        should_return_true_for_diagonal_matrix();
        should_return_true_for_positive_definite_matrix();
        should_return_true_for_non_square_matrix();
        should_return_false_for_non_singular_matrix();
        should_return_true_for_singular_matrix();
        should_return_false_for_non_singular_matrix();
        should_cast_int_matrix_to_float();
        should_cast_float_matrix_to_int();
        should_cast_int_matrix_to_double();
        should_preserve_matrix_properties_after_cast();
        should_cast_negative_values_correctly();
        should_cast_large_matrix_efficiently();
        should_allow_chaining_cast_with_operations();
        should_cast_after_matrix_operations();
        should_fill_matrix_with_value();
        should_make_identity_matrix();
        should_transpose_square_matrix_in_place();
        should_return_transposed_copy_for_non_square_matrix();
        should_check_equality_between_identical_matrices();
        should_not_be_equal_if_any_element_differs();
        should_correctly_perform_unary_minus();
        should_add_two_matrices_of_same_size();
        should_add_scalar_and_matrix();
        should_add_assign_matrix();
        should_add_assign_scalar();
        should_subtract_two_matrices_of_same_size();
        should_subtract_scalar_and_matrix();
        should_subtract_assign_matrix();
        should_subtract_assign_scalar();
        should_multiply_matrix_and_scalar();
        should_multiply_assign_scalar();
        should_divide_matrix_and_scalar();
        should_divide_assign_scalar();

        should_multiply_matrices();
        should_multiply_matrix_and_vector();
        matmul_time_test();
        should_throw_if_plu_called_on_non_square_matrix();
        should_throw_for_singular_matrix();
        should_correctly_perform_plu_decomposition_on_small_matrix();
        should_correctly_handle_identity_matrix_in_plu();
        should_correctly_decompose_upper_triangular_matrix();
        should_correctly_handle_negative_pivots_in_plu();
        plu_time_test();
        should_decompose_identity_matrix();
        should_decompose_known_small_matrix();
        should_correctly_decompose_for_known_example();
        should_decompose_diagonal_matrix();
        should_reconstruct_from_random_b_times_b_t();
        should_throw_if_non_symmetric();
        should_throw_if_not_positive_definite();
        should_auto_convert_int_matrix_to_double_in_cholesky();
        should_preserve_float_type_in_cholesky();
        should_preserve_double_type_in_cholesky();
        should_explicitly_convert_int_to_float_in_cholesky();
        should_explicitly_convert_float_to_double_in_cholesky();
        should_handle_int_identity_matrix_in_cholesky();
        should_handle_diagonal_int_matrix_in_cholesky();
        cholesky_time_test();
        return 0;
    }
};

}  // namespace maf::test
