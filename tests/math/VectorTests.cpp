#include <numeric>

#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/LinAlg.hpp"
#include "MafLib/math/linalg/Matrix.hpp"
#include "MafLib/math/linalg/Vector.hpp"

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
        ASSERT_TRUE(v.size() == 0);
        ASSERT_TRUE(v.orientation() == math::COLUMN);
    }

    void should_construct_vector_of_given_size() {
        math::Vector<int> v_col(5);
        ASSERT_SAME_TYPE(v_col, math::Vector<int>);
        ASSERT_TRUE(v_col.size() == 5);
        ASSERT_TRUE(v_col.orientation() == math::COLUMN);

        math::Vector<double> v_row(3, math::ROW);
        ASSERT_SAME_TYPE(v_row, math::Vector<double>);
        ASSERT_TRUE(v_row.size() == 3);
        ASSERT_TRUE(v_row.orientation() == math::ROW);
    }

    void should_throw_if_constructed_with_zero_size() {
        bool thrown = false;
        try {
            math::Vector<double> v(0);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_raw_data() {
        int data[3] = {10, 20, 30};
        math::Vector<int> v(3, data, math::ROW);
        ASSERT_SAME_TYPE(v, math::Vector<int>);
        ASSERT_TRUE(v.size() == 3);
        ASSERT_TRUE(v.orientation() == math::ROW);
        ASSERT_TRUE(v[0] == 10);
        ASSERT_TRUE(v[1] == 20);
        ASSERT_TRUE(v[2] == 30);
    }

    void should_throw_if_raw_data_constructor_has_zero_size() {
        bool thrown = false;
        int data[3] = {1, 2, 3};
        try {
            math::Vector<int> v(0, data);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_std_vector_copy() {
        std::vector<int> data = {5, 10, 15};
        math::Vector<int> v(3, data);
        ASSERT_SAME_TYPE(v, math::Vector<int>);
        ASSERT_TRUE(v.size() == 3);
        ASSERT_TRUE(v[1] == 10);
        data[1] = 99;
        ASSERT_TRUE(v[1] == 10);
    }

    void should_throw_if_std_vector_copy_size_mismatch() {
        std::vector<int> data = {1, 2};
        bool thrown = false;
        try {
            math::Vector<int> v(3, data);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_std_vector_move() {
        std::vector<int> data = {5, 10, 15};
        math::Vector<int> v(3, std::move(data));
        ASSERT_SAME_TYPE(v, math::Vector<int>);
        ASSERT_TRUE(v.size() == 3);
        ASSERT_TRUE(v[1] == 10);
    }

    void should_throw_if_std_vector_move_size_mismatch() {
        std::vector<int> data = {1, 2};
        bool thrown = false;
        try {
            math::Vector<int> v(3, std::move(data));
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_construct_from_std_array() {
        std::array<float, 3> data = {1.1f, 2.2f, 3.3f};
        math::Vector<float> v(3, data, math::ROW);
        ASSERT_SAME_TYPE(v, math::Vector<float>);
        ASSERT_TRUE(v.size() == 3);
        ASSERT_TRUE(v.orientation() == math::ROW);
        ASSERT_TRUE(is_close(v[1], 2.2f));
    }

    void should_throw_if_std_array_size_mismatch() {
        std::array<int, 2> data = {1, 2};
        bool thrown = false;
        try {
            math::Vector<int> v(3, data);
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    //=============================================================================
    // VECTOR ITERATORS TESTS
    //=============================================================================
    void should_access_elements_with_at_and_operator() {
        math::Vector<int> v(3);
        v[0] = 1;
        v[1] = 2;
        v.at(2) = 3;

        ASSERT_TRUE(v[0] == 1);
        ASSERT_TRUE(v.at(1) == 2);
        ASSERT_TRUE(v[2] == 3);

        const auto& cv = v;
        ASSERT_TRUE(cv[0] == 1);
        ASSERT_TRUE(cv.at(1) == 2);
    }

    void should_throw_on_out_of_bounds_access() {
        math::Vector<int> v(3);
        bool thrown_at = false;

        try {
            v.at(3) = 10;
        } catch (const std::out_of_range& e) {
            thrown_at = true;
        }

        ASSERT_TRUE(thrown_at);
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
        ASSERT_TRUE(sum == 60);

        const auto& cv = v;
        int const_sum = std::accumulate(cv.cbegin(), cv.cend(), 0);
        ASSERT_TRUE(const_sum == 60);
    }

    //=============================================================================
    // VECTOR CHECKERS TESTS
    //=============================================================================
    void should_check_if_vector_is_null() {
        math::Vector<int> v(3);
        v.fill(0);
        ASSERT_TRUE(v.is_null());

        v[1] = 1;
        ASSERT_TRUE(!v.is_null());

        math::Vector<double> vf(3);
        vf.fill(0.0);
        ASSERT_TRUE(vf.is_null());

        vf[2] = 1.0;
        ASSERT_TRUE(!vf.is_null());
    }

    //=============================================================================
    // VECTOR METHODS TESTS
    //=============================================================================
    void should_fill_vector_with_value() {
        math::Vector<int> v(10);
        v.fill(77);
        ASSERT_TRUE(v[0] == 77);
        ASSERT_TRUE(v[5] == 77);
        ASSERT_TRUE(v[9] == 77);
    }

    void should_calculate_l2_norm() {
        math::Vector<double> v(2);
        v[0] = 3.0;
        v[1] = 4.0;
        ASSERT_TRUE(is_close(v.norm(), 5.0));

        math::Vector<double> v2(3);
        v2[0] = 1.0;
        v2[1] = 2.0;
        v2[2] = 2.0;
        ASSERT_TRUE(is_close(v2.norm(), 3.0));
    }

    void should_normalize_vector_in_place() {
        math::Vector<double> v(2);
        v[0] = 3.0;
        v[1] = 4.0;
        v.normalize();

        ASSERT_TRUE(is_close(v[0], 0.6));
        ASSERT_TRUE(is_close(v[1], 0.8));
        ASSERT_TRUE(is_close(v.norm(), 1.0));
    }

    void should_transpose_vector_in_place() {
        math::Vector<int> v(3, math::COLUMN);
        ASSERT_TRUE(v.orientation() == math::COLUMN);
        v.transpose();
        ASSERT_TRUE(v.orientation() == math::ROW);
        v.transpose();
        ASSERT_TRUE(v.orientation() == math::COLUMN);
    }

    void should_return_transposed_copy() {
        math::Vector<int> v_col(3, math::COLUMN);
        auto v_row = v_col.transposed();
        ASSERT_SAME_TYPE(v_row, math::Vector<int>);

        ASSERT_TRUE(v_col.orientation() == math::COLUMN);
        ASSERT_TRUE(v_row.orientation() == math::ROW);
        ASSERT_TRUE(v_row.size() == 3);
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
        math::Vector<int> v5(2, math::ROW);
        v5[0] = 1;
        v5[1] = 2;
        math::Vector<float> v6(2);
        v6[0] = 1.F;
        v6[1] = 2.F;

        ASSERT_TRUE(v1 == v2);
        ASSERT_TRUE(!(v1 == v3));
        ASSERT_TRUE(!(v1 == v4));
        ASSERT_TRUE(!(v1 == v5));
    }

    void should_perform_unary_minus() {
        math::Vector<int> v(2);
        v[0] = 5;
        v[1] = -10;
        auto v_neg = -v;
        ASSERT_SAME_TYPE(v_neg, math::Vector<int>);
        ASSERT_TRUE(v[0] == 5);
        ASSERT_TRUE(v_neg[0] == -5);
        ASSERT_TRUE(v_neg[1] == 10);
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
        ASSERT_TRUE(v_sum.size() == 2);
        ASSERT_TRUE(is_close(v_sum[0], 11.0F));
        ASSERT_TRUE(is_close(v_sum[1], 22.0F));
    }

    void should_add_scalar_and_vector() {
        math::Vector<int> v(2);
        v[0] = 1;
        v[1] = 2;
        auto v_sum = v + 10;
        ASSERT_SAME_TYPE(v_sum, math::Vector<int>);
        ASSERT_TRUE(v_sum[0] == 11);
        ASSERT_TRUE(v_sum[1] == 12);

        auto v_sum2 = 10 + v;
        ASSERT_SAME_TYPE(v_sum2, math::Vector<int>);
        ASSERT_TRUE(v_sum2[0] == 11);
        ASSERT_TRUE(v_sum2[1] == 12);

        auto v_sum3 = 10.F + v;
        ASSERT_SAME_TYPE(v_sum3, math::Vector<float>);
        ASSERT_TRUE(is_close(v_sum3[0], 11.0F));
        ASSERT_TRUE(is_close(v_sum3[1], 12.0F));

        auto v_sum4 = v + 10.F;
        ASSERT_SAME_TYPE(v_sum4, math::Vector<float>);
        ASSERT_TRUE(is_close(v_sum4[0], 11.0F));
        ASSERT_TRUE(is_close(v_sum4[1], 12.0F));
    }

    void should_subtract_two_vectors() {
        math::Vector<int> v1(2);
        v1[0] = 10;
        v1[1] = 20;
        math::Vector<double> v2(2);
        v2[0] = 1;
        v2[1] = 2;
        auto v_diff = v1 - v2;

        ASSERT_TRUE(v_diff.size() == 2);
        ASSERT_TRUE(v_diff[0] == 9);
        ASSERT_TRUE(v_diff[1] == 18);
        ASSERT_SAME_TYPE(v_diff, math::Vector<double>);
    }

    void should_subtract_scalar_and_vector() {
        math::Vector<int> v(2);
        v[0] = 11;
        v[1] = 12;
        auto v_diff = v - 1;
        ASSERT_TRUE(v_diff[0] == 10);
        ASSERT_TRUE(v_diff[1] == 11);

        auto v_diff2 = 100.F - v;
        ASSERT_TRUE(is_close(v_diff2[0], 89.0F));
        ASSERT_TRUE(is_close(v_diff2[1], 88.0F));
        ASSERT_SAME_TYPE(v_diff, math::Vector<int>);
        ASSERT_SAME_TYPE(v_diff2, math::Vector<float>);
    }

    void should_multiply_vector_and_scalar() {
        math::Vector<int> v(2);
        v[0] = 2;
        v[1] = 3;
        auto v_prod = v * 5;
        ASSERT_TRUE(v_prod[0] == 10);
        ASSERT_TRUE(v_prod[1] == 15);

        auto v_prod2 = 5.F * v;
        ASSERT_TRUE(is_close(v_prod2[0], 10.0F));
        ASSERT_TRUE(is_close(v_prod2[1], 15.0F));
        ASSERT_SAME_TYPE(v_prod, math::Vector<int>);
        ASSERT_SAME_TYPE(v_prod2, math::Vector<float>);
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
        ASSERT_TRUE(v1.dot_product(v2) == 32);
        ASSERT_TRUE(v1.transposed() * v2 == 32);

        bool thrown = false;
        try {
            auto v3 = v1 * v2;
        } catch (const std::invalid_argument& e) {
            thrown = true;
        }
        ASSERT_TRUE(thrown);
    }

    void should_calculate_outer_product() {
        math::Vector<int> v_col(2, math::COLUMN);
        v_col[0] = 1;
        v_col[1] = 2;
        math::Vector<double> v_row(3, math::ROW);
        v_row[0] = 3;
        v_row[1] = 4;
        v_row[2] = 5;

        // (2x1) * (1x3) -> (2x3) Matrix
        // [1*3, 1*4, 1*5] = [3, 4, 5]
        // [2*3, 2*4, 2*5] = [6, 8, 10]
        auto m = v_col.outer_product(v_row);
        ASSERT_TRUE(m.row_count() == 2);
        ASSERT_TRUE(m.column_count() == 3);
        ASSERT_TRUE(m.at(0, 0) == 3);
        ASSERT_TRUE(m.at(0, 1) == 4);
        ASSERT_TRUE(m.at(0, 2) == 5);
        ASSERT_TRUE(m.at(1, 0) == 6);
        ASSERT_TRUE(m.at(1, 1) == 8);
        ASSERT_TRUE(m.at(1, 2) == 10);
        ASSERT_SAME_TYPE(m, math::Matrix<double>);
    }
    //
    //    void should_multiply_row_vector_and_matrix() {
    //        // (1x2) * (2x2) -> (1x2)
    //        math::Vector<int> v_row(2, math::Vector<int>::ROW);
    //        v_row[0] = 1;
    //        v_row[1] = 2;
    //
    //        math::Matrix<int> m(2, 2);
    //        m.at(0, 0) = 10;
    //        m.at(0, 1) = 20;
    //        m.at(1, 0) = 30;
    //        m.at(1, 1) = 40;
    //
    //        auto res = v_row * m;
    //        // res[0] = (1*10) + (2*30) = 70
    //        // res[1] = (1*20) + (2*40) = 100
    //        ASSERT_TRUE(res.size() == 2);
    //        ASSERT_TRUE(res.orientation() == math::Vector<int>::ROW);
    //        ASSERT_TRUE(res[0] == 70);
    //        ASSERT_TRUE(res[1] == 100);
    //    }
    //
    //    void should_multiply_matrix_and_column_vector() {
    //        // (2x2) * (2x1) -> (2x1)
    //        math::Matrix<int> m(2, 2);
    //        m.at(0, 0) = 10;
    //        m.at(0, 1) = 20;
    //        m.at(1, 0) = 30;
    //        m.at(1, 1) = 40;
    //
    //        math::Vector<int> v_col(2, math::Vector<int>::COLUMN);
    //        v_col[0] = 1;
    //        v_col[1] = 2;
    //
    //        auto res = m * v_col;
    //        // res[0] = (10*1) + (20*2) = 50
    //        // res[1] = (30*1) + (40*2) = 110
    //        ASSERT_TRUE(res.size() == 2);
    //        ASSERT_TRUE(res.orientation() == math::Vector<int>::COLUMN);
    //        ASSERT_TRUE(res[0] == 50);
    //        ASSERT_TRUE(res[1] == 110);
    //    }
    //
    //    void should_multiply_row_vector_and_matrix_mixed_types() {
    //        // (1x2) * (2x2) -> (1x2)
    //        math::Vector<int> v_row(2, math::Vector<int>::ROW);
    //        v_row[0] = 1;
    //        v_row[1] = 2;
    //
    //        math::Matrix<double> m(2, 2);
    //        m.at(0, 0) = 10.5;
    //        m.at(0, 1) = 20.5;
    //        m.at(1, 0) = 30.0;
    //        m.at(1, 1) = 40.0;
    //
    //        auto res = v_row * m;
    //        // res[0] = (1*10.5) + (2*30.0) = 70.5
    //        // res[1] = (1*20.5) + (2*40.0) = 100.5
    //        ASSERT_TRUE((std::is_same_v<decltype(res)::value_type, double>));
    //        ASSERT_TRUE(res.size() == 2);
    //        ASSERT_TRUE(is_close(res[0], 70.5));
    //        ASSERT_TRUE(is_close(res[1], 100.5));
    //    }
    //
    //    void should_multiply_matrix_and_column_vector_mixed_types() {
    //        // (2x2) * (2x1) -> (2x1)
    //        math::Matrix<double> m(2, 2);
    //        m.at(0, 0) = 10.5;
    //        m.at(0, 1) = 20.5;
    //        m.at(1, 0) = 30.0;
    //        m.at(1, 1) = 40.0;
    //
    //        math::Vector<int> v_col(2, math::Vector<int>::COLUMN);
    //        v_col[0] = 1;
    //        v_col[1] = 2;
    //
    //        auto res = m * v_col;
    //        // res[0] = (10.5*1) + (20.5*2) = 51.5
    //        // res[1] = (30.0*1) + (40.0*2) = 110.0
    //        ASSERT_TRUE((std::is_same_v<decltype(res)::value_type, double>));
    //        ASSERT_TRUE(res.size() == 2);
    //        ASSERT_TRUE(res.orientation() == math::Vector<double>::COLUMN);
    //        ASSERT_TRUE(is_close(res[0], 51.5));
    //        ASSERT_TRUE(is_close(res[1], 110.0));
    //    }

public:
    int run_all_tests() override {
        should_construct_empty_vector_with_zero_size();
        should_construct_vector_of_given_size();
        should_throw_if_constructed_with_zero_size();
        should_construct_from_raw_data();
        should_throw_if_raw_data_constructor_has_zero_size();
        should_construct_from_std_vector_copy();
        should_throw_if_std_vector_copy_size_mismatch();
        should_construct_from_std_vector_move();
        should_throw_if_std_vector_move_size_mismatch();
        should_construct_from_std_array();
        should_throw_if_std_array_size_mismatch();
        should_access_elements_with_at_and_operator();
        should_throw_on_out_of_bounds_access();
        should_iterate_over_elements();
        should_check_if_vector_is_null();

        should_fill_vector_with_value();
        should_calculate_l2_norm();
        should_normalize_vector_in_place();
        should_transpose_vector_in_place();
        should_return_transposed_copy();
        //    // Operators
        should_check_equality();
        should_perform_unary_minus();
        should_add_two_vectors();
        should_add_scalar_and_vector();
        should_subtract_two_vectors();
        should_subtract_scalar_and_vector();
        should_multiply_vector_and_scalar();
        should_calculate_dot_product();
        should_calculate_outer_product();
        //    should_multiply_row_vector_and_matrix();
        //    should_multiply_matrix_and_column_vector();
        //    should_multiply_row_vector_and_matrix_mixed_types();
        //    should_multiply_matrix_and_column_vector_mixed_types();
        //    std::cout << "--- Operator tests passed ---" << std::endl;

        std::cout << "=== All Vector tests passed ===" << std::endl;
        return 0;
    }
};

}  // namespace maf::test
