#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/Matrix.hpp"
#include "MafLib/math/linalg/MatrixView.hpp"
#include "MafLib/math/linalg/Vector.hpp"
#include "MafLib/math/linalg/VectorView.hpp"

namespace maf::test {

using namespace maf;
using namespace maf::math;
using namespace maf::util;

class ViewTests : public ITest {
 private:
  //=============================================================================
  // VECTOR VIEW CONSTRUCTION & ACCESS
  //=============================================================================
  void should_construct_contiguous_vector_view() {
    Vector<int> v(5);
    for (int i = 0; i < 5; ++i) v.at(i) = i + 1;
    VectorView<int> vv(v.data(), 5);

    ASSERT_TRUE(vv.size() == 5);
    ASSERT_TRUE(vv.get_increment() == 1);
    ASSERT_TRUE(vv.at(0) == 1);
    ASSERT_TRUE(vv.at(4) == 5);
  }

  void should_construct_strided_vector_view() {
    Vector<int> v(6);
    for (int i = 0; i < 6; ++i) v.at(i) = i + 1;  // [1 2 3 4 5 6]

    VectorView<int> vv(v.data(), 3, 2);  // [1 3 5]

    ASSERT_TRUE(vv.size() == 3);
    ASSERT_TRUE(vv.get_increment() == 2);
    ASSERT_TRUE(vv.at(0) == 1);
    ASSERT_TRUE(vv.at(1) == 3);
    ASSERT_TRUE(vv.at(2) == 5);
  }

  void should_modify_original_data_through_vector_view() {
    Vector<int> v(4);
    v.fill(0);

    VectorView<int> vv(v.data() + 1, 2);
    vv.at(0) = 10;
    vv.at(1) = 20;

    ASSERT_TRUE(v.at(0) == 0);
    ASSERT_TRUE(v.at(1) == 10);
    ASSERT_TRUE(v.at(2) == 20);
    ASSERT_TRUE(v.at(3) == 0);
  }

  void should_throw_on_vector_view_out_of_bounds() {
    Vector<int> v(3);
    VectorView<int> vv(v.data(), 3);

    bool thrown = false;
    try {
      vv.at(3) = 1;
    } catch (const std::out_of_range &) {
      thrown = true;
    }
    ASSERT_TRUE(thrown);
  }

  //=============================================================================
  // MATRIX VIEW CONSTRUCTION & ACCESS
  //=============================================================================
  void should_construct_matrix_view_and_access_elements() {
    Matrix<int> m(3, 4);
    int val = 1;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        m.at(i, j) = val++;
      }
    }

    // View middle 2x2 block starting at (1,1)
    MatrixView<int> mv(m.data() + 1 * m.column_count() + 1, 2, 2, m.column_count());

    ASSERT_TRUE(mv.row_count() == 2);
    ASSERT_TRUE(mv.column_count() == 2);

    ASSERT_TRUE(mv.at(0, 0) == m.at(1, 1));
    ASSERT_TRUE(mv.at(0, 1) == m.at(1, 2));
    ASSERT_TRUE(mv.at(1, 0) == m.at(2, 1));
    ASSERT_TRUE(mv.at(1, 1) == m.at(2, 2));
  }

  // CHECKED TO HERE
  void should_modify_original_matrix_through_view() {
    Matrix<int> m(3, 3);
    m.fill(0);

    MatrixView<int> mv(m.data(), 2, 2, 3);
    mv.at(0, 0) = 5;
    mv.at(1, 1) = 9;

    ASSERT_TRUE(m.at(0, 0) == 5);
    ASSERT_TRUE(m.at(1, 1) == 9);
    ASSERT_TRUE(m.at(2, 2) == 0);
  }

  void should_throw_on_matrix_view_out_of_bounds() {
    Matrix<int> m(2, 2);
    MatrixView<int> mv(m.data(), 2, 2, 2);

    bool thrown = false;
    try {
      mv.at(2, 0);
    } catch (const std::out_of_range &) {
      thrown = true;
    }
    ASSERT_TRUE(thrown);
  }

  //=============================================================================
  // GEMV TESTS (MOST IMPORTANT)
  //=============================================================================

  //  void should_compute_matrix_view_times_vector_view() {
  //    // 3x3 matrix
  //    Matrix<int> m(3, 3);
  //    int val = 1;
  //    for (size_t i = 0; i < 3; ++i)
  //      for (size_t j = 0; j < 3; ++j) m.at(i, j) = val++;
  //
  //    // Use full matrix as view
  //    MatrixView<int> mv(m.data(), 3, 3, 3);
  //
  //    Vector<int> v(3);
  //    v.at(0) = 1;
  //    v.at(1) = 2;
  //    v.at(2) = 3;
  //
  //    VectorView<int> vv(v.data(), 3);
  //
  //    auto res = mv * vv;
  //
  //    ASSERT_SAME_TYPE(res, Vector<int>);
  //    ASSERT_TRUE(res.size() == 3);
  //
  //    // manual:
  //    // [1 2 3]   [1]   [14]
  //    // [4 5 6] * [2] = [32]
  //    // [7 8 9]   [3]   [50]
  //    ASSERT_TRUE(res.at(0) == 14);
  //    ASSERT_TRUE(res.at(1) == 32);
  //    ASSERT_TRUE(res.at(2) == 50);
  //  }
  //
  //  void should_compute_matrix_view_times_vector_view_with_type_promotion() {
  //    Matrix<int> m(2, 2);
  //    m.at(0, 0) = 1;
  //    m.at(0, 1) = 2;
  //    m.at(1, 0) = 3;
  //    m.at(1, 1) = 4;
  //
  //    MatrixView<int> mv(m.data(), 2, 2, 2);
  //
  //    Vector<float> v(2);
  //    v.at(0) = 0.5f;
  //    v.at(1) = 1.5f;
  //
  //    VectorView<float> vv(v.data(), 2);
  //
  //    auto res = mv * vv;
  //
  //    ASSERT_SAME_TYPE(res, Vector<float>);
  //    ASSERT_TRUE(is_close(res.at(0), 1 * 0.5f + 2 * 1.5f));
  //    ASSERT_TRUE(is_close(res.at(1), 3 * 0.5f + 4 * 1.5f));
  //  }
  //
  //  void should_compute_vector_view_times_matrix_view() {
  //    Vector<int> v(2);
  //    v.at(0) = 2;
  //    v.at(1) = 3;
  //
  //    VectorView<int> vv(v.data(), 2);
  //
  //    Matrix<int> m(2, 3);
  //    m.at(0, 0) = 1;
  //    m.at(0, 1) = 2;
  //    m.at(0, 2) = 3;
  //    m.at(1, 0) = 4;
  //    m.at(1, 1) = 5;
  //    m.at(1, 2) = 6;
  //
  //    MatrixView<int> mv(m.data(), 2, 3, 3);
  //
  //    auto res = vv * mv;
  //
  //    ASSERT_SAME_TYPE(res, Vector<int>);
  //    ASSERT_TRUE(res.size() == 3);
  //
  //    // [2 3] * M
  //    ASSERT_TRUE(res.at(0) == 2 * 1 + 3 * 4);
  //    ASSERT_TRUE(res.at(1) == 2 * 2 + 3 * 5);
  //    ASSERT_TRUE(res.at(2) == 2 * 3 + 3 * 6);
  //  }
  //
  //  void should_compute_submatrix_view_gemv_correctly() {
  //    Matrix<int> m(4, 4);
  //    int val = 1;
  //    for (size_t i = 0; i < 4; ++i)
  //      for (size_t j = 0; j < 4; ++j) m.at(i, j) = val++;
  //
  //    // 2x3 submatrix starting at (1,1)
  //    MatrixView<int> mv(m.data() + 1 * 4 + 1, 2, 3, 4);
  //
  //    Vector<int> v(3);
  //    v.at(0) = 1;
  //    v.at(1) = 1;
  //    v.at(2) = 1;
  //
  //    auto res = mv * v;
  //
  //    ASSERT_TRUE(res.size() == 2);
  //    ASSERT_TRUE(res.at(0) == m.at(1, 1) + m.at(1, 2) + m.at(1, 3));
  //    ASSERT_TRUE(res.at(1) == m.at(2, 1) + m.at(2, 2) + m.at(2, 3));
  //  }
  //
  // public:
  //  void run_tests() override {
  //    should_construct_contiguous_vector_view();
  //    should_construct_strided_vector_view();
  //    should_modify_original_data_through_vector_view();
  //    should_throw_on_vector_view_out_of_bounds();
  //
  //    should_construct_matrix_view_and_access_elements();
  //    should_modify_original_matrix_through_view();
  //    should_throw_on_matrix_view_out_of_bounds();
  //
  //    should_compute_matrix_view_times_vector_view();
  //    should_compute_matrix_view_times_vector_view_with_type_promotion();
  //    should_compute_vector_view_times_matrix_view();
  //    should_compute_submatrix_view_gemv_correctly();
  //  }
};

}  // namespace maf::test
