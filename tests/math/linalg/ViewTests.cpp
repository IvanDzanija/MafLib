#include "ITest.hpp"
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/math/linalg/LinAlg.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::test {

using namespace maf;
using namespace maf::math;
using namespace maf::util;

class ViewTests : public ITest {
 private:
  //=============================================================================
  // VECTOR VIEW CONSTRUCTORS & ACCESS TESTS
  //=============================================================================
  void should_construct_contiguous_vector_view() {
    Vector<int> v(5);
    for (int i = 0; i < 5; ++i) v.at(i) = i + 1;

    VectorView<int> vv(v.data(), 5, v.orientation());

    assert_true(vv.size() == 5);
    assert_true(vv.get_increment() == 1);
    assert_true(vv.at(0) == 1);
    assert_true(vv.at(4) == 5);
  }

  void should_construct_strided_vector_view() {
    Vector<int> v(6);
    for (int i = 0; i < 6; ++i) v.at(i) = i + 1;  // [1 2 3 4 5 6]

    VectorView<int> vv(v.data(), 3, v.orientation(), 2);  // [1 3 5]

    assert_true(vv.size() == 3);
    assert_true(vv.get_increment() == 2);
    assert_true(vv.at(0) == 1);
    assert_true(vv.at(1) == 3);
    assert_true(vv.at(2) == 5);
  }

  void should_modify_original_data_through_vector_view() {
    Vector<int> v(4);
    v.fill(0);

    VectorView<int> vv(v.data() + 1, 2, v.orientation());
    vv.at(0) = 10;
    vv.at(1) = 20;

    assert_true(v.at(0) == 0);
    assert_true(v.at(1) == 10);
    assert_true(v.at(2) == 20);
    assert_true(v.at(3) == 0);
  }

  void should_throw_on_vector_view_out_of_bounds() {
    Vector<int> v(3);
    VectorView<int> vv(v.data(), 3, v.orientation());

    bool thrown = false;
    try {
      vv.at(3) = 1;
    } catch (const std::out_of_range &) {
      thrown = true;
    }
    assert_true(thrown);
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
    MatrixView<int> mv(m.data() + (1 * m.column_count()) + 1, 2, 2, m.column_count());

    assert_true(mv.row_count() == 2);
    assert_true(mv.column_count() == 2);

    assert_true(mv.at(0, 0) == m.at(1, 1));
    assert_true(mv.at(0, 1) == m.at(1, 2));
    assert_true(mv.at(1, 0) == m.at(2, 1));
    assert_true(mv.at(1, 1) == m.at(2, 2));
  }

  void should_modify_original_matrix_through_view() {
    Matrix<int> m(3, 3);
    m.fill(0);

    MatrixView<int> mv(m.data(), 2, 2, m.column_count());
    mv.at(0, 0) = 5;
    mv.at(1, 1) = 9;

    assert_true(m.at(0, 0) == 5);
    assert_true(m.at(1, 1) == 9);
    assert_true(m.at(2, 2) == 0);
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
    assert_true(thrown);
  }

  //=============================================================================
  // MATRIX & VECTOR VIEW OPERATORS TESTS
  //=============================================================================

  void should_compute_matrix_view_times_vector_view() {
    // INT
    Matrix<int> m(3, 3);
    int val = 1;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        m.at(i, j) = val++;
      }
    }

    MatrixView<int> mv(m.data(), 3, 3, 3);

    Vector<int> v(3);
    v.at(0) = 1;
    v.at(1) = 2;
    v.at(2) = 3;

    VectorView<int> vv(v.data(), 3, v.orientation());

    auto res = mv * vv;

    ASSERT_SAME_TYPE(res, Vector<int>);
    assert_true(res.size() == 3);

    // manual:
    // [1 2 3]   [1]   [14]
    // [4 5 6] * [2] = [32]
    // [7 8 9]   [3]   [50]
    assert_true(res.at(0) == 14);
    assert_true(res.at(1) == 32);
    assert_true(res.at(2) == 50);

    // FLOAT/DOUBLE
    Matrix<float> mf(3, 3);
    val = 1;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        mf.at(i, j) = val++;
      }
    }

    MatrixView<float> mvf(mf.data(), 3, 3, 3);

    Vector<float> vf(3);
    vf.at(0) = 1.0f;
    vf.at(1) = 2.0f;
    vf.at(2) = 3.0f;

    VectorView<float> vvf(vf.data(), 3, vf.orientation());

    auto resf = mvf * vvf;
    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(resf.size() == 3);
    assert_true(is_close(resf.at(0), 14.0f));
    assert_true(is_close(resf.at(1), 32.0f));
    assert_true(is_close(resf.at(2), 50.0f));
  }

  void should_compute_matrix_view_times_vector_view_with_type_promotion() {
    // Promote int matrix and float vector to float result
    Matrix<int> m(2, 2);
    m.at(0, 0) = 1;
    m.at(0, 1) = 2;
    m.at(1, 0) = 3;
    m.at(1, 1) = 4;

    MatrixView<int> mv(m.data(), 2, 2, 2);

    Vector<float> v(2);
    v.at(0) = 0.5f;
    v.at(1) = 1.5f;

    VectorView<float> vv(v.data(), 2, v.orientation());

    auto resf = mv * vv;

    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(is_close(resf.at(0), 1 * 0.5f + 2 * 1.5f));
    assert_true(is_close(resf.at(1), 3 * 0.5f + 4 * 1.5f));

    // Promote int matrix and double vector to double result
    Vector<double> vd(2);
    vd.at(0) = 0.5f;
    vd.at(1) = 1.5f;

    VectorView<double> vvd(vd.data(), 2, v.orientation());

    auto resd = mv * vvd;

    ASSERT_SAME_TYPE(resd, Vector<double>);
    assert_true(is_close(resd.at(0), 1 * 0.5f + 2 * 1.5f));
    assert_true(is_close(resd.at(1), 3 * 0.5f + 4 * 1.5f));

    // Promote float matrix and int vector to float result
    Matrix<float> mf(2, 2);
    mf.at(0, 0) = 1.0f;
    mf.at(0, 1) = 2.0f;
    mf.at(1, 0) = 3.0f;
    mf.at(1, 1) = 4.0f;

    MatrixView<float> mvf(mf.data(), 2, 2, 2);

    Vector<int> vi(2);
    vi.at(0) = 1;
    vi.at(1) = 2;

    VectorView<int> vvi(vi.data(), 2, vi.orientation());

    auto resf2 = mvf * vvi;

    ASSERT_SAME_TYPE(resf2, Vector<float>);
    assert_true(is_close(resf2.at(0), 1.0f * 1 + 2.0f * 2));
    assert_true(is_close(resf2.at(1), 3.0f * 1 + 4.0f * 2));

    // Promote double matrix and float vector to double result
    Matrix<double> md(2, 2);
    md.at(0, 0) = 1.0;
    md.at(0, 1) = 2.0;
    md.at(1, 0) = 3.0;
    md.at(1, 1) = 4.0;

    MatrixView<double> mvd(md.data(), 2, 2, 2);

    auto resd2 = mvd * vv;

    ASSERT_SAME_TYPE(resd2, Vector<double>);
    assert_true(is_close(resd2.at(0), 1.0 * 0.5f + 2.0 * 1.5f));
    assert_true(is_close(resd2.at(1), 3.0 * 0.5f + 4.0 * 1.5f));
  }

  void should_compute_vector_view_times_matrix_view() {
    Vector<int> v(2, Orientation::ROW);
    v.at(0) = 2;
    v.at(1) = 3;

    VectorView<int> vv(v.data(), 2, v.orientation());

    Matrix<int> m(2, 3);
    m.at(0, 0) = 1;
    m.at(0, 1) = 2;
    m.at(0, 2) = 3;
    m.at(1, 0) = 4;
    m.at(1, 1) = 5;
    m.at(1, 2) = 6;

    MatrixView<int> mv(m.data(), 2, 3, 3);

    auto res = vv * mv;

    ASSERT_SAME_TYPE(res, Vector<int>);
    assert_true(res.size() == 3);

    // [2 3] * M
    assert_true(res.at(0) == 2 * 1 + 3 * 4);
    assert_true(res.at(1) == 2 * 2 + 3 * 5);
    assert_true(res.at(2) == 2 * 3 + 3 * 6);

    // FLOAT/DOUBLE
    Vector<float> vf(2, Orientation::ROW);
    vf.at(0) = 2.0f;
    vf.at(1) = 3.0f;

    VectorView<float> vvf(vf.data(), 2, vf.orientation());

    Matrix<float> mf(2, 3);
    mf.at(0, 0) = 1.0f;
    mf.at(0, 1) = 2.0f;
    mf.at(0, 2) = 3.0f;
    mf.at(1, 0) = 4.0f;
    mf.at(1, 1) = 5.0f;
    mf.at(1, 2) = 6.0f;

    MatrixView<float> mvf(mf.data(), 2, 3, 3);

    auto resf = vvf * mvf;
    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(resf.size() == 3);
    assert_true(is_close(resf.at(0), 2.0f * 1.0f + 3.0f * 4.0f));
    assert_true(is_close(resf.at(1), 2.0f * 2.0f + 3.0f * 5.0f));
    assert_true(is_close(resf.at(2), 2.0f * 3.0f + 3.0f * 6.0f));
  }

  void should_compute_matrix_vector_subviews_correctly() {
    // MatrixView * Vector with type promotion
    Matrix<int> m(4, 4);
    int val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        m.at(i, j) = val++;
      }
    }

    // 2x3 submatrix starting at (1,1)
    // [ 6  7  8 ]
    // [10 11 12]
    MatrixView<int> mv(m.data() + 1 * 4 + 1, 2, 3, 4);

    Vector<double> v(3, Orientation::COLUMN);
    v.at(0) = 1.0;
    v.at(1) = 2.0;
    v.at(2) = 3.0;

    auto res = mv * v;

    ASSERT_SAME_TYPE(res, Vector<double>);
    assert_true(res.size() == 2);
    // [6 7 8]   [1]   [6*1 + 7*2 + 8*2]
    // [10 11 12] [2] = [10*1 + 11*2 + 12*2]
    assert_true(res.at(0) == 6 * 1 + 7 * 2 + 8 * 3);
    assert_true(res.at(1) == 10 * 1 + 11 * 2 + 12 * 3);

    // BLAS path with floats
    Matrix<float> mf(4, 4);
    val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        mf.at(i, j) = val++;
      }
    }

    // 2x3 submatrix starting at (1,1)
    // [ 6  7  8 ]
    // [10 11 12]
    MatrixView<float> mvf(mf.data() + 1 * 4 + 1, 2, 3, 4);

    Vector<float> vf2(3, Orientation::COLUMN);
    vf2.at(0) = 1.0f;
    vf2.at(1) = 2.0f;
    vf2.at(2) = 3.0f;

    auto resf = mvf * vf2;
    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(resf.size() == 2);
    assert_true(is_close(resf.at(0), 6 * 1 + 7 * 2 + 8 * 3));
    assert_true(is_close(resf.at(1), 10 * 1 + 11 * 2 + 12 * 3));

    // BLAS path with doubles
    Matrix<double> md(4, 4);
    val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        md.at(i, j) = val++;
      }
    }

    MatrixView<double> mvd(md.data() + 1 * 4 + 1, 2, 3, 4);

    Vector<double> vd2(3, Orientation::COLUMN);
    vd2.at(0) = 1.0;
    vd2.at(1) = 2.0;
    vd2.at(2) = 3.0;

    auto resd = mvd * vd2;
    ASSERT_SAME_TYPE(resd, Vector<double>);
    assert_true(resd.size() == 2);
    assert_true(is_close(resd.at(0), 6 * 1 + 7 * 2 + 8 * 3));
    assert_true(is_close(resd.at(1), 10 * 1 + 11 * 2 + 12 * 3));
  }

  void should_compute_vector_matrix_subviews_correctly() {
    // Vector * MatrixView with type promotion
    Vector<float> vf(2, Orientation::ROW);
    vf.at(0) = 1.0f;
    vf.at(1) = 2.0f;

    Matrix<int> m(4, 4);
    int val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        m.at(i, j) = val++;
      }
    }

    // 2x3 submatrix starting at (1,1)
    // [ 6  7  8 ]
    // [10 11 12]
    MatrixView<int> mv(m.data() + 1 * 4 + 1, 2, 3, 4);

    auto res2 = vf * mv;

    ASSERT_SAME_TYPE(res2, Vector<float>);
    assert_true(res2.size() == 3);
    // [1 2] * [ 6  7  8 ]
    //         [10 11 12]
    assert_true(res2.at(0) == 1 * 6 + 2 * 10);
    assert_true(res2.at(1) == 1 * 7 + 2 * 11);
    assert_true(res2.at(2) == 1 * 8 + 2 * 12);

    // BLAS path with doubles
    Vector<double> vd(2, Orientation::ROW);
    vd.at(0) = 1.0;
    vd.at(1) = 2.0;

    Matrix<double> md(4, 4);
    val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        md.at(i, j) = val++;
      }
    }

    MatrixView<double> mvd(md.data() + 1 * 4 + 1, 2, 3, 4);

    auto resd2 = vd * mvd;
    ASSERT_SAME_TYPE(resd2, Vector<double>);
    assert_true(resd2.size() == 3);
    assert_true(is_close(resd2.at(0), 1 * 6 + 2 * 10));
    assert_true(is_close(resd2.at(1), 1 * 7 + 2 * 11));
    assert_true(is_close(resd2.at(2), 1 * 8 + 2 * 12));

    // BLAS path with floats
    Matrix<float> mf(4, 4);
    val = 1;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        mf.at(i, j) = val++;
      }
    }

    MatrixView<float> mvf(mf.data() + 1 * 4 + 1, 2, 3, 4);

    auto resf2 = vf * mvf;

    ASSERT_SAME_TYPE(resf2, Vector<float>);
    assert_true(resf2.size() == 3);
    assert_true(is_close(resf2.at(0), 1 * 6 + 2 * 10));
    assert_true(is_close(resf2.at(1), 1 * 7 + 2 * 11));
    assert_true(is_close(resf2.at(2), 1 * 8 + 2 * 12));
  }

  void should_compute_strided_vector_times_weird_submatrix_with_type_promotion() {
    Matrix<int> m(5, 6);
    int val = 1;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // "Weird" 2x3 submatrix starting at (2, 1) with leading dimension 6
    // Rows:
    // r=2: [ m(2,1) m(2,2) m(2,3) ] = [14 15 16]
    // r=3: [ m(3,1) m(3,2) m(3,3) ] = [20 21 22]
    MatrixView<int> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // Build a strided ROW vector view of length 2 with increment 3:
    // backing vector: [1, 999, 999, 2, 999, 999]  -> view picks [1, 2]
    Vector<int> backing(6, Orientation::ROW);
    backing.fill(999);
    backing.at(0) = 1;
    backing.at(3) = 2;

    VectorView<int> vv(backing.data(), 2, backing.orientation(), 3);

    // Multiply: (1x2) * (2x3) => (1x3)
    // [1 2] * [14 15 16
    //          20 21 22] = [54 57 60]
    auto res = vv * mv;

    ASSERT_SAME_TYPE(res, Vector<int>);
    assert_true(res.size() == 3);
    assert_true(res.at(0) == 1 * 14 + 2 * 20);
    assert_true(res.at(1) == 1 * 15 + 2 * 21);
    assert_true(res.at(2) == 1 * 16 + 2 * 22);

    Vector<float> backing_f(6, Orientation::ROW);
    backing_f.fill(999.0f);
    backing_f.at(0) = 1.0f;
    backing_f.at(3) = 2.0f;

    VectorView<float> vvf(backing_f.data(), 2, backing_f.orientation(), 3);

    auto resf = vvf * mv;

    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(resf.size() == 3);
    assert_true(is_close(resf.at(0), 1.0f * 14 + 2.0f * 20));
    assert_true(is_close(resf.at(1), 1.0f * 15 + 2.0f * 21));
    assert_true(is_close(resf.at(2), 1.0f * 16 + 2.0f * 22));
  }

  void should_compute_strided_vector_times_weird_submatrix_blas_float_path() {
    Matrix<float> m(5, 6);
    float val = 1.0f;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // same weird 2x3 submatrix: start (2,1)
    // [14 15 16
    //  20 21 22]
    MatrixView<float> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // strided ROW vector view [1 2]
    Vector<float> backing(6, Orientation::ROW);
    backing.fill(999.0f);
    backing.at(0) = 1.0f;
    backing.at(3) = 2.0f;

    VectorView<float> vv(backing.data(), 2, backing.orientation(), 3);

    auto res = vv * mv;

    ASSERT_SAME_TYPE(res, Vector<float>);
    assert_true(res.size() == 3);
    assert_true(is_close(res.at(0), 1.0f * 14.0f + 2.0f * 20.0f));
    assert_true(is_close(res.at(1), 1.0f * 15.0f + 2.0f * 21.0f));
    assert_true(is_close(res.at(2), 1.0f * 16.0f + 2.0f * 22.0f));
  }

  void should_compute_strided_vector_times_weird_submatrix_blas_double_path() {
    Matrix<double> m(5, 6);
    double val = 1.0;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // same weird 2x3 submatrix: start (2,1)
    MatrixView<double> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // strided ROW vector view [1 2]
    Vector<double> backing(6, Orientation::ROW);
    backing.fill(999.0);
    backing.at(0) = 1.0;
    backing.at(3) = 2.0;

    VectorView<double> vv(backing.data(), 2, backing.orientation(), 3);

    auto res = vv * mv;

    ASSERT_SAME_TYPE(res, Vector<double>);
    assert_true(res.size() == 3);
    assert_true(is_close(res.at(0), 1.0 * 14.0 + 2.0 * 20.0));
    assert_true(is_close(res.at(1), 1.0 * 15.0 + 2.0 * 21.0));
    assert_true(is_close(res.at(2), 1.0 * 16.0 + 2.0 * 22.0));
  }

  void should_compute_weird_submatrix_times_strided_vector_with_type_promotion() {
    Matrix<int> m(5, 6);
    int val = 1;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // "Weird" 2x3 submatrix starting at (2,1) with leading dimension 6
    // [14 15 16
    //  20 21 22]
    MatrixView<int> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // Strided COLUMN vector view length 3 with increment 2:
    // backing: [1, 999, 2, 999, 3, 999] -> view picks [1,2,3]
    Vector<int> backing(6, Orientation::COLUMN);
    backing.fill(999);
    backing.at(0) = 1;
    backing.at(2) = 2;
    backing.at(4) = 3;

    VectorView<int> vv(backing.data(), 3, backing.orientation(), 2);

    // (2x3) * (3x1) => (2x1)
    // [14 15 16]   [1]   [14 + 30 + 48] = [92]
    // [20 21 22] * [2] = [20 + 42 + 66] = [128]
    //             [3]
    auto res = mv * vv;

    ASSERT_SAME_TYPE(res, Vector<int>);
    assert_true(res.size() == 2);
    assert_true(res.at(0) == 14 * 1 + 15 * 2 + 16 * 3);
    assert_true(res.at(1) == 20 * 1 + 21 * 2 + 22 * 3);

    Vector<float> backing_f(6, Orientation::COLUMN);
    backing_f.fill(999.0f);
    backing_f.at(0) = 1.0f;
    backing_f.at(2) = 2.0f;
    backing_f.at(4) = 3.0f;

    VectorView<float> vvf(backing_f.data(), 3, backing_f.orientation(), 2);

    auto resf = mv * vvf;

    ASSERT_SAME_TYPE(resf, Vector<float>);
    assert_true(resf.size() == 2);
    assert_true(is_close(resf.at(0), 14.0f * 1.0f + 15.0f * 2.0f + 16.0f * 3.0f));
    assert_true(is_close(resf.at(1), 20.0f * 1.0f + 21.0f * 2.0f + 22.0f * 3.0f));
  }

  void should_compute_weird_submatrix_times_strided_vector_blas_float_path() {
    Matrix<float> m(5, 6);
    float val = 1.0f;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // [14 15 16
    //  20 21 22]
    MatrixView<float> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // strided COLUMN vector view [1,2,3]
    Vector<float> backing(6, Orientation::COLUMN);
    backing.fill(999.0f);
    backing.at(0) = 1.0f;
    backing.at(2) = 2.0f;
    backing.at(4) = 3.0f;

    VectorView<float> vv(backing.data(), 3, backing.orientation(), 2);

    auto res = mv * vv;

    ASSERT_SAME_TYPE(res, Vector<float>);
    assert_true(res.size() == 2);
    assert_true(is_close(res.at(0), 14.0f * 1.0f + 15.0f * 2.0f + 16.0f * 3.0f));
    assert_true(is_close(res.at(1), 20.0f * 1.0f + 21.0f * 2.0f + 22.0f * 3.0f));
  }

  void should_compute_weird_submatrix_times_strided_vector_blas_double_path() {
    Matrix<double> m(5, 6);
    double val = 1.0;
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        m.at(i, j) = val++;
      }
    }

    // [14 15 16
    //  20 21 22]
    MatrixView<double> mv(m.data() + 2 * m.column_count() + 1, 2, 3, m.column_count());

    // strided COLUMN vector view [1,2,3]
    Vector<double> backing(6, Orientation::COLUMN);
    backing.fill(999.0);
    backing.at(0) = 1.0;
    backing.at(2) = 2.0;
    backing.at(4) = 3.0;

    VectorView<double> vv(backing.data(), 3, backing.orientation(), 2);

    auto res = mv * vv;

    ASSERT_SAME_TYPE(res, Vector<double>);
    assert_true(res.size() == 2);
    assert_true(is_close(res.at(0), 14.0 * 1.0 + 15.0 * 2.0 + 16.0 * 3.0));
    assert_true(is_close(res.at(1), 20.0 * 1.0 + 21.0 * 2.0 + 22.0 * 3.0));
  }

 public:
  std::vector<TestTask> get_tests() override {
    return {
        // Vector views
        [this] { should_construct_contiguous_vector_view(); },
        [this] { should_construct_strided_vector_view(); },
        [this] { should_modify_original_data_through_vector_view(); },
        [this] { should_throw_on_vector_view_out_of_bounds(); },
        // Matrix views
        [this] { should_construct_matrix_view_and_access_elements(); },
        [this] { should_modify_original_matrix_through_view(); },
        [this] { should_throw_on_matrix_view_out_of_bounds(); },
        // Operators
        [this] { should_compute_matrix_view_times_vector_view(); },
        [this] { should_compute_matrix_view_times_vector_view_with_type_promotion(); },
        [this] { should_compute_vector_view_times_matrix_view(); },
        [this] { should_compute_matrix_vector_subviews_correctly(); },
        [this] { should_compute_vector_matrix_subviews_correctly(); },
        [this] {
          should_compute_strided_vector_times_weird_submatrix_with_type_promotion();
        },
        [this] {
          should_compute_strided_vector_times_weird_submatrix_blas_float_path();
        },
        [this] {
          should_compute_strided_vector_times_weird_submatrix_blas_double_path();
        },
        [this] {
          should_compute_weird_submatrix_times_strided_vector_with_type_promotion();
        },
        [this] {
          should_compute_weird_submatrix_times_strided_vector_blas_float_path();
        },
        [this] {
          should_compute_weird_submatrix_times_strided_vector_blas_double_path();
        }};
  }
};

}  // namespace maf::test
