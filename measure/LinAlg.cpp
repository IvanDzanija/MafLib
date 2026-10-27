#include "Benchmark.hpp"

namespace maf::measure {
using namespace maf;
using namespace maf::math;

//=============================================================================
// MATMUL
//=============================================================================
template <typename T>
void matmul_bench(const std::vector<size_t> &sizes) {
  constexpr double FLOPS_PER_ELEMENT = 2.0;  // mul + add
  for (size_t n : sizes) {
    auto A = make_random_matrix<T>(n);
    auto B = make_random_matrix<T>(n, -10.0, 10.0, 43);

    double ms = bench_ms([&] {
      auto C = A * B;
      volatile double sink = C.at(0, 0);
      (void)sink;
    });

    double gflops = (FLOPS_PER_ELEMENT * n * n * n) / (ms * 1e6);
    print_result("matmul", "n=" + std::to_string(n), ms, gflops);
  }
}

//=============================================================================
// PLU
//=============================================================================
void plu_bench() {
  constexpr size_t n = 1000;
  auto A = make_random_matrix<double>(n);

  double ms = bench_ms([&] {
    auto decomposed = plu(A);
    volatile double sink = decomposed.L.at(0, 0);
    (void)sink;
  });

  double gflops = ((2.0 / 3.0) * n * n * n) / (ms * 1e6);
  print_result("plu", "n=" + std::to_string(n), ms, gflops);
}

//=============================================================================
// CHOLESKY
//=============================================================================
void cholesky_bench() {
  constexpr size_t n = 1000;
  constexpr double EPS = 1e-7;

  auto X = make_random_matrix<double>(n, 0.0, 1.0);
  auto A = X.transposed() * X + EPS;  // symmetric positive definite

  double ms = bench_ms([&] {
    auto L = cholesky(A);
    volatile double sink = L.at(0, 0);
    (void)sink;
  });

  double gflops = (n * n * n) / (ms * 1e6);
  print_result("cholesky", "n=" + std::to_string(n), ms, gflops);
}

//=============================================================================
// QR
//=============================================================================
void qr_bench(const std::vector<size_t> &sizes) {
  for (size_t n : sizes) {
    auto A = make_random_matrix<double>(n);

    double ms = bench_ms(
        [&] {
          auto qr = qr_decompostion(A);
          volatile double sink = qr.R.at(0, 0);
          (void)sink;
        },
        1, 2);

    double gflops = ((8.0 / 3.0) * n * n * n) / (ms * 1e6);  // Householder QR
    print_result("qr", "n=" + std::to_string(n), ms, gflops);
  }
}

//=============================================================================
// CAST
//=============================================================================
void cast_bench() {
  for (size_t n : {size_t(100), size_t(1000)}) {
    math::Matrix<int> m_int(n, n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        m_int.at(i, j) = static_cast<int>(i * n + j);
      }
    }

    double ms = bench_ms([&] {
      auto m_double = m_int.template cast<double>();
      volatile double sink = m_double.at(0, 0);
      (void)sink;
    });

    print_result("cast", "n=" + std::to_string(n) + " int->double", ms);
  }
}

//=============================================================================
// GEMV (MatrixView / VectorView)
//=============================================================================
void gemv_bench() {
#ifdef MAFLIB_OPENMP_AVAILABLE
  std::cout << "OMP threads: " << omp_get_max_threads() << "\n";
#endif
  constexpr size_t N = 4096;
  constexpr int WARMUP = 1;
  constexpr int ITERS = 3;

  // Deterministic values (same "numbers" across all type combos)
  auto mat_val = [](size_t i, size_t j) -> int {
    // small repeating pattern in [-3..3]
    return static_cast<int>((i * 1315423911u + j * 2654435761u) % 7) - 3;
  };
  auto vec_val = [](size_t i) -> int {
    return static_cast<int>((i * 2246822519u) % 7) - 3;
  };

  auto time_ms = [](auto &&fn, int iters) -> double {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (int k = 0; k < iters; ++k) fn();
    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           static_cast<double>(iters);
  };

  auto consume3 = [](const auto &v, volatile double &sink) {
    const size_t n = v.size();
    if (n == 0) {
      return;
    }
    sink += static_cast<double>(v.at(0));
    sink += static_cast<double>(v.at(n / 2));
    sink += static_cast<double>(v.at(n - 1));
  };

  std::cout << "\n=== PERF 4096: Matrix*Vector + Vector*Matrix (same values) ===\n";
  std::cout << "N=" << N << ", warmup=" << WARMUP << ", iters=" << ITERS << "\n";

  volatile double sink = 0.0;

  // ---------------------------------------------------------------------------
  // float-float
  // ---------------------------------------------------------------------------
  {
    Matrix<float> m(N, N);
    Vector<float> vc(N, Orientation::COLUMN);
    Vector<float> vr(N, Orientation::ROW);

    for (size_t i = 0; i < N; ++i) {
      const auto vv = static_cast<float>(vec_val(i));
      vc.at(i) = vv;
      vr.at(i) = vv;
      for (size_t j = 0; j < N; ++j) m.at(i, j) = static_cast<float>(mat_val(i, j));
    }

    MatrixView<float> mv(m.data(), N, N, m.column_count());
    VectorView<float> vvc(vc.data(), N, vc.orientation());
    VectorView<float> vvr(vr.data(), N, vr.orientation());

    for (int k = 0; k < WARMUP; ++k) {
      auto r1 = mv * vvc;
      auto r2 = vvr * mv;
      consume3(r1, sink);
      consume3(r2, sink);
    }

    const double ms_mv = time_ms(
        [&]() {
          auto r = mv * vvc;
          consume3(r, sink);
        },
        ITERS);

    const double ms_vm = time_ms(
        [&]() {
          auto r = vvr * mv;
          consume3(r, sink);
        },
        ITERS);

    std::cout << "float-float  | M*V: " << ms_mv << " ms/iter"
              << " | V*M: " << ms_vm << " ms/iter\n";
  }

  // ---------------------------------------------------------------------------
  // double-double
  // ---------------------------------------------------------------------------
  {
    Matrix<double> m(N, N);
    Vector<double> vc(N, Orientation::COLUMN);
    Vector<double> vr(N, Orientation::ROW);

    for (size_t i = 0; i < N; ++i) {
      const auto vv = static_cast<double>(vec_val(i));
      vc.at(i) = vv;
      vr.at(i) = vv;
      for (size_t j = 0; j < N; ++j) m.at(i, j) = static_cast<double>(mat_val(i, j));
    }

    MatrixView<double> mv(m.data(), N, N, m.column_count());
    VectorView<double> vvc(vc.data(), N, vc.orientation());
    VectorView<double> vvr(vr.data(), N, vr.orientation());

    for (int k = 0; k < WARMUP; ++k) {
      auto r1 = mv * vvc;
      auto r2 = vvr * mv;
      consume3(r1, sink);
      consume3(r2, sink);
    }

    const double ms_mv = time_ms(
        [&]() {
          auto r = mv * vvc;
          consume3(r, sink);
        },
        ITERS);

    const double ms_vm = time_ms(
        [&]() {
          auto r = vvr * mv;
          consume3(r, sink);
        },
        ITERS);

    std::cout << "double-double| M*V: " << ms_mv << " ms/iter"
              << " | V*M: " << ms_vm << " ms/iter\n";
  }

  // ---------------------------------------------------------------------------
  // int-int
  // ---------------------------------------------------------------------------
  {
    Matrix<int> m(N, N);
    Vector<int> vc(N, Orientation::COLUMN);
    Vector<int> vr(N, Orientation::ROW);

    for (size_t i = 0; i < N; ++i) {
      const int vv = vec_val(i);
      vc.at(i) = vv;
      vr.at(i) = vv;
      for (size_t j = 0; j < N; ++j) m.at(i, j) = mat_val(i, j);
    }

    MatrixView<int> mv(m.data(), N, N, m.column_count());
    VectorView<int> vvc(vc.data(), N, vc.orientation());
    VectorView<int> vvr(vr.data(), N, vr.orientation());

    for (int k = 0; k < WARMUP; ++k) {
      auto r1 = mv * vvc;
      auto r2 = vvr * mv;
      consume3(r1, sink);
      consume3(r2, sink);
    }

    const double ms_mv = time_ms(
        [&]() {
          auto r = mv * vvc;
          consume3(r, sink);
        },
        ITERS);

    const double ms_vm = time_ms(
        [&]() {
          auto r = vvr * mv;
          consume3(r, sink);
        },
        ITERS);

    std::cout << "int-int      | M*V: " << ms_mv << " ms/iter"
              << " | V*M: " << ms_vm << " ms/iter\n";
  }

  // ---------------------------------------------------------------------------
  // int-double (promotion) — same base integer matrix/vector values, vector is double
  // ---------------------------------------------------------------------------
  {
    Matrix<int> m(N, N);
    Vector<double> vc(N, Orientation::COLUMN);
    Vector<double> vr(N, Orientation::ROW);

    for (size_t i = 0; i < N; ++i) {
      const int vv_i = vec_val(i);
      vc.at(i) = static_cast<double>(vv_i);
      vr.at(i) = static_cast<double>(vv_i);
      for (size_t j = 0; j < N; ++j) m.at(i, j) = mat_val(i, j);
    }

    MatrixView<int> mv(m.data(), N, N, m.column_count());
    VectorView<double> vvc(vc.data(), N, vc.orientation());
    VectorView<double> vvr(vr.data(), N, vr.orientation());

    for (int k = 0; k < WARMUP; ++k) {
      auto r1 = mv * vvc;  // => Vector<double>
      auto r2 = vvr * mv;  // => Vector<double>
      consume3(r1, sink);
      consume3(r2, sink);
    }

    const double ms_mv = time_ms(
        [&]() {
          auto r = mv * vvc;
          consume3(r, sink);
        },
        ITERS);

    const double ms_vm = time_ms(
        [&]() {
          auto r = vvr * mv;
          consume3(r, sink);
        },
        ITERS);

    std::cout << "int-double   | M*V: " << ms_mv << " ms/iter"
              << " | V*M: " << ms_vm << " ms/iter\n";
  }

  std::cout << "sink=" << sink << "\n";
}

}  // namespace maf::measure

int main() {
#ifdef MAFLIB_OPENMP_AVAILABLE
  std::cout << "=== LinAlg measure (" << omp_get_max_threads() << " OMP threads)"
            << " ===\n";
#endif

  maf::measure::matmul_bench<float>({512, 1024, 2048});
  maf::measure::matmul_bench<double>({512, 1024, 2048});
  maf::measure::plu_bench();
  maf::measure::cholesky_bench();
  maf::measure::qr_bench({512, 1024, 2048});
  maf::measure::cast_bench();
  maf::measure::gemv_bench();

  return 0;
}
