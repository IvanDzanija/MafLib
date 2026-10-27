#pragma once
#ifdef MAFLIB_OPENMP_AVAILABLE
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "MafLib/math/linalg/LinAlg.hpp"

namespace maf::measure {

/**
 * @brief Times `fn` and returns the median run time in milliseconds.
 * @param warmup Number of untimed calls to establish steady state.
 * @param iters  Number of timed runs; the median is reported.
 */
template <typename Fn>
double bench_ms(Fn &&fn, int warmup = 1, int iters = 3) {
  using clock = std::chrono::steady_clock;
  for (int k = 0; k < warmup; ++k) {
    fn();
  }

  std::vector<double> times;
  times.reserve(iters);
  for (int k = 0; k < iters; ++k) {
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

/**
 * @brief Prints one machine-parseable result line:
 *        [PERF] name | config | <ms> ms | <gflops> GFLOPS
 */
inline void print_result(const std::string &name, const std::string &config, double ms,
                         double gflops = -1.0) {
  std::cout << "[PERF] " << std::left << std::setw(12) << name << "| " << std::setw(22)
            << config << "| " << std::fixed << std::setprecision(2) << ms << " ms | ";
  if (gflops >= 0.0) {
    std::cout << std::setprecision(2) << gflops << " GFLOPS";
  }
  std::cout << "\n";
}

/**
 * @brief Dense n x n matrix filled with uniform values in [lo, hi).
 *        Deterministic by default so timings are reproducible.
 */
template <typename T>
math::Matrix<T> make_random_matrix(size_t n, double lo = -10.0, double hi = 10.0,
                                   uint32 seed = 42) {
  math::Matrix<T> m(n, n);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(lo, hi);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      m.at(i, j) = dis(gen);
    }
  }
  return m;
}

}  // namespace maf::measure
