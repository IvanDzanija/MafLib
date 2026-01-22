#ifndef VECTOR_VIEW_H
#define VECTOR_VIEW_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {
using namespace maf::util;
/**
 * @brief A lightweight view into a subvector of an existing Vector.
 * This class does not own the data; it provides a window into
 * the data of another Vector.
 *
 * @tparam T The numeric type of the vector elements (e.g., float, double).
 *
 * @version 1.0
 * @since 2025
 */
template <Numeric T>
class VectorView {
 public:
  /** @brief The numeric type of the vector elements. */
  using value_type = T;

  /** @brief Constructs a VectorView.
   * @param data Pointer to the starting element of the subvector.
   * @param size The logical number of elements in the subvector.
   * @param inc The stride (increment) between elements (default: 1 for
   * contiguous).
   */
  VectorView(T *data, size_t size, size_t inc = 1)
      : _data(data), _size(size), _inc(inc) {}

  /** @brief Accesses element at index i with the stride. */
  [[nodiscard]] T &operator[](size_t ind) noexcept { return _data[ind * _inc]; }

  /** @brief Accesses element at index i with the stride (const). */
  [[nodiscard]] const T &operator[](size_t ind) const noexcept {
    return _data[ind * _inc];
  }

  /** @brief Gets a mutable reference to the element at index i.
   * @throws std::out_of_range if the index is invalid.
   */
  [[nodiscard]] T &at(size_t ind) {
    if (ind >= _size) {
      throw std::out_of_range("VectorView index out of bounds");
    }
    return _data[ind * _inc];
  }

  /** @brief Gets a const reference to the element at index i.
   * @throws std::out_of_range if the index is invalid.
   */
  [[nodiscard]] const T &at(size_t ind) const {
    if (ind >= _size) {
      throw std::out_of_range("VectorView index out of bounds");
    }
    return _data[ind * _inc];
  }

  /** @brief Gets the size of the VectorView. */
  [[nodiscard]] size_t size() const noexcept { return _size; }

  /** @brief Gets the increment (stride) of the VectorView. */
  [[nodiscard]] size_t increment() const noexcept { return _inc; }

 private:
  T *_data;      // Starting point
  size_t _size;  // Logical length
  size_t _inc;   // Distance between elements (1 = contiguous)
};

}  // namespace maf::math
#endif
