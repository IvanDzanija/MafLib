#ifndef UTIL_CONVERSIONS_H
#define UTIL_CONVERSIONS_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::util {
template <typename To, typename From>
[[nodiscard]] static constexpr std::vector<To> convert_if_needed(
    const std::vector<From> &src) {
  if constexpr (std::is_same_v<To, From>) {
    return src;
  } else {
    std::vector<To> dst;
    dst.reserve(src.size());
    for (const From &v : src) {
      dst.emplace_back(static_cast<To>(v));
    }
    return dst;
  }
}

/** @brief Returns a pointer to the elements of @p src as @c To.
 *
 *  Borrowing backend: if the element types already match, the returned
 *  pointer points directly into @p src (no copy is made). Otherwise the
 *  elements are converted into @p buf and a pointer to that buffer is
 *  returned.
 *
 *  @attention @p buf must outlive every use of the returned pointer.
 *
 *  @tparam To The requested element type.
 *  @tparam From The source element type.
 *  @param buf Conversion buffer used only when a conversion is needed.
 *  @param src The source vector.
 *  @return A pointer to the elements, viewed as To.
 */
template <typename To, typename From>
[[nodiscard]] static constexpr const To *to_pointer(std::vector<To> &buf,
                                                    const std::vector<From> &src) {
  if constexpr (std::is_same_v<To, From>) {
    return src.data();
  } else {
    buf = util::convert_if_needed<To>(src);
    return buf.data();
  }
}

}  // namespace maf::util

#endif
