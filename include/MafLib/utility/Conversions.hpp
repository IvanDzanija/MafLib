#ifndef UTIL_CONVERSIONS_H
#define UTIL_CONVERSIONS_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "Math.hpp"

namespace maf::util {

/** @brief Promotes an integral input type to the LAPACK-compatible float.
 *
 *  Floating types pass through untouched (this trait never demotes). Integral
 *  types promote to the smallest floating type that can hold them: types no
 *  wider than float (int, short, char, ...) promote to float, wider types
 *  (long, long long, ...) promote to double.
 */
template <Numeric T>
using float_promote_t =
    std::conditional_t<std::is_floating_point_v<T>, T,
                       std::conditional_t<(sizeof(T) <= sizeof(float)), float, double>>;

/** @brief Promotes any integral input type to double for LAPACK work.
 *
 *  Floating types pass through untouched. Use this instead of
 *  float_promote_t when the extra precision of double is wanted for
 *  every integral input.
 */
template <Numeric T>
using double_promote_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

/** @brief Promotes an integral input type to float for LAPACK work. */
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
