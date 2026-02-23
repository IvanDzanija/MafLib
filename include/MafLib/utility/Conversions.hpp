#ifndef UTIL_CONVERSIONS_H
#define UTIL_CONVERSIONS_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::util {
template <typename To, typename From>
static std::vector<To> convert_if_needed(const std::vector<From> &src) {
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

}  // namespace maf::util

#endif
