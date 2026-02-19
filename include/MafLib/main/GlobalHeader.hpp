#ifndef GLOBALHEADER_HPP
#define GLOBALHEADER_HPP

#pragma once
#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace maf {
using int8 = int8_t;
using uint8 = uint8_t;
using int16 = int16_t;
using uint16 = uint16_t;
using int32 = int32_t;
using uint32 = uint32_t;
using int64 = int64_t;
using uint64 = uint64_t;
using char8 = char8_t;
using char16 = char16_t;

}  // namespace maf

#endif  // GLOBALHEADER_HPP
