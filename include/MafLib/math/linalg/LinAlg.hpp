#ifndef LINALG_H
#define LINALG_H
#pragma once

#include "MafLib/math/Math.hpp"

namespace maf::math {

// Constants
static constexpr uint8 BLOCK_SIZE = 64;
static constexpr uint8 FLOAT_PRECISION = 5;

// Classes
template <Numeric T>
class Vector;

template <Numeric T>
class Matrix;

/** @brief Specifies if the vector behaves as a row or column vector. */
enum Orientation : uint8 { ROW, COLUMN };

// Functions

}  // namespace maf::math

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
#include "AccelerateWrappers/AccelerateWrapper.hpp"
#endif

#include "Matrix.hpp"
#include "Vector.hpp"
#endif
