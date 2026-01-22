#ifndef LINALG_H
#define LINALG_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {
using namespace maf::util;

// Classes
template <Numeric T>
class Vector;

template <Numeric T>
class Matrix;

template <Numeric T>
class VectorView;

template <Numeric T>
class MatrixView;

}  // namespace maf::math
#include "Matrix.hpp"
#include "Vector.hpp"
#endif
