#ifndef MAFLIB_MATH_LINALG_LINALG_HPP
#define MAFLIB_MATH_LINALG_LINALG_HPP
#pragma once

/**
 * @file LinAlg.hpp
 * @brief Single user-facing entry point of the linear algebra module.
 *
 * Include this (MafLib/math/linalg/LinAlg.hpp) and you get the whole linalg
 * module: Vector, Matrix, both view classes, all operators, decompositions,
 * factories and the BLAS/LAPACK backends.
 */

#include "MafLib/math/linalg/detail/BlasWrappers/LapackRoutines.hpp"
#include "MafLib/math/linalg/detail/Cholesky.hpp"
#include "MafLib/math/linalg/detail/Core.hpp"
#include "MafLib/math/linalg/detail/Matrix.hpp"
#include "MafLib/math/linalg/detail/MatrixViewOperators.hpp"
#include "MafLib/math/linalg/detail/PLU.hpp"
#include "MafLib/math/linalg/detail/QR.hpp"
#include "MafLib/math/linalg/detail/Vector.hpp"
#include "MafLib/math/linalg/detail/VectorViewOperators.hpp"
#include "MafLib/math/linalg/detail/ViewKernels.hpp"

#endif  // MAFLIB_MATH_LINALG_LINALG_HPP
