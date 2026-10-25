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

// Containers
#include "MafLib/math/linalg/detail/Matrix.hpp"
#include "MafLib/math/linalg/detail/Vector.hpp"

// 6. Constructors and factories (decompositions use e.g. identity_matrix).
#include "MafLib/math/linalg/detail/MatrixConstructors.hpp"
#include "MafLib/math/linalg/detail/MatrixFactories.hpp"

// 7. Decompositions.
#include "MafLib/math/linalg/detail/Cholesky.hpp"
#include "MafLib/math/linalg/detail/PLU.hpp"
#include "MafLib/math/linalg/detail/QR.hpp"

// 8. Member/free operator and factory definitions.
#include "MafLib/math/linalg/detail/MatrixOperators.hpp"
#include "MafLib/math/linalg/detail/MatrixViewOperators.hpp"
#include "MafLib/math/linalg/detail/VectorCheckers.hpp"
#include "MafLib/math/linalg/detail/VectorConstructors.hpp"
#include "MafLib/math/linalg/detail/VectorMethods.hpp"
#include "MafLib/math/linalg/detail/VectorOperators.hpp"
#include "MafLib/math/linalg/detail/VectorViewOperators.hpp"

#endif  // MAFLIB_MATH_LINALG_LINALG_HPP
