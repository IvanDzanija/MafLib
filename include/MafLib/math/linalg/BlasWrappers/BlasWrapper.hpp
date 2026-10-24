#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#pragma once

/**
 * @file BlasWrapper.hpp
 * @brief Single entry point for the optimized BLAS/LAPACK backends.
 *
 * On Apple the Accelerate framework is used (vecLib). On Linux an optional
 * OpenBLAS backend is used when the OPENBLAS_AVAILABLE macro is defined
 * (set up by CMake). Both backends expose their routines through their own
 * namespace:
 *  - maf::math::acc  (Accelerate, Apple)
 *  - maf::math::oblas (OpenBLAS, Linux)
 *
 * Call sites never reference a backend directly; they use the
 * maf::math::blas namespace alias, which points to whichever backend is
 * available. When no backend is available nothing is declared and callers
 * fall back to the generic implementations.
 */

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
#include <vecLib/cblas_new.h>
#include <vecLib/lapack.h>
#include <vecLib/vDSP.h>

#include "MafLib/main/GlobalHeader.hpp"

namespace maf::math::acc {}  // namespace maf::math::acc
#elifdef OPENBLAS_AVAILABLE
#include <cblas.h>

#include "MafLib/main/GlobalHeader.hpp"

namespace maf::math::oblas {}  // namespace maf::math::oblas
#endif

#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
#include "MatrixRoutines.hpp"
#include "VectorMatrixRoutines.hpp"
#include "VectorRoutines.hpp"

namespace maf::math {
namespace blas = acc;
}  // namespace maf::math
#elifdef OPENBLAS_AVAILABLE
#include "MatrixRoutines.hpp"
#include "VectorMatrixRoutines.hpp"
#include "VectorRoutines.hpp"

namespace maf::math {
namespace blas = oblas;
}  // namespace maf::math
#endif

#endif
