#ifndef ACCELERATE_WRAPPER_H
#define ACCELERATE_WRAPPER_H

#pragma once
#if defined(__APPLE__) && defined(ACCELERATE_AVAILABLE)
#include <vecLib/cblas_new.h>
#include <vecLib/lapack.h>
#include <vecLib/vDSP.h>

#include "MafLib/main/GlobalHeader.hpp"
#include "MatrixRoutines.hpp"
#include "VectorMatrixRoutines.hpp"
#include "VectorRoutines.hpp"
#endif

#endif
