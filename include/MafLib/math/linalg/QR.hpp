#ifndef QR_HPP
#define QR_HPP
#include "MafLib/main/GlobalHeader.hpp"
#include "Matrix.hpp"
#pragma once

namespace maf::math {

template <typename T>
[[nodiscard]] std::pair<Matrix<T>, Matrix<T>> QR_decompostion(const Matrix<T> &matrix) {
    if (!matrix.is_symmetric()) {
        throw std::invalid_argument(
            "Matrix must be symmetric to try QR decomposition!");
    }

    size_t n = matrix.row_count();
}

}  // namespace maf::math
#endif
