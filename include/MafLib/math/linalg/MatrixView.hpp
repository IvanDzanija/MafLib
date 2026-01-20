#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#pragma once
#include "LinAlg.hpp"
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::math {
/**
 * @brief A lightweight view into a submatrix of an existing Matrix.
 * This class does not own the data; it provides a window into
 * the data of another Matrix.
 *
 * @tparam T The numeric type of the matrix elements (e.g., float, double).
 *
 * @version 1.0
 * @since 2025
 */

template <Numeric T>
class MatrixView {
public:
    MatrixView(T *data, size_t r, size_t c, size_t s)
        : _data(data), _rows(r), _cols(c), _stride(s) {}

    /**
     * @brief Accesses the element at [row][col] with no bounds check.
     */
    [[nodiscard]] T *operator[](size_t r) noexcept {
        return _data + (r * _stride);
    }

    /**
     * @brief Accesses the element at (row, col) with no bounds check.
     */
    [[nodiscard]] const T *operator[](size_t r) const noexcept {
        return _data + (r * _stride);
    }

    /**
     * @brief Gets a mutable reference to the element at (row, col).
     * @throws std::out_of_range if the index is invalid.
     */
    T &at(size_t row, size_t col) {
        if (row >= _rows || col >= _cols) {
            throw std::out_of_range("View index out of bounds");
        }
        return _data[row * _stride + col];
    }

    /**
     * @brief Gets a const reference to the element at (row, col).
     * @throws std::out_of_range if the index is invalid.
     */
    [[nodiscard]] const T &at(size_t row, size_t col) const {
        if (row >= _rows || col >= _cols) {
            throw std::out_of_range("View index out of bounds");
        }
        return _data[row * _stride + col];
    }

    /**
     * @brief Gets a const std::span of a single row.
     * @throws std::out_of_range if the row is invalid.
     */
    [[nodiscard]] std::span<T> row_span(size_t r) {
        if (r >= _rows) {
            throw std::out_of_range("Row index out of bounds");
        }
        return std::span<T>(_data + (r * _stride), _cols);
    }

    /** @brief Returns a const span of a single row.
     * @throws std::out_of_range if the row is invalid.
     */
    [[nodiscard]] std::span<const T> row_span(size_t r) const {
        if (r >= _rows) {
            throw std::out_of_range("Row index out of bounds");
        }
        return std::span<const T>(_data + (r * _stride), _cols);
    }

    /** @brief Gets the number of rows. */
    [[nodiscard]] size_t row_count() const noexcept {
        return _rows;
    }

    /** @brief Gets the number of columns. */
    [[nodiscard]] size_t column_count() const noexcept {
        return _cols;
    }

    /** @brief Gets the stride of parent Matrix. */
    [[nodiscard]] size_t get_stride() const noexcept {
        return _stride;
    }

private:
    T *_data;        // Pointer to the start of the submatrix (top-left)
    size_t _rows;    // Logical height
    size_t _cols;    // Logical width
    size_t _stride;  // Parent Matrix width
};

}  // namespace maf::math
#endif
