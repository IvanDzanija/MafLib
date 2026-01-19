#ifndef MATRIX_CONSTRUCTORS_H
#define MATRIX_CONSTRUCTORS_H
#pragma once
#include "Matrix.hpp"

/**
 * @file MatrixConstructors.hpp
 * @brief Contains implementations for the constructors of the Matrix<T> class.
 *
 * This file is intended to be included at the *end* of Matrix.hpp and
 * should not be included directly anywhere else.
 */
namespace maf::math {
// Constructs an uninitialized matrix of size rows x cols.
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    _data.resize(rows * cols);
}

// Constructs a matrix from a raw data pointer.
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols, T *data) : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero!");
    }
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null!");
    }

    _data.assign(data, data + (rows * cols));
}

// Constructs from a std::vector, filled by rows.
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<T> &data)
    : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.assign(data.begin(), data.end());
}

// Constructs from a std::vector, move constructor.
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols, std::vector<T> &&data)
    : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix size.");
    }
    _data = std::move(data);
}

// Constructs from a nested std::vector (vector of vectors).
template <Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<std::vector<T>> &data)
    : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }

    if (data.size() != rows || data.at(0).size() != cols) {
        throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            _data.at(_get_index(i, j)) = static_cast<T>(data.at(i).at(j));
        }
    }
}

// Constructs from a std::array, filled by rows.
template <Numeric T>
template <size_t N>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::array<T, N> &data)
    : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (N != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix size.");
    }

    _data.assign(data.begin(), data.end());
}

// Constructs from a std::initializer_list, filled by rows.
template <Numeric T>
template <Numeric U>
Matrix<T>::Matrix(size_t rows, size_t cols, std::initializer_list<U> list)
    : _rows(rows), _cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be greater than zero.");
    }
    if (static_cast<size_t>(list.size()) != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix size.");
    }
    _data.assign(list.begin(), list.end());
}

}  // namespace maf::math

#endif
