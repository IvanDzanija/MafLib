#ifndef MATRIX_METHODS_H
#define MATRIX_METHODS_H
#pragma once
#include "Matrix.hpp"

namespace maf::math {
// Cast and create a new matrix
template <Numeric T>
template <Numeric U>
[[nodiscard]] Matrix<U> Matrix<T>::cast() const {
  // Replace this with explicit constructor casting when implemented
  Matrix<U> result(_rows, _cols);

  const size_t n = _data.size();
  for (size_t i = 0; i < n; ++i) {
    result.data()[i] = static_cast<U>(_data[i]);
  }
  return result;
}

// Inplace fill
template <Numeric T>
void Matrix<T>::fill(T value) {
  if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] = value;
    }
  } else {
    for (size_t i = 0; i < _data.size(); ++i) {
      _data[i] = value;
    }
  }
}

// Set to identity
template <Numeric T>
void Matrix<T>::make_identity() {
  if (!is_square()) {
    throw std::invalid_argument("Only square matrices can be set to identity!");
  }

  fill(T(0));
  for (size_t i = 0; i < _rows; ++i) {
    this->at(i, i) = T(1);
  }
}

// Performs an in-place transpose of the matrix.
template <Numeric T>
void Matrix<T>::transpose() {
  if (!is_square()) {
    throw std::invalid_argument("Matrix must be square to transpose in-place.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < _rows; i += BLOCK_SIZE) {
    for (size_t j = i; j < _cols; j += BLOCK_SIZE) {
      const size_t n = std::min(i + BLOCK_SIZE, _rows);
      const size_t m = std::min(j + BLOCK_SIZE, _cols);
      if (i == j) {
        // Diagonal block
        for (size_t k = i; k < n; ++k) {
          for (size_t l = k + 1; l < m; ++l) {
            std::swap(this->at(k, l), this->at(l, k));
          }
        }
      } else {
        // Off-diagonal block
        for (size_t k = i; k < n; ++k) {
          for (size_t l = j; l < m; ++l) {
            std::swap(this->at(k, l), this->at(l, k));
          }
        }
      }
    }
  }
}

// Creates new transposed matrix
template <Numeric T>
Matrix<T> Matrix<T>::transposed() const {
  Matrix<T> result(_cols, _rows);

#pragma omp parallel for if (_data.size() > 100 * 100)
  for (size_t i = 0; i < _rows; ++i) {
    for (size_t j = 0; j < _cols; ++j) {
      result.at(j, i) = this->at(i, j);
    }
  }
  return result;
}

}  // namespace maf::math

#endif
