#ifndef VECTOR_CONSTRUCTORS_H
#define VECTOR_CONSTRUCTORS_H
#pragma once
#include "MafLib/math/linalg/detail/Vector.hpp"

/**
 * @file VectorConstructors.hpp
 * @brief Contains implementations for the constructors of the Vector<T> class.
 *
 * It is pulled in by detail/linalg.hpp, the dependency-ordered hub
 * header of this directory. Avoid including it directly.
 */
namespace maf::math {
// Constructs an uninitialized vector of size size.
template <Numeric T>
Vector<T>::Vector(size_t size, Orientation orientation) : _orientation(orientation) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero.");
    }
    _data.resize(size);
}

// Constructs a vector from a raw data pointer.
template <Numeric T>
template <Numeric U>
Vector<T>::Vector(size_t size, const U* data, Orientation orientation)
    : _orientation(orientation) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero!");
    }
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null!");
    }

    _data.assign(data, data + size);
}

// Constructs from a std::vector, copy constructor
template <Numeric T>
template <Numeric U>
Vector<T>::Vector(size_t size, const std::vector<U>& data, Orientation orientation)
    : _orientation(orientation) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero.");
    }

    if (data.size() != size) {
        throw std::invalid_argument("Data size does not match vector size.");
    }
    _data = data;
}

// Constructs from a std::vector, move constructor
template <Numeric T>
Vector<T>::Vector(size_t size, std::vector<T>&& data, Orientation orientation)
    : _orientation(orientation) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero.");
    }

    if (data.size() != size) {
        throw std::invalid_argument("Data size does not match vector size.");
    }

    _data = std::move(data);
}

// Constructs from a std::array, copy constructor
template <Numeric T>
template <Numeric U, size_t N>
Vector<T>::Vector(size_t size, const std::array<U, N>& data, Orientation orientation)
    : _orientation(orientation) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero.");
    }
    if (N != size) {
        throw std::invalid_argument("Data size does not match vector size.");
    }

    _data.assign(data.begin(), data.end());
}

// Converting constructor
template <Numeric T>
template <Numeric U>
Vector<T>::Vector(const Vector<U>& other)
    : _orientation(other.orientation()),
      _data(other.data().begin(), other.data().end()) {}

}  // namespace maf::math

#endif
