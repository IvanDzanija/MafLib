#ifndef VECTOR_H
#define VECTOR_H
#pragma once
#include "LinAlg.hpp"

namespace maf::math {

/**
 * @brief A general-purpose mathematical vector class.
 *
 * This class implements a mathematical vector, wrapping a std::vector
 * for data storage. It supports both **ROW** and **COLUMN** orientations,
 * which is crucial for correct algebraic operations with the Matrix class.
 *
 * It is templated to support various numeric types (T) and provides
 * a range of constructors, element-wise operators, and common
 * vector operations like norm, normalization, and dot product.
 *
 * @tparam T The numeric type of the vector elements (e.g., float,
 * double, int).
 *
 * @version 1.0
 * @since 2025
 */
template <Numeric T>
class Vector {
public:
    /** @brief The numeric type of the vector's elements. */
    using value_type = T;

    // --- Constructors ---

    /**
     * @brief Default constructor.
     * @details Creates an empty vector with default COLUMN orientation.
     */
    Vector() : _orientation(COLUMN) {}

    /**
     * @brief Constructs an uninitialized vector of a given size.
     * @param size The number of elements in the vector.
     * @param orientation The vector's orientation (default: COLUMN).
     * @throws std::invalid_argument if size is zero.
     */
    Vector(size_t size, Orientation orientation = COLUMN);

    /**
     * @brief Constructs from a raw data pointer.
     * @details Elements are COPIED from the data pointer.
     * @param size The number of elements in the vector.
     * @param data Pointer to a contiguous memory.
     * @param orientation The vector's orientation (default: COLUMN).
     * @throws std::invalid_argument if size is zero or data is nullptr.
     */
    template <Numeric U>
    Vector(size_t size, const U* data, Orientation orientation = COLUMN);

    /**
     * @brief Constructs from a std::vector by copying its data.
     * @param size The number of elements. Must match data.size().
     * @param data The std::vector to copy from.
     * @param orientation The vector's orientation (default: COLUMN).
     * @throws std::invalid_argument if size is zero or data size
     * mismatches.
     */
    template <Numeric U>
    Vector(size_t size, const std::vector<U>& data, Orientation orientation = COLUMN);

    /**
     * @brief Constructs from a std::vector by moving its data.
     * @param size The number of elements. Must match data.size().
     * @param data The std::vector to move from (r-value).
     * @param orientation The vector's orientation (default: COLUMN).
     * @throws std::invalid_argument if size is zero or data size
     * mismatches.
     */
    Vector(size_t size, std::vector<T>&& data, Orientation orientation = COLUMN);

    /**
     * @brief Constructs from a std::array by copying its data.
     * @tparam U Type in the array (allows implicit conversion).
     * @tparam N Size of the array.
     * @param size The number of elements. Must match N.
     * @param data The std::array to copy from.
     * @param orientation The vector's orientation (default: COLUMN).
     * @throws std::invalid_argument if size is zero or array size
     * mismatches.
     */
    template <Numeric U, size_t N>
    Vector(size_t size, const std::array<U, N>& data, Orientation orientation = COLUMN);

    /**
     * @brief Converting constructor.
     */
    template <Numeric U>
    [[nodiscard]] Vector(const Vector<U>& other);

    // --- Iterators ---
    /** @brief Returns an iterator to the beginning. */
    [[nodiscard]] auto begin() noexcept {
        return _data.begin();
    }
    /** @brief Returns an iterator to the end. */
    [[nodiscard]] auto end() noexcept {
        return _data.end();
    }

    /** @brief Returns a const iterator to the beginning. */
    [[nodiscard]] auto begin() const noexcept {
        return _data.begin();
    }
    /** @brief Returns a const iterator to the end. */
    [[nodiscard]] auto end() const noexcept {
        return _data.end();
    }

    /** @brief Returns a const iterator to the beginning. */
    [[nodiscard]] auto cbegin() const noexcept {
        return _data.cbegin();
    }
    /** @brief Returns a const iterator to the end. */
    [[nodiscard]] auto cend() const noexcept {
        return _data.cend();
    }

    /** @brief Returns a reverse iterator to the beginning. */
    [[nodiscard]] auto rbegin() noexcept {
        return _data.rbegin();
    }
    /** @brief Returns a reverse iterator to the end. */
    [[nodiscard]] auto rend() noexcept {
        return _data.rend();
    }

    /** @brief Returns a const reverse iterator to the beginning. */
    [[nodiscard]] auto rbegin() const noexcept {
        return _data.rbegin();
    }
    /** @brief Returns a const reverse iterator to the end. */
    [[nodiscard]] auto rend() const noexcept {
        return _data.rend();
    }

    /** @brief Returns a const reverse iterator to the beginning. */
    [[nodiscard]] auto crbegin() const noexcept {
        return _data.crbegin();
    }
    /** @brief Returns a const reverse iterator to the end. */
    [[nodiscard]] auto crend() const noexcept {
        return _data.crend();
    }

    // --- Getters ---

    /**
     * @brief Gets a const reference to the underlying std::vector data
     * store.
     * @return const std::vector<T>&
     */
    [[nodiscard]] const std::vector<T>& data() const noexcept {
        return _data;
    }

    /** @brief Gets the number of elements in the vector. */
    [[nodiscard]] size_t size() const noexcept {
        return _data.size();
    }

    /** @brief Gets the vector's orientation (ROW or COLUMN). */
    [[nodiscard]] Orientation orientation() const noexcept {
        return _orientation;
    }

    /**
     * @brief Gets a mutable reference to the element at a specific index with
     * bounds check.
     * @throws std::out_of_range if the index is invalid.
     */
    T& at(size_t index) {
        return _data.at(index);
    }

    /**
     * @brief Gets a const reference to the element at a specific index with
     * bounds check.
     * @throws std::out_of_range if the index is invalid.
     */
    const T& at(size_t index) const {
        return _data.at(index);
    }

    /**
     * @brief Accesses the element at a specific index with no bounds check.
     * @throws std::out_of_range if the index is invalid.
     */
    T& operator[](size_t index) {
        return _data[index];
    }

    /**
     * @brief Accesses the element at a specific index with no bounds check.
     * @throws std::out_of_range if the index is invalid.
     */
    const T& operator[](size_t index) const {
        return _data[index];
    }

    // --- Checkers ---

    /**
     * @brief Checks if the vector is a null vector.
     * @return true if all elements are close to zero, false otherwise.
     */
    [[nodiscard]] bool is_null() const noexcept;

    // --- Methods ---

    /** @brief Fills the entire vector with a single value. */
    void fill(T value) noexcept;

    /** @brief Calculates the L2 norm (Euclidean length) of the vector. */
    [[nodiscard]] T norm() const noexcept;

    /** @brief Normalizes the vector in-place (divides by its L2 norm). */
    void normalize();

    /** @brief Transposes the vector in-place (flips orientation). */
    void transpose() noexcept;

    /** @brief Creates and returns a new vector that is the transpose of
     * this one. */
    [[nodiscard]] Vector<T> transposed() const noexcept;

    // --- Operators ---

    /**
     * @brief Checks for exact element-wise equality.
     * @details For floating-point, use `loosely_equal()`
     * @param other The vector to compare against.
     * @return true if size, orientation, and all elements are identical.
     */
    [[nodiscard]] constexpr bool operator==(const Vector& other) const noexcept;

    /**
     * @brief Unary minus. Returns a new Vector with all elements negated.
     * @return `Vector<T>`
     */
    [[nodiscard]] auto operator-() const noexcept;

    /**
     * @brief Element-wise vector addition (Vector + Vector).
     * @tparam U Numeric type of the other vector.
     * @param other The vector to add.
     * @return A new Vector of the common, promoted type.
     * @throws std::invalid_argument if dimensions or orientations do not
     * match.
     */
    template <Numeric U>
    [[nodiscard]] auto operator+(const Vector<U>& other) const;

    /**
     * @brief Element-wise scalar addition (Vector + scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to add to each element.
     * @return A new Vector of the common, promoted type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator+(const U& scalar) const noexcept;

    /**
     * @brief Element-wise vector addition assignment (Vector + Vector).
     * @tparam U Numeric type of the other vector.
     * @param other The vector to add.
     * @attention This method doesn't cast the vector if U is a broader type.
     * @throws std::invalid_argument if dimensions or orientations do not
     * match.
     */
    template <Numeric U>
    [[nodiscard]] auto operator+=(const Vector<U>& other) const;

    /**
     * @brief Element-wise scalar addition assignment (Vector + scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to add to each element.
     * @attention This method doesn't cast the vector if U is a broader type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator+=(const U& scalar) const noexcept;

    /**
     * @brief Element-wise vector subtraction (Vector - Vector).
     * @tparam U Numeric type of the other vector.
     * @param other The vector to subtract.
     * @return A new Vector of the common, promoted type.
     * @throws std::invalid_argument if dimensions or orientations do not
     * match.
     */
    template <Numeric U>
    [[nodiscard]] auto operator-(const Vector<U>& other) const;

    /**
     * @brief Element-wise scalar subtraction (Vector - scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to subtract from each element.
     * @return A new Vector of the common, promoted type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator-(const U& scalar) const noexcept;

    /**
     * @brief Element-wise vector subtraction assignment (Vector - Vector).
     * @tparam U Numeric type of the other vector.
     * @param other The vector to subtract.
     * @attention This method doesn't cast the vector if U is a broader type.
     * @throws std::invalid_argument if dimensions or orientations do not
     * match.
     */
    template <Numeric U>
    [[nodiscard]] auto operator-=(const Vector<U>& other) const;

    /**
     * @brief Element-wise scalar subtraction (Vector - scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to subtract from each element.
     * @attention This method doesn't cast the Vector if U is a broader type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator-=(const U& scalar) const noexcept;

    /**
     * @brief Element-wise scalar multiplication (Vector * scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to multiply by.
     * @return A new Vector of the common, promoted type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator*(const U& scalar) const noexcept;

    // TODO: add tests from here
    /**
     * @brief Element-wise scalar multiplication assignment (Vector * scalar).
     * @tparam U An arithmetic scalar type.
     * @param scalar The scalar value to multiply by.
     * @attention This method doesn't cast the Vector if U is a broader type.
     */
    template <Numeric U>
    [[nodiscard]] auto operator*=(const U& scalar) const noexcept;

    /**
     * @brief Calculates the dot product (inner product) of two Vectors.
     * @attention This method doesn't check for correct Vector orientations.
     * @tparam U Numeric type of the other Vector.
     * @param other The other Vector.
     * @return A scalar value of the common, promoted type.
     * @throws std::invalid_argument if Vector sizes do not match.
     */
    template <Numeric U>
    [[nodiscard]] auto dot_product(const Vector<U>& other) const;

    /**
     * @brief Outer product (Column Vector * Row Vector).
     * @details This must be a (N x 1) * (1 x M) multiplication.
     * @tparam U Numeric type of the other vector.
     * @param other The row Vector (this Vector must be a column).
     * @return A new Matrix of size (this.size x other.size).
     * @throws std::invalid_argument if orientations are not COLUMN * ROW.
     */
    template <Numeric U>
    [[nodiscard]] auto outer_product(const Vector<U>& other) const;

    /**
     * @brief Dot product (Row Vector * Column Vector).
     * @attention This must be a (1 x N) * (N x 1) multiplication.
     * @details Checks for correct orientations then calls \ref dot_product().
     * @tparam U Numeric type of the other vector.
     * @param other The row Vector (this Vector must be a column).
     * @return A new Matrix of size (this.size x other.size).
     * @throws std::invalid_argument if orientations are not ROW * COLUMN.
     * @throws std::invalid_argument if Vector sizes do not match.
     */
    template <Numeric U>
    [[nodiscard]] auto operator*(const Vector<U>& other) const;

    // TODO: Refactoring and stopped here. Continue from here
    // COMPARE BLAS ROUTINES TO OMP ONES

    /**
     * @brief Matrix-Vector multiplication (Row Vector * Matrix).
     * @details This must be a (1 x N) * (N x M) multiplication.
     * @tparam U Numeric type of the matrix.
     * @param other The matrix to multiply by (this vector must be a row).
     * @return A new row Vector of size (1 x M).
     * @throws std::invalid_argument if this is not a row vector or
     * dimensions mismatch.
     */
    template <Numeric U>
    [[nodiscard]] auto operator*(const Matrix<U>& other) const;

    // --- Printing and debugging ---

    /**
     * @brief Prints the vector contents to std::cout.
     * @details Sets floating point precision for readability.
     */
    void print() const {
        if constexpr (std::is_floating_point_v<T>) {
            std::cout << std::setprecision(FLOAT_PRECISION);
        }
        if (_orientation == COLUMN) {
            for (const T& val : _data) {
                std::cout << val << std::endl;
            }
        } else {
            for (const T& val : _data) {
                std::cout << val << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::fixed;
    }

private:
    /** @brief Stores the vector's orientation (ROW or COLUMN). */
    Orientation _orientation;

    /** @brief Internal contiguous storage for the vector elements. */
    std::vector<T> _data;

    /**
     * @brief Internal helper to invert the sign of all elements in-place.
     */
    void _invert_sign() {
        if (_data.size() > OMP_LINEAR_LIMIT) {
#pragma omp parallel for
            for (size_t i = 0; i < _data.size(); ++i) {
                _data[i] = -_data[i];
            }
        } else {
#pragma omp simd
            for (size_t i = 0; i < _data.size(); ++i) {
                _data[i] = -_data[i];
            }
        }
    }
};

}  // namespace maf::math

#include "VectorCheckers.hpp"
#include "VectorConstructors.hpp"
#include "VectorMethods.hpp"
#include "VectorOperators.hpp"

#endif
