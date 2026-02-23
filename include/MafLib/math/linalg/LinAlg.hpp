#ifndef LINALG_H
#define LINALG_H
#pragma once
#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {
using namespace maf::util;

// Classes
template <Numeric T>
class Vector;

template <Numeric T>
class Matrix;

template <Numeric T>
class VectorView;

template <Numeric T>
class MatrixView;

// Concepts
/** @brief Clean Vector concept.
 * Removes qualifiers and checks the value type of the Vector is Numeric concept.
 */
template <typename T>
concept VectorType = requires { typename std::remove_cvref_t<T>::value_type; } &&
                     Numeric<typename std::remove_cvref_t<T>::value_type> &&
                     std::same_as<std::remove_cvref_t<T>,
                                  Vector<typename std::remove_cvref_t<T>::value_type>>;

/** @brief Clean VectorViewType concept.
 * Removes qualifiers and checks the value type of the MatrixView is Numeric concept.
 */
template <typename T>
concept VectorViewType =
    requires { typename std::remove_cvref_t<T>::value_type; } &&
    Numeric<typename std::remove_cvref_t<T>::value_type> &&
    std::same_as<std::remove_cvref_t<T>,
                 VectorView<typename std::remove_cvref_t<T>::value_type>>;

/** @brief Clean concept compatible with VectorView.
 * MatrixViewCompatible is either a VectorView or a Vector that can be made into a view.
 */
template <typename T>
concept VectorViewCompatible = VectorViewType<T> || VectorType<T>;

/** @brief Clean Matrix concept.
 * Removes qualifiers and checks the value type of the Matrix is Numeric concept.
 */
template <typename T>
concept MatrixType = requires { typename std::remove_cvref_t<T>::value_type; } &&
                     Numeric<typename std::remove_cvref_t<T>::value_type> &&
                     std::same_as<std::remove_cvref_t<T>,
                                  Matrix<typename std::remove_cvref_t<T>::value_type>>;

/** @brief Clean MatrixViewType concept.
 * Removes qualifiers and checks the value type of the MatrixView is Numeric concept.
 */
template <typename T>
concept MatrixViewType =
    requires { typename std::remove_cvref_t<T>::value_type; } &&
    Numeric<typename std::remove_cvref_t<T>::value_type> &&
    std::same_as<std::remove_cvref_t<T>,
                 MatrixView<typename std::remove_cvref_t<T>::value_type>>;

/** @brief Clean concept compatible with MatrixView.
 * MatrixViewCompatible is either a MatrixView or a Matrix that can be made into a view.
 */
template <typename T>
concept MatrixViewCompatible = MatrixViewType<T> || MatrixType<T>;
}  // namespace maf::math
#include "Matrix.hpp"
#include "Vector.hpp"
#endif
