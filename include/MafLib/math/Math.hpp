#include "MafLib/main/GlobalHeader.hpp"
#include "MafLib/utility/Math.hpp"

namespace maf::math {
using namespace util;
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

/*** @brief Constant used as OMP lower bound for linear algorithms. */
constexpr static size_t OMP_LINEAR_LIMIT = 500000;

/*** @brief Constant used as OMP lower bound for quadratic algorithms. */
constexpr static size_t OMP_QUADRATIC_LIMIT = 500 * 500;

/*** @brief Constant used as OMP lower bound for cubic algorithms. */
constexpr static size_t OMP_CUBIC_LIMIT = 50 * 50;
}  // namespace maf::math
