#ifndef EXTENDEDINTEGER
#define EXTENDEDINTEGER

#pragma once
#include "MafLib/main/GlobalHeader.hpp"

namespace maf::math {
/**
 * @brief An extended integer class that supports positive and negative infinity
 *
 * This class can represent regular integers as well as positive and negative
 * infinity values. It provides arithmetic operations that handle infinity
 * according to mathematical rules.
 *
 * @version 1.0
 * @since 2025
 */
class ExtendedInt {
 public:
  enum class InfinityType : uint8 { PosInf, NegInf };

  // Factory methods
  static ExtendedInt pos_inf() { return {InfinityType::PosInf}; }
  static ExtendedInt neg_inf() { return {InfinityType::NegInf}; }

  // Constructors
  ExtendedInt() : _value(0) {}  // Default constructor initilizes value to 0
  ExtendedInt(int value) : _value(value) {}
  ExtendedInt(InfinityType value) : _value(value) {}

  // Type checking
  bool is_inf() const { return std::holds_alternative<InfinityType>(_value); }
  bool is_finite() const { return std::holds_alternative<int>(_value); }

  bool is_pos_inf() const {
    return is_inf() && std::get<InfinityType>(_value) == InfinityType::PosInf;
  }
  [[nodiscard]] bool is_neg_inf() const {
    return is_inf() && std::get<InfinityType>(_value) == InfinityType::NegInf;
  }

  // Getters and setters
  int get_value() const {
    if (is_inf()) {
      throw std::out_of_range("Value is infinity!");
    }
    return std::get<int>(_value);
  }

  ExtendedInt operator+(const ExtendedInt &other) const {
    // finite + finite
    if (is_finite() && other.is_finite()) {
      int a = get_value(), b = other.get_value();

      // Overflow checking
      if ((b > 0 && a > INT_MAX - b) || (b < 0 && a < INT_MIN - b)) {
        return b > 0 ? pos_inf() : neg_inf();
      }
      return ExtendedInt(a + b);
    }

    else if (is_inf() && other.is_finite()) {
      return ExtendedInt(get_value());
    } else if (is_finite() && other.is_inf()) {
      return ExtendedInt(other.get_value());
    }

    // Mixing infinties is not allowed!
    else if ((is_pos_inf() && other.is_neg_inf()) ||
             (is_neg_inf() && other.is_pos_inf())) {
      throw std::runtime_error("Mixing infinities is not allowed!");
    } else {
      // Positive infinity + positive infinity
      return ExtendedInt(get_value());
    }
  }

  ExtendedInt operator-(const ExtendedInt &other) const {
    // finite + finite
    if (is_finite() && other.is_finite()) {
      int a = get_value(), b = other.get_value();

      // Overflow checking
      if ((b > 0 && a < INT_MIN + b) || (b < 0 && a > INT_MAX + b)) {
        return b < 0 ? pos_inf() : neg_inf();
      }
      return ExtendedInt(a - b);
    }

    else if (is_inf() && other.is_finite()) {
      return {get_value()};
    } else if (is_finite() && other.is_inf()) {
      return {other.get_value()};
    }

    // Same infinities are not allowed!
    else if ((is_pos_inf() && other.is_pos_inf()) ||
             (is_neg_inf() && other.is_neg_inf())) {
      throw std::runtime_error("Mixing infinities is not allowed!");
    } else {
      // Positive infinity - negative infinity
      return {get_value()};
    }
  }
  bool operator==(const ExtendedInt &other) const { return _value == other._value; }

 private:
  std::variant<int, InfinityType> _value;
};
}  // namespace maf::math

#endif
