#pragma once

#include <cstdint>

#include "common.h"
#include "kernel_operator.h"
namespace xpu_ops::kernels {

__aicore__ inline uint32_t ceil_div(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

__aicore__ inline uint32_t align_to(uint32_t x, uint32_t y) { return ceil_div(x, y) * y; }

__aicore__ inline uint32_t get_tail_size(uint32_t x, uint32_t y) { return x - (ceil_div(x, y) - 1) * y; }

__aicore__ inline uint32_t floor_div(uint32_t x, uint32_t y) { return x / y; }

__aicore__ inline uint32_t floor_align_to(uint32_t x, uint32_t y) { return floor_div(x, y) * y; }

template <typename T>
__aicore__ inline T greatest_common_divisor(T a, T b) {
  T c = a;
  if (a < b) {
    a = b;
    b = c;
  }
  while (b != 0) {
    c = a;
    a = b;
    b = c % b;
  }
  return a;
}

template <typename T>
__aicore__ inline T least_common_multiple(T a, T b) {
  return a * b / greatest_common_divisor(a, b);
}

template <typename T>
__aicore__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__aicore__ inline T min(T a, T b) {
  return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T align_up(T a) {
  return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T align_down(T a, T base) {
  if (unlikely(base == 0)) {
    return a;
  }
  return a / base * base;
}

template <>
__aicore__ inline uint32_t align_up<4, uint32_t>(uint32_t a) {
  // to be Multiple of 4, result should be in a format of b(xxxx,x100).
  // This means last two bits should be zero, requiring that
  // result = num & b(1111,1100) = num & (~3).
  // &(~3) operator may reduces num into the range [num, num - 3].
  // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
  // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
  // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
  // num = a + 3 also satisfies the goal condition.
  return (a + 3) & ~3;  // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t align_up<16, uint32_t>(uint32_t a) {
  // In general, if we want to get the least multiple of b (b is the power of 2) for a,
  // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
  return (a + 15) & ~15;  // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t align_up<32, uint32_t>(uint32_t a) {
  // refer to the above comments.
  return (a + 31) & ~31;  // & ~31: set last five bits of (a+31) to be zero}
}

}  // namespace xpu_ops::kernels
