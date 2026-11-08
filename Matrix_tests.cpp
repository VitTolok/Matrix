#include <compare>
#include <cstdint>
#include <mdspan>
#include <ranges>
#include <span>
#include <type_traits>
#include "Matrix.hpp"

#if defined(__clang__)
#define STATIC_DO_FAIL(str) return false;
#else
#define STATIC_DO_FAIL(str) throw(str);
#endif

#define STATIC_ASSERT(...)                              \
  do {                                                  \
    if (!(__VA_ARGS__)) {                               \
      STATIC_DO_FAIL("failed condition: " #__VA_ARGS__) \
      return false;                                     \
    }                                                   \
  } while (false);

#if defined(__clang__)  // clang currently cannot handle this correctly due to lack of constexpr exceptions
#define STATIC_ASSERT_THROW(...)

#else
#define STATIC_ASSERT_THROW(...)                                          \
  do {                                                                    \
    bool caught = false;                                                  \
    try {                                                                 \
      {                                                                   \
        __VA_ARGS__;                                                      \
      }                                                                   \
    } catch (...) {                                                       \
      caught = true;                                                      \
    }                                                                     \
    if (!caught) {                                                        \
      STATIC_DO_FAIL("exception expected but not thrown: " #__VA_ARGS__); \
      return false;                                                       \
    }                                                                     \
  } while (false);
#endif

// #define STATIC_ASSERT_FAILURE(...)                                               \
//   do {                                                                           \
//     auto _probe = [&]<typename _T = void>() -> decltype(void(__VA_ARGS__)) {};   \
//     static_assert(!std::is_invocable_v<decltype(_probe)>, "unexpected success"); \
//   } while (false);
// all compilers suck at this apparently
#define STATIC_ASSERT_FAILURE(...)

// Stack Extents Calculation Tests
namespace matrix_stack_tests {
using matrix_implementation_detail::HStackExtents;
using matrix_implementation_detail::HStackExtentsStackable;
using matrix_implementation_detail::HStackExtentsType;
using matrix_implementation_detail::VStackExtents;
using matrix_implementation_detail::VStackExtentsStackable;
using matrix_implementation_detail::VStackExtentsType;

static_assert(
    []() constexpr -> bool {
      auto ext1 = std::extents<int32_t, 2, 3>{};
      auto ext2 = std::extents<int32_t, 2, 2>{};
      auto ext3 = std::extents<int32_t, 3, 2>{};
      auto ext4 = std::extents<int32_t, 3, 3>{};
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<decltype(ext1), decltype(ext4)>, std::extents<int32_t, 5, 3>>)
      STATIC_ASSERT(!VStackExtentsStackable<decltype(ext1), decltype(ext2)>);
      STATIC_ASSERT(!VStackExtentsStackable<decltype(ext1), decltype(ext3)>);
      STATIC_ASSERT_FAILURE(VStackExtentsType<decltype(ext1), decltype(ext2)>{};);
      STATIC_ASSERT_FAILURE(VStackExtentsType<decltype(ext1), decltype(ext3)>{};);
      STATIC_ASSERT_FAILURE(VStackExtents(ext1, ext2););
      STATIC_ASSERT_FAILURE(VStackExtents(ext1, ext3););
      STATIC_ASSERT(VStackExtents(ext1, ext4) == std::extents<int32_t, 5, 3>{});
      auto x = Matrix<int, std::extents<int, 2, 3>>{1, 2, 3, 4, 5, 6};
      auto y = Matrix<int, std::extents<int, 3, 3>>{1, 2, 3, 4, 5, 6, 7, 8, 9};
      auto xy = Matrix<int, std::extents<int, 5, 3>>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      STATIC_ASSERT(MatrixCompare(MatrixVStack(x, y), xy) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(x, y), xy, std::partial_ordering::equivalent))
      return true;
    }(),
    "VStackExtents full static extents test");
static_assert(
    []() constexpr -> bool {
      using Ext = std::extents<int16_t, std::dynamic_extent, std::dynamic_extent>;
      auto ext1 = Ext{2, 3};
      auto ext2 = Ext{2, 2};
      auto ext3 = Ext{3, 2};
      auto ext4 = Ext{3, 3};
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<decltype(ext1), decltype(ext4)>, Ext>)
      STATIC_ASSERT_THROW(VStackExtents(ext1, ext2);)
      STATIC_ASSERT_THROW(VStackExtents(ext1, ext3);)
      using Mat = Matrix<int, Ext>;
      auto m1 = Mat(std::from_range, {1, 2, 3, 4, 5, 6}, Ext{2, 3});
      auto m2 = Mat(std::from_range, {7, 8, 9, 10, 11, 12, 13, 14, 15}, Ext{3, 3});
      auto expected = Mat(std::from_range, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, Ext{5, 3});
      auto result = MatrixVStack(m1, m2);
      STATIC_ASSERT(std::is_same_v<decltype(result), Mat>)
      STATIC_ASSERT(result.Rows() == expected.Rows())
      STATIC_ASSERT(result.Columns() == expected.Columns())
      STATIC_ASSERT(MatrixCompare(result, expected) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(result, expected, std::partial_ordering::equivalent))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::greater))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::less))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::unordered))

      return true;
    }(),
    "VStackExtents full dynamic extents test");

static_assert(
    []() constexpr -> bool {
      using Ext1 = std::extents<uint16_t, 2, std::dynamic_extent>;
      using Ext2 = std::extents<uint16_t, std::dynamic_extent, 2>;
      using Ext3 = std::extents<uint16_t, 3, std::dynamic_extent>;
      using Ext4 = std::extents<uint16_t, std::dynamic_extent, 3>;

      using Ext11 = std::extents<uint16_t, 4, std::dynamic_extent>;
      using Ext12 = std::extents<uint16_t, std::dynamic_extent, 2>;
      using Ext13 = std::extents<uint16_t, 5, std::dynamic_extent>;
      using Ext14 = std::extents<uint16_t, std::dynamic_extent, 3>;
      using Ext22 = std::extents<uint16_t, std::dynamic_extent, 2>;
      using Ext23 = std::extents<uint16_t, std::dynamic_extent, 2>;
      using Ext33 = std::extents<uint16_t, 6, std::dynamic_extent>;
      using Ext34 = std::extents<uint16_t, std::dynamic_extent, 3>;
      using Ext44 = std::extents<uint16_t, std::dynamic_extent, 3>;
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext1, Ext1>, Ext11>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext1, Ext2>, Ext12>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext1, Ext3>, Ext13>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext1, Ext4>, Ext14>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext2, Ext2>, Ext22>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext2, Ext3>, Ext23>)
      STATIC_ASSERT_FAILURE(VStackExtentsType<Ext2, Ext4>{};)
      STATIC_ASSERT(!VStackExtentsStackable<Ext2, Ext4>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext3, Ext3>, Ext33>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext3, Ext4>, Ext34>)
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<Ext4, Ext4>, Ext44>)
      auto ext1 = Ext1{3};
      auto ext2 = Ext2{2};
      auto ext3 = Ext3{2};
      auto ext4 = Ext4{3};

      auto ext11 = Ext11{3};
      auto ext14 = Ext14{5};
      auto ext22 = Ext22{4};
      auto ext23 = Ext23{5};
      auto ext33 = Ext33{2};
      auto ext44 = Ext44{6};

      STATIC_ASSERT(VStackExtents(ext1, ext1) == ext11)
      STATIC_ASSERT_THROW(VStackExtents(ext1, ext2))
      STATIC_ASSERT_THROW(VStackExtents(ext1, ext3))
      STATIC_ASSERT(VStackExtents(ext1, ext4) == ext14)
      STATIC_ASSERT(VStackExtents(ext2, ext2) == ext22)
      STATIC_ASSERT(VStackExtents(ext2, ext3) == ext23)
      STATIC_ASSERT_FAILURE(VStackExtents(ext2, ext4))
      STATIC_ASSERT(VStackExtents(ext3, ext3) == ext33)
      STATIC_ASSERT_THROW(VStackExtents(ext3, ext4))
      STATIC_ASSERT(VStackExtents(ext4, ext4) == ext44)

      using Mat1 = Matrix<int64_t, Ext1>;
      using Mat2 = Matrix<int64_t, Ext2>;
      using Mat3 = Matrix<int64_t, Ext3>;
      using Mat4 = Matrix<int64_t, Ext4>;

      auto m1 = Mat1(std::from_range, {1, 2, 3, 4, 5, 6}, ext1);
      auto m2 = Mat2(std::from_range, {1, 2, 3, 4}, ext2);
      auto m3 = Mat3(std::from_range, {1, 2, 3, 4, 5, 6}, ext3);
      auto m4 = Mat4(std::from_range, {1, 2, 3, 4, 5, 6, 7, 8, 9}, ext4);

      using Mat11 = Matrix<int64_t, Ext11>;
      using Mat14 = Matrix<int64_t, Ext14>;
      using Mat22 = Matrix<int64_t, Ext22>;
      using Mat23 = Matrix<int64_t, Ext23>;
      using Mat33 = Matrix<int64_t, Ext33>;
      using Mat44 = Matrix<int64_t, Ext44>;

      auto m11 = Mat11(std::from_range, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, ext11);
      auto m14 = Mat14(std::from_range, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ext14);
      auto m22 = Mat22(std::from_range, {1, 2, 3, 4, 1, 2, 3, 4}, ext22);
      auto m23 = Mat23(std::from_range, {1, 2, 3, 4, 1, 2, 3, 4, 5, 6}, ext23);
      auto m33 = Mat33(std::from_range, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, ext33);
      auto m44 = Mat44(std::from_range, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ext44);

      STATIC_ASSERT(MatrixCompare(MatrixVStack(m1, m1), m11, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m1, m4), m14, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m2, m2), m22, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m2, m3), m23, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m3, m3), m33, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m4, m4), m44, std::partial_ordering::equivalent))

      STATIC_ASSERT(MatrixCompare(MatrixVStack(m1, m1), m11) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m1, m4), m14) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m2, m2), m22) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m2, m3), m23) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m3, m3), m33) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixVStack(m4, m4), m44) == std::partial_ordering::equivalent)

      return true;
    }(),
    "VStackExtents mixed extents test");

static_assert(
    []() constexpr -> bool {
      auto ext1 = std::extents<int32_t, 2, 3>{};
      auto ext2 = std::extents<int32_t, 2, 2>{};
      auto ext3 = std::extents<int32_t, 3, 2>{};
      auto ext4 = std::extents<int32_t, 3, 3>{};
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<decltype(ext1), decltype(ext2)>, std::extents<int32_t, 2, 5>>)
      STATIC_ASSERT(!HStackExtentsStackable<decltype(ext1), decltype(ext4)>);
      STATIC_ASSERT(!HStackExtentsStackable<decltype(ext1), decltype(ext3)>);
      STATIC_ASSERT_FAILURE(HStackExtentsType<decltype(ext1), decltype(ext4)>{};);
      STATIC_ASSERT_FAILURE(HStackExtentsType<decltype(ext1), decltype(ext3)>{};);
      STATIC_ASSERT_FAILURE(HStackExtents(ext1, ext4););
      STATIC_ASSERT_FAILURE(HStackExtents(ext1, ext3););
      STATIC_ASSERT(HStackExtents(ext1, ext2) == std::extents<int32_t, 2, 5>{});
      auto x = Matrix<int, std::extents<int, 2, 3>>{1, 2, 3, 4, 5, 6};
      auto y = Matrix<int, std::extents<int, 2, 2>>{1, 2, 3, 4};
      auto xy = Matrix<int, std::extents<int, 2, 5>>{1, 2, 3, 1, 2, 4, 5, 6, 3, 4};
      STATIC_ASSERT(MatrixCompare(MatrixHStack(x, y), xy) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(x, y), xy, std::partial_ordering::equivalent))
      return true;
    }(),
    "HStackExtents full static extents test");
static_assert(
    []() constexpr -> bool {
      using Ext = std::extents<int16_t, std::dynamic_extent, std::dynamic_extent>;
      auto ext1 = Ext{2, 3};
      auto ext2 = Ext{2, 2};
      auto ext3 = Ext{3, 2};
      auto ext4 = Ext{3, 3};
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<decltype(ext1), decltype(ext4)>, Ext>)
      STATIC_ASSERT_THROW(HStackExtents(ext1, ext4);)
      STATIC_ASSERT_THROW(HStackExtents(ext1, ext3);)
      using Mat = Matrix<int, Ext>;
      auto m1 = Mat(std::from_range, {1, 2, 3, 4, 5, 6}, Ext{2, 3});
      auto m2 = Mat(std::from_range, {7, 8, 9, 10}, Ext{2, 2});
      auto expected = Mat(std::from_range, {1, 2, 3, 7, 8, 4, 5, 6, 9, 10}, Ext{2, 5});
      auto result = MatrixHStack(m1, m2);
      STATIC_ASSERT(std::is_same_v<decltype(result), Mat>)
      STATIC_ASSERT(result.Rows() == expected.Rows())
      STATIC_ASSERT(result.Columns() == expected.Columns())
      STATIC_ASSERT(MatrixCompare(result, expected) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(result, expected, std::partial_ordering::equivalent))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::greater))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::less))
      STATIC_ASSERT(!MatrixCompare(result, expected, std::partial_ordering::unordered))

      return true;
    }(),
    "HStackExtents full dynamic extents test");

static_assert(
    []() constexpr -> bool {
      using Ext1 = std::extents<uint16_t, 2, std::dynamic_extent>;
      using Ext2 = std::extents<uint16_t, std::dynamic_extent, 2>;
      using Ext3 = std::extents<uint16_t, 3, std::dynamic_extent>;
      using Ext4 = std::extents<uint16_t, std::dynamic_extent, 3>;

      using Ext11 = std::extents<uint16_t, 2, std::dynamic_extent>;
      using Ext12 = std::extents<uint16_t, 2, std::dynamic_extent>;
      using Ext14 = std::extents<uint16_t, 2, std::dynamic_extent>;
      using Ext22 = std::extents<uint16_t, std::dynamic_extent, 4>;
      using Ext23 = std::extents<uint16_t, 3, std::dynamic_extent>;
      using Ext24 = std::extents<uint16_t, std::dynamic_extent, 5>;
      using Ext33 = std::extents<uint16_t, 3, std::dynamic_extent>;
      using Ext34 = std::extents<uint16_t, 3, std::dynamic_extent>;
      using Ext44 = std::extents<uint16_t, std::dynamic_extent, 6>;
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext1, Ext1>, Ext11>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext1, Ext2>, Ext12>)
      STATIC_ASSERT_FAILURE(HStackExtentsType<Ext1, Ext3>{};)
      STATIC_ASSERT(!HStackExtentsStackable<Ext1, Ext3>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext1, Ext4>, Ext14>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext2, Ext2>, Ext22>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext2, Ext3>, Ext23>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext2, Ext4>, Ext24>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext3, Ext3>, Ext33>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext3, Ext4>, Ext34>)
      STATIC_ASSERT(std::is_same_v<HStackExtentsType<Ext4, Ext4>, Ext44>)

      auto ext1 = Ext1{3};
      auto ext2 = Ext2{2};
      auto ext3 = Ext3{2};
      auto ext4 = Ext4{3};

      auto ext11 = Ext11{6};
      auto ext12 = Ext12{5};
      auto ext22 = Ext22{2};
      auto ext33 = Ext33{4};
      auto ext34 = Ext34{5};
      auto ext44 = Ext44{3};

      STATIC_ASSERT(HStackExtents(ext1, ext1) == ext11)
      STATIC_ASSERT(HStackExtents(ext1, ext2) == ext12)
      STATIC_ASSERT_FAILURE(HStackExtents(ext1, ext3))
      STATIC_ASSERT_THROW(HStackExtents(ext1, ext4))
      STATIC_ASSERT(HStackExtents(ext2, ext2) == ext22)
      STATIC_ASSERT_THROW(HStackExtents(ext2, ext3))
      STATIC_ASSERT_THROW(HStackExtents(ext2, ext4))
      STATIC_ASSERT(HStackExtents(ext3, ext3) == ext33)
      STATIC_ASSERT(HStackExtents(ext3, ext4) == ext34)
      STATIC_ASSERT(HStackExtents(ext4, ext4) == ext44)

      using Mat1 = Matrix<int64_t, Ext1>;
      using Mat2 = Matrix<int64_t, Ext2>;
      using Mat3 = Matrix<int64_t, Ext3>;
      using Mat4 = Matrix<int64_t, Ext4>;

      auto m1 = Mat1(std::from_range, {1, 2, 3, 4, 5, 6}, ext1);
      auto m2 = Mat2(std::from_range, {1, 2, 3, 4}, ext2);
      auto m3 = Mat3(std::from_range, {1, 2, 3, 4, 5, 6}, ext3);
      auto m4 = Mat4(std::from_range, {1, 2, 3, 4, 5, 6, 7, 8, 9}, ext4);

      using Mat11 = Matrix<int64_t, Ext11>;
      using Mat12 = Matrix<int64_t, Ext12>;
      using Mat22 = Matrix<int64_t, Ext22>;
      using Mat33 = Matrix<int64_t, Ext33>;
      using Mat34 = Matrix<int64_t, Ext34>;
      using Mat44 = Matrix<int64_t, Ext44>;

      auto m11 = Mat11(std::from_range, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}, ext11);
      auto m12 = Mat12(std::from_range, {1, 2, 3, 1, 2, 4, 5, 6, 3, 4}, ext12);
      auto m22 = Mat22(std::from_range, {1, 2, 1, 2, 3, 4, 3, 4}, ext22);
      auto m33 = Mat33(std::from_range, {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}, ext33);
      auto m34 = Mat34(std::from_range, {1, 2, 1, 2, 3, 3, 4, 4, 5, 6, 5, 6, 7, 8, 9}, ext34);
      auto m44 = Mat44(std::from_range, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9}, ext44);

      STATIC_ASSERT(MatrixCompare(MatrixHStack(m1, m1), m11, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m1, m2), m12, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m2, m2), m22, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m3, m3), m33, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m3, m4), m34, std::partial_ordering::equivalent))
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m4, m4), m44, std::partial_ordering::equivalent))

      STATIC_ASSERT(MatrixCompare(MatrixHStack(m1, m1), m11) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m1, m2), m12) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m2, m2), m22) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m3, m3), m33) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m3, m4), m34) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(MatrixHStack(m4, m4), m44) == std::partial_ordering::equivalent)

      return true;
    }(),
    "HStackExtents mixed extents test");
}  // namespace matrix_stack_tests

namespace matrix_arithmetic_tests {

using matrix_implementation_detail::MatrixExtentsStaticCompatible;
static_assert(
    []() -> bool {
      using Ext1 = std::extents<size_t, 2, 3>;
      using Ext2 = std::extents<size_t, 2, 2>;
      using Ext3 = std::extents<size_t, 3, 2>;
      using Ext4 = std::extents<size_t, 3, 3>;

      using Mat1 = Matrix<int32_t, Ext1>;
      using Mat2 = Matrix<int32_t, Ext2>;
      using Mat3 = Matrix<int32_t, Ext3>;
      using Mat4 = Matrix<int32_t, Ext4>;

      auto a1 = Mat1{1, 2, 3, 4, 5, 6};
      auto a2 = Mat2{1, 2, 3, 4};
      auto a3 = Mat3{1, 2, 3, 4, 5, 6};
      auto a4 = Mat4{1, 2, 3, 4, 5, 6, 7, 8, 9};

      const auto b1 = Mat1{3, 2, 1, 3, 2, 1};
      const auto b2 = Mat2{2, 1, 2, 1};
      const auto b3 = Mat3{2, 1, 2, 1, 2, 1};
      const auto b4 = Mat4{3, 2, 1, 3, 2, 1, 3, 2, 1};

      const auto c1 = Mat1{4, 4, 4, 7, 7, 7};
      const auto c2 = Mat2{3, 3, 5, 5};
      const auto c3 = Mat3{3, 3, 5, 5, 7, 7};
      const auto c4 = Mat4{4, 4, 4, 7, 7, 7, 10, 10, 10};

      STATIC_ASSERT_FAILURE(a1 += b2)
      STATIC_ASSERT_FAILURE(a1 += b3)
      STATIC_ASSERT_FAILURE(a1 += b2)
      STATIC_ASSERT_FAILURE(a2 += b1)
      STATIC_ASSERT_FAILURE(a2 += b3)
      STATIC_ASSERT_FAILURE(a2 += b4)
      STATIC_ASSERT_FAILURE(a3 += b1)
      STATIC_ASSERT_FAILURE(a3 += b2)
      STATIC_ASSERT_FAILURE(a3 += b4)
      STATIC_ASSERT_FAILURE(a4 += b1)
      STATIC_ASSERT_FAILURE(a4 += b2)
      STATIC_ASSERT_FAILURE(a4 += b3)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext3>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext3>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext3>)

      STATIC_ASSERT(MatrixCompare(a1 += b1, c1) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a2 += b2, c2) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a3 += b3, c3) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a4 += b4, c4) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a1 += b1, c1) == std::partial_ordering::greater)
      STATIC_ASSERT(MatrixCompare(a2 += b2, c2) == std::partial_ordering::greater)
      STATIC_ASSERT(MatrixCompare(a3 += b3, c3) == std::partial_ordering::greater)
      STATIC_ASSERT(MatrixCompare(a4 += b4, c4) == std::partial_ordering::greater)

      return true;
    }(),
    "matrix static compound add test");
static_assert(
    []() -> bool {
      using Ext1 = std::extents<size_t, 2, 3>;
      using Ext2 = std::extents<size_t, 2, 2>;
      using Ext3 = std::extents<size_t, 3, 2>;
      using Ext4 = std::extents<size_t, 3, 3>;

      using Mat1 = Matrix<int32_t, Ext1>;
      using Mat2 = Matrix<int32_t, Ext2>;
      using Mat3 = Matrix<int32_t, Ext3>;
      using Mat4 = Matrix<int32_t, Ext4>;

      auto a1 = Mat1{1, 2, 3, 4, 5, 6};
      auto a2 = Mat2{1, 2, 3, 4};
      auto a3 = Mat3{1, 2, 3, 4, 5, 6};
      auto a4 = Mat4{1, 2, 3, 4, 5, 6, 7, 8, 9};

      const auto b1 = Mat1{3, 2, 1, 3, 2, 1};
      const auto b2 = Mat2{2, 1, 2, 1};
      const auto b3 = Mat3{2, 1, 2, 1, 2, 1};
      const auto b4 = Mat4{3, 2, 1, 3, 2, 1, 3, 2, 1};

      const auto c1 = Mat1{-2, 0, 2, 1, 3, 5};
      const auto c2 = Mat2{-1, 1, 1, 3};
      const auto c3 = Mat3{-1, 1, 1, 3, 3, 5};
      const auto c4 = Mat4{-2, 0, 2, 1, 3, 5, 4, 6, 8};

      STATIC_ASSERT_FAILURE(a1 -= b2)
      STATIC_ASSERT_FAILURE(a1 -= b3)
      STATIC_ASSERT_FAILURE(a1 -= b2)
      STATIC_ASSERT_FAILURE(a2 -= b1)
      STATIC_ASSERT_FAILURE(a2 -= b3)
      STATIC_ASSERT_FAILURE(a2 -= b4)
      STATIC_ASSERT_FAILURE(a3 -= b1)
      STATIC_ASSERT_FAILURE(a3 -= b2)
      STATIC_ASSERT_FAILURE(a3 -= b4)
      STATIC_ASSERT_FAILURE(a4 -= b1)
      STATIC_ASSERT_FAILURE(a4 -= b2)
      STATIC_ASSERT_FAILURE(a4 -= b3)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext3>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext1, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext3>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext2, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext3, Ext4>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext1>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext2>)
      STATIC_ASSERT(!MatrixExtentsStaticCompatible<Ext4, Ext3>)

      STATIC_ASSERT(MatrixCompare(a1 -= b1, c1) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a2 -= b2, c2) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a3 -= b3, c3) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a4 -= b4, c4) == std::partial_ordering::equivalent)
      STATIC_ASSERT(MatrixCompare(a1 -= b1, c1) == std::partial_ordering::less)
      STATIC_ASSERT(MatrixCompare(a2 -= b2, c2) == std::partial_ordering::less)
      STATIC_ASSERT(MatrixCompare(a3 -= b3, c3) == std::partial_ordering::less)
      STATIC_ASSERT(MatrixCompare(a4 -= b4, c4) == std::partial_ordering::less)

      return true;
    }(),
    "matrix static compound sub test");
}  // namespace matrix_arithmetic_tests

int main() {};