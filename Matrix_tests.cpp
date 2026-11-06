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
#define STATIC_ASSERT_THROW(...) \
  do {                           \
  } while (false);

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
namespace {
using matrix_implementation_detail::VStackExtents;
using matrix_implementation_detail::VStackExtentsStackable;
using matrix_implementation_detail::VStackExtentsType;

static_assert(
    []() constexpr -> bool {
      auto ext1 = std::extents<int32_t, 2, 3>{};
      auto ext2 = std::extents<int32_t, 2, 2>{};
      auto ext3 = std::extents<int32_t, 3, 2>{};
      auto ext4 = std::extents<int32_t, 3, 3>{};
      auto x = Matrix<int, std::extents<int, 2, 3>>{1, 2, 3, 4, 5, 6};
      STATIC_ASSERT(std::is_same_v<VStackExtentsType<decltype(ext1), decltype(ext4)>, std::extents<int32_t, 5, 3>>)
      STATIC_ASSERT(!VStackExtentsStackable<decltype(ext1), decltype(ext2)>);
      STATIC_ASSERT(!VStackExtentsStackable<decltype(ext1), decltype(ext3)>);
      STATIC_ASSERT_FAILURE(VStackExtentsType<decltype(ext1), decltype(ext2)>{};);
      STATIC_ASSERT_FAILURE(VStackExtentsType<decltype(ext1), decltype(ext3)>{};);
      STATIC_ASSERT_FAILURE(VStackExtents(ext1, ext2););
      STATIC_ASSERT_FAILURE(VStackExtents(ext1, ext3););
      STATIC_ASSERT(VStackExtents(ext1, ext4) == std::extents<int32_t, 5, 3>{});
      //   STATIC_ASSERT(MatrixVStack(Matrix{{1, 2, 3, 4, 5, 6}, ext1}, Matrix({1, 2, 3, 4, 5, 6, 7, 8, 9}, ext3)) ==
      //                 Matrix{{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9}, std::extents<int32_t, 5, 3>{}})
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
}  // namespace

int main() {};