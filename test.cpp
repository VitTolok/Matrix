#include <cstddef>
#include <cstdint>
#include <print>
#include <span>
#include <ranges>
#include <meta>
#include "Matrix.hpp"

int main() {
  Matrix<int64_t, std::extents<size_t, 2, 3>> m{1};

  std::vector<int64_t> v{1, 2, 3, 4, 5, 6};
  std::extents<size_t, 2, std::dynamic_extent> ext{2, 3};
  Matrix<int64_t, std::extents<size_t, 2, std::dynamic_extent>> m2{std::from_range, std::move(v), ext};

  m[0, 1] = 0;
  auto m3 = MatrixTView(m);
  auto m4 = MatrixTView(m2);

  auto t = MatrixVStack(m4, m4, m4, m4);
  // std::println(std::define_static_string(std::meta::display_string_of(^^decltype(t))));
  for (size_t row = 0; row < t.Rows(); row++) {
    for (size_t col = 0; col < t.Columns(); col++) {
      std::print("{} ", t[row, col]);
    }
    std::println();
  }
  // std::fflush(stdout);

  // m.GetMDSpan();
  // std::as_const(m).GetMDSpan();
}