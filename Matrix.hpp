#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <mdspan>
#include <new>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace matrix_implementation_detail {

template <typename T, typename... Candidates>
concept OneOf = (std::same_as<T, Candidates> || ...);

template <typename T, typename Expected>
concept SomeConst = OneOf<T, const Expected &, Expected>;

template <class T>
concept ConstMatrixLike = requires(const T &cref) {
  typename T::NumericType;
  typename T::IndexType;
  typename T::ExtentsType;
  { cref[0, 0] } -> SomeConst<typename T::NumericType>;
  { cref.At(0, 0) } -> SomeConst<typename T::NumericType>;
  { cref.Rows() } -> std::same_as<typename T::IndexType>;
  { cref.Columns() } -> std::same_as<typename T::IndexType>;
  { cref.GetExtents() } -> std::same_as<typename T::ExtentsType>;
};
template <class T>
concept MutableMatrixLike = requires(T &ref) {
  typename T::NumericType;
  typename T::IndexType;
  typename T::ExtentsType;
  { ref[0, 0] } -> std::same_as<typename T::NumericType &>;
  { ref.At(0, 0) } -> std::same_as<typename T::NumericType &>;
  { std::as_const(ref).Rows() } -> std::same_as<typename T::IndexType>;
  { std::as_const(ref).Columns() } -> std::same_as<typename T::IndexType>;
  { std::as_const(ref).GetExtents() } -> std::same_as<typename T::ExtentsType>;
};
template <class T>
concept MovableMatrixLike = requires(T &&mref) {
  typename T::NumericType;
  typename T::IndexType;
  typename T::ExtentsType;
  { std::move(mref)[0, 0] } -> std::same_as<typename T::NumericType &&>;
  { std::move(mref).At(0, 0) } -> std::same_as<typename T::NumericType &&>;
  { std::as_const(mref).Rows() } -> std::same_as<typename T::IndexType>;
  { std::as_const(mref).Columns() } -> std::same_as<typename T::IndexType>;
  { std::as_const(mref).GetExtents() } -> std::same_as<typename T::ExtentsType>;
};

struct VerticalStackTagType {};
struct HorizontalStackTagType {};

template <typename Extents1, typename Extents2>
static inline constexpr auto VStackExtents(Extents1 ext1, Extents2 ext2) {
  using IndexType = typename Extents1::index_type;
  if constexpr (Extents1::static_extent(1) != std::dynamic_extent &&
                Extents2::static_extent(1) != std::dynamic_extent &&
                Extents1::static_extent(1) != Extents2::static_extent(1)) {
    static_assert(false, "the matrices have a different number of columns and cannot be stacked");
  } else {
    static constexpr size_t kRowExtent =
        (Extents1::static_extent(0) == std::dynamic_extent || Extents2::static_extent(0) == std::dynamic_extent)
            ? std::dynamic_extent
            : Extents1::static_extent(0) + Extents2::static_extent(0);
    if constexpr (Extents1::static_extent(1) == std::dynamic_extent ||
                  Extents2::static_extent(1) == std::dynamic_extent) {
      if (ext1.extent(1) != ext2.extent(1)) {
        throw std::invalid_argument("the matrices have a different number of columns and cannot be stacked");
      }
    }
    using ResultType = std::extents<IndexType, kRowExtent, Extents1::static_extent(1)>;
    static constexpr bool kIsResultStaticRows = (ResultType::static_extent(0) != std::dynamic_extent);
    static constexpr bool kIsResultStaticColumns = (ResultType::static_extent(1) != std::dynamic_extent);
    if constexpr (kIsResultStaticRows && kIsResultStaticColumns) {
      return ResultType{};
    } else if constexpr (kIsResultStaticColumns) {
      return ResultType{ext1.extent(0) + ext2.extent(0)};
    } else if constexpr (kIsResultStaticRows) {
      return ResultType{ext1.extent(1)};
    } else {
      return ResultType{ext1.extent(0) + ext2.extent(0), ext1.extent(1)};
    }
  }
}

template <typename Extents1, typename Extents2>
using VStackExtentsType = decltype(VStackExtents<Extents1, Extents2>(Extents1{}, Extents2{}));

template <typename Numeric, typename Extents>
  requires(Extents::rank() == 2 &&
           (std::is_constructible_v<std::mdspan<Numeric, Extents>, Numeric *> ||
            std::is_constructible_v<std::mdspan<Numeric, Extents>, Numeric *, int> ||
            std::is_constructible_v<std::mdspan<Numeric, Extents>, Numeric *, int, int>) &&
           std::is_same_v<Numeric, std::remove_cvref_t<Numeric>>)
class Matrix {
 private:
  static inline constexpr bool kIsStatic = (Extents::rank_dynamic() == 0);
  static inline constexpr size_t kTotalSize =
      (!kIsStatic) ? 0 : (Extents::static_extent(0) * Extents::static_extent(1));

 public:
  inline constexpr Matrix() noexcept : storage_{}, extents_{} {
  }
  template <typename... Args>
    requires(kIsStatic && (std::is_convertible_v<Args, Numeric> && ...) && sizeof...(Args) == kTotalSize)
  inline constexpr explicit Matrix(Args &&...args) noexcept : storage_{std::forward<Args>(args)...}, extents_{} {
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr explicit Matrix(const R &init)
    requires(kIsStatic)
      : storage_{}, extents_{} {
    if (init.size() != kTotalSize) {
      throw std::invalid_argument{"single range constructor: the range must match the size"};
    }
    std::ranges::copy(init.begin(), init.end(), storage_.begin());
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr Matrix(const R &range, Extents ext)
    requires(!kIsStatic)
      : storage_{}, extents_{ext} {
    if (range.size() != TotalSize(ext)) {
      throw std::invalid_argument{"range + extents copy constructor: the range must match the size"};
    }
    storage_.reserve(TotalSize(ext));
    std::ranges::copy(range, std::back_inserter(storage_));
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr explicit Matrix(R &&init)
    requires(kIsStatic)
      : storage_{}, extents_{} {
    if (init.size() != kTotalSize) {
      throw std::invalid_argument{"range + extents move constructor: the range must match the size"};
    }
    storage_.reserve(kTotalSize);
    std::ranges::move(init, storage_.begin());
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr Matrix(R &&range, Extents ext)
    requires(!kIsStatic)
      : storage_{}, extents_{ext} {
    if (range.size() != TotalSize(ext)) {
      throw std::invalid_argument{"range + extents move constructor: the range must match the size"};
    }
    storage_.reserve(TotalSize(ext));
    std::ranges::move(range, std::back_inserter(storage_));
  }
  inline constexpr Matrix(std::vector<Numeric> &&vec, Extents ext)
    requires(!kIsStatic)
      : storage_{}, extents_{ext} {
    if (vec.size() != TotalSize(ext)) {
      throw std::invalid_argument{"range + extents move constructor: the range must match the size"};
    }
    storage_ = std::move(vec);
  }
  inline constexpr explicit Matrix(Extents ext, Numeric value = Numeric{})
    requires(!kIsStatic)
      : storage_{}, extents_{ext} {
    storage_.resize(TotalSize(ext), value);
  }
  inline constexpr explicit Matrix(Numeric value = Numeric{})
    requires(kIsStatic)
      : storage_{}, extents_{} {
    storage_.fill(value);
  }
  inline constexpr Matrix(const Matrix &) = default;
  inline constexpr Matrix(Matrix &&) noexcept = default;
  inline constexpr Matrix &operator=(const Matrix &) = default;
  inline constexpr Matrix &operator=(Matrix &&) noexcept = default;
  inline constexpr ~Matrix() noexcept = default;

  using IndexType = typename Extents::index_type;
  using NumericType = Numeric;
  using ExtentsType = Extents;

  inline constexpr Numeric &operator[](IndexType row, IndexType column) & noexcept {
    return GetMDSpan()[row, column];
  }
  inline constexpr Numeric &&operator[](IndexType row, IndexType column) && noexcept {
    return std::move(GetMDSpan()[row, column]);
  }
  inline constexpr const Numeric &operator[](IndexType row, IndexType column) const & noexcept {
    return GetMDSpan()[row, column];
  }
  inline constexpr Numeric &At(IndexType row, IndexType column) & {
    return GetMDSpan().at(row, column);
  }
  inline constexpr Numeric &&At(IndexType row, IndexType column) && {
    return std::move(GetMDSpan().at(row, column));
  }
  inline constexpr const Numeric &At(IndexType row, IndexType column) const & {
    return GetMDSpan().at(row, column);
  }
  inline constexpr IndexType Rows() const noexcept {
    return extents_.extent(0);
  }
  inline constexpr IndexType Columns() const noexcept {
    return extents_.extent(1);
  }
  inline constexpr Extents GetExtents() const noexcept {
    return extents_;
  }

  template <ConstMatrixLike View>
  inline constexpr explicit Matrix(const View &view) : storage_{}, extents_{view.GetExtents()} {
    const auto total = TotalSize(extents_);
    if (TotalSize(extents_) == 0) {
      return;
    }
    if constexpr (!kIsStatic) {
      storage_.resize(total, view[0, 0]);
    }
    static constexpr size_t kCacheLineSize =
        std::max(2 * std::hardware_destructive_interference_size / sizeof(Numeric), 1uz);
    const auto rows = Rows();
    const auto cols = Columns();
    for (size_t row_maj = 0; row_maj < rows; row_maj += kCacheLineSize) {
      for (size_t col_maj = 0; col_maj < cols; col_maj += kCacheLineSize) {
        const auto row_end = std::min(row_maj + kCacheLineSize, rows);
        const auto col_end = std::min(col_maj + kCacheLineSize, cols);
        for (size_t row = row_maj; row < row_end; row++) {
          for (size_t col = col_maj; col < col_end; col++) {
            (*this)[row, col] = view[row, col];
          }
        }
      }
    }
  }

  template <typename Extents1, typename Extents2>
    requires(std::is_same_v<Extents, VStackExtentsType<Extents1, Extents2>>)
  inline constexpr Matrix(VerticalStackTagType, const Matrix<Numeric, Extents1> &matrix1,
                          const Matrix<Numeric, Extents2> &matrix2)
      : storage_{}, extents_(VStackExtents(matrix1.GetExtents(), matrix2.GetExtents())) {
    const auto total = TotalSize(extents_);
    if (total == 0) {
      return;
    }
    if constexpr (!kIsStatic) {
      storage_.resize(total, Numeric{});
    }
    const auto cols = matrix1.Columns();
    const auto rows1 = matrix1.Rows();
    const auto rows2 = matrix2.Rows();
    for (size_t row = 0; row < rows1; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)[row, col] = matrix1[row, col];
      }
    }
    for (size_t row = 0; row < rows2; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)[rows1 + row, col] = matrix2[row, col];
      }
    }
  }

 private:
  using Storage = std::conditional_t<kIsStatic, std::array<Numeric, kTotalSize>, std::vector<Numeric>>;
  Storage storage_;
  [[no_unique_address]] Extents extents_;

  inline constexpr auto GetMDSpan() & noexcept {
    if constexpr (kIsStatic) {
      return std::mdspan<Numeric, Extents>{storage_.data()};
    } else {
      return std::mdspan<Numeric, Extents>{storage_.data(), extents_};
    }
  }
  inline constexpr auto GetMDSpan() && noexcept {
    if constexpr (kIsStatic) {
      return std::mdspan<Numeric, Extents>{storage_.data()};
    } else {
      return std::mdspan<Numeric, Extents>{storage_.data(), extents_};
    }
  }
  inline constexpr auto GetMDSpan() const & noexcept {
    if constexpr (kIsStatic) {
      return std::mdspan<const Numeric, Extents>{storage_.data()};
    } else {
      return std::mdspan<const Numeric, Extents>{storage_.data(), extents_};
    }
  }

  static constexpr inline auto TotalSize(const Extents &ext) noexcept {
    static_assert(ConstMatrixLike<Matrix<Numeric, Extents>>, "base matrix missing something important");
    static_assert(MutableMatrixLike<Matrix<Numeric, Extents>>, "base matrix missing something important");
    static_assert(MovableMatrixLike<Matrix<Numeric, Extents>>, "base matrix missing something important");
    return ext.extent(0) * ext.extent(1);
  }
};

template <ConstMatrixLike T>
using TransposedExtentsType =
    std::extents<typename T::IndexType, T::ExtentsType::static_extent(1), T::ExtentsType::static_extent(0)>;

template <typename T>
struct TransposedMatrixView;

template <ConstMatrixLike T>
struct TransposedMatrixView<T> {
  using IndexType = typename T::IndexType;
  using NumericType = typename T::NumericType;
  using ExtentsType = TransposedExtentsType<T>;

  static inline constexpr bool kIsStatic = (ExtentsType::rank_dynamic() == 0);
  inline static constexpr bool kIsStaticRows = (ExtentsType::static_extent(0) != std::dynamic_extent);
  inline static constexpr bool kIsStaticColumns = (ExtentsType::static_extent(1) != std::dynamic_extent);

  inline constexpr explicit TransposedMatrixView(const T &matrix) : ref_(matrix) {
  }
  inline constexpr TransposedMatrixView(const TransposedMatrixView &) noexcept = default;
  inline constexpr TransposedMatrixView(TransposedMatrixView &&) noexcept = default;
  inline constexpr void operator=(const TransposedMatrixView &) noexcept = delete;
  inline constexpr void operator=(TransposedMatrixView &&) noexcept = delete;
  inline constexpr void operator=(const TransposedMatrixView &) const noexcept = delete;
  inline constexpr void operator=(TransposedMatrixView &&) const noexcept = delete;
  inline constexpr ~TransposedMatrixView() noexcept = default;

  inline constexpr const T &Original() const noexcept {
    return ref_;
  }
  inline constexpr const NumericType &operator[](IndexType row, IndexType column) const noexcept {
    return ref_[column, row];
  }
  inline constexpr const NumericType &At(IndexType row, IndexType column) const {
    return ref_.At(column, row);
  }
  inline constexpr IndexType Rows() const noexcept {
    return ref_.Columns();
  }
  inline constexpr IndexType Columns() const noexcept {
    return ref_.Rows();
  }
  inline constexpr ExtentsType GetExtents() const noexcept {
    if constexpr (kIsStatic) {
      return ExtentsType{};
    } else if constexpr (kIsStaticRows) {
      return ExtentsType{ref_.Rows()};
    } else if constexpr (kIsStaticColumns) {
      return ExtentsType{ref_.Columns()};
    } else {
      return ExtentsType{ref_.Columns(), ref_.Rows()};
    }
    static_assert(ConstMatrixLike<TransposedMatrixView<T>>, "transposed view is missing something important");
  }

 private:
  const T &ref_;
};

template <ConstMatrixLike T>
using TransposedMatrixType = Matrix<typename T::NumericType, TransposedExtentsType<T>>;

template <ConstMatrixLike View>
auto MatrixTView(const View &view) {
  return TransposedMatrixView<View>{view};
}
template <ConstMatrixLike View>
auto MatrixTCopy(const View &view) {
  return TransposedMatrixType<View>{MatrixTView(view)};
}

template <typename Numeric, typename Extents1, typename Extents2>
inline constexpr auto MatrixVStack(const Matrix<Numeric, Extents1> &matrix1, const Matrix<Numeric, Extents2> &matrix2)
    -> Matrix<Numeric, VStackExtentsType<Extents1, Extents2>> {
  return Matrix<Numeric, VStackExtentsType<Extents1, Extents2>>(VerticalStackTagType{}, matrix1, matrix2);
}
}  // namespace matrix_implementation_detail
using matrix_implementation_detail::Matrix;
using matrix_implementation_detail::MatrixTCopy;
using matrix_implementation_detail::MatrixVStack;