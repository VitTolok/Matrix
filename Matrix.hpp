#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <mdspan>
#include <new>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace matrix_implementation_detail {

template <typename T, typename... Candidates>
concept OneOf = (std::same_as<T, Candidates> || ...);

template <typename T, typename Expected>
concept SomeConst = OneOf<T, const Expected &, Expected>;

template <typename T>
concept Extents = requires(T t) {
  typename T::index_type;
  { T::static_extent(0) } -> std::same_as<size_t>;
  std::index_sequence<T::static_extent(0)>{};
} && std::same_as<T, std::extents<typename T::index_type, T::static_extent(0), T::static_extent(1)>>;

template <class T>
concept BaseMatrixLike =
    requires(const std::remove_cvref_t<T> &cref) {
      typename std::remove_cvref_t<T>::NumericType;
      typename std::remove_cvref_t<T>::IndexType;
      typename std::remove_cvref_t<T>::ExtentsType;
      { cref.Rows() } -> std::same_as<typename std::remove_cvref_t<T>::IndexType>;
      { cref.Columns() } -> std::same_as<typename std::remove_cvref_t<T>::IndexType>;
      { cref.GetExtents() } -> std::same_as<typename std::remove_cvref_t<T>::ExtentsType>;
    } && Extents<typename std::remove_cvref_t<T>::ExtentsType> &&
    std::same_as<typename std::remove_cvref_t<T>::IndexType, typename std::remove_cvref_t<T>::ExtentsType::index_type>;

template <class T>
concept ConstMatrixLike = BaseMatrixLike<T> && requires(const T &cref) {
  { cref[0, 0] } -> SomeConst<typename std::remove_cvref_t<T>::NumericType>;
  { cref.At(0, 0) } -> SomeConst<typename std::remove_cvref_t<T>::NumericType>;
};
template <class T>
concept MutableMatrixLike = BaseMatrixLike<T> && requires(T &ref) {
  { ref[0, 0] } -> std::same_as<typename std::remove_cvref_t<T>::NumericType &>;
  { ref.At(0, 0) } -> std::same_as<typename std::remove_cvref_t<T>::NumericType &>;
};
template <class T>
concept MovableMatrixLike = BaseMatrixLike<T> && requires(T &&mref) {
  { std::move(mref)[0, 0] } -> std::same_as<typename std::remove_cvref_t<T>::NumericType &&>;
  { std::move(mref).At(0, 0) } -> std::same_as<typename std::remove_cvref_t<T>::NumericType &&>;
};

template <BaseMatrixLike M>
using MatNumericType = typename std::remove_cvref_t<M>::NumericType;
template <BaseMatrixLike M>
using MatExtentsType = typename std::remove_cvref_t<M>::ExtentsType;
template <BaseMatrixLike M>
using MatIndexType = typename std::remove_cvref_t<M>::IndexType;

template <typename Extents1, typename Extents2>
concept MatrixExtentsStaticCompatible =
    ((Extents1::static_extent(0) == std::dynamic_extent || Extents2::static_extent(0) == std::dynamic_extent ||
      Extents1::static_extent(0) == Extents2::static_extent(0)) &&
     (Extents1::static_extent(1) == std::dynamic_extent || Extents2::static_extent(1) == std::dynamic_extent ||
      Extents1::static_extent(1) == Extents2::static_extent(1)) &&
     requires { std::common_type_t<typename Extents1::index_type, typename Extents2::index_type>{}; });

template <typename Extents1, typename Extents2>
  requires(MatrixExtentsStaticCompatible<Extents1, Extents2>)
using MatrixCommonExtents =
    std::extents<std::common_type_t<typename Extents1::index_type, typename Extents2::index_type>,
                 std::min(Extents1::static_extent(0), Extents2::static_extent(0)),
                 std::min(Extents1::static_extent(1), Extents2::static_extent(1))>;

template <typename Extents1, typename Extents2>
  requires(MatrixExtentsStaticCompatible<Extents1, Extents2>)
inline constexpr void EnforceSameExtents(Extents1 ext1, Extents2 ext2) {
  if (ext1 != ext2) {
    throw std::invalid_argument("the extents of the provided matrices are incompatible");
  }
}

template <BaseMatrixLike Mat1, BaseMatrixLike Mat2, typename Callable>
inline constexpr bool CacheLineWaltz(Mat1 &&a, Mat2 &&b, Callable &&c) noexcept(noexcept(c(a[0, 0], b[0, 0])))
  requires MatrixExtentsStaticCompatible<MatExtentsType<Mat2>, MatExtentsType<Mat2>> &&
           std::invocable<Callable, decltype(std::forward<Mat1>(a)[0, 0]), decltype(std::forward<Mat2>(b)[0, 0])>
{
  if (a.GetExtents() != b.GetExtents()) {
    std::unreachable();
  }
  static constexpr size_t kCacheLineSize1 =
      std::max(2 * std::hardware_destructive_interference_size / sizeof(MatNumericType<Mat1>), 1uz);
  static constexpr size_t kCacheLineSize2 =
      std::max(2 * std::hardware_destructive_interference_size / sizeof(MatNumericType<Mat2>), 1uz);
  static constexpr size_t kCacheLineSize = std::max(kCacheLineSize1, kCacheLineSize2);
  const auto rows = a.Rows();
  const auto cols = a.Columns();
  for (size_t row_maj = 0; row_maj < rows; row_maj += kCacheLineSize) {
    for (size_t col_maj = 0; col_maj < cols; col_maj += kCacheLineSize) {
      const auto row_end = std::min(row_maj + kCacheLineSize, static_cast<size_t>(rows));
      const auto col_end = std::min(col_maj + kCacheLineSize, static_cast<size_t>(cols));
      for (size_t row = row_maj; row < row_end; row++) {
        for (size_t col = col_maj; col < col_end; col++) {
          if constexpr (std::is_same_v<decltype(c(std::forward<Mat1>(a)[row, col], std::forward<Mat2>(b)[row, col])),
                                       bool>) {
            if (!c(std::forward<Mat1>(a)[row, col], std::forward<Mat2>(b)[row, col])) {
              return false;
            }
          } else {
            c(std::forward<Mat1>(a)[row, col], std::forward<Mat2>(b)[row, col]);
          }
        }
      }
    }
  }
  return true;
}

struct VerticalStackTagType {};
struct HorizontalStackTagType {};

template <Extents Extents1, Extents Extents2>
using VStackExtentsReturnType =
    std::extents<std::common_type_t<typename Extents1::index_type, typename Extents2::index_type>,
                 ((Extents1::static_extent(0) == std::dynamic_extent ||
                   Extents2::static_extent(0) == std::dynamic_extent)
                      ? std::dynamic_extent
                      : Extents1::static_extent(0) + Extents2::static_extent(0)),
                 std::min(Extents1::static_extent(1), Extents2::static_extent(1))>;

template <Extents Extents1, Extents Extents2>
static inline constexpr VStackExtentsReturnType<Extents1, Extents2> VStackExtents(Extents1 ext1, Extents2 ext2)
  requires(!(Extents1::static_extent(1) != std::dynamic_extent && Extents2::static_extent(1) != std::dynamic_extent &&
             Extents1::static_extent(1) != Extents2::static_extent(1)))
{
  if constexpr (Extents1::static_extent(1) == std::dynamic_extent ||
                Extents2::static_extent(1) == std::dynamic_extent) {
    if (ext1.extent(1) != ext2.extent(1)) {
      throw std::invalid_argument("the matrices have a different number of columns and cannot be stacked");
    }
  }
  using ResultType = VStackExtentsReturnType<Extents1, Extents2>;
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

template <Extents Extents1, Extents Extents2>
using HStackExtentsReturnType =
    std::extents<std::common_type_t<typename Extents1::index_type, typename Extents2::index_type>,
                 std::min(Extents1::static_extent(0), Extents2::static_extent(0)),
                 ((Extents1::static_extent(1) == std::dynamic_extent ||
                   Extents2::static_extent(1) == std::dynamic_extent)
                      ? std::dynamic_extent
                      : Extents1::static_extent(1) + Extents2::static_extent(1))>;

template <Extents Extents1, Extents Extents2>
static inline constexpr HStackExtentsReturnType<Extents1, Extents2> HStackExtents(Extents1 ext1, Extents2 ext2)
  requires(!(Extents1::static_extent(0) != std::dynamic_extent && Extents2::static_extent(0) != std::dynamic_extent &&
             Extents1::static_extent(0) != Extents2::static_extent(0)))
{
  if constexpr (Extents1::static_extent(0) == std::dynamic_extent ||
                Extents2::static_extent(0) == std::dynamic_extent) {
    if (ext1.extent(0) != ext2.extent(0)) {
      throw std::invalid_argument("the matrices have a different number of rows and cannot be stacked");
    }
  }
  using ResultType = HStackExtentsReturnType<Extents1, Extents2>;
  static constexpr bool kIsResultStaticRows = (ResultType::static_extent(0) != std::dynamic_extent);
  static constexpr bool kIsResultStaticColumns = (ResultType::static_extent(1) != std::dynamic_extent);
  if constexpr (kIsResultStaticRows && kIsResultStaticColumns) {
    return ResultType{};
  } else if constexpr (kIsResultStaticColumns) {
    return ResultType{ext1.extent(0)};
  } else if constexpr (kIsResultStaticRows) {
    return ResultType{ext1.extent(1) + ext2.extent(1)};
  } else {
    return ResultType{ext1.extent(0), ext1.extent(1) + ext2.extent(1)};
  }
}
template <typename Extents1, typename Extents2>
concept VStackExtentsStackable =
    Extents<Extents1> && Extents<Extents2> && requires() { VStackExtents(Extents1{}, Extents2{}); };
template <Extents Extents1, Extents Extents2>
  requires VStackExtentsStackable<Extents1, Extents2>
using VStackExtentsType = VStackExtentsReturnType<Extents1, Extents2>;

template <typename Extents1, typename Extents2>
concept HStackExtentsStackable =
    Extents<Extents1> && Extents<Extents2> && requires() { HStackExtents(Extents1{}, Extents2{}); };
template <Extents Extents1, Extents Extents2>
  requires HStackExtentsStackable<Extents1, Extents2>
using HStackExtentsType = HStackExtentsReturnType<Extents1, Extents2>;

template <typename Numeric, Extents Extents>
  requires(std::is_same_v<Numeric, std::remove_cvref_t<Numeric>>)
class Matrix {
 private:
  static inline constexpr bool kIsStatic = (Extents::rank_dynamic() == 0);
  static inline constexpr size_t kTotalSize =
      (!kIsStatic) ? 0 : (Extents::static_extent(0) * Extents::static_extent(1));

 public:
  inline constexpr Matrix() noexcept : storage_{}, extents_{} {
    if constexpr (!kIsStatic) {
      storage_.resize(TotalSize(extents_), Numeric{});
    }
  }
  template <typename... Args>
    requires(kIsStatic && (std::is_convertible_v<Args, Numeric> && ...) && sizeof...(Args) == kTotalSize)
  inline constexpr explicit Matrix(Args &&...args) noexcept : storage_{std::forward<Args>(args)...}, extents_{} {
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr explicit Matrix(std::from_range_t, const R &init)
    requires(kIsStatic)
      : storage_{}, extents_{} {
    if (init.size() != kTotalSize) {
      throw std::invalid_argument{"single range constructor: the range must match the size"};
    }
    std::ranges::copy(init.begin(), init.end(), storage_.begin());
  }
  template <std::ranges::sized_range R>
    requires std::is_same_v<std::remove_cvref_t<std::ranges::range_value_t<R>>, Numeric>
  inline constexpr Matrix(std::from_range_t, const R &range, Extents ext)
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
  inline constexpr explicit Matrix(std::from_range_t, R &&init)
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
  inline constexpr Matrix(std::from_range_t, R &&range, Extents ext)
    requires(!kIsStatic)
      : storage_{}, extents_{ext} {
    if (range.size() != TotalSize(ext)) {
      throw std::invalid_argument{"range + extents move constructor: the range must match the size"};
    }
    storage_.reserve(TotalSize(ext));
    std::ranges::move(range, std::back_inserter(storage_));
  }
  // vector-consuming specialization
  inline constexpr Matrix(std::from_range_t, std::vector<Numeric> &&vec, Extents ext)
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
    CacheLineWaltz(*this, view, [](auto &a, const auto &b) { a = b; });
  }

  template <BaseMatrixLike Mat1, BaseMatrixLike Mat2>
    requires(VStackExtentsStackable<MatExtentsType<Mat2>, MatExtentsType<Mat2>> &&
             std::is_same_v<Extents, VStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>> &&
             std::is_convertible_v<MatNumericType<Mat1>, NumericType> &&
             std::is_convertible_v<MatNumericType<Mat2>, NumericType>)
  inline constexpr Matrix(VerticalStackTagType, Mat1 &&a, Mat2 &&b)
      : storage_{}, extents_(VStackExtents(a.GetExtents(), b.GetExtents())) {
    const auto total = TotalSize(extents_);
    if (total == 0) {
      return;
    }
    if constexpr (!kIsStatic) {
      storage_.resize(total, Numeric{});
    }
    const auto cols = a.Columns();
    const auto rows1 = a.Rows();
    const auto rows2 = b.Rows();
    for (size_t row = 0; row < rows1; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)[row, col] = std::forward<Mat1>(a)[row, col];
      }
    }
    for (size_t row = 0; row < rows2; row++) {
      for (size_t col = 0; col < cols; col++) {
        (*this)[rows1 + row, col] = std::forward<Mat2>(b)[row, col];
      }
    }
  }

  template <BaseMatrixLike Mat1, BaseMatrixLike Mat2>
    requires(HStackExtentsStackable<MatExtentsType<Mat2>, MatExtentsType<Mat2>> &&
             std::is_same_v<Extents, HStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>> &&
             std::is_convertible_v<MatNumericType<Mat1>, NumericType> &&
             std::is_convertible_v<MatNumericType<Mat2>, NumericType>)
  inline constexpr Matrix(HorizontalStackTagType, Mat1 &&a, Mat2 &&b)
      : storage_{}, extents_(HStackExtents(a.GetExtents(), b.GetExtents())) {
    const auto total = TotalSize(extents_);
    if (total == 0) {
      return;
    }
    if constexpr (!kIsStatic) {
      storage_.resize(total, Numeric{});
    }
    const auto rows = a.Rows();
    const auto cols1 = a.Columns();
    const auto cols2 = b.Columns();
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols1; col++) {
        (*this)[row, col] = std::forward<Mat1>(a)[row, col];
      }
      for (size_t col = 0; col < cols2; col++) {
        (*this)[row, cols1 + col] = std::forward<Mat2>(b)[row, col];
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

template <BaseMatrixLike Mat1, BaseMatrixLike Mat2,
          typename Numeric = std::common_type_t<MatNumericType<Mat1>, MatNumericType<Mat2>>>
[[nodiscard]] inline constexpr auto MatrixVStack(Mat1 &&a, Mat2 &&b)
    -> Matrix<Numeric, VStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>> {
  return Matrix<Numeric, VStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>>(
      VerticalStackTagType{}, std::forward<Mat1>(a), std::forward<Mat2>(b));
}

template <BaseMatrixLike Mat1, BaseMatrixLike Mat2,
          typename Numeric = std::common_type_t<MatNumericType<Mat1>, MatNumericType<Mat2>>>
[[nodiscard]] inline constexpr auto MatrixHStack(Mat1 &&a, Mat2 &&b)
    -> Matrix<Numeric, HStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>> {
  return Matrix<Numeric, HStackExtentsType<MatExtentsType<Mat1>, MatExtentsType<Mat2>>>(
      HorizontalStackTagType{}, std::forward<Mat1>(a), std::forward<Mat2>(b));
}

/* for matrices of the same extents (the ones that have the same effective extents when this function is invoked):

returns less, if all elements of a are less than their counterparts in b

returns greater, if all elements of a are greater than in b

returns equivalent, if all elements in a are the same as in b

returns unordered, if none of the above (note that only strictly ordered matrices don't return unordered)

returns unordered, if the extents don't match*/
template <ConstMatrixLike View1, ConstMatrixLike View2>
  requires std::three_way_comparable_with<typename View1::NumericType, typename View2::NumericType>
[[nodiscard]] inline constexpr std::partial_ordering MatrixCompare(const View1 &a, const View2 &b) noexcept {
  if constexpr (!MatrixExtentsStaticCompatible<typename View1::ExtentsType, typename View2::ExtentsType>) {
    return std::partial_ordering::unordered;
  } else {
    if (a.GetExtents() != b.GetExtents()) {
      return std::partial_ordering::unordered;
    }
    std::partial_ordering result = std::partial_ordering::unordered;
    CacheLineWaltz(a, b, [&result](const auto &ai, const auto &bi) {
      const auto cmp = ai <=> bi;
      if (cmp == std::partial_ordering::unordered) {
        result = std::partial_ordering::unordered;
        return false;
      }
      if (result == std::partial_ordering::unordered) {
        result = cmp;
        return true;
      }
      if (result != cmp) {
        result = std::partial_ordering::unordered;
        return false;
      }
      return true;
    });
    return result;
  }
}
/* checks if all the corresponding elements in the matrices are ordered as required

returns false if extents don't match

through negation can be used to check unstrict relationships like <= and >= for all elements
*/
template <ConstMatrixLike View1, ConstMatrixLike View2>
  requires std::three_way_comparable_with<typename View1::NumericType, typename View2::NumericType>
[[nodiscard]] inline constexpr bool MatrixCompare(
    const View1 &a, const View2 &b,
    const std::partial_ordering ord) noexcept(noexcept(std::declval<const MatNumericType<View1> &>() <=>
                                                       std::declval<const MatNumericType<View2> &>())) {
  if constexpr (!MatrixExtentsStaticCompatible<MatExtentsType<View1>, MatExtentsType<View2>>) {
    return false;
  } else {
    if (a.GetExtents() != b.GetExtents()) {
      return false;
    }
    return CacheLineWaltz(a, b, [&ord](const auto &ai, const auto &bi) {
      const auto cmp = ai <=> bi;
      return cmp == ord;
    });
  }
}

// template <ConstMatrixLike T, typename Callable>
//   requires requires(std::remove_cvref_t<Callable> &c, const typename T::NumericType &n) { c(n); }
// struct TransformMatrixView {
//   using IndexType = typename T::IndexType;
//   using NumericType = decltype((std::declval<Callable &>())(std::declval<const typename T::NumericType &>()));
//   using ExtentsType = TransposedExtentsType<T>;

//   static inline constexpr bool kIsStatic = (ExtentsType::rank_dynamic() == 0);
//   inline static constexpr bool kIsStaticRows = (ExtentsType::static_extent(0) != std::dynamic_extent);
//   inline static constexpr bool kIsStaticColumns = (ExtentsType::static_extent(1) != std::dynamic_extent);

//   inline constexpr explicit TransformMatrixView(const T &matrix, Callable &&callable)
//       : ref_(matrix), callable_(callable) {
//   }
//   inline constexpr TransformMatrixView(const TransformMatrixView &) noexcept = default;
//   inline constexpr TransformMatrixView(TransformMatrixView &&) noexcept = default;
//   inline constexpr void operator=(const TransformMatrixView &) noexcept = delete;
//   inline constexpr void operator=(TransformMatrixView &&) noexcept = delete;
//   inline constexpr void operator=(const TransformMatrixView &) const noexcept = delete;
//   inline constexpr void operator=(TransformMatrixView &&) const noexcept = delete;
//   inline constexpr ~TransformMatrixView() noexcept = default;

//   inline constexpr const T &Original() const noexcept {
//     return ref_;
//   }
//   inline constexpr const NumericType &operator[](IndexType row, IndexType column) const noexcept {
//     return ref_[column, row];
//   }
//   inline constexpr const NumericType &At(IndexType row, IndexType column) const {
//     return ref_.At(column, row);
//   }
//   inline constexpr IndexType Rows() const noexcept {
//     return ref_.Columns();
//   }
//   inline constexpr IndexType Columns() const noexcept {
//     return ref_.Rows();
//   }
//   inline constexpr ExtentsType GetExtents() const noexcept {
//     if constexpr (kIsStatic) {
//       return ExtentsType{};
//     } else if constexpr (kIsStaticRows) {
//       return ExtentsType{ref_.Rows()};
//     } else if constexpr (kIsStaticColumns) {
//       return ExtentsType{ref_.Columns()};
//     } else {
//       return ExtentsType{ref_.Columns(), ref_.Rows()};
//     }
//     static_assert(ConstMatrixLike<TransposedMatrixView<T>>, "transposed view is missing something important");
//   }

//  private:
//   const T &ref_;
//   Callable &&callable_;
// };

// template <ConstMatrixLike Matrix1, ConstMatrixLike Matrix2>
// Matrix
}  // namespace matrix_implementation_detail
using matrix_implementation_detail::Matrix;
using matrix_implementation_detail::MatrixCompare;
using matrix_implementation_detail::MatrixTCopy;
using matrix_implementation_detail::MatrixVStack;