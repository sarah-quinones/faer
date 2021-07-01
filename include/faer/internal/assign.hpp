#ifndef FAER_ASSIGN_HPP_OQN5H84AS
#define FAER_ASSIGN_HPP_OQN5H84AS

#include "faer/internal/expr/eval.hpp"
#include "faer/traits/core.hpp"
#include "faer/internal/expr/cursor.hpp"

#include <veg/type_traits/assignable.hpp>
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {

template <bool Optimizable, StorageOrder O, StorageStride S>
struct linear_assign_impl;

template <>
struct linear_assign_impl<false, StorageOrder::col_major, StorageStride::contiguous> {
	template <typename Out, typename Expr>
	HEDLEY_ALWAYS_INLINE static VEG_CPP14(constexpr) void apply(
			meta::uncvref_t<Out>* const outptr,
			Expr&& expr,
			i64 const nrows,
			i64 const ncols,
			i64 const /*row_stride*/,
			i64 const /*col_stride*/) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
		for (i64 i = 0; i < nrows * ncols; ++i) {
			static_cast<Out>(outptr[i]) = meta::matrix_traits<Expr>::read_seq(VEG_FWD(expr), i);
		}
	}
};
template <>
struct linear_assign_impl<false, StorageOrder::row_major, StorageStride::contiguous>
		: linear_assign_impl<false, StorageOrder::col_major, StorageStride::contiguous> {};

template <>
struct linear_assign_impl<false, StorageOrder::col_major, StorageStride::inner> {
	template <typename Out, typename Expr>
	HEDLEY_ALWAYS_INLINE static VEG_CPP14(constexpr) void apply(
			meta::uncvref_t<Out>* const outptr,
			Expr&& expr,
			i64 const nrows,
			i64 const ncols,
			i64 const row_stride,
			i64 const col_stride) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
		for (i64 j = 0; j < ncols; ++j) {
			for (i64 i = 0; i < nrows; ++i) {
				static_cast<Out>(outptr[j * col_stride + i * row_stride]) =
						meta::matrix_traits<Expr>::read(VEG_FWD(expr), i, j);
			}
		}
	}
};
template <>
struct linear_assign_impl<false, StorageOrder::row_major, StorageStride::inner> {
	template <typename Out, typename Expr>
	HEDLEY_ALWAYS_INLINE static VEG_CPP14(constexpr) void apply(
			meta::uncvref_t<Out>* const outptr,
			Expr&& expr,
			i64 const nrows,
			i64 const ncols,
			i64 const row_stride,
			i64 const col_stride) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
		for (i64 i = 0; i < nrows; ++i) {
			for (i64 j = 0; j < ncols; ++j) {
				static_cast<Out>(outptr[j * col_stride + i * row_stride]) =
						meta::matrix_traits<Expr>::read(VEG_FWD(expr), i, j);
			}
		}
	}
};
template <>
struct linear_assign_impl<false, StorageOrder::mixed, StorageStride::inner> {
	template <typename Out, typename Expr>
	HEDLEY_ALWAYS_INLINE static VEG_CPP14(constexpr) void apply(
			meta::uncvref_t<Out>* const outptr,
			Expr&& expr,
			i64 const nrows,
			i64 const ncols,
			i64 const row_stride,
			i64 const col_stride) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
		veg::unused(outptr, expr, nrows, ncols, row_stride, col_stride);
		VEG_ASSERT_ELSE("unimplemented", false);
	}
};

template <StorageOrder O, StorageStride S>
struct linear_assign_impl<true, O, S> {
	template <typename Out, typename Expr>
	HEDLEY_ALWAYS_INLINE static VEG_CPP14(constexpr) void apply(
			meta::uncvref_t<Out>* const outptr,
			Expr&& expr,
			i64 const nrows,
			i64 const ncols,
			i64 const row_stride,
			i64 const col_stride) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
		(void)row_stride, (void)col_stride;
		auto dest = expr::ContiguousCursor<meta::uncvref_t<Out>>{outptr};
		auto src = meta::matrix_traits<Expr>::to_simd(meta::MetaTag<expr::ContiguousCursor>{}, expr);
		expr::evaluate_contiguous(nrows * ncols, dest, src);
	}
};

template <StorageOrder O>
struct linear_assign_impl<false, O, StorageStride::outer> : linear_assign_impl<false, O, StorageStride::inner> {};

template <StorageOrder O, StorageStride S, typename Out, typename Expr>
HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void assign_impl(
		meta::uncvref_t<Out>* const outptr,
		Expr&& expr,
		i64 const nrows,
		i64 const ncols,
		i64 const row_stride,
		i64 const col_stride) noexcept(VEG_CONCEPT(nothrow_assignable<Out&, meta::Read<Expr>>)) {
	linear_assign_impl<meta::matrix_traits<Expr>::optimizable, O, S>::template apply<Out>( //
			outptr,
			VEG_FWD(expr),
			nrows,
			ncols,
			row_stride,
			col_stride);
}

} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_ASSIGN_HPP_OQN5H84AS */
