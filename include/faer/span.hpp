#ifndef FAER_CONTIGUOUS_HPP_J6FW3CD8S
#define FAER_CONTIGUOUS_HPP_J6FW3CD8S

#include "faer/traits/core.hpp"
#include "faer/internal/assign.hpp"
#include "faer/internal/expr/reader.hpp"

#include <veg/internal/delete_special_members.hpp>
#include <veg/type_traits/assignable.hpp>
#include "faer/internal/prologue.hpp"

namespace fae {
namespace meta {

template <typename T>
struct delete_if_rref : veg::internal::NoCopyAssign, veg::internal::NoMoveAssign {};

template <typename T>
struct delete_if_rref<T&&> : veg::internal::NoCopyCtor {};
} // namespace meta

template <typename T, typename NRows, typename NCols, StorageOrder O, StorageStride S>
struct Span;

namespace internal {
namespace adl {
struct AdlBase {};
} // namespace adl
template <typename T, typename NRows, typename NCols, StorageOrder O, StorageStride S>
struct SpanBase;

template <typename T, typename NRows, typename NCols, StorageOrder O>
struct SpanBase<T, NRows, NCols, O, StorageStride::inner> : meta::delete_if_rref<T> {
	VEG_CHECK_CONCEPT(reference<T>);
	using Pointer = meta::unref_t<T>*;

	friend struct fae::Span<T, NRows, NCols, O, StorageStride::inner>;

private:
	internal::idx::Compressed<Pointer, NRows, NCols> self;
	i64 _outer_stride;
	i64 _inner_stride;

public:
	HEDLEY_ALWAYS_INLINE constexpr SpanBase(
			Pointer data, NRows rows, NCols cols, Dyn outer_stride, Dyn inner_stride) noexcept
			: self(data, rows, cols), _outer_stride(outer_stride), _inner_stride(inner_stride) {}
	HEDLEY_ALWAYS_INLINE constexpr auto outer_stride() const noexcept -> Dyn { return {_outer_stride}; }
	HEDLEY_ALWAYS_INLINE constexpr auto inner_stride() const noexcept -> Dyn { return {_inner_stride}; }
	HEDLEY_ALWAYS_INLINE constexpr auto col_stride() const noexcept -> Dyn {
		return O == col_major ? Dyn{_outer_stride} : Dyn{_inner_stride};
	}
	HEDLEY_ALWAYS_INLINE constexpr auto row_stride() const noexcept -> Dyn {
		return O == row_major ? Dyn{_outer_stride} : Dyn{_inner_stride};
	}
};

template <typename T, typename NRows, typename NCols, StorageOrder O>
struct SpanBase<T, NRows, NCols, O, StorageStride::outer> : meta::delete_if_rref<T> {
	VEG_CHECK_CONCEPT(reference<T>);
	using Pointer = meta::unref_t<T>*;

	friend struct fae::Span<T, NRows, NCols, O, StorageStride::outer>;

private:
	internal::idx::Compressed<Pointer, NRows, NCols> self;
	i64 _outer_stride;

	HEDLEY_ALWAYS_INLINE constexpr auto _dim_stride(meta::true_type /*tag*/) const noexcept -> Dyn {
		return {_outer_stride};
	}
	HEDLEY_ALWAYS_INLINE constexpr auto _dim_stride(meta::false_type /*tag*/) const noexcept -> Fix<1> { return {}; }

public:
	HEDLEY_ALWAYS_INLINE constexpr SpanBase(
			Pointer data, NRows rows, NCols cols, Dyn outer_stride, Dyn /*inner_stride*/ = {}) noexcept
			: self(data, rows, cols), _outer_stride(outer_stride) {}
	HEDLEY_ALWAYS_INLINE constexpr auto outer_stride() const noexcept -> Dyn { return {_outer_stride}; }
	HEDLEY_ALWAYS_INLINE constexpr auto inner_stride() const noexcept -> Fix<1> { return {}; }
	HEDLEY_ALWAYS_INLINE constexpr auto col_stride() const noexcept -> meta::conditional_t<O == col_major, Dyn, Fix<1>> {
		return _dim_stride(meta::bool_constant<O == col_major>{});
	}
	HEDLEY_ALWAYS_INLINE constexpr auto row_stride() const noexcept -> meta::conditional_t<O == row_major, Dyn, Fix<1>> {
		return _dim_stride(meta::bool_constant<O == row_major>{});
	}
};

template <typename T, typename NRows, typename NCols, StorageOrder O>
struct SpanBase<T, NRows, NCols, O, StorageStride::contiguous> : meta::delete_if_rref<T> {
	VEG_CHECK_CONCEPT(reference<T>);
	using Pointer = meta::unref_t<T>*;

	friend struct fae::Span<T, NRows, NCols, O, StorageStride::contiguous>;

private:
	internal::idx::Compressed<Pointer, NRows, NCols> self;

	HEDLEY_ALWAYS_INLINE constexpr auto _outer_stride(meta::true_type /*col_major*/) const noexcept -> NRows {
		return self.n0();
	}
	HEDLEY_ALWAYS_INLINE constexpr auto _outer_stride(meta::false_type /*col_major*/) const noexcept -> NCols {
		return self.n1();
	}

	HEDLEY_ALWAYS_INLINE constexpr auto _dim_stride(meta::true_type /*tag*/) const noexcept
			-> meta::conditional_t<O == col_major, NRows, NCols> {
		return {outer_stride()};
	}
	HEDLEY_ALWAYS_INLINE constexpr auto _dim_stride(meta::false_type /*tag*/) const noexcept -> Fix<1> { return {}; }

public:
	HEDLEY_ALWAYS_INLINE constexpr SpanBase(
			Pointer data, NRows rows, NCols cols, Dyn /*outer_stride*/ = {}, Dyn /*inner_stride*/ = {}) noexcept
			: self(data, rows, cols) {}
	HEDLEY_ALWAYS_INLINE constexpr auto outer_stride() const noexcept
			-> meta::conditional_t<O == col_major, NRows, NCols> {
		return _outer_stride(meta::bool_constant<O == col_major>{});
	}
	HEDLEY_ALWAYS_INLINE constexpr auto inner_stride() const noexcept -> Fix<1> { return {}; }
	HEDLEY_ALWAYS_INLINE constexpr auto col_stride() const noexcept
			-> meta::conditional_t<O == col_major, NRows, Fix<1>> {
		return _dim_stride(meta::bool_constant<O == col_major>{});
	}
	HEDLEY_ALWAYS_INLINE constexpr auto row_stride() const noexcept
			-> meta::conditional_t<O == row_major, NCols, Fix<1>> {
		return _dim_stride(meta::bool_constant<O == row_major>{});
	}
};
} // namespace internal
template <typename T, typename NRows, typename NCols, StorageOrder O, StorageStride S>
struct Span : internal::SpanBase<T, NRows, NCols, O, S>, internal::adl::AdlBase {
	Span(Span const&) = default;
	Span(Span&&) noexcept = default;
	auto operator=(Span) & -> Span& = delete;

private:
	using Base = internal::SpanBase<T, NRows, NCols, O, S>;

public:
	using Base::SpanBase;

	HEDLEY_ALWAYS_INLINE constexpr auto data() const noexcept -> typename Base::Pointer { return Base::self.value; }
	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> NRows { return Base::self.n0(); }
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> NCols { return Base::self.n1(); }

	VEG_TEMPLATE(
			typename Expr,
			requires(
					FAER_CONCEPT(rows_maybe_eq<meta::Rows<Span>, meta::Rows<Expr>>) && //
					FAER_CONCEPT(cols_maybe_eq<meta::Cols<Span>, meta::Cols<Expr>>) && //
					FAER_CONCEPT(readable<Expr>) &&                                    //
					VEG_CONCEPT(assignable<T, meta::Read<Expr>>)),
			VEG_CPP14(constexpr) /* NOLINT */
			HEDLEY_ALWAYS_INLINE auto noalias_assign,
			(expr, Expr&&))
	const noexcept(
			(meta::matrix_traits<meta::ReadExpr<Expr>>::nothrow_read &&
	     VEG_CONCEPT(nothrow_assignable<T, meta::Read<Expr>>))) {
		using expr_traits = meta::matrix_traits<meta::ReadExpr<Expr>>;
		auto&& ex = meta::read_expr(VEG_FWD(expr));

		internal::assign_impl<
				meta::common_matrix_order(O, expr_traits::storage_order),
				meta::common_matrix_stride(S, expr_traits::storage_stride),
				T>(                                                    //
				data(),                                                //
				VEG_FWD(ex),                                           //
				i64{meta::coerce_eq(nrows(), expr_traits::nrows(ex))}, //
				i64{meta::coerce_eq(ncols(), expr_traits::ncols(ex))}, //
				i64{this->row_stride()},                               //
				i64{this->col_stride()}                                //
		);
	}
};

namespace nb {
struct span_col_major {
	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols))
	const noexcept->Span<T&, NRows, NCols, col_major, StorageStride::contiguous> { return {data, nrows, ncols}; }

	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols),
			(outer_stride, Dyn))
	const noexcept->Span<T&, NRows, NCols, col_major, StorageStride::outer> { return {data, nrows, ncols, outer_stride}; }

	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols),
			(outer_stride, Dyn),
			(inner_stride, Dyn))
	const noexcept->Span<T&, NRows, NCols, col_major, StorageStride::inner> {
		return {data, nrows, ncols, outer_stride, inner_stride};
	}
};

struct span_row_major {
	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols))
	const noexcept->Span<T&, NRows, NCols, row_major, StorageStride::contiguous> { return {data, nrows, ncols}; }

	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols),
			(outer_stride, Dyn))
	const noexcept->Span<T&, NRows, NCols, row_major, StorageStride::outer> { return {data, nrows, ncols, outer_stride}; }

	VEG_TEMPLATE(
			(typename T, typename NRows, typename NCols),
			requires(
					VEG_CONCEPT(complete<T>) &&  //
					VEG_CONCEPT(index<NRows>) && //
					VEG_CONCEPT(index<NCols>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(data, T*),
			(nrows, NRows),
			(ncols, NCols),
			(outer_stride, Dyn),
			(inner_stride, Dyn))
	const noexcept->Span<T&, NRows, NCols, row_major, StorageStride::inner> {
		return {data, nrows, ncols, outer_stride, inner_stride};
	}
};
} // namespace nb
VEG_NIEBLOID(span_row_major);
VEG_NIEBLOID(span_col_major);

namespace meta {
template <typename T, typename NRows, typename NCols, StorageOrder O, StorageStride S>
struct span_matrix_traits {
	using Type = Span<T, NRows, NCols, O, S>;

	template <category_e C>
	using Fwd = Span<veg::meta::apply_category_t<C, T>, NRows, NCols, O, S>;
	template <bool CanMove>
	using ReadFwd = Span<
			meta::conditional_t<VEG_CONCEPT(rvalue_reference<T>) && CanMove, T&&, uncvref_t<T> const&>,
			NRows,
			NCols,
			O,
			S>;

	template <category_e C>
	HEDLEY_ALWAYS_INLINE static constexpr auto fwd(Category<C> /*tag*/, Type s) noexcept -> Fwd<C> {
		return {s.data(), s.nrows(), s.ncols(), s.outer_stride(), s.inner_stride()};
	}
	template <bool B>
	HEDLEY_ALWAYS_INLINE static constexpr auto read_fwd(bool_constant<B> /*tag*/, Type s) noexcept -> ReadFwd<B> {
		return {s.data(), (nrows)(s), (ncols)(s), s.outer_stride(), s.inner_stride()};
	}

	template <typename NRows_>
	using SubRows = Span<T, NRows_, NCols, O, S>;
	template <typename NCols_>
	using SubCols = Span<T, NRows, NCols_, O, S>;

	template <typename NRows_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Type s, i64 start_row, NRows_ nrows) noexcept -> SubRows<NRows_> {
		return {
				s.data() + i64{s.row_stride()} * start_row,
				nrows,
				s.ncols(),
				s.outer_stride(),
				s.inner_stride(),
		};
	}

	template <typename NCols_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Type s, i64 start_col, NCols_ ncols) noexcept -> SubCols<NCols_> {
		return {
				s.data() + i64{s.col_stride()} * start_col,
				s.nrows(),
				ncols,
				s.outer_stride(),
				s.inner_stride(),
		};
	}

	using Rows = NRows;
	using Cols = NCols;

	HEDLEY_ALWAYS_INLINE static constexpr auto nrows(Type const& s) noexcept -> Rows { return s.nrows(); }
	HEDLEY_ALWAYS_INLINE static constexpr auto ncols(Type const& s) noexcept -> Cols { return s.ncols(); }

	using Read = T&&;

	HEDLEY_ALWAYS_INLINE static constexpr auto read(Type s, i64 i, i64 j) noexcept -> Read {
		return Read(s.data()[i64{s.row_stride()} * i + i64{s.col_stride()} * j]);
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto read_seq(Type s, i64 i) noexcept -> Read { return Read(s.data()[i]); }

	using Transpose = Span<T, NRows, NCols, StorageOrder(-int(O)), S>;

	HEDLEY_ALWAYS_INLINE static constexpr auto trans(Type s) noexcept -> Transpose {
		return {s.data(), s.ncols(), s.nrows(), s.outer_stride(), s.inner_stride()};
	}

	HEDLEY_ALWAYS_INLINE static constexpr auto
	to_simd(meta::MetaTag<internal::expr::ContiguousCursor> /*unused*/, Type expr) noexcept
			-> internal::expr::ReadExpr<internal::expr::ContiguousCursor<uncvref_t<T>>, false> {
		return {const_cast<uncvref_t<T>*>(expr.data())};
	}

	static constexpr auto storage_order = O;
	static constexpr auto storage_stride = S;
	static constexpr bool nothrow_read = true;
	static constexpr bool specialized = true;
	static constexpr bool optimizable = VEG_CONCEPT(floating_point<uncvref_t<T>>);
};

template <typename T, typename NRows, typename NCols, StorageOrder O>
struct contig_span_matrix_traits;

template <typename T, typename NRows, typename NCols>
struct contig_span_matrix_traits<T, NRows, NCols, col_major>
		: span_matrix_traits<T, NRows, NCols, col_major, StorageStride::contiguous> {
	using Type = Span<T, NRows, NCols, col_major, StorageStride::contiguous>;

	template <typename NRows_>
	using SubRows = conditional_t<
			VEG_CONCEPT(same<eq_type<NRows, NRows_>, Boolean<yes>>),
			Type,
			Span<T, NRows_, NCols, col_major, StorageStride::outer>>;

	template <typename NRows_>
	HEDLEY_ALWAYS_INLINE static constexpr auto _subr(Type s, i64 start_row, NRows_ nrows, false_type /*tag*/) noexcept
			-> Span<T, NRows_, NCols, col_major, StorageStride::outer> {
		return {s.data() + i64{s.nrows() * Dyn(start_row)}, nrows, s.ncols(), s.nrows()};
	}
	template <typename NRows_>
	HEDLEY_ALWAYS_INLINE static constexpr auto
	_subr(Type s, i64 /*start_row*/, NRows_ /*nrows*/, true_type /*tag*/) noexcept -> Type {
		return s;
	}

	template <typename NRows_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Type s, i64 start_row, NRows_ nrows) noexcept -> SubRows<NRows_> {
		return _subr(s, start_row, nrows, bool_constant<VEG_CONCEPT(same<eq_type<NRows, NRows_>, Boolean<yes>>)>{});
	}

	template <typename NCols_>
	using SubCols = Span<T, NRows, NCols_, col_major, StorageStride::contiguous>;

	template <typename NCols_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subc(Type s, i64 start_col, NCols_ ncols) noexcept -> SubCols<NCols_> {
		return {s.data() + start_col, s.nrows(), ncols};
	}
};

template <typename T, typename NRows, typename NCols>
struct contig_span_matrix_traits<T, NRows, NCols, row_major>
		: span_matrix_traits<T, NRows, NCols, col_major, StorageStride::contiguous> {
	using trans_traits /* uh.. are human traits? */ =
			matrix_traits<Span<T, NCols, NRows, col_major, StorageStride::contiguous>>;
	using Type = Span<T, NRows, NCols, row_major, StorageStride::contiguous>;
	using TypeT = Span<T, NCols, NRows, col_major, StorageStride::contiguous>;

	template <typename NRows_>
	using SubRows = typename trans_traits::template SubCols<NRows_>;
	template <typename NCols_>
	using SubCols = typename trans_traits::template SubRows<NCols_>;

	template <typename NCols_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subc(Type s, i64 start_col, NCols_ ncols) noexcept -> SubCols<NCols_> {
		TypeT trans{s.data(), s.ncols(), s.nrows()};
		auto sub_trans = trans_traits::subr(trans, start_col, ncols);
		return trans_traits::trans(sub_trans);
	}
	template <typename NRows_>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Type s, i64 start_col, NRows_ nrows) noexcept -> SubRows<NRows_> {
		TypeT trans{s.data(), s.ncols(), s.nrows()};
		auto sub_trans = trans_traits::subc(trans, start_col, nrows);
		return trans_traits::trans(sub_trans);
	}
};

template <typename T, typename NRows, typename NCols, StorageOrder O, StorageStride S>
struct matrix_traits<Span<T, NRows, NCols, O, S>> : span_matrix_traits<T, NRows, NCols, O, S> {};

template <typename T, typename NRows, typename NCols, StorageOrder O>
struct matrix_traits<Span<T, NRows, NCols, O, StorageStride::contiguous>>
		: contig_span_matrix_traits<T, NRows, NCols, O> {};

} // namespace meta
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_CONTIGUOUS_HPP_J6FW3CD8S */
