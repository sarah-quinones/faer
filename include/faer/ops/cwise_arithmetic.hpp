#ifndef FAER_ADD_HPP_C5YNQULZS
#define FAER_ADD_HPP_C5YNQULZS

#include "faer/traits/core.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
using veg::i64;

template <typename Expr>
struct Neg {
	struct RawParts {
		Expr expr;
	} unsafe_self;

	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> typename meta::matrix_traits<Neg>::Rows {
		return meta::matrix_traits<Neg>::nrows(*this);
	}
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> typename meta::matrix_traits<Neg>::Cols {
		return meta::matrix_traits<Neg>::ncols(*this);
	}
};

template <typename Lhs, typename Rhs>
struct Add {
	struct RawParts {
		Lhs lhs;
		Rhs rhs;
	} unsafe_self;

	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> typename meta::matrix_traits<Add>::Rows {
		return meta::matrix_traits<Add>::nrows(*this);
	}
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> typename meta::matrix_traits<Add>::Cols {
		return meta::matrix_traits<Add>::ncols(*this);
	}
};
template <typename Lhs, typename Rhs>
struct Sub {
	struct RawParts {
		Lhs lhs;
		Rhs rhs;
	} unsafe_self;

	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> typename meta::matrix_traits<Sub>::Rows {
		return meta::matrix_traits<Sub>::nrows(*this);
	}
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> typename meta::matrix_traits<Sub>::Cols {
		return meta::matrix_traits<Sub>::ncols(*this);
	}
};
template <typename Lhs, typename Rhs>
struct Mul {
	struct RawParts {
		Lhs lhs;
		Rhs rhs;
	} unsafe_self;

	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> typename meta::matrix_traits<Mul>::Rows {
		return meta::matrix_traits<Mul>::nrows(*this);
	}
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> typename meta::matrix_traits<Mul>::Cols {
		return meta::matrix_traits<Mul>::ncols(*this);
	}
};
template <typename Lhs, typename Rhs>
struct Div {
	struct RawParts {
		Lhs lhs;
		Rhs rhs;
	} unsafe_self;

	HEDLEY_ALWAYS_INLINE constexpr auto nrows() const noexcept -> typename meta::matrix_traits<Div>::Rows {
		return meta::matrix_traits<Div>::nrows(*this);
	}
	HEDLEY_ALWAYS_INLINE constexpr auto ncols() const noexcept -> typename meta::matrix_traits<Div>::Cols {
		return meta::matrix_traits<Div>::ncols(*this);
	}
};

namespace meta {
template <typename Op, template <typename...> class Tpl, typename... Exprs>
struct cwise_traits {
	template <bool CanMove>
	using ReadFwd = Tpl<typename matrix_traits<Exprs>::template ReadFwd<CanMove>...>;

	using Rows = CoerceEq<typename matrix_traits<Exprs>::Rows...>;
	using Cols = CoerceEq<typename matrix_traits<Exprs>::Cols...>;

	using Read = detected_or_t<void, Op::template Type, Read<Exprs>...>;

	template <typename NRows>
	using SubRows = Tpl<typename matrix_traits<Exprs>::template SubRows<NRows>...>;
	template <typename NCols>
	using SubCols = Tpl<typename matrix_traits<Exprs>::template SubCols<NCols>...>;
	using Transpose = Tpl<typename matrix_traits<Exprs>::Transpose...>;

	static constexpr bool nothrow_read =                  //
			VEG_ALL_OF(matrix_traits<Exprs>::nothrow_read) && //
			(detected_or_t<false_type, Op::template nothrow, typename matrix_traits<Exprs>::Read...>::value);
	static constexpr bool specialized = true;
	static constexpr bool optimizable = VEG_ALL_OF(matrix_traits<Exprs>::optimizable);
};

template <typename Op, template <typename, typename> class Tpl, typename Lhs, typename Rhs>
struct binop_cwise_traits : cwise_traits<Op, Tpl, Lhs, Rhs> {
	using base = cwise_traits<Op, Tpl, Lhs, Rhs>;

	static constexpr auto storage_order =
			meta::common_matrix_order(matrix_traits<Lhs>::storage_order, matrix_traits<Rhs>::storage_order);
	static constexpr auto storage_stride =
			meta::common_matrix_stride(matrix_traits<Lhs>::storage_stride, matrix_traits<Rhs>::storage_stride);

	template <bool B>
	HEDLEY_ALWAYS_INLINE static constexpr auto read_fwd(bool_constant<B> tag, Tpl<Lhs, Rhs>&& expr) noexcept ->
			typename base::template ReadFwd<B> {
		return {
				matrix_traits<Lhs>::read_fwd(tag, VEG_FWD(expr).unsafe_self.lhs),
				matrix_traits<Rhs>::read_fwd(tag, VEG_FWD(expr).unsafe_self.rhs),
		};
	}

	HEDLEY_ALWAYS_INLINE static constexpr auto nrows(Tpl<Lhs, Rhs> const& expr) noexcept -> typename base::Rows {
		return meta::coerce_eq( //
				matrix_traits<Lhs>::nrows(expr.unsafe_self.lhs),
				matrix_traits<Rhs>::nrows(expr.unsafe_self.rhs));
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto ncols(Tpl<Lhs, Rhs> const& expr) noexcept -> typename base::Cols {
		return meta::coerce_eq( //
				matrix_traits<Lhs>::ncols(expr.unsafe_self.lhs),
				matrix_traits<Rhs>::ncols(expr.unsafe_self.rhs));
	}

	HEDLEY_ALWAYS_INLINE static constexpr auto read_seq(Tpl<Lhs, Rhs>&& expr, i64 i) noexcept(base::nothrow_read) ->
			typename base::Read {
		return Op::apply(
				matrix_traits<Lhs>::read_seq(VEG_FWD(expr).unsafe_self.lhs, i),
				matrix_traits<Rhs>::read_seq(VEG_FWD(expr).unsafe_self.rhs, i));
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto read(Tpl<Lhs, Rhs>&& expr, i64 i, i64 j) noexcept(base::nothrow_read) ->
			typename base::Read {
		return Op::apply(
				matrix_traits<Lhs>::read(VEG_FWD(expr).unsafe_self.lhs, i, j),
				matrix_traits<Rhs>::read(VEG_FWD(expr).unsafe_self.rhs, i, j));
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto trans(Tpl<Lhs, Rhs>&& expr) noexcept -> typename base::Transpose {
		return {
				matrix_traits<Lhs>::trans(VEG_FWD(expr).unsafe_self.lhs),
				matrix_traits<Rhs>::trans(VEG_FWD(expr).unsafe_self.rhs),
		};
	}

	template <typename NRows>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Tpl<Lhs, Rhs>&& expr, i64 start_row, NRows nrows) noexcept ->
			typename base::template SubRows<NRows> {
		return {
				matrix_traits<Lhs>::subr(VEG_FWD(expr).unsafe_self.lhs, start_row, nrows),
				matrix_traits<Rhs>::subr(VEG_FWD(expr).unsafe_self.rhs, start_row, nrows),
		};
	}
	template <typename NCols>
	HEDLEY_ALWAYS_INLINE static constexpr auto subc(Tpl<Lhs, Rhs>&& expr, i64 start_col, NCols ncols) noexcept ->
			typename base::template SubCols<NCols> {
		return {
				matrix_traits<Lhs>::subc(VEG_FWD(expr).unsafe_self.lhs, start_col, ncols),
				matrix_traits<Rhs>::subc(VEG_FWD(expr).unsafe_self.rhs, start_col, ncols),
		};
	}

	template <template <typename...> class Cursor>
	HEDLEY_ALWAYS_INLINE static constexpr auto to_simd(meta::MetaTag<Cursor> tag, Tpl<Lhs, Rhs> const& expr) noexcept
			FAER_DECLTYPE_RET(Op::simd_apply( //
					matrix_traits<Lhs>::to_simd(tag, expr.unsafe_self.lhs),
					matrix_traits<Rhs>::to_simd(tag, expr.unsafe_self.rhs)));
};

template <typename Op, template <typename> class Tpl, typename Expr>
struct unop_cwise_traits : cwise_traits<Op, Tpl, Expr> {
	using base = cwise_traits<Op, Tpl, Expr>;
	static constexpr auto storage_order = matrix_traits<Expr>::storage_order;
	static constexpr auto storage_stride = matrix_traits<Expr>::storage_stride;

	template <bool B>
	HEDLEY_ALWAYS_INLINE static constexpr auto read_fwd(bool_constant<B> tag, Tpl<Expr>&& expr) noexcept ->
			typename base::template ReadFwd<B> {
		return {matrix_traits<Expr>::read_fwd(tag, VEG_FWD(expr).unsafe_self.expr)};
	}

	HEDLEY_ALWAYS_INLINE static constexpr auto nrows(Tpl<Expr> const& expr) noexcept -> typename base::Rows {
		return matrix_traits<Expr>::nrows(expr.unsafe_self.expr);
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto ncols(Tpl<Expr> const& expr) noexcept -> typename base::Cols {
		return matrix_traits<Expr>::ncols(expr.unsafe_self.expr);
	}

	HEDLEY_ALWAYS_INLINE static constexpr auto read_seq(Tpl<Expr>&& expr, i64 i) noexcept(base::nothrow_read) ->
			typename base::Read {
		return Op::apply(matrix_traits<Expr>::read_seq(VEG_FWD(expr).unsafe_self.expr, i));
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto read(Tpl<Expr>&& expr, i64 i, i64 j) noexcept(base::nothrow_read) ->
			typename base::Read {
		return Op::apply(matrix_traits<Expr>::read(VEG_FWD(expr).unsafe_self.expr, i, j));
	}
	HEDLEY_ALWAYS_INLINE static constexpr auto trans(Tpl<Expr>&& expr) noexcept -> typename base::Transpose {
		return {matrix_traits<Expr>::trans(VEG_FWD(expr).unsafe_self.expr)};
	}
	template <typename NRows>
	HEDLEY_ALWAYS_INLINE static constexpr auto subr(Tpl<Expr>&& expr, i64 start_row, NRows nrows) noexcept ->
			typename base::template SubRows<NRows> {
		return {
				matrix_traits<Expr>::subr(VEG_FWD(expr).unsafe_self.expr, start_row, nrows),
		};
	}
	template <typename NCols>
	HEDLEY_ALWAYS_INLINE static constexpr auto subc(Tpl<Expr>&& expr, i64 start_col, NCols ncols) noexcept ->
			typename base::template SubCols<NCols> {
		return {
				matrix_traits<Expr>::subc(VEG_FWD(expr).unsafe_self.expr, start_col, ncols),
		};
	}

	template <template <typename...> class Cursor>
	HEDLEY_ALWAYS_INLINE static constexpr auto to_simd(meta::MetaTag<Cursor> tag, Tpl<Expr> const& expr) noexcept
			FAER_DECLTYPE_RET(Op::simd_apply(matrix_traits<Expr>::to_simd(tag, expr.unsafe_self.expr)));
};

template <typename T>
struct matrix_traits<Neg<T>> : unop_cwise_traits<neg, Neg, T> {};

template <typename Lhs, typename Rhs>
struct matrix_traits<Add<Lhs, Rhs>> : binop_cwise_traits<add, Add, Lhs, Rhs> {};
template <typename Lhs, typename Rhs>
struct matrix_traits<Sub<Lhs, Rhs>> : binop_cwise_traits<sub, Sub, Lhs, Rhs> {};
template <typename Lhs, typename Rhs>
struct matrix_traits<Mul<Lhs, Rhs>> : binop_cwise_traits<mul, Mul, Lhs, Rhs> {};
template <typename Lhs, typename Rhs>
struct matrix_traits<Div<Lhs, Rhs>> : binop_cwise_traits<div, Div, Lhs, Rhs> {};
} // namespace meta

namespace ops {
namespace nb {
struct neg {
	VEG_TEMPLATE(
			typename Expr,
			requires(
					FAER_CONCEPT(readable<Expr>) && //
					FAER_CONCEPT(negatable<meta::Read<Expr>>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(expr, Expr&&))
	const noexcept->Neg<meta::ReadExpr<Expr>> { return {meta::read_expr(VEG_FWD(expr))}; }
};

struct add {
	VEG_TEMPLATE(
			(typename Lhs, typename Rhs),
			requires(
					FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
					FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
					FAER_CONCEPT(readable<Lhs>) &&                                   //
					FAER_CONCEPT(readable<Rhs>) &&                                   //
					FAER_CONCEPT(addable<meta::Read<Lhs>, meta::Read<Rhs>>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(lhs, Lhs&&),
			(rhs, Rhs&&))
	const noexcept->Add<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
		return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
	}
};
struct sub {
	VEG_TEMPLATE(
			(typename Lhs, typename Rhs),
			requires(
					FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
					FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
					FAER_CONCEPT(readable<Lhs>) &&                                   //
					FAER_CONCEPT(readable<Rhs>) &&                                   //
					FAER_CONCEPT(subtractible<meta::Read<Lhs>, meta::Read<Rhs>>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(lhs, Lhs&&),
			(rhs, Rhs&&))
	const noexcept->Sub<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
		return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
	}
};
struct cwise_mul {
	VEG_TEMPLATE(
			(typename Lhs, typename Rhs),
			requires(
					FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
					FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
					FAER_CONCEPT(readable<Lhs>) &&                                   //
					FAER_CONCEPT(readable<Rhs>) &&                                   //
					FAER_CONCEPT(multipliable<meta::Read<Lhs>, meta::Read<Rhs>>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(lhs, Lhs&&),
			(rhs, Rhs&&))
	const noexcept->Mul<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
		return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
	}
};
struct cwise_div {
	VEG_TEMPLATE(
			(typename Lhs, typename Rhs),
			requires(
					FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
					FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
					FAER_CONCEPT(readable<Lhs>) &&                                   //
					FAER_CONCEPT(readable<Rhs>) &&                                   //
					FAER_CONCEPT(divisible<meta::Read<Lhs>, meta::Read<Rhs>>)),
			HEDLEY_ALWAYS_INLINE constexpr auto
			operator(),
			(lhs, Lhs&&),
			(rhs, Rhs&&))
	const noexcept->Div<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
		return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
	}
};
} // namespace nb
VEG_NIEBLOID(add);
VEG_NIEBLOID(sub);
VEG_NIEBLOID(cwise_mul);
VEG_NIEBLOID(cwise_div);
} // namespace ops

namespace internal {
namespace adl {
VEG_TEMPLATE(
		(typename Lhs, typename Rhs),
		requires(
				FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
				FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
				FAER_CONCEPT(readable<Lhs>) &&                                   //
				FAER_CONCEPT(readable<Rhs>) &&                                   //
				FAER_CONCEPT(addable<meta::Read<Lhs>, meta::Read<Rhs>>)),
		HEDLEY_ALWAYS_INLINE constexpr auto
		operator+,
		(lhs, Lhs&&),
		(rhs, Rhs&&))
noexcept -> Add<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
	return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
}

VEG_TEMPLATE(
		(typename Lhs, typename Rhs),
		requires(
				FAER_CONCEPT(rows_maybe_eq<meta::Rows<Lhs>, meta::Rows<Rhs>>) && //
				FAER_CONCEPT(cols_maybe_eq<meta::Cols<Lhs>, meta::Cols<Rhs>>) && //
				FAER_CONCEPT(readable<Lhs>) &&                                   //
				FAER_CONCEPT(readable<Rhs>) &&                                   //
				FAER_CONCEPT(subtractible<meta::Read<Lhs>, meta::Read<Rhs>>)),
		HEDLEY_ALWAYS_INLINE constexpr auto
		operator-,
		(lhs, Lhs&&),
		(rhs, Rhs&&))
noexcept -> Sub<meta::ReadExpr<Lhs>, meta::ReadExpr<Rhs>> {
	return {meta::read_expr(VEG_FWD(lhs)), meta::read_expr(VEG_FWD(rhs))};
}
} // namespace adl
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_ADD_HPP_C5YNQULZS */
