#ifndef FAER_SCALAR_HPP_UGNW3SP6S
#define FAER_SCALAR_HPP_UGNW3SP6S

#include "faer/internal/expr/add.hpp"
#include "faer/internal/expr/mul.hpp"

#include <veg/type_traits/constructible.hpp>
#include <veg/util/index.hpp>
#include <veg/type_traits/value_category.hpp>
#include <veg/type_traits/primitives.hpp>
#include "faer/internal/prologue.hpp"

#define FAER_CONCEPT(...) VEG_CONCEPT_MACRO(::fae::concepts, __VA_ARGS__)
#define FAER_CHECK_CONCEPT(...) VEG_CHECK_CONCEPT_MACRO(::fae::concepts, __VA_ARGS__)
#define FAER_CHECK_CONCEPT(...) VEG_CHECK_CONCEPT_MACRO(::fae::concepts, __VA_ARGS__)

#define FAER_DELETE_COPY(Class)                                                                                        \
	~Class() = default;                                                                                                  \
	Class(Class const&) = delete;                   /* NOLINT */                                                         \
	Class(Class&&) = delete;                        /* NOLINT */                                                         \
	auto operator=(Class const&)&->Class& = delete; /* NOLINT */                                                         \
	auto operator=(Class&&)&->Class& = delete       /* NOLINT */

#define FAER_DELETE_ASSIGN(Class)                                                                                      \
	~Class() = default;                                                                                                  \
	Class(Class const&) = default;                  /* NOLINT */                                                         \
	Class(Class&&) = default;                       /* NOLINT */                                                         \
	auto operator=(Class const&)&->Class& = delete; /* NOLINT */                                                         \
	auto operator=(Class&&)&->Class& = delete       /* NOLINT */

#define FAER_EXPLICIT_COPY(Class)                                                                                      \
	~Class() = default;                                                                                                  \
	explicit Class(Class const&) = default;          /* NOLINT */                                                        \
	Class(Class&&) = default;                        /* NOLINT */                                                        \
	auto operator=(Class const&)&->Class& = default; /* NOLINT */                                                        \
	auto operator=(Class&&)&->Class& = default       /* NOLINT */

namespace fae {
using veg::i64;
using veg::Unsafe;
using veg::unsafe;
using veg::Fix;
using veg::Dyn;
using veg::Boolean;

using veg::yes;
using veg::no;
using veg::maybe;

enum struct StorageOrder : signed char {
	col_major = 1,
	row_major = -1,
	mixed = 0,
};
enum struct StorageStride : unsigned char {
	inner,
	outer,
	contiguous,
};

constexpr StorageOrder col_major = StorageOrder::col_major;
constexpr StorageOrder row_major = StorageOrder::row_major;

template <typename A, typename B = void>
struct number_traits {
	static constexpr bool commutative_add = true;
	static constexpr bool commutative_mul = false;
};
template <typename A>
struct number_traits<A> {
	using AliasedType = A;
};
template <typename T>
struct number_traits<T, T> {
	static constexpr bool commutative_add = true;
	static constexpr bool commutative_mul = false;
};
template <>
struct number_traits<double, double> {
	static constexpr bool commutative_add = true;
	static constexpr bool commutative_mul = true;
};
template <>
struct number_traits<float, float> {
	static constexpr bool commutative_add = true;
	static constexpr bool commutative_mul = true;
};

namespace meta {
template <template <typename...> class F>
struct MetaTag {
	template <typename... Ts>
	using Apply = F<Ts...>;
};

using veg::meta::bool_constant;
using veg::meta::true_type;
using veg::meta::false_type;
using veg::meta::detected_t;
using veg::meta::conditional_t;
using veg::meta::uncvref_t;
using veg::meta::unref_t;
using veg::meta::category_e;
using veg::meta::constant;
} // namespace meta

namespace internal {
namespace idx {
template <typename T, typename N0, typename N1>
struct Compressed {
	T value;
	N0 _n0;
	N1 _n1;
};

template <typename T, i64 N0, i64 N1>
struct Compressed<T, Fix<N0>, Fix<N1>> {
	T value;
	HEDLEY_ALWAYS_INLINE constexpr Compressed(T _value, Fix<N0> /*n0*/, Fix<N1> /*n1*/) noexcept
			: value(VEG_FWD(_value)) {}

	HEDLEY_ALWAYS_INLINE constexpr auto n0() const noexcept -> Fix<N0> { return {}; }
	HEDLEY_ALWAYS_INLINE constexpr auto n1() const noexcept -> Fix<N1> { return {}; }
};

template <typename T, i64 N0, typename N1>
struct Compressed<T, Fix<N0>, N1> {
	T value;
	N1 _n1;
	HEDLEY_ALWAYS_INLINE constexpr Compressed(T _value, Fix<N0> /*n0*/, N1 n1) noexcept
			: value(VEG_FWD(_value)), _n1(n1) {}

	HEDLEY_ALWAYS_INLINE constexpr auto n0() const noexcept -> Fix<N0> { return {}; }
	HEDLEY_ALWAYS_INLINE constexpr auto n1() const noexcept -> N1 { return _n1; }
};

template <typename T, typename N0, i64 N1>
struct Compressed<T, N0, Fix<N1>> {
	T value;
	N0 _n0;
	HEDLEY_ALWAYS_INLINE constexpr Compressed(T _value, N0 n0, Fix<N1> /*n1*/) noexcept
			: value(VEG_FWD(_value)), _n0(n0) {}

	HEDLEY_ALWAYS_INLINE constexpr auto n0() const noexcept -> N0 { return _n0; }
	HEDLEY_ALWAYS_INLINE constexpr auto n1() const noexcept -> Fix<N1> { return {}; }
};

template <typename T, typename U = T>
struct coerce_eq;

template <typename T>
struct coerce_eq<T, T> {
	using Type = T;
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(T a, T /*b*/) noexcept -> Type { return a; }
};

template <typename T>
struct coerce_eq<Dyn, T> {
	using Type = T;
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(Dyn /*a*/, T b) noexcept -> Type { return b; }
};

template <typename T>
struct coerce_eq<T, Dyn> {
	using Type = T;
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(T a, Dyn /*b*/) noexcept -> Type { return a; }
};

template <typename L, typename R>
using eq_expr = decltype(VEG_DECLVAL(L) == VEG_DECLVAL(R));

} // namespace idx
} // namespace internal

namespace meta {
template <typename L, typename R>
using eq_type = meta::detected_t<internal::idx::eq_expr, L, R>;

template <typename L, typename R, bool IsIdx = VEG_CONCEPT(index<L>) && VEG_CONCEPT(index<R>)>
struct maybe_eq : false_type {};
template <typename L, typename R>
struct maybe_eq<L, R, true> : bool_constant<decltype(L{} == R{})::type::value != veg::ternary_e::no> {};

template <category_e C>
using Category = veg::meta::constant<category_e, C>;

template <typename... Ts>
using CoerceEq = typename internal::idx::coerce_eq<Ts...>::Type;

template <typename T, typename U>
HEDLEY_ALWAYS_INLINE constexpr auto coerce_eq(T a, U b) noexcept -> CoerceEq<T, U> {
	return internal::idx::coerce_eq<T, U>::apply(a, b);
}

constexpr auto common_matrix_order(StorageOrder a, StorageOrder b) -> StorageOrder {
	return (a == b) ? a : StorageOrder::mixed;
}
constexpr auto common_matrix_stride(StorageStride a, StorageStride b) -> StorageStride {
	return int(a) < int(b) ? a : b;
}

template <typename T>
struct matrix_traits {
	template <bool CanMove>
	using ReadFwd = void;

	using Rows = void;
	using Cols = void;

	using Read = void;

	template <typename NRows>
	using SubRows = void;
	template <typename NCols>
	using SubCols = void;
	using Transpose = void;

	static constexpr auto storage_order = StorageOrder::mixed;
	static constexpr auto storage_stride = StorageStride::inner;
	static constexpr bool nothrow_read = false;
	static constexpr bool specialized = false;
	static constexpr bool optimizable = false;
};

template <typename T>
using ReadExpr = typename matrix_traits<uncvref_t<T>>::template ReadFwd<!VEG_CONCEPT(lvalue_reference<T>)>;
template <typename T>
HEDLEY_ALWAYS_INLINE constexpr auto read_expr(T&& t) noexcept -> ReadExpr<T> {
	return matrix_traits<uncvref_t<T>>::read_fwd(
			bool_constant < !VEG_CONCEPT(lvalue_reference<T>) && !VEG_CONCEPT(const_type<T>) > {},
			const_cast<uncvref_t<T>&&>(t));
}

template <typename T>
using Rows = typename matrix_traits<ReadExpr<T>>::Rows;
template <typename T>
using Cols = typename matrix_traits<ReadExpr<T>>::Cols;

template <typename T>
using Read = typename matrix_traits<ReadExpr<T>>::Read;

using veg::meta::detected_t;
using veg::meta::detected_or_t;

#define VEG_BINOP_ARITH(Name, Op)                                                                                      \
	struct Name {                                                                                                        \
		template <typename Lhs, typename Rhs>                                                                              \
		using Type = decltype(VEG_DECLVAL(Lhs) Op VEG_DECLVAL(Rhs));                                                       \
                                                                                                                       \
		template <typename Lhs, typename Rhs>                                                                              \
		using nothrow = bool_constant<FAER_IS_NOEXCEPT(VEG_DECLVAL_NOEXCEPT(Lhs) Op VEG_DECLVAL_NOEXCEPT(Rhs))>;           \
                                                                                                                       \
		template <typename Lhs, typename Rhs>                                                                              \
		HEDLEY_ALWAYS_INLINE static constexpr auto apply(Lhs&& lhs, Rhs&& rhs)                                             \
				VEG_DEDUCE_RET(VEG_FWD(lhs) Op VEG_FWD(rhs));                                                                  \
                                                                                                                       \
		template <typename Lhs, typename Rhs>                                                                              \
		HEDLEY_ALWAYS_INLINE static constexpr auto simd_apply(Lhs const& lhs, Rhs const& rhs) noexcept                     \
				FAER_DECLTYPE_RET(internal::expr::Name(lhs, rhs));                                                             \
	}

VEG_BINOP_ARITH(add, +);
VEG_BINOP_ARITH(sub, -);
VEG_BINOP_ARITH(mul, *);
VEG_BINOP_ARITH(div, /);

struct neg {
	template <typename T>
	using Type = decltype(-VEG_DECLVAL(T));

	template <typename T>
	using nothrow = bool_constant<FAER_IS_NOEXCEPT(-VEG_DECLVAL_NOEXCEPT(T))>;

	template <typename T>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(T&& t) VEG_DEDUCE_RET(-VEG_FWD(t));

	template <typename T>
	HEDLEY_ALWAYS_INLINE static constexpr auto simd_apply(T const& expr) noexcept FAER_DECLTYPE_RET(expr.negate());
};
} // namespace meta

namespace concepts {
VEG_DEF_CONCEPT(
		typename T,
		scalar,
		(VEG_CONCEPT(constructible<T>) &&              //
     VEG_CONCEPT(nothrow_move_constructible<T>) && //
     VEG_CONCEPT(copy_constructible<T>)));

VEG_DEF_CONCEPT(typename T, readable, !VEG_CONCEPT(void_type<meta::Read<T>>));

VEG_DEF_CONCEPT(typename T, negatable, VEG_CONCEPT(detected<meta::neg::template Type, T>));
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), addable, VEG_CONCEPT(detected<meta::add::template Type, Lhs, Rhs>));
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), subtractible, VEG_CONCEPT(detected<meta::sub::template Type, Lhs, Rhs>));
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), multipliable, VEG_CONCEPT(detected<meta::mul::template Type, Lhs, Rhs>));
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), divisible, VEG_CONCEPT(detected<meta::div::template Type, Lhs, Rhs>));

VEG_DEF_CONCEPT(typename T, nothrow_negatable, meta::neg::template nothrow<T>::value);
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), nothrow_addable, meta::add::template nothrow<Lhs, Rhs>::value);
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), nothrow_subtractible, meta::sub::template nothrow<Lhs, Rhs>::value);
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), nothrow_multipliable, meta::mul::template nothrow<Lhs, Rhs>::value);
VEG_DEF_CONCEPT((typename Lhs, typename Rhs), nothrow_divisible, meta::div::template nothrow<Lhs, Rhs>::value);

VEG_DEF_CONCEPT((typename N, typename M), rows_maybe_eq, meta::maybe_eq<N, M>::value);
VEG_DEF_CONCEPT((typename N, typename M), cols_maybe_eq, meta::maybe_eq<N, M>::value);
VEG_DEF_CONCEPT((typename N, typename M), depth_maybe_eq, meta::maybe_eq<N, M>::value);

} // namespace concepts
}; // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_SCALAR_HPP_UGNW3SP6S */
