#ifndef FAER_MUL_HPP_YUBOEAQRS
#define FAER_MUL_HPP_YUBOEAQRS

#include "faer/internal/expr/core.hpp"
#include "faer/internal/expr/broadcast.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename Impl1, typename Impl2>
struct MulExprImpl {
	Impl1 impl1;
	Impl2 impl2;

	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl1::Element, typename Impl2::Element>));
	FAER_STATIC_ASSERT_DEV(is_broadcast<Impl1>::value || VEG_CONCEPT(same<typename Impl1::WithoutScalar, void>));
	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl2::WithoutScalar, void>));

	using Element = typename Impl1::Element;
	static constexpr i64 packet_size = internal::min_(Impl1::packet_size, Impl2::packet_size);

	static constexpr bool preserve_zero = Impl1::preserve_zero && Impl2::preserve_zero;
	static constexpr bool has_fma = true;
	static constexpr bool has_nfma = true;

	using WithoutScalar = veg::meta::conditional_t<is_broadcast<Impl1>::value, Impl2, void>;

	static constexpr i64 estimated_register_cost_base =
			Impl1::estimated_register_cost_base + Impl2::estimated_register_cost_base;
	static constexpr i64 estimated_register_cost_per_chain =
			Impl1::estimated_register_cost_per_chain + Impl2::estimated_register_cost_per_chain;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		impl1.template advance<N, A>();
		impl2.template advance<N, A>();
	}
	template <i64 N>
	HEDLEY_ALWAYS_INLINE constexpr auto read_pack() const noexcept -> typename _simd::sized_pack_type<Element, N>::Pack {
		return _simd::sized_pack_traits<Element, N>::mul( //
				impl1.template read_pack<N>(),
				impl2.template read_pack<N>());
	}

	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto extract_scalar() const noexcept -> Element {
		return impl1.extract_scalar();
	}
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto extract_expr() const noexcept -> Impl2 { return impl2; }
};

template <typename Impl1, typename Impl2, bool FlipSign>
struct MulExpr {
	using Impl = MulExprImpl<Impl1, Impl2>;
	Impl impl;

	static constexpr bool flip_sign = FlipSign;

	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> MulExpr<Impl1, Impl2, !FlipSign> {
		return {impl};
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Impl>
struct InvExprImpl {
	Impl impl;

	using Element = typename Impl::Element;
	static constexpr i64 packet_size = Impl::packet_size;

	static constexpr bool preserve_zero = false;
	static constexpr bool has_fma = true;
	static constexpr bool has_nfma = true;

	using WithoutScalar = void;

	static constexpr i64 estimated_register_cost_base = Impl::estimated_register_cost_base;
	static constexpr i64 estimated_register_cost_per_chain = Impl::estimated_register_cost_per_chain;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		impl.template advance<N, A>();
	}
	template <i64 N>
	HEDLEY_ALWAYS_INLINE constexpr auto read_pack() const noexcept -> typename _simd::sized_pack_type<Element, N>::Pack {
		return _simd::sized_pack_traits<Element, N>::div( //
				_simd::sized_pack_traits<Element, N>::broadcast(Element(1)),
				impl.template read_pack<N>());
	}
};

template <typename ImplI, bool FlipSign>
struct InvExpr {
	using Impl = InvExprImpl<ImplI>;
	Impl impl;

	static constexpr bool flip_sign = FlipSign;
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> InvExpr<ImplI, !FlipSign> { return {impl}; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct mul_impl0 {
	// both have inner exprs, neither is broadcast
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                       //
					BroadcastExprImpl<typename E1::Impl::Element>,
					MulExprImpl<typename E1::Impl::WithoutScalar, typename E2::Impl::WithoutScalar>,
					E1::flip_sign != E2::flip_sign> {
		auto s1 = e1.impl.extract_scalar();
		auto s2 = e2.impl.extract_scalar();
		return {{{s1 * s2}, {e1.impl.extract_expr(), e2.impl.extract_expr()}}};
	}
};
template <typename Extracted1, typename Extracted2>
struct mul_impl : mul_impl0 {};

struct mul_impl1 {
	// both have inner exprs, lhs is broadcast
	template <typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto
	apply(BroadcastExpr<typename E2::Impl::Element> const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                   //
					BroadcastExprImpl<typename E2::Impl::Element>,
					typename E2::Impl::WithoutScalar,
					E2::flip_sign> {
		auto s1 = e1.impl.extract_scalar();
		auto s2 = e2.impl.extract_scalar();
		return {{{s1 * s2}, e2.impl.extract_expr()}};
	}
};
template <typename Extracted2>
struct mul_impl<One, Extracted2> : mul_impl1 {};

struct mul_impl2 {
	// both have inner exprs, rhs is broadcast
	template <typename E1>
	HEDLEY_ALWAYS_INLINE static constexpr auto
	apply(E1 const& e1, BroadcastExpr<typename E1::Impl::Element> const& e2) noexcept //
			-> MulExpr<                                                                   //
					BroadcastExprImpl<typename E1::Impl::Element>,
					typename E1::Impl::WithoutScalar,
					E1::flip_sign> {
		auto s1 = e1.impl.extract_scalar();
		auto s2 = e2.impl.extract_scalar();
		return {{{s1 * s2}, e1.impl.extract_expr()}};
	}
};
template <typename Extracted1>
struct mul_impl<Extracted1, One> : mul_impl2 {};

template <>
struct mul_impl<One, One> {
	// both have inner exprs, both broadcast
	template <typename T>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(BroadcastExpr<T> const& e1, BroadcastExpr<T> const& e2) noexcept //
			-> BroadcastExpr<T> {
		auto s1 = e1.impl.extract_scalar();
		auto s2 = e2.impl.extract_scalar();
		return {s1 * s2};
	}
};

struct mul_impl4 {
	// lhs has no inner expr
	// rhs has inner expr, not broadcast
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                       //
					BroadcastExprImpl<typename E1::Impl::Element>,
					MulExprImpl<typename E1::Impl, typename E2::Impl::WithoutScalar>,
					E1::flip_sign != E2 ::flip_sign> {
		auto s2 = e2.impl.extract_scalar();
		return {{{s2}, {e1.impl, e2.impl.extract_expr()}}};
	}
};
template <typename Extracted2>
struct mul_impl<void, Extracted2> : mul_impl4 {};

struct mul_impl5 {
	// lhs has inner expr, not broadcast
	// rhs has no inner expr
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                       //
					BroadcastExprImpl<typename E1::Impl::Element>,
					MulExprImpl<typename E1::Impl::WithoutScalar, typename E2::Impl>,
					E1::flip_sign != E2::flip_sign> {
		auto s1 = e1.impl.extract_scalar();
		return {{{s1}, {e1.impl.extract_expr(), e2.impl}}};
	}
};
template <typename Extracted1>
struct mul_impl<Extracted1, void> : mul_impl5 {};

template <>
struct mul_impl<void, One> {
	// lhs has no inner expr
	// rhs has inner expr, broadcast
	template <typename E1>
	HEDLEY_ALWAYS_INLINE static constexpr auto
	apply(E1 const& e1, BroadcastExpr<typename E1::Impl::Element> const& e2) noexcept //
			-> MulExpr<                                                                   //
					BroadcastExprImpl<typename E1::Impl::Element>,
					typename E1::Impl,
					E1::flip_sign> {
		return {{{e2.impl.extract_scalar()}, {e1.impl}}};
	}
};

template <>
struct mul_impl<One, void> {
	// lhs has inner expr, broadcast
	// rhs has no inner expr
	template <typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto
	apply(BroadcastExpr<typename E2::Impl::Element> const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                   //
					BroadcastExprImpl<typename E2::Impl::Element>,
					typename E2::Impl,
					E2::flip_sign> {
		return {{{e1.impl.extract_scalar()}, {e2.impl}}};
	}
};

template <>
struct mul_impl<void, void> {
	// neiter has inner expr
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept //
			-> MulExpr<                                                                       //
					typename E1::Impl,
					typename E2::Impl,
					E1::flip_sign != E2::flip_sign> {
		return {{e1.impl, e2.impl}};
	}
};

template <typename E1, typename E2>
HEDLEY_ALWAYS_INLINE constexpr auto mul(E1 const& e1, E2 const& e2) noexcept
		FAER_DECLTYPE_RET(mul_impl<typename E1::Impl::WithoutScalar, typename E2::Impl::WithoutScalar>::apply(e1, e2));

template <typename E1, typename E2>
HEDLEY_ALWAYS_INLINE constexpr auto div(E1 const& e1, E2 const& e2) noexcept
		FAER_DECLTYPE_RET(expr::mul(e1, InvExpr<typename E2::Impl, E2::flip_sign>{e2.impl}));

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_MUL_HPP_YUBOEAQRS */
