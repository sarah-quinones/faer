#ifndef FAER_ADD_HPP_MBU17CRKS
#define FAER_ADD_HPP_MBU17CRKS

#include "faer/internal/expr/core.hpp"
#include "faer/internal/expr/broadcast.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename Impl1, typename Impl2>
struct AddExprImpl {
	Impl1 impl1;
	Impl2 impl2;

	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl1::Element, typename Impl2::Element>));

	using Element = typename Impl1::Element;
	static constexpr i64 packet_size = internal::min_(Impl1::packet_size, Impl2::packet_size);

	static constexpr bool preserve_zero = Impl1::preserve_zero && Impl2::preserve_zero;
	static constexpr bool has_fma = false;
	static constexpr bool has_nfma = false;

	using WithoutScalar = void;

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
		return _simd::sized_pack_traits<Element, N>::add( //
				impl1.template read_pack<N>(),
				impl2.template read_pack<N>());
	}
};

template <typename Impl1, typename Impl2, bool FlipSign>
struct AddExpr {
	using Impl = AddExprImpl<Impl1, Impl2>;
	Impl impl;

	static constexpr bool flip_sign = FlipSign;

	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> AddExpr<Impl1, Impl2, !FlipSign> {
		return {impl};
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Impl1, typename Impl2>
struct SubExprImpl {
	Impl1 impl1;
	Impl2 impl2;

	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl1::Element, typename Impl2::Element>));

	using Element = typename Impl1::Element;
	static constexpr i64 packet_size = internal::min_(Impl1::packet_size, Impl2::packet_size);

	static constexpr bool preserve_zero = Impl1::preserve_zero && Impl2::preserve_zero;
	static constexpr bool has_fma = false;
	static constexpr bool has_nfma = false;

	using WithoutScalar = void;

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
		return _simd::sized_pack_traits<Element, N>::sub( //
				impl1.template read_pack<N>(),
				impl2.template read_pack<N>());
	}
};

template <typename Impl1, typename Impl2>
struct SubExpr {
	using Impl = SubExprImpl<Impl1, Impl2>;
	Impl impl;

	static constexpr bool flip_sign = false;

	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> SubExpr<Impl2, Impl1> {
		return {{impl.impl2, impl.impl1}};
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////

// sign1 * a * b + sign3 * c
template <typename Impl1, typename Impl2, typename Impl3, bool Flip1, bool Flip3>
struct FMAddExprImpl {
	Impl1 impl1;
	Impl2 impl2;
	Impl3 impl3;

	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl1::Element, typename Impl2::Element>));
	FAER_STATIC_ASSERT_DEV(VEG_CONCEPT(same<typename Impl1::Element, typename Impl3::Element>));

	using Element = typename Impl1::Element;
	static constexpr i64 packet_size =
			internal::min_(internal::min_(Impl1::packet_size, Impl2::packet_size), Impl3::packet_size);

	static constexpr bool preserve_zero = Impl1::preserve_zero && Impl2::preserve_zero && Impl3::preserve_zero;
	static constexpr bool has_fma = false;
	static constexpr bool has_nfma = false;

	using WithoutScalar = void;

	static constexpr i64 estimated_register_cost_base = Impl1::estimated_register_cost_base + //
	                                                    Impl2::estimated_register_cost_base +
	                                                    Impl3::estimated_register_cost_base;
	static constexpr i64 estimated_register_cost_per_chain = Impl1::estimated_register_cost_per_chain +
	                                                         Impl2::estimated_register_cost_per_chain +
	                                                         Impl3::estimated_register_cost_per_chain;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		impl1.template advance<N, A>();
		impl2.template advance<N, A>();
		impl3.template advance<N, A>();
	}

	template <i64 N>
	HEDLEY_ALWAYS_INLINE constexpr auto read_pack() const noexcept -> typename _simd::sized_pack_type<Element, N>::Pack {
		using traits = _simd::sized_pack_traits<Element, N>;
		constexpr auto* fn = Flip1 //
		                         ? (Flip3 ? traits::fnmsub : traits::fnmadd)
		                         : (Flip3 ? traits::fmsub : traits::fmadd);
		return fn( //
				impl1.template read_pack<N>(),
				impl2.template read_pack<N>(),
				impl3.template read_pack<N>());
	}
};

template <typename Impl1, typename Impl2, typename Impl3, bool Flip1, bool Flip3>
struct FMAddExpr {
	using Impl = FMAddExpr<Impl1, Impl2, Impl3, Flip1, Flip3>;
	Impl impl;

	static constexpr bool flip_sign = false;

	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept
			-> FMAddExpr<Impl1, Impl2, Impl3, !Flip1, !Flip3> {
		return {{impl.impl1, impl.impl2, impl.impl3}};
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////

struct broadcast_add_impl {
	template <typename T>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(BroadcastExpr<T> const& e1, BroadcastExpr<T> const& e2) noexcept
			-> BroadcastExpr<T> {
		return {{e1.impl.s + e2.impl.s}};
	}
};

struct generic_add_impl_same_sign {
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept
			-> AddExpr<typename E1::Impl, typename E2::Impl, E1::flip_sign && E2::flip_sign> {
		FAER_STATIC_ASSERT_DEV(E1::flip_sign == E2::flip_sign);
		return {{e1.impl, e2.impl}};
	}
};

struct generic_add_impl_diff_sign0 {
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept
			-> SubExpr<typename E1::Impl, typename E2::Impl> {
		FAER_STATIC_ASSERT_DEV(E2::flip_sign && !E1::flip_sign);
		return {{e1.impl, e2.impl}};
	}
};
struct generic_add_impl_diff_sign1 {
	template <typename E1, typename E2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1 const& e1, E2 const& e2) noexcept
			-> SubExpr<typename E2::Impl, typename E1::Impl> {
		FAER_STATIC_ASSERT_DEV(E1::flip_sign && !E2::flip_sign);
		return {{e2.impl, e1.impl}};
	}
};
struct fmadd_impl0 {
	template <typename E1_2, typename E3>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E1_2 const& e1_2, E3 const& e3) noexcept -> FMAddExpr<
			decltype(e1_2.impl.impl1),
			decltype(e1_2.impl.impl2),
			typename E3::Impl,
			E1_2::flip_sign,
			E3::flip_sign> {
		return {{e1_2.impl.impl1, e1_2.impl.impl2, e3.impl}};
	}
};
struct fmadd_impl1 {
	template <typename E3, typename E1_2>
	HEDLEY_ALWAYS_INLINE static constexpr auto apply(E3 const& e3, E1_2 const& e1_2) noexcept -> FMAddExpr<
			decltype(e1_2.impl.impl1),
			decltype(e1_2.impl.impl2),
			typename E3::Impl,
			E1_2::flip_sign,
			E3::flip_sign> {
		return {{e1_2.impl.impl1, e1_2.impl.impl2, e3.impl}};
	}
};

template <typename E1, typename E2>
HEDLEY_ALWAYS_INLINE static constexpr auto add(E1 const& e1, E2 const& e2) noexcept FAER_DECLTYPE_RET( //
		veg::meta::conditional_t<                                                                          //
				(E1::Impl::has_fma || E2::Impl::has_fma),                                                      //
				veg::meta::conditional_t<E1::Impl::has_fma, fmadd_impl0, fmadd_impl1>,                         //
				veg::meta::conditional_t<                                                                      //
						((is_broadcast<typename E1::Impl>::value &&                                                //
              is_broadcast<typename E2::Impl>::value)),                                                //
						broadcast_add_impl,                                                                        //
                                                                                                       //
						veg::meta::conditional_t<                                                                  //
								E1::flip_sign == E2::flip_sign,                                                        //
								generic_add_impl_same_sign,                                                            //
								veg::meta::conditional_t<                                                              //
										E2::flip_sign,                                                                     //
										generic_add_impl_diff_sign0,                                                       //
										generic_add_impl_diff_sign1                                                        //
										>                                                                                  //
								>                                                                                      //
						>                                                                                          //
				>::apply(e1, e2)                                                                               //
);

template <typename E1, typename E2>
HEDLEY_ALWAYS_INLINE static constexpr auto sub(E1 const& e1, E2 const& e2) noexcept
		FAER_DECLTYPE_RET(expr::add(e1, e2.negate()));

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_ADD_HPP_MBU17CRKS */
