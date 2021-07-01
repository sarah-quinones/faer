#ifndef FAER_CAST_HPP_GTRPKQEAS
#define FAER_CAST_HPP_GTRPKQEAS

#include "faer/internal/expr/core.hpp"

#include <veg/type_traits/core.hpp>
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename Impl, typename To>
struct CastExprImpl {
	Impl e;

	FAER_STATIC_ASSERT_DEV(!VEG_CONCEPT(same<typename Impl::Element, To>));
	FAER_STATIC_ASSERT_DEV(!VEG_CONCEPT(const_type<To>));

	using Element = To;
	static constexpr i64 packet_size = internal::min_(Impl::packet_size, FAER_PACK_SIZE / i64{sizeof(To)});

	static constexpr bool preserve_zero = Impl::preserve_zero;
	static constexpr bool has_fma = false;
	static constexpr bool has_fnma = false;

	using WithoutScalar = void;

	static constexpr i64 estimated_register_cost_base = Impl::estimated_register_cost_base;
	static constexpr i64 estimated_register_cost_per_chain = Impl::estimated_register_cost_per_chain;

	using value_type = To;
	using read_type = To;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		e.template advance<N, A>();
	}
	template <i64 N>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto read_pack() const noexcept ->
			typename _simd::sized_pack_type<Element, N>::Pack {
		return _simd::sized_pack_traits<typename Impl::Element, N>::cvt(veg::Tag<To>{}, e.template get_pack<N>());
	}
};

template <typename ExprImpl, typename To, bool FlipSign>
struct CastExpr {
	using Impl = CastExprImpl<ExprImpl, To>;
	Impl impl;

	static constexpr bool flip_sign = FlipSign;
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> CastExpr<ExprImpl, To, !FlipSign> {
		return {impl};
	}
};

template <typename To, typename Expr>
HEDLEY_ALWAYS_INLINE constexpr auto cast(Expr const& e) noexcept -> veg::meta::conditional_t<
		VEG_CONCEPT(same<To, typename Expr::Impl::Element>),
		Expr,
		CastExpr<typename Expr::Impl, To, Expr::flip_sign>> {
	return {e.impl};
}

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_CAST_HPP_GTRPKQEAS */
