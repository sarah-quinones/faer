#ifndef FAER_SCAL_MUL_HPP_PPMZYT37S
#define FAER_SCAL_MUL_HPP_PPMZYT37S

#include "faer/internal/expr/core.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename T>
struct BroadcastExprImpl {
	using Element = T;
	Element s;

	static constexpr i64 packet_size = FAER_PACK_SIZE / i64{sizeof(Element)};

	static constexpr bool preserve_zero = true;
	static constexpr bool has_fma = false;
	static constexpr bool has_nfma = false;

	using WithoutScalar = One;

	static constexpr i64 estimated_register_cost_base = 0;
	static constexpr i64 estimated_register_cost_per_chain = 1;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {}
	template <i64 N>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto read_pack() const noexcept -> _simd::sized_pack_type<Element, N> {
		return _simd::sized_pack_traits<Element, N>::broadcast(s);
	}
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto extract_scalar() const noexcept -> Element { return s; }
};

template <typename T>
struct BroadcastExpr {
	using Impl = BroadcastExprImpl<T>;
	Impl impl;

	static constexpr bool flip_sign = false;
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> BroadcastExpr { return {{-impl.s}}; }
};

template <typename T>
struct is_broadcast : veg::meta::false_type {};
template <typename T>
struct is_broadcast<BroadcastExprImpl<T>> : veg::meta::true_type {};

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_SCAL_MUL_HPP_PPMZYT37S */
