#ifndef FAER_READER_HPP_AJHHWEY4S
#define FAER_READER_HPP_AJHHWEY4S

#include "faer/internal/expr/core.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename Cursor>
struct ReadExprImpl {
	Cursor cursor;

	using Element = typename Cursor::Element;
	static constexpr i64 packet_size = FAER_PACK_SIZE / 8 / i64{sizeof(Element)};

	static constexpr bool preserve_zero = true;
	static constexpr bool has_fma = false;
	static constexpr bool has_nfma = false;

	using WithoutScalar = void;

	static constexpr i64 estimated_register_cost_base = 0;
	static constexpr i64 estimated_register_cost_per_chain = 1;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		cursor.template advance<N, A>();
	}
	template <i64 N>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto read_pack() const noexcept ->
			typename _simd::sized_pack_type<Element, N>::Pack {
		return _simd::sized_pack_traits<Element, N>::load(cursor.ptr);
	}
};

template <typename Cursor, bool FlipSign>
struct ReadExpr {
	using Impl = ReadExprImpl<Cursor>;
	Impl impl;

	static constexpr bool flip_sign = FlipSign;
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) auto negate() const noexcept -> ReadExpr<Cursor, !FlipSign> {
		return {impl};
	}
};

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_READER_HPP_AJHHWEY4S */
