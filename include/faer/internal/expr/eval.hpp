#ifndef FAER_EVAL_HPP_XB0DQFJNS
#define FAER_EVAL_HPP_XB0DQFJNS

#include "faer/internal/expr/cursor.hpp"
#include "faer/internal/expr/add.hpp"
#include "faer/internal/expr/mul.hpp"
#include "faer/internal/expr/broadcast.hpp"
#include "faer/internal/expr/reader.hpp"
#include "faer/internal/expr/cast.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename ImplI>
struct ApplySignExpr {
	struct Impl {
		ImplI impli;

		using Element = typename ImplI::Element;
		static constexpr i64 packet_size = ImplI::packet_size;

		static constexpr bool preserve_zero = ImplI::preserve_zero;

		static constexpr i64 estimated_register_cost_base = ImplI::estimated_register_cost_base;
		static constexpr i64 estimated_register_cost_per_chain = ImplI::estimated_register_cost_per_chain;

		template <i64 N, advance_e A>
		HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
			impli.template advance<N, A>();
		}

		template <i64 N>
		HEDLEY_ALWAYS_INLINE constexpr auto read_pack() const noexcept -> _simd::sized_pack_type<Element, N> {
			return _simd::sized_pack_traits<Element, N>::neg(impli.template read_pack<N>());
		}
	};
	Impl impl;
};

template <typename Dest, typename Src>
HEDLEY_ALWAYS_INLINE
VEG_CPP14(constexpr) void evaluate_contiguous(i64 const n, ContiguousCursor<Dest>& dest, Src& src) {
	using T = typename Src::Impl::Element;

	veg::meta::conditional_t<Src::flip_sign, ApplySignExpr<Src>, Src> eval{src};

	auto& src_ = eval.impl;

	constexpr i64 packet_size = internal::max_(Src::Impl::packet_size, 1);
	constexpr i64 max_packet_size = FAER_PACK_SIZE / i64{sizeof(T)};

	constexpr i64 c0 = Src::Impl::estimated_register_cost_base;
	constexpr i64 c1 = Src::Impl::estimated_register_cost_per_chain;
	constexpr i64 n_chains = (FAER_N_REGISTERS - c0 - 2) / c1;

	FAER_STATIC_ASSERT_DEV(packet_size <= max_packet_size);

	i64 const n0 = internal::round_down(n, n_chains * packet_size);
	i64 const n1 = internal::round_down(n, packet_size);

	i64 i = 0;
	{
		using traits = _simd::sized_pack_traits<T, packet_size>;
		using Pack = typename traits::Pack;

		Pack packs[n_chains];
		for (; i < n0; i += packet_size * n_chains) {
			for (i64 k = 0; k < n_chains; ++k) {
				packs[k] = src_.template read_pack<i64{traits::size}>();
			}
			for (i64 k = 0; k < n_chains; ++k) {
				traits::store(ptr::incr(dest.ptr, k * i64{traits::size}), packs[k]);
			}
			for (i64 k = 0; k < n_chains; ++k) {
				src_.template advance<i64{traits::size}, advance_e::inner>();
				dest.template advance<i64{traits::size}, advance_e::inner>();
			}
		}
	}

	{
		using traits = _simd::sized_pack_traits<T, packet_size>;
		using Pack = typename traits::Pack;

		Pack pack;
		for (; i < n1; i += packet_size) {
			pack = src_.template read_pack<i64{traits::size}>();
			traits::store(dest.ptr, pack);
			src_.template advance<i64{traits::size}, advance_e::inner>();
			dest.template advance<i64{traits::size}, advance_e::inner>();
		}
	}

#if FAER_HAS_HALF_PACK

	FAER_IF(max_packet_size / 2 < packet_size) {
		i64 const n2 = internal::round_down(n, max_packet_size / 2);
		using traits = _simd::pack_half_traits<T>;
		using Pack = typename traits::Pack;

		Pack pack;
		for (; i < n2; i += max_packet_size / 2) {
			pack = src_.template read_pack<i64{traits::size}>();
			traits::store(dest.ptr, pack);
			src_.template advance<i64{traits::size}, advance_e::inner>();
			dest.template advance<i64{traits::size}, advance_e::inner>();
		}
	}
#endif

#if FAER_HAS_QUARTER_PACK

	FAER_IF(max_packet_size / 4 < packet_size) {
		i64 const n3 = internal::round_down(n, max_packet_size / 4);
		using traits = _simd::pack_quarter_traits<T>;
		using Pack = typename traits::Pack;

		Pack pack;
		for (; i < n3; i += max_packet_size / 4) {
			pack = src_.template read_pack<i64{traits::size}>();
			traits::store(dest.ptr, pack);
			src_.template advance<i64{traits::size}, advance_e::inner>();
			dest.template advance<i64{traits::size}, advance_e::inner>();
		}
	}
#endif

	{
		using traits = _simd::sized_pack_traits<T, 1>;
		using Pack = typename traits::Pack;

		Pack pack;
		for (; i < n; ++i) {
			pack = src_.template read_pack<i64{traits::size}>();
			traits::store(dest.ptr, pack);
			src_.template advance<i64{traits::size}, advance_e::inner>();
			dest.template advance<i64{traits::size}, advance_e::inner>();
		}
	}
}

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_EVAL_HPP_XB0DQFJNS */
