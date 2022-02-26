#ifndef FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS
#define FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS

#include "faer/internal/simd.hpp"
#include "faer/internal/enums.hpp"
#include "faer/internal/helpers.hpp"

namespace fae {
namespace _detail {

namespace _matvec {
template <Order>
struct MatVecVectorizedImpl;

template <usize N, typename T>
struct Broadcast {
	simd::Pack<T, N>* rhs_packs;
	T const* rhs;
	usize depth_outer;

	VEG_INLINE void operator()(usize iter) const noexcept { rhs_packs[iter].broadcast(rhs + depth_outer + iter); }
};

template <usize N, typename T>
struct CastUnit {
	simd::Pack<T, N> const* rhs_packs;
	simd::Pack<T, 1>* rhs_packs_unit;

	VEG_INLINE void operator()(usize iter) const noexcept { rhs_packs_unit[iter] = rhs_packs[iter].cast_unit(); }
};

template <usize N, typename T>
struct LoadLhs {
	simd::Pack<T, N>* lhs_packs;
	usize depth_outer;
	T const* lhs_plus_row;
	isize lhs_stride_bytes;
	VEG_INLINE void operator()(usize iter) const noexcept {
		lhs_packs[iter].load_unaligned(_detail::incr( //
				lhs_plus_row,
				isize(depth_outer + iter) * lhs_stride_bytes));
	}
};

template <usize N, typename T>
struct LhsMulRhs {
	simd::Pack<T, N>* lhs_packs;
	simd::Pack<T, N> const* rhs_packs;
	VEG_INLINE void operator()(usize iter) const noexcept { lhs_packs[iter].mul(lhs_packs[iter], rhs_packs[iter]); }
};

template <>
struct MatVecVectorizedImpl<Order::COLMAJOR> {
	template <usize N, typename T>
	static void fn(usize m, usize k, T* dest, T const* lhs, T const* rhs, isize lhs_stride) noexcept {
		usize k_unroll = k / 4 * 4;

		constexpr usize simd_alignment = sizeof(T) * N;
		std::uintptr_t first_aligned_row =
				(((std::uintptr_t(dest) + (simd_alignment - 1)) / simd_alignment * simd_alignment) - std::uintptr_t(dest)) /
				sizeof(T);
		std::uintptr_t last_aligned_row =
				((std::uintptr_t(dest + m) / simd_alignment * simd_alignment) - std::uintptr_t(dest)) / sizeof(T);

		if (first_aligned_row >= last_aligned_row) {
			first_aligned_row = 0;
			last_aligned_row = 0;
		}

		isize lhs_stride_bytes = lhs_stride * isize{sizeof(T)};

		usize depth = 0;

		FAER_NO_UNROLL
		for (; depth < k_unroll; depth += 4) {
			simd::Pack<T, N> rhs_packs[4];
			simd::Pack<T, 1> rhs_packs_unit[4];

			_detail::unroll<4>(Broadcast<N, T>{rhs_packs, rhs, depth});
			_detail::unroll<4>(CastUnit<N, T>{rhs_packs, rhs_packs_unit});

			FAER_NO_UNROLL
			for (usize row = 0; row < first_aligned_row; ++row) {
				simd::Pack<T, 1> lhs_packs[4];
				simd::Pack<T, 1> dest_pack;

				dest_pack.load_aligned(dest + row);
				_detail::unroll<4>(LoadLhs<1, T>{lhs_packs, depth, lhs + row, lhs_stride_bytes});
				_detail::unroll<4>(LhsMulRhs<1, T>{lhs_packs, rhs_packs_unit});

				lhs_packs[0].add(lhs_packs[0], lhs_packs[1]);
				lhs_packs[2].add(lhs_packs[2], lhs_packs[3]);
				lhs_packs[0].add(lhs_packs[0], lhs_packs[2]);
				dest_pack.add(dest_pack, lhs_packs[0]);
				dest_pack.store_aligned(dest + row);
			}

			FAER_NO_UNROLL
			for (usize row = first_aligned_row; row < last_aligned_row; row += N) {
				simd::Pack<T, N> lhs_packs[4];
				simd::Pack<T, N> dest_pack;

				dest_pack.load_aligned(dest + row);
				_detail::unroll<4>(LoadLhs<N, T>{lhs_packs, depth, lhs + row, lhs_stride_bytes});
				_detail::unroll<4>(LhsMulRhs<N, T>{lhs_packs, rhs_packs});

				lhs_packs[0].add(lhs_packs[0], lhs_packs[1]);
				lhs_packs[2].add(lhs_packs[2], lhs_packs[3]);
				lhs_packs[0].add(lhs_packs[0], lhs_packs[2]);
				dest_pack.add(dest_pack, lhs_packs[0]);
				dest_pack.store_aligned(dest + row);
			}

			FAER_NO_UNROLL
			for (usize row = last_aligned_row; row < m; ++row) {
				simd::Pack<T, 1> lhs_packs[4];
				simd::Pack<T, 1> dest_pack;

				dest_pack.load_aligned(dest + row);
				_detail::unroll<4>(LoadLhs<1, T>{lhs_packs, depth, lhs + row, lhs_stride_bytes});
				_detail::unroll<4>(LhsMulRhs<1, T>{lhs_packs, rhs_packs_unit});

				lhs_packs[0].add(lhs_packs[0], lhs_packs[1]);
				lhs_packs[2].add(lhs_packs[2], lhs_packs[3]);
				lhs_packs[0].add(lhs_packs[0], lhs_packs[2]);
				dest_pack.add(dest_pack, lhs_packs[0]);
				dest_pack.store_aligned(dest + row);
			}
		}

		FAER_NO_UNROLL
		for (; depth < k; ++depth) {
			simd::Pack<T, N> rhs_packs[1];
			simd::Pack<T, 1> rhs_packs_unit[1];

			Broadcast<N, T>{rhs_packs, rhs, depth}(0);
			CastUnit<N, T>{rhs_packs, rhs_packs_unit}(0);

			FAER_NO_UNROLL
			for (usize row = 0; row < first_aligned_row; ++row) {
				simd::Pack<T, 1> lhs_packs[1];
				simd::Pack<T, 1> dest_pack;

				dest_pack.load_aligned(dest + row);
				lhs_packs[0].load_unaligned(_detail::incr(lhs + row, isize(depth) * lhs_stride_bytes));
				dest_pack.fmadd(simd::Pos{}, simd::Pos{}, lhs_packs[0], rhs_packs_unit[0], dest_pack);
				dest_pack.store_aligned(dest + row);
			}

			FAER_NO_UNROLL
			for (usize row = first_aligned_row; row < last_aligned_row; row += N) {
				simd::Pack<T, N> lhs_packs[1];
				simd::Pack<T, N> dest_pack;

				dest_pack.load_aligned(dest + row);
				lhs_packs[0].load_unaligned(_detail::incr(lhs + row, isize(depth) * lhs_stride_bytes));

				dest_pack.fmadd(simd::Pos{}, simd::Pos{}, lhs_packs[0], rhs_packs[0], dest_pack);
				dest_pack.store_aligned(dest + row);
			}

			FAER_NO_UNROLL
			for (usize row = last_aligned_row; row < m; ++row) {
				simd::Pack<T, 1> lhs_packs[1];
				simd::Pack<T, 1> dest_pack;

				dest_pack.load_aligned(dest + row);
				lhs_packs[0].load_unaligned(_detail::incr(lhs + row, isize(depth) * lhs_stride_bytes));
				dest_pack.fmadd(simd::Pos{}, simd::Pos{}, lhs_packs[0], rhs_packs_unit[0], dest_pack);
				dest_pack.store_aligned(dest + row);
			}
		}
	}
};
} // namespace _matvec

template <Order LHS, usize N, typename T>
void matvec_large_vectorized( //
		usize m,
		usize k,
		T* dest,
		T const* lhs,
		T const* rhs,
		isize lhs_stride) noexcept {
	_matvec::MatVecVectorizedImpl<LHS>::template fn<N, T>(m, k, dest, lhs, rhs, lhs_stride);
}

} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS */
