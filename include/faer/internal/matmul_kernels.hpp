#ifndef FAER_MATMUL_KERNELS_HPP_YVPVWNUCS
#define FAER_MATMUL_KERNELS_HPP_YVPVWNUCS

#include "faer/internal/simd.hpp"
#include "faer/internal/helpers.hpp"
#include <veg/type_traits/core.hpp>

namespace fae {
namespace _detail {
namespace _kernel {
template <typename T, usize N>
struct KernelIterLoadLhs {
	T const* packed_lhs_iter;
	simd::Pack<T, N>* lhs;
	VEG_INLINE void operator()(usize iter) const noexcept { lhs[iter].load_aligned(packed_lhs_iter + iter * N); }
};
template <typename T, usize N>
struct KernelIterFma {
	simd::Pack<T, N>* accum_iter;
	simd::Pack<T, N> const* lhs;
	simd::Pack<T, N> const* rhs;
	usize nr;
	VEG_INLINE void operator()(usize iter) const noexcept {
		accum_iter[iter * nr].fmadd( //
				simd::Pos{},
				simd::Pos{},
				lhs[iter],
				*rhs,
				accum_iter[iter * nr]);
	}
};
template <typename T, usize N, usize MR>
struct KernelIterLoadRhsFma {
	T const* packed_rhs_iter;
	simd::Pack<T, N>* accum;
	simd::Pack<T, N> const* lhs;
	simd::Pack<T, N>* rhs;
	usize nr;
	VEG_INLINE void operator()(usize iter) const noexcept {
		rhs->broadcast(packed_rhs_iter + iter);
		_detail::unroll<MR / N>(KernelIterFma<T, N>{accum + iter, lhs, rhs, nr});
	}
};

template <typename T, usize N, usize MR, usize NR>
struct KernelIter {
	T const* packed_lhs;
	isize lhs_stride_bytes;
	T const* packed_rhs;
	isize rhs_stride_bytes;
	simd::Pack<T, N>* accum;
	simd::Pack<T, N>* lhs;
	simd::Pack<T, N>* rhs;

	VEG_INLINE void operator()(usize iter) const noexcept {
		_detail::unroll<MR / N>(KernelIterLoadLhs<T, N>{
				_detail::incr(packed_lhs, isize(iter) * lhs_stride_bytes),
				lhs,
		});
		_detail::unroll<NR>(KernelIterLoadRhsFma<T, N, MR>{
				_detail::incr(packed_rhs, isize(iter) * rhs_stride_bytes),
				accum,
				lhs,
				rhs,
				NR,
		});
	}
};

template <typename T, usize N>
struct ZeroAccum {
	simd::Pack<T, N>* accum;
	VEG_INLINE void operator()(usize iter) const noexcept { accum[iter].zero(); }
};
struct Prefetch {
	void const* p;
	VEG_INLINE void operator()(usize iter) const noexcept {
		simde_mm_prefetch(static_cast<char const*>(p) + iter * FAER_CACHE_LINE_BYTES, SIMDE_MM_HINT_T0); // NOLINT
	}
};

template <typename T, usize N>
struct DestUpdateInner {
	simd::Pack<T, N> const* accum_iter;
	simd::Pack<T, N> const* factor_pack;
	T* dest_iter;
	isize dest_stride_bytes;

	VEG_INLINE void operator()(usize iter) const noexcept {
		T* dest_target = _detail::incr(dest_iter, isize(iter) * dest_stride_bytes);
		simd::Pack<T, N> tmp;
		tmp.load_unaligned(dest_target);
		tmp.fmadd(simd::Pos{}, simd::Pos{}, *factor_pack, accum_iter[iter], tmp);
		tmp.store_unaligned(dest_target);
	}
};

template <typename T, usize N, usize NR>
struct DestUpdateOuter {
	simd::Pack<T, N> const* accum;
	simd::Pack<T, N> const* factor_pack;
	T* dest;
	isize dest_stride_bytes;
	VEG_INLINE void operator()(usize iter) const noexcept {
		_detail::unroll<NR>(DestUpdateInner<T, N>{
				accum + NR * iter,
        factor_pack,
				dest + N * iter,
				dest_stride_bytes,
		});
	}
};
} // namespace _kernel
} // namespace _detail

namespace simd {
template <usize MR, usize NR, usize N, usize K_UNROLL = 4, typename T>
VEG_NO_INLINE void packed_inner_kernel(
		T* dest,
		isize dest_stride_bytes,
		T const* packed_lhs,
		isize lhs_stride_bytes,
		T const* packed_rhs,
		isize rhs_stride_bytes,
		usize k,
		T const* factor) noexcept {

	using Pack = simd::Pack<T, N>;

	Pack accum[NR * (MR / N)];
	_detail::unroll<NR*(MR / N)>(_detail::_kernel::ZeroAccum<T, N>{accum});

	Pack lhs[MR / N];
	Pack rhs;

	usize k_unroll = k / K_UNROLL;
	usize k_leftover = k % K_UNROLL;

	usize depth = k_unroll;
	if (depth != 0) {
		FAER_NO_UNROLL
		while (true) {
			_detail::unroll<K_UNROLL>(_detail::_kernel::KernelIter<T, N, MR, NR>{
					packed_lhs,
					lhs_stride_bytes,
					packed_rhs,
					rhs_stride_bytes,
					accum,
					lhs,
					&rhs,
			});

			packed_lhs = _detail::incr(packed_lhs, isize{K_UNROLL} * lhs_stride_bytes);
			packed_rhs = _detail::incr(packed_rhs, isize{K_UNROLL} * rhs_stride_bytes);

			--depth;
			if (depth == 0) {
				break;
			}
		}
	}

	if constexpr (K_UNROLL != 1) {
		depth = k_leftover;
		if (depth != 0) {
			FAER_NO_UNROLL
			while (true) {
				_detail::unroll<1>(_detail::_kernel::KernelIter<T, N, MR, NR>{
						packed_lhs,
						lhs_stride_bytes,
						packed_rhs,
						rhs_stride_bytes,
						accum,
						lhs,
						&rhs,
				});
				packed_lhs = _detail::incr(packed_lhs, lhs_stride_bytes);
				packed_rhs = _detail::incr(packed_rhs, rhs_stride_bytes);

				--depth;
				if (depth == 0) {
					break;
				}
			}
		}
	}

	simd::Pack<T, N> factor_pack;
	factor_pack.load_unaligned(factor);
	_detail::unroll<MR / N>(_detail::_kernel::DestUpdateOuter<T, N, NR>{
			accum,
			veg::mem::addressof(factor_pack),
			dest,
			dest_stride_bytes,
	});
}
} // namespace simd
} // namespace fae

#endif /* end of include guard FAER_MATMUL_KERNELS_HPP_YVPVWNUCS */
