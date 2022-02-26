#ifndef FAER_PACK_OPERANDS_HPP_ZPP0EJKJS
#define FAER_PACK_OPERANDS_HPP_ZPP0EJKJS

#include "faer/internal/helpers.hpp"
#include "faer/internal/enums.hpp"

namespace fae {
namespace _detail {
template <typename T, usize N>
struct LoadStore {
	T* dest;
	T const* src;
	VEG_INLINE auto operator()(usize iter) const noexcept {
		simd::Pack<T, N> tmp;
		tmp.load_unaligned(src + iter * N);
		tmp.store_aligned(dest + iter * N);
	}
};

constexpr auto largest_pack_size_that_divides(usize n, usize max) noexcept -> usize {
	return usize{1} << _detail::min_u( //
						 _detail::count_trailing_zeros(n),
						 _detail::count_trailing_zeros(max));
}

template <typename T, usize N>
struct Load {
	simd::Pack<T, N>* tmp;
	T const* src_iter;
	isize src_stride_bytes;

	VEG_INLINE void operator()(usize iter) noexcept {
		tmp[iter].load_unaligned(_detail::incr(src_iter, isize(iter) * src_stride_bytes));
	}
};
template <typename T, usize N>
struct Store {
	simd::Pack<T, N>* tmp;
	T* dest_iter;
	usize width;

	VEG_INLINE void operator()(usize iter) noexcept { tmp[iter].store_aligned(dest_iter + iter * width); }
};
template <typename T, usize N>
struct PackTransIter {
	simd::Pack<T, N>* tmp;
	T const* src;
	isize src_stride_bytes;
	T* dest;
	usize width;

	VEG_INLINE void operator()(usize iter) noexcept {
		_detail::unroll<N>(Load<T, N>{
				tmp,
				_detail::incr(src, isize(iter * N) * (src_stride_bytes)),
				src_stride_bytes,
		});
		simd::Pack<T, N>::trans(tmp);
		_detail::unroll<N>(Store<T, N>{
				tmp,
				dest + iter * N,
				width,
		});
	}
};

template <usize N, usize WIDTH, typename T>
void pack_cis_inner_loop(T* dest, T const* src, isize src_stride_bytes, usize k) noexcept {

	FAER_NO_UNROLL
	for (usize i = 0; i < k; ++i) {
		_detail::unroll<WIDTH / N>(LoadStore<T, N>{dest, src});
		src = _detail::incr(src, src_stride_bytes);
		dest += WIDTH;
	}
}

template <usize ACTUAL_N, usize WIDTH, typename T>
void pack_trans_inner_loop_impl(T* dest, T const* src, isize src_stride_bytes, usize k) noexcept {
	usize k_unroll = k / ACTUAL_N;
	usize k_leftover = k % ACTUAL_N;

	usize i = k_unroll;
	if (i != 0) {
		simd::Pack<T, ACTUAL_N> tmp[ACTUAL_N];

		FAER_NO_UNROLL
		while (true) {
			_detail::unroll<WIDTH / ACTUAL_N>(PackTransIter<T, ACTUAL_N>{
					tmp,
					src,
					src_stride_bytes,
					dest,
					WIDTH,
			});
			src += ACTUAL_N;
			dest += ACTUAL_N * WIDTH;

			--i;
			if (i == 0) {
				break;
			}
		}
	}

	if constexpr (ACTUAL_N != 1) {
		i = k_leftover;
		if (i != 0) {
			simd::Pack<T, 1> tmp[1];

			FAER_NO_UNROLL
			while (true) {
				_detail::unroll<WIDTH>(PackTransIter<T, 1>{
						tmp,
						src,
						src_stride_bytes,
						dest,
						WIDTH,
				});
				++src;
				dest += WIDTH;

				--i;
				if (i == 0) {
					break;
				}
			}
		}
	}
}

template <usize N, usize WIDTH, typename T>
void pack_trans_inner_loop(T* dest, T const* src, isize src_stride_bytes, usize k) noexcept {
	constexpr usize ACTUAL_N = _detail::largest_pack_size_that_divides(WIDTH, N);
	_detail::pack_trans_inner_loop_impl<ACTUAL_N, WIDTH, T>(dest, src, src_stride_bytes, k);
}

template <usize N, usize WIDTH, typename T>
void pack_cis( //
		T* dest,
		T const* src,
		isize src_stride,
		isize dest_stride,
		usize m,
		usize k) noexcept {
	VEG_ASSERT(m % WIDTH == 0);

	isize src_stride_bytes = src_stride * isize{sizeof(T)};
	isize dest_stride_bytes = dest_stride * isize{sizeof(T)};

	usize i = 0;

	usize peeled_main_pack = m / WIDTH * WIDTH;
	FAER_NO_UNROLL
	for (; i < peeled_main_pack; i += WIDTH) {
		_detail::pack_cis_inner_loop<N, WIDTH, T>(dest, src, src_stride_bytes, k);
		src += WIDTH;
		dest = _detail::incr(dest, dest_stride_bytes);
	}
}

template <usize N, usize WIDTH, typename T>
void pack_trans( //
		T* dest,
		T const* src,
		isize src_stride,
		isize dest_stride,
		usize m,
		usize k) noexcept {
	VEG_ASSERT(m % WIDTH == 0);

	isize src_stride_bytes = src_stride * isize{sizeof(T)};
	isize dest_stride_bytes = dest_stride * isize{sizeof(T)};

	usize i = 0;

	FAER_NO_UNROLL
	for (; i < m; i += WIDTH) {
		_detail::pack_trans_inner_loop<N, WIDTH, T>(dest, src, src_stride_bytes, k);
		src = _detail::incr(src, isize{WIDTH} * src_stride_bytes);
		dest = _detail::incr(dest, dest_stride_bytes);
	}
}

template <bool IS_LHS, bool IS_COLMAJOR>
struct PackOperand {
	template <usize N, usize MR, typename T>
	static void fn( //
			T* dest,
			T const* src,
			isize src_stride,
			isize dest_stride,
			usize m,
			usize k) noexcept {
		_detail::pack_trans<N, MR, T>(dest, src, src_stride, dest_stride, m, k);
	}
};

template <bool BOTH_EQUAL>
struct PackOperand<BOTH_EQUAL, BOTH_EQUAL> {
	template <usize N, usize MR, typename T>
	static void fn( //
			T* dest,
			T const* src,
			isize src_stride,
			isize dest_stride,
			usize m,
			usize k) noexcept {
		_detail::pack_cis<N, MR, T>(dest, src, src_stride, dest_stride, m, k);
	}
};
} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_PACK_OPERANDS_HPP_ZPP0EJKJS */
