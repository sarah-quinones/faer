#ifndef FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS
#define FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS

#include "faer/internal/simd.hpp"
#include "faer/internal/enums.hpp"
#include "faer/internal/matmul_kernels.hpp"
#include "faer/internal/helpers.hpp"

namespace fae {
namespace _detail {

namespace _matvec {
template <Order>
struct MatVecVectorizedImpl;

template <>
struct MatVecVectorizedImpl<Order::COLMAJOR> {
	template <usize N, typename T>
	static void fn(usize m, usize k, T* dest, T const* lhs, T const* rhs, isize lhs_stride, T const factor) noexcept {
		constexpr usize simd_alignment = sizeof(T) * N;

		usize const aligned_start = _detail::min2(m, _detail::offset_to_align(dest, simd_alignment));
		usize const aligned_count = (m - aligned_start) / N * N;
		usize const aligned_end = aligned_start + aligned_count;

		simd::Pack<T, N> factor_pack;
		factor_pack.broadcast(veg::mem::addressof(factor));

		usize const depth_step = 4;
		isize const lhs_stride_bytes = lhs_stride * isize{sizeof(T)};

		for (usize depth_outer = 0; depth_outer < k;) {
			usize k_chunk = _detail::min2(depth_step, k - depth_outer);
			T const* lhs_ptr = _detail::incr(lhs, isize(depth_outer) * lhs_stride_bytes);
			T const* rhs_ptr = rhs + depth_outer;

			for (usize row = 0; row < aligned_start; ++row) {
				simd::packed_inner_kernel<1, 1, 1, 4, T>( //
						dest + row,
						0,
						lhs_ptr + row,
						lhs_stride_bytes,
						rhs_ptr,
						1 * isize{sizeof(T)},
						k_chunk,
						reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
			}

			{
				usize row = aligned_start;
				for (; row < aligned_start + _detail::round_down(aligned_count, 8 * N); row += 8 * N) {
					simd::packed_inner_kernel<8 * N, 1, N, 4, T>( //
							dest + row,
							0,
							lhs_ptr + row,
							lhs_stride_bytes,
							rhs_ptr,
							1 * isize{sizeof(T)},
							k_chunk,
							reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
				}
				if (row < aligned_start + _detail::round_down(aligned_count, 4 * N)) {
					simd::packed_inner_kernel<4 * N, 1, N, 4, T>( //
							dest + row,
							0,
							lhs_ptr + row,
							lhs_stride_bytes,
							rhs_ptr,
							1 * isize{sizeof(T)},
							k_chunk,
							reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
					row += 4 * N;
				}
				if (row < aligned_start + _detail::round_down(aligned_count, 2 * N)) {
					simd::packed_inner_kernel<2 * N, 1, N, 4, T>( //
							dest + row,
							0,
							lhs_ptr + row,
							lhs_stride_bytes,
							rhs_ptr,
							1 * isize{sizeof(T)},
							k_chunk,
							reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
					row += 2 * N;
				}
				if (row < aligned_end) {
					simd::packed_inner_kernel<N, 1, N, 4, T>( //
							dest + row,
							0,
							lhs_ptr + row,
							lhs_stride_bytes,
							rhs_ptr,
							1 * isize{sizeof(T)},
							k_chunk,
							reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
				}
			}

			for (usize row = aligned_end; row < m; ++row) {
				simd::packed_inner_kernel<1, 1, 1, 4, T>( //
						dest + row,
						0,
						lhs_ptr + row,
						lhs_stride_bytes,
						rhs_ptr,
						1 * isize{sizeof(T)},
						k_chunk,
						reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
			}

			depth_outer += k_chunk;
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
		isize lhs_stride,
		veg::DoNotDeduce<T> const factor) noexcept {
	_matvec::MatVecVectorizedImpl<LHS>::template fn<N, T>(m, k, dest, lhs, rhs, lhs_stride, factor);
}

} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_MATVEC_MUL_LARGE_HPP_A27FWQFKS */
