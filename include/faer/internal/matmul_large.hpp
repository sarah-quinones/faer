#ifndef FAER_MATMUL_LARGE_HPP_9SB10RHQS
#define FAER_MATMUL_LARGE_HPP_9SB10RHQS

#include "faer/internal/matmul_kernels.hpp"
#include "faer/internal/pack_operands.hpp"
#include "faer/internal/cache.hpp"

#include <veg/memory/dynamic_stack.hpp>

namespace fae {
namespace _detail {

template <usize N, usize MR, usize NR, typename T>
auto matmul_large_vectorized_req( //
		veg::Tag<T> /*tag*/,
		usize /*m*/,
		usize /*n*/,
		usize /*k*/) noexcept -> veg::dynstack::StackReq {

	KernelParams params = kernel_params<MR, NR, sizeof(T)>();
	auto kc = params.kc;
	auto mc = params.mc;
	auto nc = params.nc;
	isize simd_alignment = isize(N * sizeof(T));
	return veg::dynstack::StackReq{(1U << 25U) + isize(mc) * isize(kc) * isize{sizeof(T)}, simd_alignment} &
	       veg::dynstack::StackReq{isize(nc) * isize(kc) * isize{sizeof(T)}, simd_alignment};
}

// dest is colmajor
template <Order LHS, Order RHS, usize N, usize MR, usize NR, typename T>
FAER_MINSIZE void matmul_large_vectorized( //
		usize m,
		usize n,
		usize k,
		T* dest,
		T const* lhs,
		T const* rhs,
		isize dest_stride,
		isize lhs_stride,
		isize rhs_stride,
		veg::DoNotDeduce<T> factor,
		veg::dynstack::DynStackMut stack) noexcept {
	KernelParams params = kernel_params<MR, NR, sizeof(T)>();
	auto kc = params.kc;
	auto mc = params.mc;
	auto nc = params.nc;
	VEG_ASSERT(mc % MR == 0);
	VEG_ASSERT(nc % NR == 0);

	VEG_ASSERT(m % MR == 0);
	VEG_ASSERT(n % NR == 0);

	simd::Pack<T, N> factor_pack;
	factor_pack.broadcast(veg::mem::addressof(factor));

	isize simd_alignment = isize(N * sizeof(T));
	auto packed_rhs_stride = isize(kc * NR);
	auto packed_lhs_stride = isize(kc * MR);

	auto _packed_rhs =
			stack.make_new_for_overwrite(veg::Tag<T>{}, packed_rhs_stride * isize(nc / NR), simd_alignment).unwrap();
	auto _packed_lhs =
			stack.make_new_for_overwrite(veg::Tag<T>{}, packed_lhs_stride * isize(mc / MR), simd_alignment).unwrap();

	usize col_outer = 0;
	FAER_NO_UNROLL
	while (col_outer != n) {
		usize n_chunk = _detail::min2(nc, n - col_outer);
		usize depth_outer = 0;
		FAER_NO_UNROLL
		while (depth_outer != k) {
			usize k_chunk = _detail::min2(kc, k - depth_outer);

			auto packed_rhs = _packed_rhs.ptr_mut();

			PackOperand<false, RHS == Order::COLMAJOR>::template fn<N, NR, T>( //
					packed_rhs,
					MemAccess<RHS>::fn(rhs, depth_outer, col_outer, rhs_stride),
					rhs_stride,
					packed_rhs_stride,
					n_chunk,
					k_chunk);

			usize row_outer = 0;
			FAER_NO_UNROLL
			while (row_outer != m) {
				usize m_chunk = min2(mc, m - row_outer);

				auto packed_lhs = _packed_lhs.ptr_mut();
				PackOperand<true, LHS == Order::COLMAJOR>::template fn<N, MR, T>( //
						packed_lhs,
						MemAccess<LHS>::fn(lhs, row_outer, depth_outer, lhs_stride),
						lhs_stride,
						packed_lhs_stride,
						m_chunk,
						k_chunk);

				for (usize col_inner = 0; col_inner < n_chunk; col_inner += NR) {
					for (usize row_inner = 0; row_inner < m_chunk; row_inner += MR) {
						simd::packed_inner_kernel<MR, NR, N>(
								MemAccess<Order::COLMAJOR>::fn(dest, row_outer + row_inner, col_outer + col_inner, dest_stride),
								dest_stride * isize{sizeof(T)},
								packed_lhs + row_inner * kc,
								MR * isize{sizeof(T)},
								packed_rhs + col_inner * kc,
								NR * isize{sizeof(T)},
								k_chunk,
								reinterpret_cast<T const*>(veg::mem::addressof(factor_pack)));
					}
				}

				row_outer += m_chunk;
			}
			depth_outer += k_chunk;
		}

		col_outer += n_chunk;
	}
}
} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_MATMUL_LARGE_HPP_9SB10RHQS */
