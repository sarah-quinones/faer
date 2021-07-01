#ifndef FAER_MAT_MUL_SMALL_HPP_Q5ZPJKE2S
#define FAER_MAT_MUL_SMALL_HPP_Q5ZPJKE2S

#include "faer/internal/simd.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace _matmul_smol {

template <typename PackTraits, i64 N>
HEDLEY_ALWAYS_INLINE void matmul_c_N_pack(
		i64 const i,
		typename PackTraits::Type const alpha,
		int const use_alpha, // 0: use alpha, 1: add mul, -1 sub mul
		typename PackTraits::Type* const HEDLEY_RESTRICT dst,
		i64 const dst_outer,
		typename PackTraits::Type const* const lhs,
		bool const lhs_contig,
		i64 const lhs_col_stride,
		i64 const lhs_row_stride,
		typename PackTraits::Type const* const rhs,
		i64 const rhs_col_stride,
		i64 const rhs_row_stride,
		i64 const /*rows*/,
		i64 const cols) noexcept {

	using pack_traits = PackTraits;
	using Pack = typename pack_traits::Pack;
	using T = typename pack_traits::Type;

	Pack l[usize{N}];
	for (i64 idx = 0; idx < N; ++idx) {
		l[idx] = pack_traits::maybe_gather(
				lhs_contig, ptr::incr(lhs, lhs_row_stride * i + lhs_col_stride * idx), lhs_row_stride);
	}

	for (i64 j = 0; j < cols; ++j) {
		Pack c = pack_traits::zero();

		for (i64 k = 0; k < N; ++k) {
			Pack r = pack_traits::broadcast(ptr::read(ptr::incr(rhs, rhs_col_stride * j + k * rhs_row_stride)));
			c = pack_traits::fmadd(l[k], r, c);
		}

		T* const addr = ptr::incr(dst, i + j * dst_outer);
		pack_traits::store(
				addr, pack_traits::maybe_fmadd(use_alpha, pack_traits::broadcast(alpha), c, pack_traits::load(addr)));
	}
}

// dst and lhs are col major
template <i64 N, typename T>
HEDLEY_ALWAYS_INLINE void matmul_c_N(
		T const alpha,
		int const use_alpha, // 0: use alpha, 1: add mul, -1 sub mul
		T* const HEDLEY_RESTRICT dst,
		i64 const dst_outer,
		T const* const lhs,
		bool const lhs_contig,
		i64 const lhs_col_stride,
		i64 const lhs_row_stride,
		T const* const rhs,
		i64 const rhs_col_stride,
		i64 const rhs_row_stride,
		i64 const rows,
		i64 const cols) noexcept {

	using pack_traits = _simd::pack_traits<T>;
	constexpr i64 pack_size = pack_traits::size;
	i64 const rp = rows / pack_size * pack_size;
	i64 const rem = rows - rp;

	i64 i = 0;
	for (; i < rp; i += pack_size) {
		_matmul_smol::matmul_c_N_pack<pack_traits, N>(
				i,
				alpha,
				use_alpha,
				dst,
				dst_outer,
				lhs,
				lhs_contig,
				lhs_col_stride,
				lhs_row_stride,
				rhs,
				rhs_col_stride,
				rhs_row_stride,
				rows,
				cols);
	}

#if FAER_HAS_HALF_PACK
	if (rem >= pack_size / 2) {
		_matmul_smol::matmul_c_N_pack<_simd::pack_half_traits<T>, N>(
				i,
				alpha,
				use_alpha,
				dst,
				dst_outer,
				lhs,
				lhs_contig,
				lhs_col_stride,
				lhs_row_stride,
				rhs,
				rhs_col_stride,
				rhs_row_stride,
				rows,
				cols);
		i += pack_size / 2;
	}
#endif

	for (; i < rows; ++i) {
		_matmul_smol::matmul_c_N_pack<_simd::sized_pack_traits<T, 1>, N>(
				i,
				alpha,
				use_alpha,
				dst,
				dst_outer,
				lhs,
				lhs_contig,
				lhs_col_stride,
				lhs_row_stride,
				rhs,
				rhs_col_stride,
				rhs_row_stride,
				rows,
				cols);
	}
}

template <typename T>
struct MatMulParams {
	T alpha;
	int use_alpha;
	T* HEDLEY_RESTRICT dst;
	i64 const dst_outer;
	T const* lhs;
	bool const lhs_contig;
	i64 const lhs_col_stride;
	i64 const lhs_row_stride;
	T const* rhs;
	i64 const rhs_col_stride;
	i64 const rhs_row_stride;
	i64 const rows;
	i64 const cols;

	i64 const& d;

	template <i64 N>
	HEDLEY_ALWAYS_INLINE void apply() const noexcept {
		_matmul_smol::matmul_c_N<N>(
				alpha,
				use_alpha,
				dst,
				dst_outer,
				lhs + lhs_col_stride * d,
				lhs_contig,
				lhs_col_stride,
				lhs_row_stride,
				rhs + rhs_row_stride * d,
				rhs_col_stride,
				rhs_row_stride,
				rows,
				cols);
	}
};

template <typename T>
HEDLEY_ALWAYS_INLINE void matmul_c(
		T alpha,
		int use_alpha,
		T* HEDLEY_RESTRICT dst,
		i64 const dst_outer,
		T const* lhs,
		bool const lhs_contig,
		i64 const lhs_col_stride,
		i64 const lhs_row_stride,
		T const* rhs,
		i64 const rhs_col_stride,
		i64 const rhs_row_stride,
		i64 const rows,
		i64 const cols,
		i64 depth) noexcept {

	i64 const d8 = depth / 8 * 8;
	i64 const rem = depth - d8;

	i64 d = 0;

	MatMulParams<T> matmul_fn{
			alpha,
			use_alpha,
			dst,
			dst_outer,
			lhs + lhs_col_stride * d,
			lhs_contig,
			lhs_col_stride,
			lhs_row_stride,
			rhs + rhs_row_stride * d,
			rhs_col_stride,
			rhs_row_stride,
			rows,
			cols,
			d,
	};

	for (; d < d8; d += 8) {
		matmul_fn.template apply<8>();
	}
	switch (rem) {
	case 7:
		matmul_fn.template apply<7>();
		return;
	case 6:
		matmul_fn.template apply<6>();
		return;
	case 5:
		matmul_fn.template apply<5>();
		return;
	case 4:
		matmul_fn.template apply<4>();
		return;
	case 3:
		matmul_fn.template apply<3>();
		return;
	case 2:
		matmul_fn.template apply<2>();
		return;
	case 1:
		matmul_fn.template apply<1>();
	default:
		return;
	}
}

} // namespace _matmul_smol
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_MAT_MUL_SMALL_HPP_Q5ZPJKE2S */
