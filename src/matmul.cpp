#include "faer/internal/simd.hpp"
#include "faer/internal/mat_mul_real.hpp"

#include "query_cache.hpp"

#include <veg/util/timer.hpp>
#include <veg/tuple.hpp>
#include <veg/functional/curry.hpp>
#include <veg/util/assert.hpp>
#include <veg/internal/narrow.hpp>
#include <veg/functional/utils.hpp>
#include <veg/memory/placement.hpp>
#include <veg/memory/aligned_alloc.hpp>
#include <veg/functional/copy_fn.hpp>

#include <simde/simde-features.h>
#include <simde/x86/sse.h>

#include <iostream>
#include <algorithm>
#include <cmath>

#include <alloca.h>

#include "faer/internal/prologue.hpp"

#undef EIGEN_ALIGNED_ALLOCA

#ifdef SIMDE_ARCH_X86
#define ASM_COMMENT(X) __asm__("#" X) // NOLINT
#else
#define ASM_COMMENT(X)
#endif

namespace fae {
using namespace veg::literals;
using veg::i64;
namespace internal {
namespace _simd {
inline void prefetch(void const* mem) noexcept {
	simde_mm_prefetch(mem, SIMDE_MM_HINT_T0); // NOLINT
}
} // namespace _simd
} // namespace internal

namespace FAER_ABI_VERSION {

namespace _internal {
namespace _matmul {

namespace _simd = internal::_simd;
namespace ptr = internal::ptr;

// column major
template <typename T, bool Contiguous>
struct DataMapper;

template <typename T, bool Contiguous>
struct LinearMapper;

template <typename T>
struct DataMapper<T, true> {
	T* data;
	i64 stride;

	HEDLEY_ALWAYS_INLINE DataMapper(T* _data, i64 _stride, i64 /*_incr*/) noexcept : data{_data}, stride{_stride} {}

	VEG_NODISCARD auto ptr(i64 i, i64 j) const noexcept -> T* { return internal::ptr::incr(data, i + j * stride); }
	VEG_NODISCARD auto submapper(i64 i, i64 j) const noexcept -> DataMapper { return {ptr(i, j), stride, 1}; }
	VEG_NODISCARD auto linear_mapper(i64 i, i64 j) const noexcept -> LinearMapper<T, true> { return {ptr(i, j)}; }
};

template <typename T>
struct DataMapper<T, false> {
	T* data;
	i64 stride;
	i64 incr;

	DataMapper(T* _data, i64 _stride, i64 _incr) noexcept : data{_data}, stride{_stride}, incr{_incr} {}

	VEG_NODISCARD auto ptr(i64 i, i64 j) const noexcept -> T* { return internal::ptr::incr(data, i * incr + j * stride); }
	VEG_NODISCARD auto submapper(i64 i, i64 j) const noexcept -> DataMapper { return {ptr(i, j), stride, incr}; }
	VEG_NODISCARD auto linear_mapper(i64 i, i64 j) const noexcept -> LinearMapper<T, false> { return {ptr(i, j), incr}; }
};

template <typename T>
struct LinearMapper<T, true> {
	T* data;

	VEG_NODISCARD HEDLEY_ALWAYS_INLINE auto ptr(i64 i) const noexcept -> T* { return internal::ptr::incr(data, i); }
	VEG_NODISCARD HEDLEY_ALWAYS_INLINE auto load(i64 i) const noexcept -> _simd::Pack<T> {
		return _simd::pack_traits<T>::load(ptr(i));
	}
	HEDLEY_ALWAYS_INLINE void store(i64 i, _simd::Pack<T> p) const noexcept {
		return _simd::pack_traits<T>::store(ptr(i), p);
	}
};

template <typename T>
struct LinearMapper<T, false> {
	T* data;
	i64 incr;

	VEG_NODISCARD auto ptr(i64 i) const noexcept -> T* { return internal::ptr::incr(data, i * incr); }
	VEG_NODISCARD HEDLEY_ALWAYS_INLINE auto load(i64 i) const noexcept -> _simd::Pack<T> {
		return _simd::pack_traits<T>::gather(ptr(i), incr);
	}
	HEDLEY_ALWAYS_INLINE void store(i64 i, _simd::Pack<T> p) const noexcept {
		_simd::pack_traits<T>::scatter(ptr(i), p, incr);
	}
};

template <typename T>
struct matmul_traits {
	static constexpr unsigned nr = 4;
	static constexpr unsigned mr = 3 * fae::internal::_simd::pack_traits<T>::size;
	static constexpr int LhsProgress = _simd::pack_traits<T>::size;
	static constexpr int RhsProgress = 1;
};

// pack a block of the lhs
// The traversal is as follows (mr==4):
//   0  4  8 12 ...
//   1  5  9 13 ...
//   2  6 10 14 ...
//   3  7 11 15 ...
//
//  16 20 24 28 ...
//  17 21 25 29 ...
//  18 22 26 30 ...
//  19 23 27 31 ...
//
//  32 33 34 35 ...
//  36 36 38 39 ...
template <typename T>
HEDLEY_NEVER_INLINE void pack_lhs_colmajor(T* blockA, T const* lhs, i64 depth, i64 rows, i64 stride) {
	using Pack = _simd::Pack<T>;
	using pack_traits = _simd::pack_traits<T>;
	constexpr int PacketSize = pack_traits::size;
	constexpr int Pack1 = matmul_traits<T>::mr;
	constexpr int Pack2 = matmul_traits<T>::LhsProgress;

	ASM_COMMENT("PRODUCT PACK LHS");
	i64 count = 0;

	i64 const peeled_mc3 = Pack1 >= 3 * PacketSize ? (internal::round_down)(rows, 3 * PacketSize) : 0;
	i64 const peeled_mc2 =
			Pack1 >= 2 * PacketSize ? peeled_mc3 + (internal::round_down)(rows - peeled_mc3, 2 * PacketSize) : 0;
	i64 const peeled_mc1 = Pack1 >= 1 * PacketSize ? (internal::round_down)(rows, 1 * PacketSize) : 0;
	i64 const peeled_mc0 = Pack2 >= 1 * PacketSize ? peeled_mc1 : Pack2 > 1 ? (internal::round_down)(rows, Pack2) : 0;

	i64 i = 0;

	// Pack 3 packets
	if (Pack1 >= 3 * PacketSize) {
		for (; i < peeled_mc3; i += 3 * PacketSize) {
			for (i64 k = 0; k < depth; k++) {
				Pack A = pack_traits::load(ptr::incr(lhs, i + 0 * PacketSize + k * stride));
				Pack B = pack_traits::load(ptr::incr(lhs, i + 1 * PacketSize + k * stride));
				Pack C = pack_traits::load(ptr::incr(lhs, i + 2 * PacketSize + k * stride));
				pack_traits::store(ptr::incr(blockA, count + 0 * PacketSize), A);
				pack_traits::store(ptr::incr(blockA, count + 1 * PacketSize), B);
				pack_traits::store(ptr::incr(blockA, count + 2 * PacketSize), C);
				count += 3 * PacketSize;
			}
		}
	}
	// Pack 2 packets
	if (Pack1 >= 2 * PacketSize) {
		for (; i < peeled_mc2; i += 2 * PacketSize) {
			for (i64 k = 0; k < depth; k++) {
				Pack A = pack_traits::load(ptr::incr(lhs, i + 0 * PacketSize + k * stride));
				Pack B = pack_traits::load(ptr::incr(lhs, i + 1 * PacketSize + k * stride));
				pack_traits::store(ptr::incr(blockA, count + 0 * PacketSize), A);
				pack_traits::store(ptr::incr(blockA, count + 1 * PacketSize), B);
				count += 2 * PacketSize;
			}
		}
	}
	// Pack 1 packets
	if (Pack1 >= 1 * PacketSize) {
		for (; i < peeled_mc1; i += 1 * PacketSize) {
			for (i64 k = 0; k < depth; k++) {
				Pack A = pack_traits::load(ptr::incr(lhs, i + 0 * PacketSize + k * stride));
				pack_traits::store(ptr::incr(blockA, count + 0 * PacketSize), A);
				count += PacketSize;
			}
		}
	}
	// Pack scalars
	if (Pack2 < PacketSize && Pack2 > 1) {
		for (; i < peeled_mc0; i += Pack2) {
			for (i64 k = 0; k < depth; k++) {
				for (i64 w = 0; w < Pack2; w++) {
					ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, i + w + k * stride)));
				}
			}
		}
	}
	for (; i < rows; i++) {
		for (i64 k = 0; k < depth; k++) {
			ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, i + k * stride)));
		}
	}
}

template <typename T>
HEDLEY_NEVER_INLINE void pack_lhs_rowmajor(T* blockA, T const* lhs, i64 depth, i64 rows, i64 stride) {
	using Pack = _simd::Pack<T>;
	using pack_traits = _simd::pack_traits<T>;
	constexpr int PacketSize = pack_traits::size;
	constexpr int Pack1 = matmul_traits<T>::mr;
	constexpr int Pack2 = matmul_traits<T>::LhsProgress;

	ASM_COMMENT("PRODUCT PACK LHS");
	i64 count = 0;

	int pack = Pack1;
	i64 i = 0;
	while (pack > 0) {
		i64 remaining_rows = rows - i;
		i64 peeled_mc = i + (remaining_rows / pack) * pack;
		for (; i < peeled_mc; i += pack) {
			i64 const peeled_k = (depth / PacketSize) * PacketSize;
			i64 k = 0;
			if (pack >= PacketSize) {
				for (; k < peeled_k; k += PacketSize) {
					for (i64 m = 0; m < pack; m += PacketSize) {

						Pack kernel[usize(PacketSize)];

						for (int p = 0; p < PacketSize; ++p) {
							kernel[p] = // NOLINT
									pack_traits::load(ptr::incr(lhs, stride * (i + p + m) + k));
						}

						pack_traits::transpose(internal::_simd::NumTag<sizeof(kernel) / sizeof(kernel[0])>{}, kernel);

						for (int p = 0; p < PacketSize; ++p) {
							pack_traits::store(blockA + count + m + pack * p, kernel[p]); // NOLINT
						}
					}
					count += PacketSize * pack;
				}
			}
			for (; k < depth; k++) {
				i64 w = 0;
				for (; w < pack - 3; w += 4) {
					ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * (i + w + 0) + k)));
					ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * (i + w + 1) + k)));
					ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * (i + w + 2) + k)));
					ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * (i + w + 3) + k)));
				}
				if ((pack % 4) != 0) {
					for (; w < pack; ++w) {
						ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * (i + w) + k)));
					}
				}
			}
		}

		pack -= PacketSize;
		if (pack < Pack2 && (pack + PacketSize) != Pack2) {
			pack = Pack2;
		}
	}

	for (; i < rows; i++) {
		for (i64 k = 0; k < depth; k++) {
			ptr::write(ptr::incr(blockA, count++), ptr::read(ptr::incr(lhs, stride * i + k)));
		}
	}
}

// copy a complete panel of the rhs
// this version is optimized for column major matrices
// The traversal order is as follow: (nr==4):
//  0  1  2  3   12 13 14 15   24 25 26
//  4  5  6  7   16 17 18 19   27 28 29
//  8  9 10 11   20 21 22 23   30 31 32
template <typename T>
HEDLEY_NEVER_INLINE void pack_rhs_colmajor(T* blockB, T const* rhs, i64 depth, i64 cols, i64 stride) {
	static_assert(matmul_traits<T>::nr == 4, ".");

	using Pack = _simd::Pack<T>;
	using pack_traits = _simd::pack_traits<T>;
	constexpr int PacketSize = pack_traits::size;

	ASM_COMMENT("PRODUCT PACK RHS COLMAJOR");
	i64 packet_cols4 = matmul_traits<T>::nr >= 4 ? (cols / 4) * 4 : 0;
	i64 count = 0;
	i64 const peeled_k = (depth / PacketSize) * PacketSize;

	for (i64 j2 = 0; j2 < packet_cols4; j2 += 4) {
		// skip what we have before
		T const* dm0 = ptr::incr(rhs, 0 + (j2 + 0) * stride);
		T const* dm1 = ptr::incr(rhs, 0 + (j2 + 1) * stride);
		T const* dm2 = ptr::incr(rhs, 0 + (j2 + 2) * stride);
		T const* dm3 = ptr::incr(rhs, 0 + (j2 + 3) * stride);

		i64 k = 0;
		if ((PacketSize % 4) == 0) {
			for (; k < peeled_k; k += PacketSize) {
				Pack kernel[(PacketSize % 4) == 0 ? 4 : usize(PacketSize)];

				kernel[0] = pack_traits::load(ptr::incr(dm0, k));
				kernel[1 % PacketSize] = pack_traits::load(ptr::incr(dm1, k));
				kernel[2 % PacketSize] = pack_traits::load(ptr::incr(dm2, k));
				kernel[3 % PacketSize] = pack_traits::load(ptr::incr(dm3, k));

				pack_traits::transpose(internal::_simd::NumTag<sizeof(kernel) / sizeof(kernel[0])>{}, kernel);

				pack_traits::store(ptr::incr(blockB, count + 0 * PacketSize), kernel[0]);
				pack_traits::store(ptr::incr(blockB, count + 1 * PacketSize), kernel[1 % PacketSize]);
				pack_traits::store(ptr::incr(blockB, count + 2 * PacketSize), kernel[2 % PacketSize]);
				pack_traits::store(ptr::incr(blockB, count + 3 * PacketSize), kernel[3 % PacketSize]);

				count += 4 * PacketSize;
			}
		}
		for (; k < depth; k++) {
			ptr::write(ptr::incr(blockB, count + 0), ptr::read(ptr::incr(dm0, k)));
			ptr::write(ptr::incr(blockB, count + 1), ptr::read(ptr::incr(dm1, k)));
			ptr::write(ptr::incr(blockB, count + 2), ptr::read(ptr::incr(dm2, k)));
			ptr::write(ptr::incr(blockB, count + 3), ptr::read(ptr::incr(dm3, k)));
			count += 4;
		}
	}

	for (i64 k = 0; k < depth; ++k) {
		for (i64 j2 = packet_cols4; j2 < cols; ++j2) {
			ptr::write(ptr::incr(blockB, count++), ptr::read(ptr::incr(rhs, k + j2 * stride)));
		}
	}
}

template <typename T>
HEDLEY_NEVER_INLINE void pack_rhs_rowmajor(T* blockB, T const* rhs, i64 depth, i64 cols, i64 stride) {
	using Pack = _simd::Pack<T>;
	using pack_traits = _simd::pack_traits<T>;
	constexpr int PacketSize = pack_traits::size;

	ASM_COMMENT("PRODUCT PACK RHS ROWMAJOR");
	i64 packet_cols4 = matmul_traits<T>::nr >= 4 ? (cols / 4) * 4 : 0;
	i64 count = 0;

	if (matmul_traits<T>::nr >= 4) {
		for (i64 j2 = 0; j2 < packet_cols4; j2 += 4) {

			for (i64 k = 0; k < depth; k++) {
				if (PacketSize == 4) {
					Pack A = pack_traits::load(ptr::incr(rhs, k * stride + j2));
					pack_traits::store(ptr::incr(blockB, count), A);
					count += PacketSize;
				} else {
					T const* dm0 = ptr::incr(rhs, k * stride + j2);
					ptr::write(ptr::incr(blockB, count + 0), ptr::read(ptr::incr(dm0, 0)));
					ptr::write(ptr::incr(blockB, count + 1), ptr::read(ptr::incr(dm0, 1)));
					ptr::write(ptr::incr(blockB, count + 2), ptr::read(ptr::incr(dm0, 2)));
					ptr::write(ptr::incr(blockB, count + 3), ptr::read(ptr::incr(dm0, 3)));
					count += 4;
				}
			}
		}
	}
	// copy the remaining columns one row at a time
	for (i64 k = 0; k < depth; k++) {
		for (i64 j2 = packet_cols4; j2 < cols; ++j2) {
			ptr::write(ptr::incr(blockB, count++), ptr::read(ptr::incr(rhs, k * stride + j2)));
		}
	}
}

constexpr i64 pk = 8;

template <template <typename...> class F, typename T, typename Seq>
struct apply_seq;

template <template <typename...> class F, typename T, usize... Is>
struct apply_seq<F, T, veg::meta::index_sequence<Is...>> {
	using type = F<veg::internal::meta_::discard_1st<decltype(Is), T>...>;
};

template <typename T, i64 N>
using TupN = typename apply_seq<veg::Tuple, T, veg::meta::make_index_sequence<usize{N}>>::type;

template <i64... Is>
HEDLEY_ALWAYS_INLINE constexpr auto iseq_impl(veg::meta::integer_sequence<i64, Is...> /*seq*/)
		-> TupN<i64, i64{sizeof...(Is)}> {
	return {
			veg::direct,
			Is...,
	};
}

template <i64 N>
HEDLEY_ALWAYS_INLINE constexpr auto iseq() -> TupN<i64, N> {
	return iseq_impl(veg::meta::make_integer_sequence<i64, N>{});
}

#if defined(__clang__)
#define FAER_INLINE_LAMBDA __attribute__((always_inline)) noexcept
#elif defined(__GNUC__)
#define FAER_INLINE_LAMBDA noexcept __attribute__((always_inline))
#else
#define FAER_INLINE_LAMBDA
#endif

namespace nb {
struct tuple_for_each_zip {
	// template <typename Fn, typename... Args>
	// HEDLEY_ALWAYS_INLINE auto operator()(Fn&& f, Args&&... args) const VEG_DEDUCE_RET(veg::tuple::for_each(
	// 		veg::tuple::zip(VEG_FWD(args)...), veg::fn::copy_fn(veg::fn::rcurry_fwd(veg::tuple::unpack, VEG_FWD(f)))));
	template <typename Fn, typename... Args>
	HEDLEY_ALWAYS_INLINE auto operator()(Fn&& f, Args&&... args) const -> decltype(auto) {
		return veg::tuple::for_each(
				veg::tuple::zip(VEG_FWD(args)...), veg::fn::copy_fn(veg::fn::curry_fwd(veg::tuple::unpack, VEG_FWD(f))));
	}
};
struct as_ref {
	template <typename T>
	HEDLEY_ALWAYS_INLINE auto operator()(T&& arg) const VEG_DEDUCE_RET(VEG_FWD(arg).as_ref());
};
} // namespace nb
VEG_NIEBLOID(tuple_for_each_zip);
VEG_NIEBLOID(as_ref);

template <typename T, bool Contiguous>
struct Gebp {
	DataMapper<T, Contiguous> const res;
	T const* const blockA;
	T const* const blockB;
	i64 const depth;
	T const alpha;
	i64 const strideA;
	i64 const strideB;
	i64 peeled_kc;
	using LinMap = LinearMapper<T, Contiguous>;

	static constexpr unsigned nr = matmul_traits<T>::nr;
	static constexpr unsigned mr = matmul_traits<T>::mr;
	static constexpr int LhsProgress = matmul_traits<T>::LhsProgress;
	static constexpr int RhsProgress = matmul_traits<T>::RhsProgress;

	template <i64 N, i64 M>
	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<N> /*N*/, veg::Fix<M> /*M*/, i64 i, i64 j2) const noexcept {
		{
			using Pack = _simd::Pack<T>;
			constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
			using pack_traits = _simd::pack_traits<T>;

			// We selected a 3*LhsProgress x nr micro block of res which
			// is entirely stored into 3 x nr registers.

			T const* blA = ptr::incr(blockA, i * strideA);
			_simd::prefetch(blA);

			// gets res block as register
			TupN<TupN<Pack, N>, M> C{};

			TupN<LinMap, M> r = veg::tuple::map(iseq<M>(), [&](i64 k) { return res.linear_mapper(i, j2 + k); });
			veg::tuple::for_each(r.as_ref(), [](LinMap& ri) FAER_INLINE_LAMBDA { _simd::prefetch(ri.data); });

			// performs "inner" products
			T const* blB = ptr::incr(blockB, j2 * strideB);
			_simd::prefetch(blB);
			TupN<Pack, N - 1> A;

			struct _gebp_one_step_fn {
				TupN<TupN<Pack, N>&, M> C;
				TupN<Pack&, N> A;
				Pack& B_0;
				T const* blA;
				T const* blB;

				HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
					{
						ASM_COMMENT("begin step of gebp micro kernel NpXM");
						ASM_COMMENT("Note: these asm comments work around bug 935!");
						_simd::prefetch(ptr::incr(blA, (3 * (K) + 16) * LhsProgress));
#ifdef SIMDE_ARCH_ARM
						_simd::prefetch(ptr::incr(blB, (2 * (K) + 16) * RhsProgress));
#endif

						struct _fn0 {
							T const* blA;
							i64 K;

							HEDLEY_ALWAYS_INLINE void operator()(Pack& Ai, i64 n) const noexcept {
								Ai = pack_traits::load(ptr::incr(blA, (n + N * K) * LhsProgress));
							}
						};

						struct _fn1 {
							Pack& B_0;

							HEDLEY_ALWAYS_INLINE void operator()(Pack& Ai, Pack& Cij) const noexcept {
								Cij = pack_traits::fmadd(Ai, B_0, Cij);
							}
						};

						struct _fn2 {
							TupN<Pack&, N> A;
							T const* blA;
							T const* blB;
							i64 K;
							Pack& B_0;
							HEDLEY_ALWAYS_INLINE void operator()(TupN<Pack, N>& Ci, i64 m) const noexcept {
								tuple_for_each_zip(_fn0{blA, K}, A.as_ref(), iseq<N>());
								B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (m + M * (K)) * RhsProgress)));
								tuple_for_each_zip(_fn1{B_0}, A.as_ref(), Ci.as_ref());
							}
						};

						tuple_for_each_zip(_fn2{A.as_ref(), blA, blB, K, B_0}, veg::clone(C), iseq<M>());

						ASM_COMMENT("end step of gebp micro kernel NpXM");
					}
				}
			};

			for (i64 k = 0; k < peeled_kc; k += pk) {
				_simd::prefetch(blB);
				Pack B_0;
				veg::Tuple<Pack> AN;

				auto gebp_onestep = _gebp_one_step_fn{
						C.as_ref(),
						veg::tuple::cat(A.as_ref(), AN.as_ref()),
						B_0,
						blA,
						blB,
				};
				veg::tuple::for_each(iseq<pk>(), gebp_onestep);

				blB = ptr::incr(blB, pk * M * RhsProgress);
				blA = ptr::incr(blA, pk * N * LhsProgress);
			}
			// process remaining peeled loop
			for (i64 k = peeled_kc; k < depth; k++) {
				Pack B_0;
				veg::Tuple<Pack> AN;
				auto gebp_onestep = _gebp_one_step_fn{C.as_ref(), veg::tuple::cat(A.as_ref(), AN.as_ref()), B_0, blA, blB};
				gebp_onestep(0);
				blB = ptr::incr(blB, M * RhsProgress);
				blA = ptr::incr(blA, N * LhsProgress);
			}

			TupN<Pack, N> R;

			Pack alphav = pack_traits::broadcast(alpha);

			struct _load_fn {
				LinMap& ri;
				HEDLEY_ALWAYS_INLINE void operator()(Pack& Ri, i64 n) const noexcept { Ri = ri.load(n * pack_size); }
			};
			struct _fmadd_fn {
				Pack const& alphav;
				HEDLEY_ALWAYS_INLINE void operator()(Pack& Ri, Pack& Cij) const noexcept {
					Ri = pack_traits::fmadd(Cij, alphav, Ri);
				}
			};
			struct _store_fn {
				LinMap& ri;
				HEDLEY_ALWAYS_INLINE void operator()(Pack& Ri, i64 n) const noexcept { ri.store(n * pack_size, Ri); }
			};

			struct _all_fn {
				TupN<Pack, N>& R;
				Pack const& alphav;

				HEDLEY_ALWAYS_INLINE void operator()(TupN<Pack, N>& Ci, LinMap& ri) const noexcept {
					tuple_for_each_zip(_load_fn{ri}, R.as_ref(), iseq<N>());
					tuple_for_each_zip(_fmadd_fn{alphav}, R.as_ref(), Ci.as_ref());
					tuple_for_each_zip(_store_fn{ri}, R.as_ref(), iseq<N>());
				}
			};

			tuple_for_each_zip(_all_fn{R, alphav}, C.as_ref(), r.as_ref());
		}
	}

	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<3> /*N*/, veg::Fix<4> /*M*/, i64 i, i64 j2) const noexcept {

		using Pack = _simd::Pack<T>;
		constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
		using pack_traits = _simd::pack_traits<T>;

		// We selected a 3*LhsProgress x nr micro block of res which
		// is entirely stored into 3 x nr registers.

		T const* blA = ptr::incr(blockA, i * strideA);
		_simd::prefetch(blA);

		// gets res block as register
		Pack C0{};
		Pack C1{};
		Pack C2{};
		Pack C3{};
		Pack C4{};
		Pack C5{};
		Pack C6{};
		Pack C7{};
		Pack C8{};
		Pack C9{};
		Pack C10{};
		Pack C11{};

		LinMap r0 = res.linear_mapper(i, j2 + 0);
		LinMap r1 = res.linear_mapper(i, j2 + 1);
		LinMap r2 = res.linear_mapper(i, j2 + 2);
		LinMap r3 = res.linear_mapper(i, j2 + 3);
		_simd::prefetch(r0.data);
		_simd::prefetch(r1.data);
		_simd::prefetch(r2.data);
		_simd::prefetch(r3.data);

		// performs "inner" products
		T const* blB = ptr::incr(blockB, j2 * strideB);
		_simd::prefetch(blB);
		Pack A0;
		Pack A1;

		using PackRef = Pack&;
		struct _gebp_one_step_fn {
			PackRef C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11;
			PackRef A0, A1, A2;
			PackRef B_0;
			T const* blA;
			T const* blB;

			HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
				{

					ASM_COMMENT("begin step of gebp micro kernel 3pX4");
					ASM_COMMENT("Note: these asm comments work around bug 935!");
					_simd::prefetch(ptr::incr(blA, (3 * K + 16) * LhsProgress));
#ifdef SIMDE_ARCH_ARM
					_simd::prefetch(ptr::incr(blB, (4 * K + 16) * RhsProgress));
#endif
					A0 = pack_traits::load(ptr::incr(blA, (0 + 3 * K) * LhsProgress));
					A1 = pack_traits::load(ptr::incr(blA, (1 + 3 * K) * LhsProgress));
					A2 = pack_traits::load(ptr::incr(blA, (2 + 3 * K) * LhsProgress));

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (0 + 4 * K) * RhsProgress)));
					C0 = pack_traits::fmadd(A0, B_0, C0);
					C4 = pack_traits::fmadd(A1, B_0, C4);
					C8 = pack_traits::fmadd(A2, B_0, C8);

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (1 + 4 * K) * RhsProgress)));
					C1 = pack_traits::fmadd(A0, B_0, C1);
					C5 = pack_traits::fmadd(A1, B_0, C5);
					C9 = pack_traits::fmadd(A2, B_0, C9);

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (2 + 4 * K) * RhsProgress)));
					C2 = pack_traits::fmadd(A0, B_0, C2);
					C6 = pack_traits::fmadd(A1, B_0, C6);
					C10 = pack_traits::fmadd(A2, B_0, C10);

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (3 + 4 * K) * RhsProgress)));
					C3 = pack_traits::fmadd(A0, B_0, C3);
					C7 = pack_traits::fmadd(A1, B_0, C7);
					C11 = pack_traits::fmadd(A2, B_0, C11);

					ASM_COMMENT("end step of gebp micro kernel 3pX4");
				}
			}
		};

		for (i64 k = 0; k < peeled_kc; k += pk) {
			_simd::prefetch(blB);
			Pack B_0;
			Pack A2;

			auto const gebp_onestep =
					_gebp_one_step_fn{C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, A0, A1, A2, B_0, blA, blB};
			veg::tuple::for_each(iseq<pk>(), gebp_onestep);

			blB = ptr::incr(blB, pk * 4 * RhsProgress);
			blA = ptr::incr(blA, pk * 3 * LhsProgress);
		}
		// process remaining peeled loop
		for (i64 k = peeled_kc; k < depth; k++) {
			Pack B_0;
			Pack A2;
			auto const gebp_onestep =
					_gebp_one_step_fn{C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, A0, A1, A2, B_0, blA, blB};
			gebp_onestep(0);
			blB = ptr::incr(blB, 4 * RhsProgress);
			blA = ptr::incr(blA, 3 * LhsProgress);
		}

		Pack R0;
		Pack R1;
		Pack R2;

		auto alphav = pack_traits::broadcast(alpha);

		R0 = r0.load(0 * pack_size);
		R1 = r0.load(1 * pack_size);
		R2 = r0.load(2 * pack_size);
		R0 = pack_traits::fmadd(C0, alphav, R0);
		R1 = pack_traits::fmadd(C4, alphav, R1);
		R2 = pack_traits::fmadd(C8, alphav, R2);
		r0.store(0 * pack_size, R0);
		r0.store(1 * pack_size, R1);
		r0.store(2 * pack_size, R2);

		R0 = r1.load(0 * pack_size);
		R1 = r1.load(1 * pack_size);
		R2 = r1.load(2 * pack_size);
		R0 = pack_traits::fmadd(C1, alphav, R0);
		R1 = pack_traits::fmadd(C5, alphav, R1);
		R2 = pack_traits::fmadd(C9, alphav, R2);
		r1.store(0 * pack_size, R0);
		r1.store(1 * pack_size, R1);
		r1.store(2 * pack_size, R2);

		R0 = r2.load(0 * pack_size);
		R1 = r2.load(1 * pack_size);
		R2 = r2.load(2 * pack_size);
		R0 = pack_traits::fmadd(C2, alphav, R0);
		R1 = pack_traits::fmadd(C6, alphav, R1);
		R2 = pack_traits::fmadd(C10, alphav, R2);
		r2.store(0 * pack_size, R0);
		r2.store(1 * pack_size, R1);
		r2.store(2 * pack_size, R2);

		R0 = r3.load(0 * pack_size);
		R1 = r3.load(1 * pack_size);
		R2 = r3.load(2 * pack_size);
		R0 = pack_traits::fmadd(C3, alphav, R0);
		R1 = pack_traits::fmadd(C7, alphav, R1);
		R2 = pack_traits::fmadd(C11, alphav, R2);
		r3.store(0 * pack_size, R0);
		r3.store(1 * pack_size, R1);
		r3.store(2 * pack_size, R2);
	}

	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<3> /*N*/, veg::Fix<3> /*M*/, i64 i, i64 j2) const noexcept {

		using Pack = _simd::Pack<T>;
		constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
		using pack_traits = _simd::pack_traits<T>;

		// We selected a 3*LhsProgress x nr micro block of res which
		// is entirely stored into 3 x nr registers.

		T const* blA = ptr::incr(blockA, i * strideA);
		_simd::prefetch(blA);

		// gets res block as register
		Pack C0{};
		Pack C1{};
		Pack C2{};
		Pack C4{};
		Pack C5{};
		Pack C6{};
		Pack C8{};
		Pack C9{};
		Pack C10{};

		LinMap r0 = res.linear_mapper(i, j2 + 0);
		LinMap r1 = res.linear_mapper(i, j2 + 1);
		LinMap r2 = res.linear_mapper(i, j2 + 2);
		_simd::prefetch(r0.data);
		_simd::prefetch(r1.data);
		_simd::prefetch(r2.data);

		// performs "inner" products
		T const* blB = ptr::incr(blockB, j2 * strideB);
		_simd::prefetch(blB);
		Pack A0;
		Pack A1;

		using PackRef = Pack&;
		struct _gebp_one_step_fn {
			PackRef C0, C1, C2, C4, C5, C6, C8, C9, C10;
			PackRef A0, A1, A2;
			PackRef B_0;
			T const* blA;
			T const* blB;

			HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
				{

					ASM_COMMENT("begin step of gebp micro kernel 3pX3");
					ASM_COMMENT("Note: these asm comments work around bug 935!");
					_simd::prefetch(ptr::incr(blA, (3 * K + 16) * LhsProgress));
#ifdef SIMDE_ARCH_ARM
					_simd::prefetch(ptr::incr(blB, (3 * K + 16) * RhsProgress));
#endif
					A0 = pack_traits::load(ptr::incr(blA, (0 + 3 * K) * LhsProgress));
					A1 = pack_traits::load(ptr::incr(blA, (1 + 3 * K) * LhsProgress));
					A2 = pack_traits::load(ptr::incr(blA, (2 + 3 * K) * LhsProgress));

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (0 + 3 * K) * RhsProgress)));
					C0 = pack_traits::fmadd(A0, B_0, C0);
					C4 = pack_traits::fmadd(A1, B_0, C4);
					C8 = pack_traits::fmadd(A2, B_0, C8);

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (1 + 3 * K) * RhsProgress)));
					C1 = pack_traits::fmadd(A0, B_0, C1);
					C5 = pack_traits::fmadd(A1, B_0, C5);
					C9 = pack_traits::fmadd(A2, B_0, C9);

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (2 + 3 * K) * RhsProgress)));
					C2 = pack_traits::fmadd(A0, B_0, C2);
					C6 = pack_traits::fmadd(A1, B_0, C6);
					C10 = pack_traits::fmadd(A2, B_0, C10);
					ASM_COMMENT("end step of gebp micro kernel 3pX3");
				}
			}
		};

		for (i64 k = 0; k < peeled_kc; k += pk) {
			_simd::prefetch(blB);
			Pack B_0;
			Pack A2;

			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C2, C4, C5, C6, C8, C9, C10, A0, A1, A2, B_0, blA, blB};
			veg::tuple::for_each(iseq<pk>(), gebp_onestep);

			blB = ptr::incr(blB, pk * 3 * RhsProgress);
			blA = ptr::incr(blA, pk * 3 * LhsProgress);
		}
		// process remaining peeled loop
		for (i64 k = peeled_kc; k < depth; k++) {
			Pack B_0;
			Pack A2;
			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C2, C4, C5, C6, C8, C9, C10, A0, A1, A2, B_0, blA, blB};
			gebp_onestep(0);
			blB = ptr::incr(blB, 3 * RhsProgress);
			blA = ptr::incr(blA, 3 * LhsProgress);
		}

		Pack R0;
		Pack R1;
		Pack R2;

		auto alphav = pack_traits::broadcast(alpha);

		R0 = r0.load(0 * pack_size);
		R1 = r0.load(1 * pack_size);
		R2 = r0.load(2 * pack_size);
		R0 = pack_traits::fmadd(C0, alphav, R0);
		R1 = pack_traits::fmadd(C4, alphav, R1);
		R2 = pack_traits::fmadd(C8, alphav, R2);
		r0.store(0 * pack_size, R0);
		r0.store(1 * pack_size, R1);
		r0.store(2 * pack_size, R2);

		R0 = r1.load(0 * pack_size);
		R1 = r1.load(1 * pack_size);
		R2 = r1.load(2 * pack_size);
		R0 = pack_traits::fmadd(C1, alphav, R0);
		R1 = pack_traits::fmadd(C5, alphav, R1);
		R2 = pack_traits::fmadd(C9, alphav, R2);
		r1.store(0 * pack_size, R0);
		r1.store(1 * pack_size, R1);
		r1.store(2 * pack_size, R2);

		R0 = r2.load(0 * pack_size);
		R1 = r2.load(1 * pack_size);
		R2 = r2.load(2 * pack_size);
		R0 = pack_traits::fmadd(C2, alphav, R0);
		R1 = pack_traits::fmadd(C6, alphav, R1);
		R2 = pack_traits::fmadd(C10, alphav, R2);
		r2.store(0 * pack_size, R0);
		r2.store(1 * pack_size, R1);
		r2.store(2 * pack_size, R2);
	}

	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<3> /*N*/, veg::Fix<2> /*M*/, i64 i, i64 j2) const noexcept {

		using Pack = _simd::Pack<T>;
		constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
		using pack_traits = _simd::pack_traits<T>;

		// We selected a 3*LhsProgress x nr micro block of res which
		// is entirely stored into 3 x nr registers.

		T const* blA = ptr::incr(blockA, i * strideA);
		_simd::prefetch(blA);

		// gets res block as register
		Pack C0{};
		Pack C1{};
		Pack C4{};
		Pack C5{};
		Pack C8{};
		Pack C9{};

		LinMap r0 = res.linear_mapper(i, j2 + 0);
		LinMap r1 = res.linear_mapper(i, j2 + 1);
		_simd::prefetch(r0.data);
		_simd::prefetch(r1.data);

		// performs "inner" products
		T const* blB = ptr::incr(blockB, j2 * strideB);
		_simd::prefetch(blB);
		Pack A0;
		Pack A1;

		using PackRef = Pack&;
		struct _gebp_one_step_fn {
			PackRef C0, C1, C4, C5, C8, C9;
			PackRef A0, A1, A2;
			PackRef B_0, B_1;
			T const* blA;
			T const* blB;

			HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
				{

					ASM_COMMENT("begin step of gebp micro kernel 3pX2");
					ASM_COMMENT("Note: these asm comments work around bug 935!");
					A0 = pack_traits::load(ptr::incr(blA, (0 + 3 * K) * LhsProgress));
					A1 = pack_traits::load(ptr::incr(blA, (1 + 3 * K) * LhsProgress));
					A2 = pack_traits::load(ptr::incr(blA, (2 + 3 * K) * LhsProgress));

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (0 + 2 * K) * RhsProgress)));
					B_1 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (1 + 2 * K) * RhsProgress)));

					C0 = pack_traits::fmadd(A0, B_0, C0);
					C4 = pack_traits::fmadd(A1, B_0, C4);
					C8 = pack_traits::fmadd(A2, B_0, C8);

					C1 = pack_traits::fmadd(A0, B_1, C1);
					C5 = pack_traits::fmadd(A1, B_1, C5);
					C9 = pack_traits::fmadd(A2, B_1, C9);

					ASM_COMMENT("end step of gebp micro kernel 3pX2");
				}
			}
		};

		for (i64 k = 0; k < peeled_kc; k += pk) {
			_simd::prefetch(blB);
			Pack B_0;
			Pack B_1;
			Pack A2;

			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C4, C5, C8, C9, A0, A1, A2, B_0, B_1, blA, blB};
			veg::tuple::for_each(iseq<pk>(), gebp_onestep);

			blB = ptr::incr(blB, pk * 2 * RhsProgress);
			blA = ptr::incr(blA, pk * 3 * LhsProgress);
		}
		// process remaining peeled loop
		for (i64 k = peeled_kc; k < depth; k++) {
			Pack B_0;
			Pack B_1;
			Pack A2;
			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C4, C5, C8, C9, A0, A1, A2, B_0, B_1, blA, blB};
			gebp_onestep(0);
			blB = ptr::incr(blB, 2 * RhsProgress);
			blA = ptr::incr(blA, 3 * LhsProgress);
		}

		Pack R0;
		Pack R1;
		Pack R2;

		auto alphav = pack_traits::broadcast(alpha);

		R0 = r0.load(0 * pack_size);
		R1 = r0.load(1 * pack_size);
		R2 = r0.load(2 * pack_size);
		R0 = pack_traits::fmadd(C0, alphav, R0);
		R1 = pack_traits::fmadd(C4, alphav, R1);
		R2 = pack_traits::fmadd(C8, alphav, R2);
		r0.store(0 * pack_size, R0);
		r0.store(1 * pack_size, R1);
		r0.store(2 * pack_size, R2);

		R0 = r1.load(0 * pack_size);
		R1 = r1.load(1 * pack_size);
		R2 = r1.load(2 * pack_size);
		R0 = pack_traits::fmadd(C1, alphav, R0);
		R1 = pack_traits::fmadd(C5, alphav, R1);
		R2 = pack_traits::fmadd(C9, alphav, R2);
		r1.store(0 * pack_size, R0);
		r1.store(1 * pack_size, R1);
		r1.store(2 * pack_size, R2);
	}

	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<3> /*N*/, veg::Fix<1> /*M*/, i64 i, i64 j2) const noexcept {

		using Pack = _simd::Pack<T>;
		constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
		using pack_traits = _simd::pack_traits<T>;

		// We selected a 3*LhsProgress x nr micro block of res which
		// is entirely stored into 3 x nr registers.

		T const* blA = ptr::incr(blockA, i * strideA);
		_simd::prefetch(blA);

		// gets res block as register
		Pack C0{};
		Pack C4{};
		Pack C8{};

		LinMap r0 = res.linear_mapper(i, j2 + 0);
		_simd::prefetch(r0.data);

		// performs "inner" products
		T const* blB = ptr::incr(blockB, j2 * strideB);
		_simd::prefetch(blB);
		Pack A0;
		Pack A1;

		using PackRef = Pack&;
		struct _gebp_one_step_fn {
			PackRef C0, C4, C8;
			PackRef A0, A1, A2;
			PackRef B_0;
			T const* blA;
			T const* blB;

			HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
				{

					ASM_COMMENT("begin step of gebp micro kernel 3pX1");
					ASM_COMMENT("Note: these asm comments work around bug 935!");
					_simd::prefetch(ptr::incr(blA, (3 * K + 16) * LhsProgress));
#ifdef SIMDE_ARCH_ARM
					_simd::prefetch(ptr::incr(blB, (1 * K + 16) * RhsProgress));
#endif
					A0 = pack_traits::load(ptr::incr(blA, (0 + 3 * K) * LhsProgress));
					A1 = pack_traits::load(ptr::incr(blA, (1 + 3 * K) * LhsProgress));
					A2 = pack_traits::load(ptr::incr(blA, (2 + 3 * K) * LhsProgress));

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (0 + 1 * K) * RhsProgress)));
					C0 = pack_traits::fmadd(A0, B_0, C0);
					C4 = pack_traits::fmadd(A1, B_0, C4);
					C8 = pack_traits::fmadd(A2, B_0, C8);

					ASM_COMMENT("end step of gebp micro kernel 3pX1");
				}
			}
		};

		for (i64 k = 0; k < peeled_kc; k += pk) {
			_simd::prefetch(blB);
			Pack B_0;
			Pack A2;

			auto const gebp_onestep = _gebp_one_step_fn{C0, C4, C8, A0, A1, A2, B_0, blA, blB};
			veg::tuple::for_each(iseq<pk>(), gebp_onestep);

			blB = ptr::incr(blB, pk * 1 * RhsProgress);
			blA = ptr::incr(blA, pk * 3 * LhsProgress);
		}
		// process remaining peeled loop
		for (i64 k = peeled_kc; k < depth; k++) {
			Pack B_0;
			Pack A2;
			auto const gebp_onestep = _gebp_one_step_fn{C0, C4, C8, A0, A1, A2, B_0, blA, blB};
			gebp_onestep(0);
			blB = ptr::incr(blB, 1 * RhsProgress);
			blA = ptr::incr(blA, 3 * LhsProgress);
		}

		Pack R0;
		Pack R1;
		Pack R2;

		auto alphav = pack_traits::broadcast(alpha);

		R0 = r0.load(0 * pack_size);
		R1 = r0.load(1 * pack_size);
		R2 = r0.load(2 * pack_size);
		R0 = pack_traits::fmadd(C0, alphav, R0);
		R1 = pack_traits::fmadd(C4, alphav, R1);
		R2 = pack_traits::fmadd(C8, alphav, R2);
		r0.store(0 * pack_size, R0);
		r0.store(1 * pack_size, R1);
		r0.store(2 * pack_size, R2);
	}

	HEDLEY_ALWAYS_INLINE void operator()(veg::Fix<2> /*N*/, veg::Fix<4> /*M*/, i64 i, i64 j2) const noexcept {

		using Pack = _simd::Pack<T>;
		constexpr i64 pack_size = sizeof(Pack) / sizeof(T);
		using pack_traits = _simd::pack_traits<T>;

		// We selected a 3*LhsProgress x nr micro block of res which
		// is entirely stored into 3 x nr registers.

		T const* blA = ptr::incr(blockA, i * strideA);
		_simd::prefetch(blA);

		// gets res block as register
		Pack C0{};
		Pack C1{};
		Pack C2{};
		Pack C3{};
		Pack C4{};
		Pack C5{};
		Pack C6{};
		Pack C7{};

		LinMap r0 = res.linear_mapper(i, j2 + 0);
		LinMap r1 = res.linear_mapper(i, j2 + 1);
		LinMap r2 = res.linear_mapper(i, j2 + 2);
		LinMap r3 = res.linear_mapper(i, j2 + 3);
		_simd::prefetch(r0.data);
		_simd::prefetch(r1.data);
		_simd::prefetch(r2.data);
		_simd::prefetch(r3.data);

		// performs "inner" products
		T const* blB = ptr::incr(blockB, j2 * strideB);
		_simd::prefetch(ptr::incr(blB, 0));
		Pack A0;
		Pack A1;

		using PackRef = Pack&;
		struct _gebp_one_step_fn {
			PackRef C0, C1, C2, C3, C4, C5, C6, C7;
			PackRef A0, A1;
			PackRef B_0, B_1, B_2, B_3;
			T const* blA;
			T const* blB;

			HEDLEY_ALWAYS_INLINE void operator()(i64 K) const noexcept {
				{

					ASM_COMMENT("begin step of gebp micro kernel 3pX4");
					ASM_COMMENT("Note: these asm comments work around bug 935!");
					A0 = pack_traits::load(ptr::incr(blA, (0 + 2 * K) * LhsProgress));
					A1 = pack_traits::load(ptr::incr(blA, (1 + 2 * K) * LhsProgress));

					B_0 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (0 + 4 * K) * RhsProgress)));
					B_1 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (1 + 4 * K) * RhsProgress)));
					B_2 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (2 + 4 * K) * RhsProgress)));
					B_3 = pack_traits::broadcast(ptr::read(ptr::incr(blB, (3 + 4 * K) * RhsProgress)));

					C0 = pack_traits::fmadd(A0, B_0, C0);
					C4 = pack_traits::fmadd(A1, B_0, C4);

					C1 = pack_traits::fmadd(A0, B_1, C1);
					C5 = pack_traits::fmadd(A1, B_1, C5);

					C2 = pack_traits::fmadd(A0, B_2, C2);
					C6 = pack_traits::fmadd(A1, B_2, C6);

					C3 = pack_traits::fmadd(A0, B_3, C3);
					C7 = pack_traits::fmadd(A1, B_3, C7);

					ASM_COMMENT("end step of gebp micro kernel 3pX4");
				}
			}
		};

		for (i64 k = 0; k < peeled_kc; k += pk) {
			Pack B_0;
			Pack B_1;
			Pack B_2;
			Pack B_3;

			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C2, C3, C4, C5, C6, C7, A0, A1, B_0, B_1, B_2, B_3, blA, blB};
			constexpr auto seq1 = iseq<pk / 2>();
			struct _add {
				i64 i;
				constexpr auto operator()(i64 j) const noexcept -> i64 { return i + j; }
			};
			constexpr auto seq2 = veg::tuple::map(veg::clone(seq1), _add{4});

			_simd::prefetch(ptr::incr(blB, 48 + 0));
			veg::tuple::for_each(iseq<pk / 2>(), gebp_onestep);
			_simd::prefetch(ptr::incr(blB, 48 + 16));
			veg::tuple::for_each(veg::clone(seq2), gebp_onestep);

			blB = ptr::incr(blB, pk * 4 * RhsProgress);
			blA = ptr::incr(blA, pk * 2 * LhsProgress);
		}
		// process remaining peeled loop
		for (i64 k = peeled_kc; k < depth; k++) {
			Pack B_0;
			Pack B_1;
			Pack B_2;
			Pack B_3;
			auto const gebp_onestep = _gebp_one_step_fn{C0, C1, C2, C3, C4, C5, C6, C7, A0, A1, B_0, B_1, B_2, B_3, blA, blB};
			gebp_onestep(0);
			blB = ptr::incr(blB, 4 * RhsProgress);
			blA = ptr::incr(blA, 2 * LhsProgress);
		}

		Pack R0;
		Pack R1;
		Pack R2;
		Pack R3;

		auto alphav = pack_traits::broadcast(alpha);

		R0 = r0.load(0 * pack_size);
		R1 = r0.load(1 * pack_size);
		R2 = r1.load(0 * pack_size);
		R3 = r1.load(1 * pack_size);

		R0 = pack_traits::fmadd(C0, alphav, R0);
		R1 = pack_traits::fmadd(C4, alphav, R1);
		R2 = pack_traits::fmadd(C1, alphav, R2);
		R3 = pack_traits::fmadd(C5, alphav, R3);

		r0.store(0 * pack_size, R0);
		r0.store(1 * pack_size, R1);
		r1.store(0 * pack_size, R2);
		r1.store(1 * pack_size, R3);

		R0 = r2.load(0 * pack_size);
		R1 = r2.load(1 * pack_size);
		R2 = r3.load(0 * pack_size);
		R3 = r3.load(1 * pack_size);

		R0 = pack_traits::fmadd(C2, alphav, R0);
		R1 = pack_traits::fmadd(C6, alphav, R1);
		R2 = pack_traits::fmadd(C3, alphav, R2);
		R3 = pack_traits::fmadd(C7, alphav, R3);

		r2.store(0 * pack_size, R0);
		r2.store(1 * pack_size, R1);
		r3.store(0 * pack_size, R2);
		r3.store(1 * pack_size, R3);
	}
};

template <typename T, bool Contiguous>
HEDLEY_NEVER_INLINE void eigen_gebp_mul(
		i64 const l1,
		DataMapper<T, Contiguous> const res,
		T const* blockA,
		T const* blockB,
		i64 rows,
		i64 depth,
		i64 cols,
		T alpha,
		i64 strideA,
		i64 strideB) {

	using Pack = _simd::Pack<T>;
	using pack_traits = _simd::pack_traits<T>;

	static constexpr unsigned nr = matmul_traits<T>::nr;
	static constexpr unsigned mr = matmul_traits<T>::mr;
	static constexpr int LhsProgress = matmul_traits<T>::LhsProgress;

	i64 const packet_cols4 = nr >= 4 ? (cols / 4) * 4 : 0;

	i64 const peeled_mc3 = mr >= 3 * LhsProgress ? (rows / (3 * LhsProgress)) * (3 * LhsProgress) : 0;
	i64 const peeled_mc2 =
			mr >= 2 * LhsProgress ? peeled_mc3 + ((rows - peeled_mc3) / (2 * LhsProgress)) * (2 * LhsProgress) : 0;
	i64 const peeled_mc1 = mr >= 1 * LhsProgress ? (rows / (1 * LhsProgress)) * (1 * LhsProgress) : 0;

	auto const peeled_kc =
			static_cast<i64>((static_cast<unsigned long long>(depth)) & ~(static_cast<unsigned long long>(pk - 1U)));

	Gebp<T, Contiguous> const gebp{
			res,
			blockA,
			blockB,
			depth,
			alpha,
			strideA,
			strideB,
			peeled_kc,
	};
	if (mr >= 3 * LhsProgress && peeled_mc3 != 0) {
		// Here, the general idea is to loop on each largest micro horizontal
		// panel of the lhs (3*LhsProgress x depth) and on each largest
		// micro vertical panel of the rhs (depth * nr). Blocking sizes, i.e.,
		// 'depth' has been computed so that the micro horizontal panel of the lhs
		// fit in L1. However, if depth is too small, we can extend the number of
		// rows of these horizontal panels. This actual number of rows is computed
		// as follow:

		// The max(1, ...) here is needed because we may be using blocking params
		// larger than what our known l1 cache size suggests we should be using:
		// either because our known l1 cache size is inaccurate (e.g. on Android,
		// we can only guess), or because we are testing specific blocking sizes.
		i64 const actual_panel_rows =
				(3 * LhsProgress) * std::max<i64>(
																1,
																((l1 - i64(sizeof(T)) * mr * nr - depth * nr * i64(sizeof(T))) /
		                             (depth * i64(sizeof(T)) * 3 * LhsProgress)));

		for (i64 i1 = 0; i1 < peeled_mc3; i1 += actual_panel_rows) {
			i64 const actual_panel_end = (std::min)(i1 + actual_panel_rows, peeled_mc3);
			for (i64 j2 = 0; j2 < packet_cols4; j2 += nr) {
				for (i64 i = i1; i < actual_panel_end; i += 3 * LhsProgress) {
					gebp(3_c, 4_c, i, j2);
				}
			}

			i64 j2 = packet_cols4;
			switch (cols - packet_cols4) {
			case 0:
				break;
			case 1:
				for (i64 i = i1; i < actual_panel_end; i += 3 * LhsProgress) {
					gebp(3_c, 1_c, i, j2);
				}
				break;
			case 2:
				for (i64 i = i1; i < actual_panel_end; i += 3 * LhsProgress) {
					gebp(3_c, 2_c, i, j2);
				}
				break;
			case 3:
				for (i64 i = i1; i < actual_panel_end; i += 3 * LhsProgress) {
					gebp(3_c, 3_c, i, j2);
				}
				break;
			}
		}
	}

	//---------- Process 2 * LhsProgress rows at once ----------
	if (mr >= 2 * LhsProgress && peeled_mc2 != peeled_mc3) {
		// The max(1, ...) here is needed because we may be using blocking params
		// larger than what our known l1 cache size suggests we should be using:
		// either because our known l1 cache size is inaccurate (e.g. on Android,
		// we can only guess), or because we are testing specific blocking sizes.
		i64 actual_panel_rows = (2 * LhsProgress) * std::max<i64>(
																										1,
																										((l1 - i64(sizeof(T)) * mr * nr - depth * nr * i64(sizeof(T))) /
		                                                 (depth * i64(sizeof(T)) * 2 * LhsProgress)));

		for (i64 i1 = peeled_mc3; i1 < peeled_mc2; i1 += actual_panel_rows) {
			i64 actual_panel_end = (std::min)(i1 + actual_panel_rows, peeled_mc2);
			for (i64 j2 = 0; j2 < packet_cols4; j2 += nr) {
				for (i64 i = i1; i < actual_panel_end; i += 2 * LhsProgress) {
					gebp(2_c, 4_c, i, j2);
				}
			}

			i64 j2 = packet_cols4;
			switch (cols - packet_cols4) {
			case 0:
				break;
			case 1:
				for (i64 i = i1; i < actual_panel_end; i += 2 * LhsProgress) {
					gebp(2_c, 1_c, i, j2);
				}
				break;
			case 2:
				for (i64 i = i1; i < actual_panel_end; i += 2 * LhsProgress) {
					gebp(2_c, 2_c, i, j2);
				}
				break;
			case 3:
				for (i64 i = i1; i < actual_panel_end; i += 2 * LhsProgress) {
					gebp(2_c, 3_c, i, j2);
				}
				break;
			}
		}
	}
	//---------- Process 1 * LhsProgress rows at once ----------
	if (mr >= 1 * LhsProgress && peeled_mc1 != peeled_mc2) {
		// loops on each largest micro horizontal panel of lhs (1*LhsProgress x
		// depth)
		for (i64 i = peeled_mc2; i < peeled_mc1; i += 1 * LhsProgress) {
			// loops on each largest micro vertical panel of rhs (depth * nr)
			for (i64 j2 = 0; j2 < packet_cols4; j2 += nr) {
				gebp(1_c, 4_c, i, j2);
			}

			i64 j2 = packet_cols4;
			switch (cols - packet_cols4) {
			case 0:
				break;
			case 1:
				gebp(1_c, 1_c, i, j2);
				break;
			case 2:
				gebp(1_c, 2_c, i, j2);
				break;
			case 3:
				gebp(1_c, 3_c, i, j2);
				break;
			}
		}
	}
	//---------- Process remaining rows, 1 at once ----------
	if (peeled_mc1 < rows) {
		// loop on each panel of the rhs
		for (i64 j2 = 0; j2 < packet_cols4; j2 += nr) {
			// loop on each row of the lhs (1*LhsProgress x depth)
			for (i64 i = peeled_mc1; i < rows; i += 1) {
				T const* blA = ptr::incr(blockA, i * strideA);
				_simd::prefetch(blA);
				T const* blB = ptr::incr(blockB, j2 * strideB);

				// The following piece of code wont work for 512 bit registers
				// Moreover, if LhsProgress==8 it assumes that there is a half packet
				// of the same size as nr (which is currently 4) for the return type.
#if FAER_HAS_HALF_PACK
				using PackHalf = _simd::PackHalf<T>;
				using pack_half_traits = _simd::pack_half_traits<T>;
				if ((LhsProgress % 4) == 0 && (LhsProgress <= 8) && (LhsProgress != 8 || pack_half_traits::size == nr)) {

					Pack C0{};
					Pack C1{};
					Pack C2{};
					Pack C3{};

					i64 const spk = (std::max)(1, LhsProgress / 4);
					i64 const endk = (depth / spk) * spk;
					i64 const endk4 = (depth / (spk * 4)) * (spk * 4);

					i64 k = 0;
					for (; k < endk4; k += 4 * spk) {
						Pack B_0;
						Pack B_1;
						Pack A0;
						Pack A1;

						B_0 = pack_traits::load(ptr::incr(blB, 0 * LhsProgress));
						B_1 = pack_traits::load(ptr::incr(blB, 1 * LhsProgress));

						A0 = pack_traits::broadcast_blocks_of_4(ptr::incr(blA, 0 * spk));
						A1 = pack_traits::broadcast_blocks_of_4(ptr::incr(blA, 1 * spk));
						C0 = pack_traits::fmadd(A0, B_0, C0);
						C1 = pack_traits::fmadd(A1, B_1, C1);

						B_0 = pack_traits::load(ptr::incr(blB, 2 * LhsProgress));
						B_1 = pack_traits::load(ptr::incr(blB, 3 * LhsProgress));
						A0 = pack_traits::broadcast_blocks_of_4(ptr::incr(blA, 2 * spk));
						A1 = pack_traits::broadcast_blocks_of_4(ptr::incr(blA, 3 * spk));
						C2 = pack_traits::fmadd(A0, B_0, C2);
						C3 = pack_traits::fmadd(A1, B_1, C3);

						blB = ptr::incr(blB, 4 * LhsProgress);
						blA = ptr::incr(blA, 4 * spk);
					}
					C0 = pack_traits::add(pack_traits::add(C0, C1), pack_traits::add(C2, C3));
					for (; k < endk; k += spk) {
						Pack B_0;
						Pack A0;

						B_0 = pack_traits::load(blB);
						A0 = pack_traits::broadcast_blocks_of_4(blA);
						C0 = pack_traits::fmadd(B_0, A0, C0);

						blB = ptr::incr(blB, LhsProgress);
						blA = ptr::incr(blA, spk);
					}
					if (LhsProgress == 8) {
						// 256bit floats or 512bit doubles

						// Special case where we have to first reduce the accumulation
						// register C0

						auto R = pack_half_traits::gather(res.ptr(i, j2), res.stride);
						PackHalf alphav = pack_half_traits::broadcast(alpha);

						if (depth - endk > 0) {
							// We have to handle the last row of the rhs which corresponds
							// to a half-packet
							PackHalf b0;
							PackHalf a0;
							b0 = pack_half_traits::load(blB);
							a0 = pack_half_traits::broadcast(ptr::read(blA));

							PackHalf c00 = pack_traits::first_half(C0);
							PackHalf c01 = pack_traits::second_half(C0);
							PackHalf c0 = pack_half_traits::add(c00, c01);

							c0 = pack_half_traits::fmadd(b0, a0, c0);
							R = pack_half_traits::fmadd(c0, alphav, R);
						} else {
							PackHalf c00 = pack_traits::first_half(C0);
							PackHalf c01 = pack_traits::second_half(C0);
							PackHalf c0 = pack_half_traits::add(c00, c01);
							R = pack_half_traits::fmadd(c0, alphav, R);
						}
						pack_half_traits::scatter(res.ptr(i, j2), R, res.stride);
					} else {
						Pack R = pack_traits::gather(res.ptr(i, j2), res.stride);
						Pack alphav = pack_traits::broadcast(alpha);
						R = pack_traits::fmadd(C0, alphav, R);
						pack_traits::scatter(res.ptr(i, j2), R, res.stride);
					}
				} else // scalar path
#endif
				{
					// get a 1 x 4 res block as registers
					T C0{};
					T C1{};
					T C2{};
					T C3{};

					for (i64 k = 0; k < depth; k++) {
						T A0 = ptr::read(ptr::incr(blA, k));
						T B_0 = ptr::read(ptr::incr(blB, 0));
						T B_1 = ptr::read(ptr::incr(blB, 1));

						C0 = std::fma(A0, B_0, C0);
						C1 = std::fma(A0, B_1, C1);

						B_0 = ptr::read(ptr::incr(blB, 2));
						B_1 = ptr::read(ptr::incr(blB, 3));

						C2 = std::fma(A0, B_0, C2);
						C3 = std::fma(A0, B_1, C3);

						blB = ptr::incr(blB, 4);
					}
					ptr::write(res.ptr(i, j2 + 0), std::fma(alpha, C0, ptr::read(res.ptr(i, j2 + 0))));
					ptr::write(res.ptr(i, j2 + 1), std::fma(alpha, C1, ptr::read(res.ptr(i, j2 + 1))));
					ptr::write(res.ptr(i, j2 + 2), std::fma(alpha, C2, ptr::read(res.ptr(i, j2 + 2))));
					ptr::write(res.ptr(i, j2 + 3), std::fma(alpha, C3, ptr::read(res.ptr(i, j2 + 3))));
				}
			}
		}

		// remaining columns
		for (i64 j2 = packet_cols4; j2 < cols; j2++) {
			// loop on each row of the lhs (1*LhsProgress x depth)
			for (i64 i = peeled_mc1; i < rows; i += 1) {
				T const* blA = ptr::incr(blockA, i * strideA);
				_simd::prefetch(blA);
				// gets a 1 x 1 res block as registers
				T C0{};
				T const* blB = ptr::incr(blockB, packet_cols4 * strideB + (j2 - packet_cols4));

				for (i64 k = 0; k < depth; k++) {
					T A0 = ptr::read(ptr::incr(blA, k));
					T B_0 = ptr::read(ptr::incr(blB, k * (cols - packet_cols4)));
					C0 = std::fma(A0, B_0, C0);
				}
				ptr::write(res.ptr(i, j2), std::fma(alpha, C0, ptr::read(res.ptr(i, j2))));
			}
		}
	}
}

HEDLEY_NEVER_INLINE void eigen_product_blocking_heuristic(
		i64& k, i64& m, i64& n, i64 nr, i64 mr, i64 type_size, i64 KcFactor = 1, i64 num_threads = 1) {
	using ull = unsigned long long;

	// Explanations:
	// Let's recall that the product algorithms form mc x kc vertical panels A' on the lhs and
	// kc x nc blocks B' on the rhs. B' has to fit into L2/L3 cache. Moreover, A' is processed
	// per mr x kc horizontal small panels where mr is the blocking size along the m dimension
	// at the register level. This small horizontal panel has to stay within L1 cache.
	auto caches = (cache_size)();
	std::ptrdiff_t l1 = caches.l1;
	std::ptrdiff_t l2 = caches.l2;
	std::ptrdiff_t l3 = caches.l3;

	if (num_threads > 1) {
		i64 const kdiv = KcFactor * (mr * type_size + nr * type_size);
		i64 const ksub = mr * nr * type_size;
		i64 const kr = 8;
		// Increasing k gives us more time to prefetch the content of the "C"
		// registers. However once the latency is hidden there is no point in
		// increasing the value of k, so we'll cap it at 320 (value determined
		// experimentally).
		// To avoid that k vanishes, we make k_cache at least as big as kr
		i64 const k_cache = (std::max)(kr, (std::min)((l1 - ksub) / kdiv, i64(320)));
		if (k_cache < k) {
			k = k_cache - (k_cache % kr);
			VEG_INTERNAL_ASSERT_PRECONDITION(k > 0);
		}

		i64 const n_cache = (l2 - l1) / (nr * type_size * k);
		i64 const n_per_thread = (internal::round_up)(n, num_threads);
		if (n_cache <= n_per_thread) {
			// Don't exceed the capacity of the l2 cache.
			VEG_INTERNAL_ASSERT_PRECONDITION(n_cache >= i64(nr));
			n = n_cache - (n_cache % nr);
			VEG_INTERNAL_ASSERT_PRECONDITION(n > 0);
		} else {
			n = (std::min)(n, (n_per_thread + nr - 1) - ((n_per_thread + nr - 1) % nr));
		}

		if (l3 > l2) {
			// l3 is shared between all cores, so we'll give each thread its own chunk of l3.
			i64 const m_cache = (l3 - l2) / (type_size * k * num_threads);
			i64 const m_per_thread = (internal::round_up)(m, num_threads);
			if (m_cache < m_per_thread && m_cache >= i64(mr)) {
				m = m_cache - (m_cache % mr);
				VEG_INTERNAL_ASSERT_PRECONDITION(m > 0);
			} else {
				m = (std::min)(m, (m_per_thread + mr - 1) - ((m_per_thread + mr - 1) % mr));
			}
		}
	} else {
		// In unit tests we do not want to use extra large matrices,
		// so we reduce the cache size to check the blocking strategy is not flawed

		// Early return for small problems because the computation below are time consuming for small problems.
		// Perhaps it would make more sense to consider k*n*m??
		// Note that for very tiny problem, this function should be bypassed anyway
		// because we use the coefficient-based implementation for them.

		i64 const k_peeling = 8;
		i64 const k_div = KcFactor * (mr * type_size + nr * type_size);
		i64 const k_sub = mr * nr * type_size;

		// ---- 1st level of blocking on L1, yields kc ----

		// Blocking on the third dimension (i.e., k) is chosen so that an horizontal panel
		// of size mr x kc of the lhs plus a vertical panel of kc x nr of the rhs both fits within L1 cache.
		// We also include a register-level block of the result (mx x nr).
		// (In an ideal world only the lhs panel would stay in L1)
		// Moreover, kc has to be a multiple of 8 to be compatible with loop peeling, leading to a maximum blocking size
		// of:
		i64 const max_kc = (std::max)(i64(ull((l1 - k_sub) / k_div) & (~(ull(k_peeling) - 1))), i64(1));
		i64 const old_k = k;
		if (k > max_kc) {
			// We are really blocking on the third dimension:
			// -> reduce blocking size to make sure the last block is as large as possible
			//    while keeping the same number of sweeps over the result.
			k = (k % max_kc) == 0 ? max_kc
			                      : max_kc - k_peeling * ((max_kc - 1 - (k % max_kc)) / (k_peeling * (k / max_kc + 1)));

#ifdef VEG_INTERNAL_ASSERTIONS
			VEG_ASSERT_ELSE("the number of sweeps has to remain the same", ((old_k / k) == (old_k / max_kc)));
#endif
		}

		// ---- 2nd level of blocking on max(L2,L3), yields nc ----

		i64 const actual_l2 = 1572864; // == 1.5 MB

		// Here, nc is chosen such that a block of kc x nc of the rhs fit within half of L2.
		// The second half is implicitly reserved to access the result and lhs coefficients.
		// When k<max_kc, then nc can arbitrarily growth. In practice, it seems to be fruitful
		// to limit this growth: we bound nc to growth by a factor x1.5.
		// However, if the entire lhs block fit within L1, then we are not going to block on the rows at all,
		// and it becomes fruitful to keep the packed rhs blocks in L1 if there is enough remaining space.
		i64 max_nc = 0;
		i64 const lhs_bytes = m * k * type_size;
		i64 const remaining_l1 = l1 - k_sub - lhs_bytes;
		if (remaining_l1 >= i64(nr * type_size) * k) {
			// L1 blocking
			max_nc = remaining_l1 / (k * type_size);
		} else {
			// L2 blocking
			max_nc = (3 * actual_l2) / (2 * 2 * max_kc * type_size);
		}
		// WARNING Below, we assume that Traits::nr is a power of two.
		VEG_INTERNAL_ASSERT_PRECONDITION(nr > 0);
		VEG_INTERNAL_ASSERT_PRECONDITION((unsigned(nr) & (unsigned(nr) - 1U)) == 0);
		i64 nc = i64(ull((std::min)(actual_l2 / (2 * k * type_size), max_nc)) & (~(ull(nr) - 1)));
		if (n > nc) {
			// We are really blocking over the columns:
			// -> reduce blocking size to make sure the last block is as large as possible
			//    while keeping the same number of sweeps over the packed lhs.
			//    Here we allow one more sweep if this gives us a perfect match, thus the commented "-1"
			n = (n % nc) == 0 ? nc : (nc - nr * ((nc /*-1*/ - (n % nc)) / (nr * (n / nc + 1))));
		} else if (old_k == k) {
			// So far, no blocking at all, i.e., kc==k, and nc==n.
			// In this case, let's perform a blocking over the rows such that the packed lhs data is kept in cache L1/L2
			// TODO: part of this blocking strategy is now implemented within the kernel itself, so the L1-based heuristic
			// here should be obsolete.
			i64 problem_size = k * n * type_size;
			i64 actual_lm = actual_l2;
			i64 max_mc = m;
			if (problem_size <= 1024) {
				// problem is small enough to keep in L1
				// Let's choose m such that lhs's block fit in 1/3 of L1
				actual_lm = l1;
			} else if (l3 != 0 && problem_size <= 32768) {
				// we have both L2 and L3, and problem is small enough to be kept in L2
				// Let's choose m such that lhs's block fit in 1/3 of L2
				actual_lm = l2;
				max_mc = (std::min)(i64(576), max_mc);
			}
			i64 mc = (std::min)(actual_lm / (3 * k * type_size), max_mc);
			if (mc > mr) {
				mc -= mc % mr;
			} else if (mc == 0) {
				return;
			}
			m = (m % mc) == 0 ? mc : (mc - mr * ((mc /*-1*/ - (m % mc)) / (mr * (m / mc + 1))));
		}
	}
}

template <typename T>
struct MatMulVars {
	T* dst;
	i64 dst_outer;
	i64 dst_inner;
	T const* lhs;
	i64 lhs_outer;
	bool lhs_colmajor;
	T const* rhs;
	i64 rhs_outer;
	bool rhs_colmajor;
	T alpha;
	i64 rows;
	i64 cols;
	i64 depth;
};

constexpr usize max_bytes = 2 * 128U * 1024U;
constexpr usize max_align = 64;

template <typename T>
HEDLEY_NEVER_INLINE void eigen_matmul_colmajor2(MatMulVars<T> const& vars, i64 nc, i64 mc, i64 kc, T* block) {

	T* blockA = static_cast<T*>(block);                                             // kc * mc
	T* blockB = static_cast<T*>(veg::mem::align_next(max_align, blockA + kc * mc)); // kc * nc

	DataMapper<T, true> res_cont{vars.dst, vars.dst_outer, vars.dst_inner};
	DataMapper<T, false> res_incr{vars.dst, vars.dst_outer, vars.dst_inner};

	bool const pack_rhs_once = mc != vars.rows && kc == vars.depth && nc == vars.cols;
	i64 const l1 = cache_size().l1;

	// For each horizontal panel of the rhs, and corresponding panel of the
	// lhs...
	for (i64 i2 = 0; i2 < vars.rows; i2 += mc) {
		i64 const actual_mc = (std::min)(i2 + mc, vars.rows) - i2;

		for (i64 k2 = 0; k2 < vars.depth; k2 += kc) {
			i64 const actual_kc = (std::min)(k2 + kc, vars.depth) - k2;

			// OK, here we have selected one horizontal panel of rhs and one
			// vertical panel of lhs.
			// => Pack lhs's panel into a sequential chunk of memory (L2/L3
			// caching) Note that this panel will be read as many times as the
			// number of blocks in the rhs's horizontal panel which is, in
			// practice, a very low number.

			if (vars.lhs_colmajor) {
				_matmul::pack_lhs_colmajor(
						blockA, ptr::incr(vars.lhs, i2 + k2 * vars.lhs_outer), actual_kc, actual_mc, vars.lhs_outer);
			} else {
				_matmul::pack_lhs_rowmajor(
						blockA, ptr::incr(vars.lhs, vars.lhs_outer * i2 + k2), actual_kc, actual_mc, vars.lhs_outer);
			}

			// For each kc x nc block of the rhs's horizontal panel...
			for (i64 j2 = 0; j2 < vars.cols; j2 += nc) {
				i64 const actual_nc = (std::min)(j2 + nc, vars.cols) - j2;

				// We pack the rhs's block into a sequential chunk of memory (L2
				// caching) Note that this block will be read a very high number of
				// times, which is equal to the number of micro horizontal panel of
				// the large rhs's panel (e.g., rows/12 times).
				if ((!pack_rhs_once) || i2 == 0) {

					if (vars.rhs_colmajor) {
						_matmul::pack_rhs_colmajor(
								blockB, ptr::incr(vars.rhs, k2 + j2 * vars.rhs_outer), actual_kc, actual_nc, vars.rhs_outer);
					} else {
						_matmul::pack_rhs_rowmajor(
								blockB, ptr::incr(vars.rhs, vars.rhs_outer * k2 + j2), actual_kc, actual_nc, vars.rhs_outer);
					}
				}

				if (vars.dst_inner == 1) {
					(_matmul::eigen_gebp_mul)(
							l1,
							res_cont.submapper(i2, j2),
							blockA,
							blockB,
							actual_mc,
							actual_kc,
							actual_nc,
							vars.alpha,
							actual_kc,
							actual_kc);
				} else {
					(_matmul::eigen_gebp_mul)(
							l1,
							res_incr.submapper(i2, j2),
							blockA,
							blockB,
							actual_mc,
							actual_kc,
							actual_nc,
							vars.alpha,
							actual_kc,
							actual_kc);
				}
			}
		}
	}
}

template <typename T>
HEDLEY_ALWAYS_INLINE void eigen_matmul_colmajor(MatMulVars<T> const& vars, i64 nc, i64 mc, i64 kc) {
	bool use_stack = veg::nb::narrow<usize>{}(i64(sizeof(T)) * (kc * (nc + mc)) + i64(max_align)) < max_bytes;
	i64 alloc_size = i64(sizeof(T)) * (kc * (mc + nc));

	void* alloc = static_cast<T*>(
			use_stack ? veg::mem::align_next(max_align, ::alloca(usize(alloc_size + 2 * i64(max_align))))
								: veg::mem::aligned_alloc(max_align, alloc_size + i64(max_align)));

	(eigen_matmul_colmajor2)(vars, nc, mc, kc, static_cast<T*>(veg::mem::align_next(max_align, alloc)));
}

template <typename T>
HEDLEY_ALWAYS_INLINE void eigen_matmul(
		T* dst,
		i64 dst_outer,
		i64 dst_inner,
		bool dst_colmajor,
		T const* lhs,
		i64 lhs_outer,
		bool lhs_colmajor,
		T const* rhs,
		i64 rhs_outer,
		bool rhs_colmajor,
		T alpha,
		i64 rows,
		i64 cols,
		i64 depth) {

	static constexpr unsigned nr = matmul_traits<T>::nr;
	static constexpr unsigned mr = matmul_traits<T>::mr;

	i64 mc = rows;
	i64 nc = cols;
	i64 kc = depth;
	if ((std::max)(mc, (std::max)(nc, kc)) >= 48) {
		(eigen_product_blocking_heuristic)(kc, mc, nc, nr, mr, i64{sizeof(T)});
	}

	if (dst_colmajor) {
		_matmul::eigen_matmul_colmajor<T>(
				{dst,
		     dst_outer,
		     dst_inner,
		     lhs,
		     lhs_outer,
		     lhs_colmajor,
		     rhs,
		     rhs_outer,
		     rhs_colmajor,
		     alpha,
		     rows,
		     cols,
		     depth},
				nc,
				mc,
				kc);
	} else {
		_matmul::eigen_matmul_colmajor<T>(
				{dst,
		     dst_outer,
		     dst_inner,
		     rhs,
		     rhs_outer,
		     !rhs_colmajor,
		     lhs,
		     lhs_outer,
		     !lhs_colmajor,
		     alpha,
		     cols,
		     rows,
		     depth},
				mc,
				nc,
				kc);
	}
}

} // namespace _matmul
} // namespace _internal

void eigen_matmul<f32>::apply(
		f32* dst,
		i64 dst_outer,
		i64 dst_inner,
		bool dst_colmajor,
		f32 const* lhs,
		i64 lhs_outer,
		bool lhs_colmajor,
		f32 const* rhs,
		i64 rhs_outer,
		bool rhs_colmajor,
		f32 alpha,
		i64 rows,
		i64 cols,
		i64 depth) {
	_internal::_matmul::eigen_matmul(
			dst,
			dst_outer,
			dst_inner,
			dst_colmajor,
			lhs,
			lhs_outer,
			lhs_colmajor,
			rhs,
			rhs_outer,
			rhs_colmajor,
			alpha,
			rows,
			cols,
			depth);
}

void eigen_matmul<f64>::apply(
		f64* dst,
		i64 dst_outer,
		i64 dst_inner,
		bool dst_colmajor,
		f64 const* lhs,
		i64 lhs_outer,
		bool lhs_colmajor,
		f64 const* rhs,
		i64 rhs_outer,
		bool rhs_colmajor,
		f64 alpha,
		i64 rows,
		i64 cols,
		i64 depth) {
	_internal::_matmul::eigen_matmul(
			dst,
			dst_outer,
			dst_inner,
			dst_colmajor,
			lhs,
			lhs_outer,
			lhs_colmajor,
			rhs,
			rhs_outer,
			rhs_colmajor,
			alpha,
			rows,
			cols,
			depth);
}

} // namespace FAER_ABI_VERSION
} // namespace fae
#include "faer/internal/epilogue.hpp"
