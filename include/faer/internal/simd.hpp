#ifndef FAER_SIMD_HPP_ZAODKO1TS
#define FAER_SIMD_HPP_ZAODKO1TS

#if defined(__clang__) && defined(__AVX2__) && defined(__FMA__)
#include "faer/internal/simd_clang_avx2_fma.hpp"
#else
#include "faer/internal/simd_macros.hpp"
#endif
#include <veg/type_traits/tags.hpp>

#include <cstring>

#include "faer/internal/prologue.hpp"

HEDLEY_DIAGNOSTIC_PUSH
#ifdef __clang__
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"

static_assert(static_cast<unsigned char>(-1) == 255, ".");
static_assert(sizeof(float) == 4, ".");
static_assert(sizeof(double) == 8, ".");

#if FAER_PACK_SIZE >= 512
#define FAER_N_REGISTERS 32
#else
#define FAER_N_REGISTERS 16
#endif

#if FAER_PACK_SIZE >= 512
#define FAER_HAS_QUARTER_PACK 1
#else
#define FAER_HAS_QUARTER_PACK 0
#endif

#if FAER_PACK_SIZE >= 256
#define FAER_HAS_HALF_PACK 1
#else
#define FAER_HAS_HALF_PACK 0
#endif

namespace fae {

using veg::i64;
using veg::usize;

namespace internal {
HEDLEY_ALWAYS_INLINE constexpr auto max_(i64 n, i64 m) noexcept -> i64 {
	return n > m ? n : m;
}
HEDLEY_ALWAYS_INLINE constexpr auto min_(i64 n, i64 m) noexcept -> i64 {
	return n < m ? n : m;
}
HEDLEY_ALWAYS_INLINE constexpr auto round_down(i64 n, i64 k) noexcept -> i64 {
	return (n / k) * k;
}
HEDLEY_ALWAYS_INLINE constexpr auto round_up(i64 n, i64 k) noexcept -> i64 {
	return ((n + k - 1) / k) * k;
}

namespace ptr {
template <typename T>
auto read(T const* ptr) noexcept -> T {
	T out;
	std::memcpy(&out, ptr, sizeof(T));
	return out;
}

template <typename T>
void write(T* ptr, T value) noexcept {
	std::memcpy(ptr, &value, sizeof(T));
}

template <typename T>
auto incr(T* ptr, i64 n) noexcept -> T* {
	using Ptr = T*;
	using VoidPtr = void*;
	using BytePtr = unsigned char*;
	return Ptr(VoidPtr(BytePtr(ptr) + n * i64{sizeof(T)}));
}
} // namespace ptr

namespace _simd {
template <usize N>
struct NumTag {};

template <typename T, usize N>
struct sized_pack_type;

template <typename T>
using Pack = typename sized_pack_type<T, FAER_PACK_SIZE / 8 / sizeof(T)>::Pack;
template <typename T>
using PackHalf = typename sized_pack_type<T, FAER_PACK_SIZE / 8 / (1 + FAER_HAS_HALF_PACK) / sizeof(T)>::Pack;
template <typename T>
using PackQuarter = typename sized_pack_type<T, FAER_PACK_SIZE / 8 / (1 + 3 * FAER_HAS_QUARTER_PACK) / sizeof(T)>::Pack;

template <typename T, usize N>
struct sized_pack_traits;

template <typename T>
using pack_traits = sized_pack_traits<T, FAER_PACK_SIZE / 8 / sizeof(T)>;
template <typename T>
using pack_half_traits = sized_pack_traits<T, FAER_PACK_SIZE / 8 / (1 + FAER_HAS_HALF_PACK) / sizeof(T)>;
template <typename T>
using pack_quarter_traits = sized_pack_traits<T, FAER_PACK_SIZE / 8 / (1 + 3 * FAER_HAS_QUARTER_PACK) / sizeof(T)>;

#define FAER_IMPL_LOAD_STORE                                                                                           \
	HEDLEY_ALWAYS_INLINE                                                                                                 \
	static auto(load)(Type const* src) noexcept->Pack {                                                                  \
		Pack out;                                                                                                          \
		(::std::memcpy)(&out, src, sizeof(out));                                                                           \
		return out;                                                                                                        \
	}                                                                                                                    \
	HEDLEY_ALWAYS_INLINE                                                                                                 \
	static void(store)(Type * dst, Pack const& src) noexcept { (::std::memcpy)(dst, &src, sizeof(src)); }                \
	VEG_NOM_SEMICOLON

#define FAER_IMPL_SIMD_1(Fn, FnImpl)                                                                                   \
	HEDLEY_ALWAYS_INLINE static auto(Fn)(Pack const& a) noexcept->Pack { return {FnImpl(a.v)}; }                         \
	VEG_NOM_SEMICOLON
#define FAER_IMPL_SIMD_2(Fn, FnImpl)                                                                                   \
	HEDLEY_ALWAYS_INLINE static auto(Fn)(Pack const& a, Pack const& b) noexcept->Pack { return {FnImpl(a.v, b.v)}; }     \
	VEG_NOM_SEMICOLON
#define FAER_IMPL_SIMD_3(Fn, FnImpl)                                                                                   \
	HEDLEY_ALWAYS_INLINE static auto(Fn)(Pack const& a, Pack const& b, Pack const& c) noexcept->Pack {                   \
		return {FnImpl(a.v, b.v, c.v)};                                                                                    \
	}                                                                                                                    \
	VEG_NOM_SEMICOLON
#define FAER_IMPL_SIMD_CVT(Target, FnImpl)                                                                             \
	HEDLEY_ALWAYS_INLINE static auto(cvt)(                                                                               \
			::veg::Tag<Target> /*tag*/, Pack const& a) noexcept->sized_pack_type<Target, (size)>::Pack {                     \
		return {FnImpl(a.v)};                                                                                              \
	}                                                                                                                    \
	VEG_NOM_SEMICOLON

template <typename T>
struct sized_pack_type<T, 1> {
	using Pack = T;
};

template <>
struct sized_pack_type<float, 2> {
	struct Pack {
		FAER_m128 v;
	};
};
template <>
struct sized_pack_type<float, 4> {
	struct Pack {
		FAER_m128 v;
	};
};
template <>
struct sized_pack_type<float, 8> {
	struct Pack {
		FAER_m256 v;
	};
};

template <>
struct sized_pack_type<double, 2> {
	struct Pack {
		FAER_m128d v;
	};
};
template <>
struct sized_pack_type<double, 4> {
	struct Pack {
		FAER_m256d v;
	};
};

#define FAER_IMPL_SIMD_FLOAT(T, Size, Prefix, Suffix, ...)                                                             \
	template <>                                                                                                          \
	struct sized_pack_traits<T, Size> {                                                                                  \
		using Type = T;                                                                                                    \
		static constexpr usize size = Size;                                                                                \
		using Pack = ::fae::internal::_simd::sized_pack_type<Type, size>::Pack;                                            \
                                                                                                                       \
		FAER_IMPL_LOAD_STORE;                                                                                              \
                                                                                                                       \
		HEDLEY_ALWAYS_INLINE                                                                                               \
		static auto(broadcast)(Type value) noexcept -> Pack { return {FAER_##Prefix##_set1_p##Suffix(value)}; }            \
                                                                                                                       \
		HEDLEY_ALWAYS_INLINE                                                                                               \
		static auto(zero)() noexcept -> Pack { return {FAER_##Prefix##_setzero_p##Suffix()}; }                             \
                                                                                                                       \
		FAER_IMPL_SIMD_1(neg, FAER_##Prefix##_neg_p##Suffix);                                                              \
		FAER_IMPL_SIMD_2(add, FAER_##Prefix##_add_p##Suffix);                                                              \
		FAER_IMPL_SIMD_2(sub, FAER_##Prefix##_sub_p##Suffix);                                                              \
		FAER_IMPL_SIMD_2(mul, FAER_##Prefix##_mul_p##Suffix);                                                              \
		FAER_IMPL_SIMD_2(div, FAER_##Prefix##_div_p##Suffix);                                                              \
                                                                                                                       \
		FAER_IMPL_SIMD_3(fmadd, FAER_##Prefix##_fmadd_p##Suffix);                                                          \
		FAER_IMPL_SIMD_3(fmsub, FAER_##Prefix##_fmsub_p##Suffix);                                                          \
		FAER_IMPL_SIMD_3(fnmadd, FAER_##Prefix##_fnmadd_p##Suffix);                                                        \
		FAER_IMPL_SIMD_3(fnmsub, FAER_##Prefix##_fnmsub_p##Suffix);                                                        \
                                                                                                                       \
		HEDLEY_ALWAYS_INLINE static auto maybe_fmadd(int use_a, Pack a, Pack b, Pack c) noexcept -> Pack {                 \
			switch (use_a) {                                                                                                 \
			case 1:                                                                                                          \
				return sized_pack_traits::add(b, c);                                                                           \
			case -1:                                                                                                         \
				return sized_pack_traits::sub(c, b);                                                                           \
			default:                                                                                                         \
				return sized_pack_traits::fmadd(a, b, c);                                                                      \
			}                                                                                                                \
		}                                                                                                                  \
                                                                                                                       \
		__VA_ARGS__                                                                                                        \
	}

#define FAER_MAYBE_MASK                                                                                                \
	HEDLEY_ALWAYS_INLINE static auto maybe_mask_load(bool use_mask, Type const* src, i64 N) noexcept->Pack {             \
		if (use_mask) {                                                                                                    \
			return sized_pack_traits::mask_load(src, N);                                                                     \
		} else {                                                                                                           \
			return sized_pack_traits::load(src);                                                                             \
		}                                                                                                                  \
	}                                                                                                                    \
	HEDLEY_ALWAYS_INLINE static void maybe_mask_store(bool use_mask, Type* dest, Pack src, i64 N) noexcept {             \
		if (use_mask) {                                                                                                    \
			sized_pack_traits::mask_store(dest, src, N);                                                                     \
		} else {                                                                                                           \
			sized_pack_traits::store(dest, src);                                                                             \
		}                                                                                                                  \
	}                                                                                                                    \
	VEG_NOM_SEMICOLON

#define FAER_MAYBE_GATHER                                                                                              \
	HEDLEY_ALWAYS_INLINE static auto maybe_gather(bool contig, Type const* src, i64 stride) noexcept->Pack {             \
		if (contig) {                                                                                                      \
			return sized_pack_traits::load(src);                                                                             \
		} else {                                                                                                           \
			return sized_pack_traits::gather(src, stride);                                                                   \
		}                                                                                                                  \
	}                                                                                                                    \
	VEG_NOM_SEMICOLON

template <typename T>
struct sized_pack_traits<T, 1> {
	using Type = T;
	static constexpr usize size = 1;
	using Pack = T;

	FAER_IMPL_LOAD_STORE;
	HEDLEY_ALWAYS_INLINE static auto broadcast(Type value) noexcept -> Pack { return value; }
	HEDLEY_ALWAYS_INLINE static auto zero() noexcept -> Pack { return Pack{0}; }

	HEDLEY_ALWAYS_INLINE static auto neg(Pack a) noexcept -> Pack { return -a; }
	HEDLEY_ALWAYS_INLINE static auto add(Pack a, Pack b) noexcept -> Pack { return a + b; }
	HEDLEY_ALWAYS_INLINE static auto sub(Pack a, Pack b) noexcept -> Pack { return a - b; }
	HEDLEY_ALWAYS_INLINE static auto mul(Pack a, Pack b) noexcept -> Pack { return a * b; }
	HEDLEY_ALWAYS_INLINE static auto div(Pack a, Pack b) noexcept -> Pack { return a / b; }

	HEDLEY_ALWAYS_INLINE static auto fmadd(Pack a, Pack b, Pack c) noexcept -> Pack {
#ifdef __clang__
#pragma STDC FP_CONTRACT ON
#endif
		return a * b + c;
	}
	HEDLEY_ALWAYS_INLINE static auto fmsub(Pack a, Pack b, Pack c) noexcept -> Pack {
		return sized_pack_traits::fmadd(a, b, -c);
	}
	HEDLEY_ALWAYS_INLINE static auto fnmadd(Pack a, Pack b, Pack c) noexcept -> Pack {
		return sized_pack_traits::fmadd(-a, b, c);
	}
	HEDLEY_ALWAYS_INLINE static auto fnmsub(Pack a, Pack b, Pack c) noexcept -> Pack {
		return sized_pack_traits::fmadd(-a, b, -c);
	}

	HEDLEY_ALWAYS_INLINE static auto maybe_fmadd(int use_a, Pack a, Pack b, Pack c) noexcept -> Pack {
		switch (use_a) {
		case 1:
			return sized_pack_traits::add(b, c);
		case -1:
			return sized_pack_traits::sub(c, b);
		default:
			return sized_pack_traits::fmadd(a, b, c);
		}
	}

	HEDLEY_ALWAYS_INLINE static auto gather(Type const* src, i64 /*stride*/) noexcept -> Pack {
		return sized_pack_traits::load(src);
	}

	HEDLEY_ALWAYS_INLINE static auto maybe_gather(bool /*contig*/, Type const* src, i64 /*stride*/) noexcept -> Pack {
		return sized_pack_traits::load(src);
	}

	HEDLEY_ALWAYS_INLINE static auto maybe_mask_load(bool /*use_mask*/, Type const* src, i64 /*N*/) noexcept -> Pack {
		return sized_pack_traits::load(src);
	}
	HEDLEY_ALWAYS_INLINE static void maybe_mask_store(bool /*use_mask*/, Type* dest, Pack src, i64 /*N*/) noexcept {
		sized_pack_traits::store(dest, src);
	}
};

FAER_IMPL_SIMD_FLOAT(
		double, 2, mm, d, FAER_IMPL_SIMD_CVT(float, FAER_mm_cvtpd_ps);

		HEDLEY_ALWAYS_INLINE static auto mask_load(Type const* src, i64 /*N*/) noexcept->Pack {
			// N must be 1
			return {FAER_mm_load_sd(src)};
		}

		HEDLEY_ALWAYS_INLINE static void mask_store(Type* dest, Pack src, i64 /*N*/) noexcept {
			// N must be 1
			FAER_mm_store_sd(dest, src.v);
		}

		FAER_MAYBE_MASK;
		FAER_MAYBE_GATHER;

		HEDLEY_ALWAYS_INLINE static auto gather(Type const* src, i64 stride) noexcept->Pack {
			return {FAER_mm_set_pd( //
					ptr::read(ptr::incr(src, 1 * stride)),
					ptr::read(ptr::incr(src, 0 * stride)))};
		}

		HEDLEY_ALWAYS_INLINE static void scatter(Type* dest, Pack p, i64 stride) noexcept {
			Type clone[size];
			std::memcpy(&clone, &p, sizeof(clone));
			ptr::write(ptr::incr(dest, 0 * stride), clone[0]);
			ptr::write(ptr::incr(dest, 1 * stride), clone[1]);
		}

		HEDLEY_ALWAYS_INLINE static void transpose(NumTag<2>, Pack* kernel) {
			FAER_m128d tmp = FAER_mm_unpackhi_pd(kernel[0].v, kernel[1].v);
			kernel[0] = {FAER_mm_unpacklo_pd(kernel[0].v, kernel[1].v)};
			kernel[1] = {tmp};
		}

);
FAER_IMPL_SIMD_FLOAT(
		double,
		4,
		mm256,
		d,

		FAER_IMPL_SIMD_CVT(float, FAER_mm256_cvtpd_ps);

		HEDLEY_ALWAYS_INLINE static auto mask(i64 N)
				->FAER_m256i {
					FAER_m256i const masks[] = {
							FAER_mm256_set_epi64x(0, 0, 0, -1),
							FAER_mm256_set_epi64x(0, 0, -1, -1),
							FAER_mm256_set_epi64x(0, -1, -1, -1),
					};
					return masks[N];
				}

		HEDLEY_ALWAYS_INLINE static auto mask_load(Type const* src, i64 N) noexcept->Pack {
			return {FAER_mm256_maskload_pd(src, mask(N - 1))};
		}

		HEDLEY_ALWAYS_INLINE static void mask_store(Type* dest, Pack src, i64 N) noexcept {
			FAER_mm256_maskstore_pd(dest, mask(N - 1), src.v);
		}

		FAER_MAYBE_MASK;
		FAER_MAYBE_GATHER;

		HEDLEY_ALWAYS_INLINE static auto first_half(Pack a) noexcept->sized_pack_type<Type, size / 2>::Pack {
			return {FAER_mm256_extractf128_pd(a.v, 0)};
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto second_half(Pack a) noexcept->sized_pack_type<Type, size / 2>::Pack {
			return {FAER_mm256_extractf128_pd(a.v, 1)};
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto broadcast_blocks_of_4(Type const* src) noexcept->Pack {
			return sized_pack_traits::broadcast(ptr::read(src));
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto gather(Type const* src, i64 stride) noexcept->Pack {
			return {FAER_mm256_set_pd(

					ptr::read(ptr::incr(src, 3 * stride)),
					ptr::read(ptr::incr(src, 2 * stride)),
					ptr::read(ptr::incr(src, 1 * stride)),
					ptr::read(ptr::incr(src, 0 * stride))

							)};
		}

		HEDLEY_ALWAYS_INLINE static void scatter(Type* dest, Pack p, i64 stride) noexcept {
			Type clone[size];
			std::memcpy(&clone, &p, sizeof(clone));
			ptr::write(ptr::incr(dest, 0 * stride), clone[0]);
			ptr::write(ptr::incr(dest, 1 * stride), clone[1]);
			ptr::write(ptr::incr(dest, 2 * stride), clone[2]);
			ptr::write(ptr::incr(dest, 3 * stride), clone[3]);
		}

		HEDLEY_ALWAYS_INLINE static void transpose(NumTag<4>, Pack* kernel) {
			FAER_m256d T0 = FAER_mm256_shuffle_pd(kernel[0].v, kernel[1].v, 15);
			FAER_m256d T1 = FAER_mm256_shuffle_pd(kernel[0].v, kernel[1].v, 0);
			FAER_m256d T2 = FAER_mm256_shuffle_pd(kernel[2].v, kernel[3].v, 15);
			FAER_m256d T3 = FAER_mm256_shuffle_pd(kernel[2].v, kernel[3].v, 0);

			kernel[1].v = FAER_mm256_permute2f128_pd(T0, T2, 32);
			kernel[3].v = FAER_mm256_permute2f128_pd(T0, T2, 49);
			kernel[0].v = FAER_mm256_permute2f128_pd(T1, T3, 32);
			kernel[2].v = FAER_mm256_permute2f128_pd(T1, T3, 49);
		}

);

FAER_IMPL_SIMD_FLOAT(
		float,
		4,
		mm,
		s,

		FAER_IMPL_SIMD_CVT(double, FAER_mm256_cvtps_pd);

		HEDLEY_ALWAYS_INLINE static auto mask(i64 N)
				->FAER_m128i {
					FAER_m128i const masks[] = {
							FAER_mm_set_epi32(0, 0, 0, -1),
							FAER_mm_set_epi32(0, 0, -1, -1),
							FAER_mm_set_epi32(0, -1, -1, -1),
					};
					return masks[N];
				}

		HEDLEY_ALWAYS_INLINE static auto mask_load(Type const* src, i64 N) noexcept->Pack {
			return {FAER_mm_maskload_ps(src, mask(N - 1))};
		}

		HEDLEY_ALWAYS_INLINE static void mask_store(Type* dest, Pack src, i64 N) noexcept {
			FAER_mm_maskstore_ps(dest, mask(N - 1), src.v);
		}

		FAER_MAYBE_MASK;
		FAER_MAYBE_GATHER;

		HEDLEY_ALWAYS_INLINE static auto broadcast_blocks_of_4(Type const* src) noexcept->Pack {
			return sized_pack_traits::broadcast(ptr::read(src));
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto gather(Type const* src, i64 stride) noexcept->Pack {
			return {FAER_mm_set_ps(

					ptr::read(ptr::incr(src, 3 * stride)),
					ptr::read(ptr::incr(src, 2 * stride)),
					ptr::read(ptr::incr(src, 1 * stride)),
					ptr::read(ptr::incr(src, 0 * stride))

							)};
		}

		HEDLEY_ALWAYS_INLINE static void scatter(Type* dest, Pack p, i64 stride) noexcept {
			Type clone[size];
			std::memcpy(&clone, &p, sizeof(clone));
			ptr::write(ptr::incr(dest, 0 * stride), clone[0]);
			ptr::write(ptr::incr(dest, 1 * stride), clone[1]);
			ptr::write(ptr::incr(dest, 2 * stride), clone[2]);
			ptr::write(ptr::incr(dest, 3 * stride), clone[3]);
		}

		HEDLEY_ALWAYS_INLINE static void transpose(NumTag<4>, Pack* kernel) {
			FAER_m128 tmp3 = FAER_mm_unpacklo_ps((kernel[0].v), (kernel[1].v));
			FAER_m128 tmp2 = FAER_mm_unpacklo_ps((kernel[2].v), (kernel[3].v));
			FAER_m128 tmp1 = FAER_mm_unpackhi_ps((kernel[0].v), (kernel[1].v));
			FAER_m128 tmp0 = FAER_mm_unpackhi_ps((kernel[2].v), (kernel[3].v));
			(kernel[0].v) = FAER_mm_movelh_ps(tmp0, tmp2);
			(kernel[1].v) = FAER_mm_movehl_ps(tmp2, tmp0);
			(kernel[2].v) = FAER_mm_movelh_ps(tmp1, tmp3);
			(kernel[3].v) = FAER_mm_movehl_ps(tmp3, tmp1);
		}

);
FAER_IMPL_SIMD_FLOAT(
		float,
		8,
		mm256,
		s,
		// TODO: PR to simde
		/* FAER_IMPL_SIMD_CVT(double, FAER_mm512_cvtps_pd);*/

		HEDLEY_ALWAYS_INLINE static auto mask(i64 N)
				->FAER_m256i {
					static FAER_m256i const masks[] = {
							FAER_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
							FAER_mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
							FAER_mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
							FAER_mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
							FAER_mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
							FAER_mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
							FAER_mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
					};
					return masks[N];
				}

		HEDLEY_ALWAYS_INLINE static auto mask_load(Type const* src, i64 N) noexcept->Pack {
			return {FAER_mm256_maskload_ps(src, mask(N - 1))};
		}

		HEDLEY_ALWAYS_INLINE static void mask_store(Type* dest, Pack src, i64 N) noexcept {
			FAER_mm256_maskstore_ps(dest, mask(N - 1), src.v);
		}

		FAER_MAYBE_MASK;
		FAER_MAYBE_GATHER;

		HEDLEY_ALWAYS_INLINE static auto first_half(Pack a) noexcept->sized_pack_type<Type, size / 2>::Pack {
			return {FAER_mm256_extractf128_ps(a.v, 0)};
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto second_half(Pack a) noexcept->sized_pack_type<Type, size / 2>::Pack {
			return {FAER_mm256_extractf128_ps(a.v, 1)};
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto broadcast_blocks_of_4(Type const* src) noexcept->Pack {
			auto _0 = sized_pack_traits<Type, size / 2>::broadcast(ptr::read(src));
			auto _1 = sized_pack_traits<Type, size / 2>::broadcast(ptr::read(ptr::incr(src, 1)));
			return {FAER_mm256_set_m128(_1.v, _0.v)};
		} VEG_NOM_SEMICOLON;

		HEDLEY_ALWAYS_INLINE static auto gather(Type const* src, i64 stride) noexcept->Pack {
			return {FAER_mm256_set_ps(

					ptr::read(ptr::incr(src, 7 * stride)),
					ptr::read(ptr::incr(src, 6 * stride)),
					ptr::read(ptr::incr(src, 5 * stride)),
					ptr::read(ptr::incr(src, 4 * stride)),
					ptr::read(ptr::incr(src, 3 * stride)),
					ptr::read(ptr::incr(src, 2 * stride)),
					ptr::read(ptr::incr(src, 1 * stride)),
					ptr::read(ptr::incr(src, 0 * stride))

							)};
		}

		HEDLEY_ALWAYS_INLINE static void scatter(Type* dest, Pack p, i64 stride) noexcept {
			Type clone[size];
			std::memcpy(&clone, &p, sizeof(clone));
			ptr::write(ptr::incr(dest, 0 * stride), clone[0]);
			ptr::write(ptr::incr(dest, 1 * stride), clone[1]);
			ptr::write(ptr::incr(dest, 2 * stride), clone[2]);
			ptr::write(ptr::incr(dest, 3 * stride), clone[3]);
			ptr::write(ptr::incr(dest, 4 * stride), clone[4]);
			ptr::write(ptr::incr(dest, 5 * stride), clone[5]);
			ptr::write(ptr::incr(dest, 6 * stride), clone[6]);
			ptr::write(ptr::incr(dest, 7 * stride), clone[7]);
		}

		HEDLEY_ALWAYS_INLINE static void transpose(NumTag<4>, Pack* kernel) {
			FAER_m256 T0 = FAER_mm256_unpacklo_ps(kernel[0].v, kernel[1].v);
			FAER_m256 T1 = FAER_mm256_unpackhi_ps(kernel[0].v, kernel[1].v);
			FAER_m256 T2 = FAER_mm256_unpacklo_ps(kernel[2].v, kernel[3].v);
			FAER_m256 T3 = FAER_mm256_unpackhi_ps(kernel[2].v, kernel[3].v);

			FAER_m256 S0 = FAER_mm256_shuffle_ps(T0, T2, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S1 = FAER_mm256_shuffle_ps(T0, T2, FAER_MM_SHUFFLE(3, 2, 3, 2));
			FAER_m256 S2 = FAER_mm256_shuffle_ps(T1, T3, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S3 = FAER_mm256_shuffle_ps(T1, T3, FAER_MM_SHUFFLE(3, 2, 3, 2));

			kernel[0].v = FAER_mm256_permute2f128_ps(S0, S1, 0x20);
			kernel[1].v = FAER_mm256_permute2f128_ps(S2, S3, 0x20);
			kernel[2].v = FAER_mm256_permute2f128_ps(S0, S1, 0x31);
			kernel[3].v = FAER_mm256_permute2f128_ps(S2, S3, 0x31);
		}

		HEDLEY_ALWAYS_INLINE static void transpose(NumTag<8>, Pack* kernel) {
			FAER_m256 T0 = FAER_mm256_unpacklo_ps(kernel[0].v, kernel[1].v);
			FAER_m256 T1 = FAER_mm256_unpackhi_ps(kernel[0].v, kernel[1].v);
			FAER_m256 T2 = FAER_mm256_unpacklo_ps(kernel[2].v, kernel[3].v);
			FAER_m256 T3 = FAER_mm256_unpackhi_ps(kernel[2].v, kernel[3].v);
			FAER_m256 T4 = FAER_mm256_unpacklo_ps(kernel[4].v, kernel[5].v);
			FAER_m256 T5 = FAER_mm256_unpackhi_ps(kernel[4].v, kernel[5].v);
			FAER_m256 T6 = FAER_mm256_unpacklo_ps(kernel[6].v, kernel[7].v);
			FAER_m256 T7 = FAER_mm256_unpackhi_ps(kernel[6].v, kernel[7].v);
			FAER_m256 S0 = FAER_mm256_shuffle_ps(T0, T2, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S1 = FAER_mm256_shuffle_ps(T0, T2, FAER_MM_SHUFFLE(3, 2, 3, 2));
			FAER_m256 S2 = FAER_mm256_shuffle_ps(T1, T3, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S3 = FAER_mm256_shuffle_ps(T1, T3, FAER_MM_SHUFFLE(3, 2, 3, 2));
			FAER_m256 S4 = FAER_mm256_shuffle_ps(T4, T6, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S5 = FAER_mm256_shuffle_ps(T4, T6, FAER_MM_SHUFFLE(3, 2, 3, 2));
			FAER_m256 S6 = FAER_mm256_shuffle_ps(T5, T7, FAER_MM_SHUFFLE(1, 0, 1, 0));
			FAER_m256 S7 = FAER_mm256_shuffle_ps(T5, T7, FAER_MM_SHUFFLE(3, 2, 3, 2));
			kernel[0].v = FAER_mm256_permute2f128_ps(S0, S4, 0x20);
			kernel[1].v = FAER_mm256_permute2f128_ps(S1, S5, 0x20);
			kernel[2].v = FAER_mm256_permute2f128_ps(S2, S6, 0x20);
			kernel[3].v = FAER_mm256_permute2f128_ps(S3, S7, 0x20);
			kernel[4].v = FAER_mm256_permute2f128_ps(S0, S4, 0x31);
			kernel[5].v = FAER_mm256_permute2f128_ps(S1, S5, 0x31);
			kernel[6].v = FAER_mm256_permute2f128_ps(S2, S6, 0x31);
			kernel[7].v = FAER_mm256_permute2f128_ps(S3, S7, 0x31);
		}

);

} // namespace _simd
} // namespace internal
} // namespace fae

HEDLEY_DIAGNOSTIC_POP
#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_SIMD_HPP_ZAODKO1TS */
