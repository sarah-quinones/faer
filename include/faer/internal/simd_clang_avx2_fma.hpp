#ifndef FAER_SIMD_GCC_AVX2_HPP_0REL7ACHS
#define FAER_SIMD_GCC_AVX2_HPP_0REL7ACHS

#include <veg/internal/macros.hpp>
#include <cstring>
#include "faer/internal/prologue.hpp"

HEDLEY_DIAGNOSTIC_PUSH
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"

typedef float __v4sf __attribute__((__vector_size__(16)));        // NOLINT
typedef double __v2df __attribute__((__vector_size__(16)));       // NOLINT
typedef int __v4si __attribute__((__vector_size__(16)));          // NOLINT
typedef unsigned int __v4su __attribute__((__vector_size__(16))); // NOLINT

typedef float __v8sf __attribute__((__vector_size__(32)));        // NOLINT
typedef double __v4df __attribute__((__vector_size__(32)));       // NOLINT
typedef int __v8si __attribute__((__vector_size__(32)));          // NOLINT
typedef unsigned int __v8su __attribute__((__vector_size__(32))); // NOLINT
typedef long long __v4di __attribute__((__vector_size__(32)));    // NOLINT

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));      // NOLINT
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));    // NOLINT
typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16))); // NOLINT

typedef float __m256 __attribute__((__vector_size__(32), __aligned__(32)));      // NOLINT
typedef double __m256d __attribute__((__vector_size__(32), __aligned__(32)));    // NOLINT
typedef long long __m256i __attribute__((__vector_size__(32), __aligned__(32))); // NOLINT

#if defined(__AVX512F__)
#define FAER_PACK_SIZE 512
#else
#define FAER_PACK_SIZE 256
#endif

#define FAER_MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w)) /* NOLINT */
#define FAER_m128 __m128
#define FAER_m128d __m128d
#define FAER_m256 __m256
#define FAER_m256d __m256d
#define FAER_m128i __m128i
#define FAER_m256i __m256i

namespace fae {
namespace internal {
namespace _intrin {
HEDLEY_ALWAYS_INLINE auto mm256_castps128_ps256(__m128 a) noexcept -> __m256 {
	return __builtin_shufflevector((__v4sf)a, (__v4sf)a, 0, 1, 2, 3, -1, -1, -1, -1); // NOLINT
}

HEDLEY_ALWAYS_INLINE auto mm_load_sd(double const* p) noexcept -> __m128d {
	double d; // NOLINT
	(::std::memcpy)(&d, p, sizeof(d));
	return __extension__(__m128d){d, 0};
}

HEDLEY_ALWAYS_INLINE void mm_store_sd(double* p, __m128d a) noexcept {
	(::std::memcpy)(p, &a, sizeof(double));
}

} // namespace _intrin
} // namespace internal
} // namespace fae

#define FAER_mm256_castps128_ps256(a) (::fae::internal::_intrin::mm256_castps128_ps256(a))

#define FAER_mm256_insertf128_ps(V1, V2, M)                                                                            \
	((__m256)__builtin_ia32_vinsertf128_ps256((__v8sf)(__m256)(V1), (__v4sf)(__m128)(V2), (int)(M)))

#define FAER_mm_set_epi32(e3, e2, e1, e0) (__extension__(__m128i)(__v4si){e0, e1, e2, e3})
#define FAER_mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0)                                                           \
	(__extension__(__m256i)(__v8si){e0, e1, e2, e3, e4, e5, e6, e7})
#define FAER_mm256_set_epi64x(e3, e2, e1, e0) (__extension__(__m256i)(__v4di){e0, e1, e2, e3})
#define FAER_mm256_set_m128(e1, e0) FAER_mm256_insertf128_ps(FAER_mm256_castps128_ps256(e0), e1, 1)

#define FAER_mm_set_ps(e3, e2, e1, e0) (__extension__(__m128){e0, e1, e2, e3})
#define FAER_mm_set_pd(e1, e0) (__extension__(__m128d){e0, e1})
#define FAER_mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0) (__extension__(__m256){e0, e1, e2, e3, e4, e5, e6, e7})
#define FAER_mm256_set_pd(e3, e2, e1, e0) (__extension__(__m256d){e0, e1, e2, e3})

#define FAER_mm_set1_ps(w) FAER_mm_set_ps(w, w, w, w)
#define FAER_mm_set1_pd(w) FAER_mm_set_pd(w, w)
#define FAER_mm256_set1_ps(w) FAER_mm256_set_ps(w, w, w, w, w, w, w, w)
#define FAER_mm256_set1_pd(w) FAER_mm256_set_pd(w, w, w, w)

#define FAER_mm_setzero_ps() FAER_mm_set1_ps(0)
#define FAER_mm_setzero_pd() FAER_mm_set1_pd(0)
#define FAER_mm256_setzero_ps() FAER_mm256_set1_ps(0)
#define FAER_mm256_setzero_pd() FAER_mm256_set1_pd(0)

#define FAER_mm_xor_ps(a, b) ((__m128)((__v4su)(a) ^ (__v4su)(b)))
#define FAER_mm_xor_pd(a, b) ((__m128d)((__v4su)(a) ^ (__v4su)(b)))
#define FAER_mm256_xor_ps(a, b) ((__m256)((__v8su)(a) ^ (__v8su)(b)))
#define FAER_mm256_xor_pd(a, b) ((__m256d)((__v8su)(a) ^ (__v8su)(b)))

#define FAER_mm_neg_ps(V) FAER_mm_xor_ps(V, FAER_mm_set1_ps((-0.0F)))
#define FAER_mm_neg_pd(V) FAER_mm_xor_pd(V, FAER_mm_set1_pd((-0.0)))
#define FAER_mm256_neg_ps(V) FAER_mm256_xor_ps(V, FAER_mm256_set1_ps((-0.0F)))
#define FAER_mm256_neg_pd(V) FAER_mm256_xor_pd(V, FAER_mm256_set1_pd((-0.0)))

#define FAER_mm_add_ps(a, b) ((__m128)((__v4sf)(a) + (__v4sf)(b)))
#define FAER_mm_sub_ps(a, b) ((__m128)((__v4sf)(a) - (__v4sf)(b)))
#define FAER_mm_mul_ps(a, b) ((__m128)((__v4sf)(a) * (__v4sf)(b)))
#define FAER_mm_div_ps(a, b) ((__m128)((__v4sf)(a) / (__v4sf)(b)))
#define FAER_mm_add_pd(a, b) ((__m128d)((__v2df)(a) + (__v2df)(b)))
#define FAER_mm_sub_pd(a, b) ((__m128d)((__v2df)(a) - (__v2df)(b)))
#define FAER_mm_mul_pd(a, b) ((__m128d)((__v2df)(a) * (__v2df)(b)))
#define FAER_mm_div_pd(a, b) ((__m128d)((__v2df)(a) / (__v2df)(b)))

#define FAER_mm256_add_ps(a, b) ((__m256)((__v8sf)(a) + (__v8sf)(b)))
#define FAER_mm256_sub_ps(a, b) ((__m256)((__v8sf)(a) - (__v8sf)(b)))
#define FAER_mm256_mul_ps(a, b) ((__m256)((__v8sf)(a) * (__v8sf)(b)))
#define FAER_mm256_div_ps(a, b) ((__m256)((__v8sf)(a) / (__v8sf)(b)))
#define FAER_mm256_add_pd(a, b) ((__m256d)((__v4df)(a) + (__v4df)(b)))
#define FAER_mm256_sub_pd(a, b) ((__m256d)((__v4df)(a) - (__v4df)(b)))
#define FAER_mm256_mul_pd(a, b) ((__m256d)((__v4df)(a) * (__v4df)(b)))
#define FAER_mm256_div_pd(a, b) ((__m256d)((__v4df)(a) / (__v4df)(b)))

#define FAER_mm_fmadd_ps(a, b, c) ((__m128)__builtin_ia32_vfmaddps((__v4sf)(a), (__v4sf)(b), (__v4sf)(c)))
#define FAER_mm_fmsub_ps(a, b, c) ((__m128)__builtin_ia32_vfmaddps((__v4sf)(a), (__v4sf)(b), -(__v4sf)(c)))
#define FAER_mm_fnmadd_ps(a, b, c) ((__m128)__builtin_ia32_vfmaddps(-(__v4sf)(a), (__v4sf)(b), (__v4sf)(c)))
#define FAER_mm_fnmsub_ps(a, b, c) ((__m128)__builtin_ia32_vfmaddps(-(__v4sf)(a), (__v4sf)(b), -(__v4sf)(c)))
#define FAER_mm_fmadd_pd(a, b, c) ((__m128d)__builtin_ia32_vfmaddpd((__v2df)(a), (__v2df)(b), (__v2df)(c)))
#define FAER_mm_fmsub_pd(a, b, c) ((__m128d)__builtin_ia32_vfmaddpd((__v2df)(a), (__v2df)(b), -(__v2df)(c)))
#define FAER_mm_fnmadd_pd(a, b, c) ((__m128d)__builtin_ia32_vfmaddpd(-(__v2df)(a), (__v2df)(b), (__v2df)(c)))
#define FAER_mm_fnmsub_pd(a, b, c) ((__m128d)__builtin_ia32_vfmaddpd(-(__v2df)(a), (__v2df)(b), -(__v2df)(c)))

#define FAER_mm256_fmadd_ps(a, b, c) ((__m256)__builtin_ia32_vfmaddps256((__v8sf)(a), (__v8sf)(b), (__v8sf)(c)))
#define FAER_mm256_fmsub_ps(a, b, c) ((__m256)__builtin_ia32_vfmaddps256((__v8sf)(a), (__v8sf)(b), -(__v8sf)(c)))
#define FAER_mm256_fnmadd_ps(a, b, c) ((__m256)__builtin_ia32_vfmaddps256(-(__v8sf)(a), (__v8sf)(b), (__v8sf)(c)))
#define FAER_mm256_fnmsub_ps(a, b, c) ((__m256)__builtin_ia32_vfmaddps256(-(__v8sf)(a), (__v8sf)(b), -(__v8sf)(c)))
#define FAER_mm256_fmadd_pd(a, b, c) ((__m256d)__builtin_ia32_vfmaddpd256((__v4df)(a), (__v4df)(b), (__v4df)(c)))
#define FAER_mm256_fmsub_pd(a, b, c) ((__m256d)__builtin_ia32_vfmaddpd256((__v4df)(a), (__v4df)(b), -(__v4df)(c)))
#define FAER_mm256_fnmadd_pd(a, b, c) ((__m256d)__builtin_ia32_vfmaddpd256(-(__v4df)(a), (__v4df)(b), (__v4df)(c)))
#define FAER_mm256_fnmsub_pd(a, b, c) ((__m256d)__builtin_ia32_vfmaddpd256(-(__v4df)(a), (__v4df)(b), -(__v4df)(c)))

#define FAER_mm256_cvtpd_ps(a) ((__m128)__builtin_ia32_cvtpd2ps256((__v4df)(a)))
#define FAER_mm256_cvtps_pd(a) ((__m256d) __builtin_convertvector((__v4sf)(a), __v4df))
#define FAER_mm_cvtpd_ps(a) ((__m128)__builtin_ia32_cvtpd2ps((__v2df)(a)))

#define FAER_mm256_extractf128_pd(V, M) ((__m128d)__builtin_ia32_vextractf128_pd256((__v4df)(__m256d)(V), (int)(M)))
#define FAER_mm256_extractf128_ps(V, M) ((__m128)__builtin_ia32_vextractf128_ps256((__v8sf)(__m256)(V), (int)(M)))

#define FAER_mm_load_sd(p) (::fae::internal::_intrin::mm_load_sd(p))
#define FAER_mm_store_sd(p, a) (::fae::internal::_intrin::mm_store_sd(p, a))

#define FAER_mm_maskload_ps(p, m) ((__m128)__builtin_ia32_maskloadps((const __v4sf*)(p), (__v4si)(m))) /* NOLINT */
#define FAER_mm_maskstore_ps(p, m, a)                                                                                  \
	(__builtin_ia32_maskstoreps((__v4sf*)(p), (__v4si)(m), (__v4sf)(a))) /***** NOLINT *****/

#define FAER_mm256_maskload_pd(p, m)                                                                                   \
	(__m256d) __builtin_ia32_maskloadpd256((const __v4df*)(p), (__v4di)(m)) /* NOLINT */
#define FAER_mm256_maskload_ps(p, m)                                                                                   \
	((__m256)__builtin_ia32_maskloadps256((const __v8sf*)(p), (__v8si)(m))) /* NOLINT */
#define FAER_mm256_maskstore_pd(p, m, a)                                                                               \
	(__builtin_ia32_maskstorepd256((__v4df*)(p), (__v4di)(m), (__v4df)(a))) /* NOLINT */
#define FAER_mm256_maskstore_ps(p, m, a)                                                                               \
	(__builtin_ia32_maskstoreps256((__v8sf*)(p), (__v8si)(m), (__v8sf)(a))) /* NOLINT */

#define FAER_mm256_permute2f128_pd(V1, V2, M)                                                                          \
	((__m256d)__builtin_ia32_vperm2f128_pd256((__v4df)(__m256d)(V1), (__v4df)(__m256d)(V2), (int)(M)))
#define FAER_mm256_permute2f128_ps(V1, V2, M)                                                                          \
	((__m256)__builtin_ia32_vperm2f128_ps256((__v8sf)(__m256)(V1), (__v8sf)(__m256)(V2), (int)(M)))

#define FAER_mm_movehl_ps(a, b) ((__m128)__builtin_shufflevector((__v4sf)(a), (__v4sf)(b), 6, 7, 2, 3))
#define FAER_mm_movelh_ps(a, b) ((__m128)__builtin_shufflevector((__v4sf)(a), (__v4sf)(b), 0, 1, 4, 5))

#define FAER_mm_unpackhi_ps(a, b) ((__m128)__builtin_shufflevector((__v4sf)(a), (__v4sf)(b), 2, 6, 3, 7))
#define FAER_mm_unpacklo_ps(a, b) ((__m128)__builtin_shufflevector((__v4sf)(a), (__v4sf)(b), 0, 4, 1, 5))
#define FAER_mm_unpackhi_pd(a, b) ((__m128d)__builtin_shufflevector((__v2df)(a), (__v2df)(b), 1, 2 + 1))
#define FAER_mm_unpacklo_pd(a, b) ((__m128d)__builtin_shufflevector((__v2df)(a), (__v2df)(b), 0, 2 + 0))

#define FAER_mm256_unpackhi_ps(a, b)                                                                                   \
	((__m256)__builtin_shufflevector((__v8sf)(a), (__v8sf)(b), 2, 10, 2 + 1, 10 + 1, 6, 14, 6 + 1, 14 + 1))
#define FAER_mm256_unpacklo_ps(a, b)                                                                                   \
	((__m256)__builtin_shufflevector((__v8sf)(a), (__v8sf)(b), 0, 8, 0 + 1, 8 + 1, 4, 12, 4 + 1, 12 + 1))

#define FAER_mm256_shuffle_pd(a, b, mask)                                                                              \
	((__m256d)__builtin_ia32_shufpd256((__v4df)(__m256d)(a), (__v4df)(__m256d)(b), (int)(mask)))

#define FAER_mm256_shuffle_ps(a, b, mask)                                                                              \
	(__m256) __builtin_ia32_shufps256((__v8sf)(__m256)(a), (__v8sf)(__m256)(b), (int)(mask))

HEDLEY_DIAGNOSTIC_POP

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_SIMD_GCC_AVX2_HPP_0REL7ACHS */
