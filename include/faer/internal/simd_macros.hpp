#ifndef FAER_SIMD_MACROS_HPP_SUK0JK8XS
#define FAER_SIMD_MACROS_HPP_SUK0JK8XS

#include <simde/simde-features.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <simde/x86/sse.h>

#define FAER_mm_prefetch simde_mm_prefetch
#define FAER_PACK_SIZE SIMDE_NATURAL_VECTOR_SIZE

#define FAER_MM_SHUFFLE SIMDE_MM_SHUFFLE
#define FAER_m128 simde__m128
#define FAER_m128d simde__m128d
#define FAER_m256 simde__m256
#define FAER_m256d simde__m256d
#define FAER_m128i simde__m128i
#define FAER_m256i simde__m256i

#define FAER_mm256_add_pd simde_mm256_add_pd
#define FAER_mm256_add_ps simde_mm256_add_ps
#define FAER_mm256_cvtpd_ps simde_mm256_cvtpd_ps
#define FAER_mm256_cvtps_pd simde_mm256_cvtps_pd
#define FAER_mm256_div_pd simde_mm256_div_pd
#define FAER_mm256_div_ps simde_mm256_div_ps
#define FAER_mm256_extractf128_pd simde_mm256_extractf128_pd
#define FAER_mm256_extractf128_ps simde_mm256_extractf128_ps
#define FAER_mm256_fmadd_pd simde_mm256_fmadd_pd
#define FAER_mm256_fmadd_ps simde_mm256_fmadd_ps
#define FAER_mm256_fmsub_pd simde_mm256_fmsub_pd
#define FAER_mm256_fmsub_ps simde_mm256_fmsub_ps
#define FAER_mm256_fnmadd_pd simde_mm256_fnmadd_pd
#define FAER_mm256_fnmadd_ps simde_mm256_fnmadd_ps
#define FAER_mm256_fnmsub_pd simde_mm256_fnmsub_pd
#define FAER_mm256_fnmsub_ps simde_mm256_fnmsub_ps
#define FAER_mm256_maskload_pd simde_mm256_maskload_pd
#define FAER_mm256_maskload_ps simde_mm256_maskload_ps
#define FAER_mm256_maskstore_pd simde_mm256_maskstore_pd
#define FAER_mm256_maskstore_ps simde_mm256_maskstore_ps
#define FAER_mm256_mul_pd simde_mm256_mul_pd
#define FAER_mm256_mul_ps simde_mm256_mul_ps
#define FAER_mm256_neg_pd(V) simde_mm256_xor_pd(V, simde_mm256_set1_pd(SIMDE_FLOAT64_C(-0.0)))
#define FAER_mm256_neg_ps(V) simde_mm256_xor_ps(V, simde_mm256_set1_ps(SIMDE_FLOAT32_C(-0.0)))
#define FAER_mm256_permute2f128_pd simde_mm256_permute2f128_pd
#define FAER_mm256_permute2f128_ps simde_mm256_permute2f128_ps
#define FAER_mm256_set1_pd simde_mm256_set1_pd
#define FAER_mm256_set1_ps simde_mm256_set1_ps
#define FAER_mm256_set_epi32 simde_mm256_set_epi32
#define FAER_mm256_set_epi64x simde_mm256_set_epi64x
#define FAER_mm256_set_m128 simde_mm256_set_m128
#define FAER_mm256_set_pd simde_mm256_set_pd
#define FAER_mm256_set_ps simde_mm256_set_ps
#define FAER_mm256_setzero_pd simde_mm256_setzero_pd
#define FAER_mm256_setzero_ps simde_mm256_setzero_ps
#define FAER_mm256_shuffle_pd simde_mm256_shuffle_pd
#define FAER_mm256_shuffle_ps simde_mm256_shuffle_ps
#define FAER_mm256_sub_pd simde_mm256_sub_pd
#define FAER_mm256_sub_ps simde_mm256_sub_ps
#define FAER_mm256_unpackhi_ps simde_mm256_unpackhi_ps
#define FAER_mm256_unpacklo_ps simde_mm256_unpacklo_ps

#define FAER_mm_add_pd simde_mm_add_pd
#define FAER_mm_add_ps simde_mm_add_ps
#define FAER_mm_cvtpd_ps simde_mm_cvtpd_ps
#define FAER_mm_div_pd simde_mm_div_pd
#define FAER_mm_div_ps simde_mm_div_ps
#define FAER_mm_fmadd_pd simde_mm_fmadd_pd
#define FAER_mm_fmadd_ps simde_mm_fmadd_ps
#define FAER_mm_fmsub_pd simde_mm_fmsub_pd
#define FAER_mm_fmsub_ps simde_mm_fmsub_ps
#define FAER_mm_fnmadd_pd simde_mm_fnmadd_pd
#define FAER_mm_fnmadd_ps simde_mm_fnmadd_ps
#define FAER_mm_fnmsub_pd simde_mm_fnmsub_pd
#define FAER_mm_fnmsub_ps simde_mm_fnmsub_ps
#define FAER_mm_load_sd simde_mm_load_sd
#define FAER_mm_maskload_ps simde_mm_maskload_ps
#define FAER_mm_maskstore_ps simde_mm_maskstore_ps
#define FAER_mm_movehl_ps simde_mm_movehl_ps
#define FAER_mm_movelh_ps simde_mm_movelh_ps
#define FAER_mm_mul_pd simde_mm_mul_pd
#define FAER_mm_mul_ps simde_mm_mul_ps
#define FAER_mm_neg_pd(V) simde_mm_xor_pd(V, simde_mm_set1_pd(SIMDE_FLOAT64_C(-0.0)))
#define FAER_mm_neg_ps(V) simde_mm_xor_ps(V, simde_mm_set1_ps(SIMDE_FLOAT32_C(-0.0)))
#define FAER_mm_set1_pd simde_mm_set1_pd
#define FAER_mm_set1_ps simde_mm_set1_ps
#define FAER_mm_set_epi32 simde_mm_set_epi32
#define FAER_mm_set_pd simde_mm_set_pd
#define FAER_mm_set_ps simde_mm_set_ps
#define FAER_mm_setzero_pd simde_mm_setzero_pd
#define FAER_mm_setzero_ps simde_mm_setzero_ps
#define FAER_mm_store_sd simde_mm_store_sd
#define FAER_mm_sub_pd simde_mm_sub_pd
#define FAER_mm_sub_ps simde_mm_sub_ps
#define FAER_mm_unpackhi_pd simde_mm_unpackhi_pd
#define FAER_mm_unpackhi_ps simde_mm_unpackhi_ps
#define FAER_mm_unpacklo_pd simde_mm_unpacklo_pd
#define FAER_mm_unpacklo_ps simde_mm_unpacklo_ps

#endif /* end of include guard FAER_SIMD_MACROS_HPP_SUK0JK8XS */
