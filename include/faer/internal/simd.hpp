#ifndef FAER_SIMD_HPP_RA2KIJ1TS
#define FAER_SIMD_HPP_RA2KIJ1TS

#include <cstring>
#include <simde/x86/fma.h>
#include <simde/x86/sse4.2.h>
#include <simde/x86/avx2.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <veg/util/assert.hpp>
#include <veg/slice.hpp>

namespace fae {
using veg::usize;
using veg::isize;
using f32 = float;
using f64 = double;

namespace simd {
template <typename T, usize N>
struct Pack;
enum struct Sign {
	pos,
	neg,
};
using Pos = veg::meta::constant<Sign, Sign::pos>;
using Neg = veg::meta::constant<Sign, Sign::neg>;

template <>
struct Pack<f32, 1> {
	simde__m128 _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm_setzero_ps(); }
	VEG_INLINE void broadcast(f32 const* ptr) noexcept { _ = simde_mm_load_ss(ptr); }
	VEG_INLINE void load_unaligned(f32 const* ptr) noexcept { _ = simde_mm_load_ss(ptr); }
	VEG_INLINE void load_aligned(f32 const* ptr) noexcept { _ = simde_mm_load_ss(ptr); }
	VEG_INLINE void store_unaligned(f32* ptr) const noexcept { simde_mm_store_ss(ptr, _); }
	VEG_INLINE void store_aligned(f32* ptr) const noexcept { simde_mm_store_ss(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f32, 1> { return *this; }

	VEG_INLINE void add(P lhs, P rhs) noexcept { _ = simde_mm_add_ss(lhs._, rhs._); }
	VEG_INLINE void sub(P lhs, P rhs) noexcept { _ = simde_mm_sub_ss(lhs._, rhs._); }
	VEG_INLINE void mul(P lhs, P rhs) noexcept { _ = simde_mm_mul_ss(lhs._, rhs._); }
	VEG_INLINE void div(P lhs, P rhs) noexcept { _ = simde_mm_div_ss(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmadd_ss(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmadd_ss(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmsub_ss(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmsub_ss(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept -> f32 { return simde_mm_cvtss_f32(_); }
	VEG_INLINE static void trans(Pack* /*p*/) noexcept {}
};
template <>
struct Pack<f32, 2> {
	simde__m128 _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm_setzero_ps(); }
	VEG_INLINE void broadcast(f32 const* ptr) noexcept { _ = simde_mm_set1_ps(*ptr); }
	VEG_INLINE void load_unaligned(f32 const* ptr) noexcept {
		f64 tmp; // NOLINT
		std::memcpy(&tmp, ptr, sizeof(f64));
		_ = simde_mm_castpd_ps(simde_mm_set1_pd(tmp));
	}
	VEG_INLINE void load_aligned(f32 const* ptr) noexcept { load_unaligned(ptr); }
	VEG_INLINE void store_unaligned(f32* ptr) const noexcept { std::memcpy(ptr, &_, sizeof(f64)); }
	VEG_INLINE void store_aligned(f32* ptr) const noexcept {
		simde_mm_store_sd(reinterpret_cast<f64*>(ptr), simde_mm_castps_pd(_));
	}
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f32, 1> { return {_}; }

	VEG_INLINE void add(P lhs, P rhs) noexcept { _ = simde_mm_add_ps(lhs._, rhs._); }
	VEG_INLINE void sub(P lhs, P rhs) noexcept { _ = simde_mm_sub_ps(lhs._, rhs._); }
	VEG_INLINE void mul(P lhs, P rhs) noexcept { _ = simde_mm_mul_ps(lhs._, rhs._); }
	VEG_INLINE void div(P lhs, P rhs) noexcept { _ = simde_mm_div_ps(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmsub_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmsub_ps(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept {
		simde__m128 _0_1_0_1 = _;
		simde__m128 _1_1_1_1 = simde_mm_movehdup_ps(_0_1_0_1);
		simde__m128 _01_x_x_x = simde_mm_add_ss(_0_1_0_1, _1_1_1_1);
		return simde_mm_cvtss_f32(_01_x_x_x);
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		simde__m128 p0 = p[0]._;
		simde__m128 p1 = p[1]._;

		auto tmp0 = simde_mm_unpacklo_ps(p0, p1);

		p[0]._ = simde_mm_movelh_ps(tmp0, tmp0);
		p[1]._ = simde_mm_movehl_ps(tmp0, tmp0);
	}
};
template <>
struct Pack<f32, 4> {
	simde__m128 _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm_setzero_ps(); }
	VEG_INLINE void broadcast(f32 const* ptr) noexcept { _ = simde_mm_set1_ps(*ptr); }
	VEG_INLINE void load_unaligned(f32 const* ptr) noexcept { _ = simde_mm_loadu_ps(ptr); }
	VEG_INLINE void load_aligned(f32 const* ptr) noexcept { _ = simde_mm_load_ps(ptr); }
	VEG_INLINE void store_unaligned(f32* ptr) const noexcept { simde_mm_storeu_ps(ptr, _); }
	VEG_INLINE void store_aligned(f32* ptr) const noexcept { simde_mm_store_ps(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f32, 1> { return {_}; }

	VEG_INLINE void add(P lhs, P rhs) noexcept { _ = simde_mm_add_ps(lhs._, rhs._); }
	VEG_INLINE void sub(P lhs, P rhs) noexcept { _ = simde_mm_sub_ps(lhs._, rhs._); }
	VEG_INLINE void mul(P lhs, P rhs) noexcept { _ = simde_mm_mul_ps(lhs._, rhs._); }
	VEG_INLINE void div(P lhs, P rhs) noexcept { _ = simde_mm_div_ps(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmsub_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmsub_ps(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept {
		simde__m128 _0_1_2_3 = _;
		simde__m128 _1_1_3_3 = simde_mm_movehdup_ps(_0_1_2_3);
		simde__m128 _01_x_23_x = simde_mm_add_ps(_0_1_2_3, _1_1_3_3);
		simde__m128 _23_x_xx_x = simde_mm_movehl_ps(_1_1_3_3, _01_x_23_x);
		simde__m128 _0123_x_x_x = simde_mm_add_ss(_01_x_23_x, _23_x_xx_x);
		return simde_mm_cvtss_f32(_0123_x_x_x);
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		SIMDE_MM_TRANSPOSE4_PS /* NOLINT */ (p[0]._, p[1]._, p[2]._, p[3]._);
	}
};
template <>
struct Pack<f32, 8> {
	simde__m256 _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm256_setzero_ps(); }
	VEG_INLINE void broadcast(f32 const* ptr) noexcept { _ = simde_mm256_set1_ps(*ptr); }
	VEG_INLINE void load_unaligned(f32 const* ptr) noexcept { _ = simde_mm256_loadu_ps(ptr); }
	VEG_INLINE void load_aligned(f32 const* ptr) noexcept { _ = simde_mm256_load_ps(ptr); }
	VEG_INLINE void store_unaligned(f32* ptr) const noexcept { simde_mm256_storeu_ps(ptr, _); }
	VEG_INLINE void store_aligned(f32* ptr) const noexcept { simde_mm256_store_ps(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f32, 1> { return {simde_mm256_castps256_ps128(_)}; }

	VEG_INLINE void add(P lhs, P rhs) noexcept { _ = simde_mm256_add_ps(lhs._, rhs._); }
	VEG_INLINE void sub(P lhs, P rhs) noexcept { _ = simde_mm256_sub_ps(lhs._, rhs._); }
	VEG_INLINE void mul(P lhs, P rhs) noexcept { _ = simde_mm256_mul_ps(lhs._, rhs._); }
	VEG_INLINE void div(P lhs, P rhs) noexcept { _ = simde_mm256_div_ps(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fnmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fmsub_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fnmsub_ps(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept -> f32 {
		Pack<f32, 4> lo = {simde_mm256_castps256_ps128(_)};
		Pack<f32, 4> hi = {simde_mm256_extractf128_ps(_, 1)};
		lo.add(lo, hi);
		return lo.horizontal_add();
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		simde__m256 T0 = simde_mm256_unpacklo_ps(p[0]._, p[1]._);
		simde__m256 T1 = simde_mm256_unpackhi_ps(p[0]._, p[1]._);
		simde__m256 T2 = simde_mm256_unpacklo_ps(p[2]._, p[3]._);
		simde__m256 T3 = simde_mm256_unpackhi_ps(p[2]._, p[3]._);
		simde__m256 T4 = simde_mm256_unpacklo_ps(p[4]._, p[5]._);
		simde__m256 T5 = simde_mm256_unpackhi_ps(p[4]._, p[5]._);
		simde__m256 T6 = simde_mm256_unpacklo_ps(p[6]._, p[7]._);
		simde__m256 T7 = simde_mm256_unpackhi_ps(p[6]._, p[7]._);

		simde__m256 S0 = simde_mm256_shuffle_ps(T0, T2, SIMDE_MM_SHUFFLE /* NOLINT */ (1, 0, 1, 0));
		simde__m256 S1 = simde_mm256_shuffle_ps(T0, T2, SIMDE_MM_SHUFFLE /* NOLINT */ (3, 2, 3, 2));
		simde__m256 S2 = simde_mm256_shuffle_ps(T1, T3, SIMDE_MM_SHUFFLE /* NOLINT */ (1, 0, 1, 0));
		simde__m256 S3 = simde_mm256_shuffle_ps(T1, T3, SIMDE_MM_SHUFFLE /* NOLINT */ (3, 2, 3, 2));
		simde__m256 S4 = simde_mm256_shuffle_ps(T4, T6, SIMDE_MM_SHUFFLE /* NOLINT */ (1, 0, 1, 0));
		simde__m256 S5 = simde_mm256_shuffle_ps(T4, T6, SIMDE_MM_SHUFFLE /* NOLINT */ (3, 2, 3, 2));
		simde__m256 S6 = simde_mm256_shuffle_ps(T5, T7, SIMDE_MM_SHUFFLE /* NOLINT */ (1, 0, 1, 0));
		simde__m256 S7 = simde_mm256_shuffle_ps(T5, T7, SIMDE_MM_SHUFFLE /* NOLINT */ (3, 2, 3, 2));

		p[0]._ = simde_mm256_permute2f128_ps(S0, S4, 0x20);
		p[1]._ = simde_mm256_permute2f128_ps(S1, S5, 0x20);
		p[2]._ = simde_mm256_permute2f128_ps(S2, S6, 0x20);
		p[3]._ = simde_mm256_permute2f128_ps(S3, S7, 0x20);
		p[4]._ = simde_mm256_permute2f128_ps(S0, S4, 0x31);
		p[5]._ = simde_mm256_permute2f128_ps(S1, S5, 0x31);
		p[6]._ = simde_mm256_permute2f128_ps(S2, S6, 0x31);
		p[7]._ = simde_mm256_permute2f128_ps(S3, S7, 0x31);
	}
};
#ifdef __AVX512F__
template <>
struct Pack<f32, 16> {
	__m512 _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = _mm512_setzero_ps(); }
	VEG_INLINE void broadcast(f32 const* ptr) noexcept { _ = _mm512_set1_ps(*ptr); }
	VEG_INLINE void load_unaligned(f32 const* ptr) noexcept { _ = _mm512_loadu_ps(ptr); }
	VEG_INLINE void load_aligned(f32 const* ptr) noexcept { _ = _mm512_load_ps(ptr); }
	VEG_INLINE void store_unaligned(f32* ptr) const noexcept { _mm512_storeu_ps(ptr, _); }
	VEG_INLINE void store_aligned(f32* ptr) const noexcept { _mm512_store_ps(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f32, 1> { return {_mm512_castps512_ps128(_)}; }

	VEG_INLINE void add(P lhs, P rhs) noexcept { _ = _mm512_add_ps(lhs._, rhs._); }
	VEG_INLINE void sub(P lhs, P rhs) noexcept { _ = _mm512_sub_ps(lhs._, rhs._); }
	VEG_INLINE void mul(P lhs, P rhs) noexcept { _ = _mm512_mul_ps(lhs._, rhs._); }
	VEG_INLINE void div(P lhs, P rhs) noexcept { _ = _mm512_div_ps(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = _mm512_fmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = _mm512_fnmadd_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = _mm512_fmsub_ps(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = _mm512_fnmsub_ps(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept -> f32 {
		Pack<f32, 8> lo = {_mm512_castps512_ps256(_)};
		Pack<f32, 8> hi = {_mm512_extractf32x8_ps(_, 1)};
		lo.add(lo, hi);
		return lo.horizontal_add();
	}
	VEG_INLINE static void trans(Pack* /*p*/) noexcept { VEG_UNIMPLEMENTED(); }
};
#endif

template <>
struct Pack<f64, 1> {
	simde__m128d _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm_setzero_pd(); }
	VEG_INLINE void broadcast(f64 const* ptr) noexcept { _ = simde_mm_load_sd(ptr); }
	VEG_INLINE void load_unaligned(f64 const* ptr) noexcept { _ = simde_mm_load_sd(ptr); }
	VEG_INLINE void load_aligned(f64 const* ptr) noexcept { _ = simde_mm_load_sd(ptr); }
	VEG_INLINE void store_unaligned(f64* ptr) const noexcept { simde_mm_store_sd(ptr, _); }
	VEG_INLINE void store_aligned(f64* ptr) const noexcept { simde_mm_store_sd(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f64, 1> { return *this; }

	VEG_INLINE void add(Pack lhs, Pack rhs) noexcept { _ = simde_mm_add_sd(lhs._, rhs._); }
	VEG_INLINE void sub(Pack lhs, Pack rhs) noexcept { _ = simde_mm_sub_sd(lhs._, rhs._); }
	VEG_INLINE void mul(Pack lhs, Pack rhs) noexcept { _ = simde_mm_mul_sd(lhs._, rhs._); }
	VEG_INLINE void div(Pack lhs, Pack rhs) noexcept { _ = simde_mm_div_sd(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmadd_sd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmadd_sd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmsub_sd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmsub_sd(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept -> f64 { return simde_mm_cvtsd_f64(_); }
	VEG_INLINE static void trans(Pack* /*p*/) noexcept {}
};
template <>
struct Pack<f64, 2> {
	simde__m128d _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm_setzero_pd(); }
	VEG_INLINE void broadcast(f64 const* ptr) noexcept { _ = simde_mm_set1_pd(*ptr); }
	VEG_INLINE void load_unaligned(f64 const* ptr) noexcept { _ = simde_mm_loadu_pd(ptr); }
	VEG_INLINE void load_aligned(f64 const* ptr) noexcept { _ = simde_mm_load_pd(ptr); }
	VEG_INLINE void store_unaligned(f64* ptr) const noexcept { simde_mm_storeu_pd(ptr, _); }
	VEG_INLINE void store_aligned(f64* ptr) const noexcept { simde_mm_store_pd(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f64, 1> { return {_}; }

	VEG_INLINE void add(Pack lhs, Pack rhs) noexcept { _ = simde_mm_add_pd(lhs._, rhs._); }
	VEG_INLINE void sub(Pack lhs, Pack rhs) noexcept { _ = simde_mm_sub_pd(lhs._, rhs._); }
	VEG_INLINE void mul(Pack lhs, Pack rhs) noexcept { _ = simde_mm_mul_pd(lhs._, rhs._); }
	VEG_INLINE void div(Pack lhs, Pack rhs) noexcept { _ = simde_mm_div_pd(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fmsub_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm_fnmsub_pd(lhs._, rhs._, accum._); }
	VEG_NODISCARD VEG_INLINE auto horizontal_add() const noexcept -> f64 {
		simde__m128d lo = _;
		simde__m128d hi = simde_mm_castps_pd(simde_mm_movehl_ps(simde_mm_undefined_ps(), simde_mm_castpd_ps(_)));
		return simde_mm_cvtsd_f64(simde_mm_add_sd(lo, hi));
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		simde__m128d tmp = simde_mm_unpackhi_pd(p[0]._, p[1]._);
		p[0]._ = simde_mm_unpacklo_pd(p[0]._, p[1]._);
		p[1]._ = tmp;
	}
};
template <>
struct Pack<f64, 4> {
	simde__m256d _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = simde_mm256_setzero_pd(); }
	VEG_INLINE void broadcast(f64 const* ptr) noexcept { _ = simde_mm256_set1_pd(*ptr); }
	VEG_INLINE void load_unaligned(f64 const* ptr) noexcept { _ = simde_mm256_loadu_pd(ptr); }
	VEG_INLINE void load_aligned(f64 const* ptr) noexcept { _ = simde_mm256_load_pd(ptr); }
	VEG_INLINE void store_unaligned(f64* ptr) const noexcept { simde_mm256_storeu_pd(ptr, _); }
	VEG_INLINE void store_aligned(f64* ptr) const noexcept { simde_mm256_store_pd(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f64, 1> { return {simde_mm256_castpd256_pd128(_)}; }

	VEG_INLINE void add(Pack lhs, Pack rhs) noexcept { _ = simde_mm256_add_pd(lhs._, rhs._); }
	VEG_INLINE void sub(Pack lhs, Pack rhs) noexcept { _ = simde_mm256_sub_pd(lhs._, rhs._); }
	VEG_INLINE void mul(Pack lhs, Pack rhs) noexcept { _ = simde_mm256_mul_pd(lhs._, rhs._); }
	VEG_INLINE void div(Pack lhs, Pack rhs) noexcept { _ = simde_mm256_div_pd(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fnmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fmsub_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = simde_mm256_fnmsub_pd(lhs._, rhs._, accum._); }
	VEG_NODISCARD auto horizontal_add() const noexcept -> f64 {
		Pack<f64, 2> lo = {simde_mm256_castpd256_pd128(_)};
		Pack<f64, 2> hi = {simde_mm256_extractf128_pd(_, 1)};
		lo.add(lo, hi);
		return lo.horizontal_add();
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		simde__m256d T0 = simde_mm256_shuffle_pd(p[0]._, p[1]._, 0x0);
		simde__m256d T1 = simde_mm256_shuffle_pd(p[0]._, p[1]._, 0xF);
		simde__m256d T2 = simde_mm256_shuffle_pd(p[2]._, p[3]._, 0x0);
		simde__m256d T3 = simde_mm256_shuffle_pd(p[2]._, p[3]._, 0xF);

		p[0]._ = simde_mm256_permute2f128_pd(T0, T2, 0x20);
		p[2]._ = simde_mm256_permute2f128_pd(T0, T2, 0x31);
		p[1]._ = simde_mm256_permute2f128_pd(T1, T3, 0x20);
		p[3]._ = simde_mm256_permute2f128_pd(T1, T3, 0x31);
	}
};
#ifdef __AVX512F__
template <>
struct Pack<f64, 8> {
	__m512d _;
	using P = Pack;

	VEG_INLINE void zero() noexcept { _ = _mm512_setzero_pd(); }
	VEG_INLINE void broadcast(f64 const* ptr) noexcept { _ = _mm512_set1_pd(*ptr); }
	VEG_INLINE void load_unaligned(f64 const* ptr) noexcept { _ = _mm512_loadu_pd(ptr); }
	VEG_INLINE void load_aligned(f64 const* ptr) noexcept { _ = _mm512_load_pd(ptr); }
	VEG_INLINE void store_unaligned(f64* ptr) const noexcept { _mm512_storeu_pd(ptr, _); }
	VEG_INLINE void store_aligned(f64* ptr) const noexcept { _mm512_store_pd(ptr, _); }
	VEG_NODISCARD VEG_INLINE auto cast_unit() const noexcept -> Pack<f64, 1> { return {_mm512_castpd512_pd128(_)}; }

	VEG_INLINE void add(Pack lhs, Pack rhs) noexcept { _ = _mm512_add_pd(lhs._, rhs._); }
	VEG_INLINE void sub(Pack lhs, Pack rhs) noexcept { _ = _mm512_sub_pd(lhs._, rhs._); }
	VEG_INLINE void mul(Pack lhs, Pack rhs) noexcept { _ = _mm512_mul_pd(lhs._, rhs._); }
	VEG_INLINE void div(Pack lhs, Pack rhs) noexcept { _ = _mm512_div_pd(lhs._, rhs._); }
	VEG_INLINE void fmadd(Pos, Pos, P lhs, P rhs, P accum) noexcept { _ = _mm512_fmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Pos, P lhs, P rhs, P accum) noexcept { _ = _mm512_fnmadd_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Pos, Neg, P lhs, P rhs, P accum) noexcept { _ = _mm512_fmsub_pd(lhs._, rhs._, accum._); }
	VEG_INLINE void fmadd(Neg, Neg, P lhs, P rhs, P accum) noexcept { _ = _mm512_fnmsub_pd(lhs._, rhs._, accum._); }
	VEG_NODISCARD auto horizontal_add() const noexcept -> f64 {
		Pack<f64, 4> lo = {_mm512_castpd512_pd256(_)};
		Pack<f64, 4> hi = {_mm512_extractf64x4_pd(_, 1)};
		lo.add(lo, hi);
		return lo.horizontal_add();
	}
	VEG_INLINE static void trans(Pack* p) noexcept {
		__m512 T0 = _mm512_unpacklo_pd(p[0]._, p[1]._);
		__m512 T1 = _mm512_unpackhi_pd(p[0]._, p[1]._);
		__m512 T2 = _mm512_unpacklo_pd(p[2]._, p[3]._);
		__m512 T3 = _mm512_unpackhi_pd(p[2]._, p[3]._);
		__m512 T4 = _mm512_unpacklo_pd(p[4]._, p[5]._);
		__m512 T5 = _mm512_unpackhi_pd(p[4]._, p[5]._);
		__m512 T6 = _mm512_unpacklo_pd(p[6]._, p[7]._);
		__m512 T7 = _mm512_unpackhi_pd(p[6]._, p[7]._);

		__m512 S0 = _mm512_shuffle_f64x2(T0, T2, 0x88);
		__m512 S1 = _mm512_shuffle_f64x2(T0, T2, 0xdd);
		__m512 S2 = _mm512_shuffle_f64x2(T1, T3, 0x88);
		__m512 S3 = _mm512_shuffle_f64x2(T1, T3, 0xdd);
		__m512 S4 = _mm512_shuffle_f64x2(T4, T6, 0x88);
		__m512 S5 = _mm512_shuffle_f64x2(T4, T6, 0xdd);
		__m512 S6 = _mm512_shuffle_f64x2(T5, T7, 0x88);
		__m512 S7 = _mm512_shuffle_f64x2(T5, T7, 0xdd);

		p[0]._ = _mm512_shuffle_f64x2(S0, S4, 0x88);
		p[2]._ = _mm512_shuffle_f64x2(S1, S5, 0x88);
		p[1]._ = _mm512_shuffle_f64x2(S2, S6, 0x88);
		p[3]._ = _mm512_shuffle_f64x2(S3, S7, 0x88);
		p[4]._ = _mm512_shuffle_f64x2(S0, S4, 0xdd);
		p[6]._ = _mm512_shuffle_f64x2(S1, S5, 0xdd);
		p[5]._ = _mm512_shuffle_f64x2(S2, S6, 0xdd);
		p[7]._ = _mm512_shuffle_f64x2(S3, S7, 0xdd);
	}
};
#endif

template <typename T>
struct NativePackSize : veg::meta::constant<usize, (SIMDE_NATURAL_VECTOR_SIZE / 8) / sizeof(T)> {};
} // namespace simd
} // namespace fae
namespace veg {
namespace fmt {
template <typename T, usize N>
struct Debug<fae::simd::Pack<T, N>> {
	static void to_string(BufferMut out, Ref<fae::simd::Pack<T, N>> r) {
		dbg_to(
				out,
				ref(Slice<T>{
						unsafe,
						from_raw_parts,
						reinterpret_cast<T const*>(veg::mem::addressof(r.get())),
						isize{N},
				}));
	}
};
} // namespace fmt
} // namespace veg

#endif /* end of include guard FAER_SIMD_HPP_RA2KIJ1TS */
