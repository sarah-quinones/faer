#include "faer/internal/helpers.hpp"
#include <faer/internal/cache.hpp>
#include <simde/simde-arch.h>
#include <veg/util/assert.hpp>

#if defined(SIMDE_ARCH_X86) || defined(SIMDE_ARCH_AMD64)
#include <cpuinfo_x86.h>
#elif defined(SIMDE_ARCH_ARM)
#include <cpuinfo_arm.h>
#else
#error "unrecognized arch"
#endif

namespace fae {
namespace _detail {
auto cache_size_impl() noexcept -> CacheInfo {
	auto info_array =
#if defined(SIMDE_ARCH_X86) || defined(SIMDE_ARCH_AMD64)
			cpu_features::GetX86CacheInfo();
#elif defined(SIMDE_ARCH_ARM)
#endif

	CacheInfo out{};
	for (usize k = 0; k < usize(info_array.size); ++k) {
		auto& info = info_array.levels[k]; // NOLINT
		if (info.cache_type == cpu_features::CacheType::CPU_FEATURE_CACHE_DATA ||
		    info.cache_type == cpu_features::CacheType::CPU_FEATURE_CACHE_UNIFIED) {
			switch (info.level) {
			case 1: {
				out.l1_cache_bytes = usize(info.cache_size);
				out.l1_assoc = usize(info.ways);
				out.l1_line_bytes = usize(info.line_size);
				break;
			}
			case 2: {
				out.l2_cache_bytes = usize(info.cache_size);
				out.l2_assoc = usize(info.ways);
				out.l2_line_bytes = usize(info.line_size);
				break;
			}
			case 3: {
				out.l3_cache_bytes = usize(info.cache_size);
				out.l3_assoc = usize(info.ways);
				out.l3_line_bytes = usize(info.line_size);
				break;
			}
			default:
				VEG_UNIMPLEMENTED();
			}
		}
	}
	return out;
}

constexpr auto div_up(usize n, usize k) noexcept -> usize {
	return _detail::round_up(n, k) / k;
}

constexpr auto gcd(usize a, usize b) noexcept -> usize;

constexpr auto gcd2(usize a, usize b) noexcept -> usize {
	// assume a >= b
	return b == 0 ? a : gcd(a % b, b);
}
constexpr auto gcd(usize a, usize b) noexcept -> usize {
	return _detail::gcd2( //
			_detail::max2(a, b),
			_detail::min2(a, b));
}

constexpr auto mc_from_lhs_l2_assoc(usize lhs_l2_assoc, usize l2_cache_bytes, usize l2_assoc, usize kc) -> usize {
	return (lhs_l2_assoc * l2_cache_bytes / l2_assoc) / (sizeof(f64) * kc);
}

auto kernel_params_impl(usize mr, usize nr, usize sizeof_T) noexcept -> KernelParams {
	auto cache = _detail::cache_size();
	VEG_ASSERT_ALL_OF( //
			cache.l1_assoc > 2,
			cache.l2_assoc > 2,
			cache.l3_assoc > 2);
	usize l1_cache_bytes = cache.l1_cache_bytes;
	usize l2_cache_bytes = cache.l2_cache_bytes;
	usize l3_cache_bytes = cache.l3_cache_bytes;

	usize l1_line_bytes = cache.l1_line_bytes;
	usize l2_line_bytes = cache.l2_line_bytes;
	usize l3_line_bytes = cache.l3_line_bytes;

	usize l1_assoc = cache.l1_assoc;
	usize l2_assoc = cache.l2_assoc;
	usize l3_assoc = cache.l3_assoc;

	usize l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);
	usize l2_n_sets = l2_cache_bytes / (l2_line_bytes * l2_assoc);
	usize l3_n_sets = l3_cache_bytes / (l3_line_bytes * l3_assoc);
	veg::unused(l2_n_sets, l3_n_sets);

	// requires
	// A micropanels must occupy different cache sets
	// so that loading a micropanel evicts the previous one
	// => byte stride must be multiple of n_sets×line_bytes
	//
	// => mr×kc×scalar_bytes == C_A × l1_line_bytes × l1_n_sets
	//
	// l1 must be able to hold A micropanel, B micropanel + set for C update
	//
	// => C_A + C_B <= l1_assoc -1

	// a×n = b×m
	// find lcm of a, b
	// n = lcm / a = b/gcd(a,b)
	// m = lcm / b = a/gcd(a,b)
	usize kc_0 = l1_line_bytes * l1_n_sets / _detail::gcd(mr * sizeof_T, l1_line_bytes * l1_n_sets);
	usize C_lhs = mr * sizeof_T / _detail::gcd(mr * sizeof_T, l1_line_bytes * l1_n_sets);
	usize C_rhs = nr * kc_0 * sizeof_T / (l1_line_bytes * l1_n_sets);
	usize kc_multiplier = (l1_assoc - 1) / (C_lhs + C_rhs);
	usize auto_kc = kc_0 * kc_multiplier;

	// l2 cache must hold
	//  - B micropanel: nr×kc
	//  - C update? 1 assoc degree
	//  - A macropanel: mc×kc
	// mc×kc×scalar_bytes
	usize rhs_micropanel_bytes = nr * auto_kc * sizeof(f64);
	usize rhs_l2_assoc = div_up(rhs_micropanel_bytes, l2_cache_bytes / l2_assoc);
	usize lhs_l2_assoc = (l2_assoc - 1 - rhs_l2_assoc);

	// usize auto_mc = round_down(mc_from_lhs_l2_assoc(lhs_l2_assoc, l2_cache_bytes, l2_assoc, auto_kc), mr);
	usize auto_mc = round_down(mc_from_lhs_l2_assoc(lhs_l2_assoc - 1, l2_cache_bytes, l2_assoc, auto_kc), mr);
	// usize auto_mc = round_down(mc_from_lhs_l2_assoc(l2_assoc / 2, l2_cache_bytes, l2_assoc, auto_kc), mr);

	// l3 cache must hold
	//  - B macropanel: nc×kc
	//  - A macropanel: mc×kc
	//  - C update? 1 assoc degree
	usize lhs_macropanel_bytes = auto_mc * auto_kc * sizeof(f64);
	usize lhs_l3_assoc = div_up(lhs_macropanel_bytes, l3_cache_bytes / l3_assoc);
	usize rhs_l3_assoc = (l3_assoc - 1 - lhs_l3_assoc);
	usize rhs_macropanel_max_bytes = rhs_l3_assoc * l3_cache_bytes / l3_assoc;

	usize auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof(f64) * auto_kc), nr);
	return {auto_kc, auto_mc, auto_nc};
}
} // namespace _detail
} // namespace fae
