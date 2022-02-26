#ifndef FAER_CACHE_HPP_WKXJ6U9KS
#define FAER_CACHE_HPP_WKXJ6U9KS

#include <veg/type_traits/core.hpp>

namespace fae {
using usize = decltype(sizeof(0));
namespace _detail {
struct CacheInfo {
	usize l1_cache_bytes, l2_cache_bytes, l3_cache_bytes;
	usize l1_assoc, l2_assoc, l3_assoc;
	usize l1_line_bytes, l2_line_bytes, l3_line_bytes;
	VEG_REFLECT( //
			CacheInfo,
			l1_cache_bytes,
			l2_cache_bytes,
			l3_cache_bytes,
			l1_assoc,
			l2_assoc,
			l3_assoc,
			l1_line_bytes,
			l2_line_bytes,
			l3_line_bytes);
};

struct KernelParams {
	usize kc, mc, nc;
	VEG_REFLECT(KernelParams, kc, mc, nc);
};

auto cache_size_impl() noexcept -> CacheInfo;
auto kernel_params_impl(usize mr, usize nr, usize sizeof_T) noexcept -> KernelParams;

inline auto cache_size() noexcept -> CacheInfo {
	static const auto caches = (cache_size_impl)();
	return caches;
}
template <usize MR, usize NR, usize SIZEOF_T>
auto kernel_params() noexcept -> KernelParams {
	static const auto params = (kernel_params_impl)(MR, NR, SIZEOF_T);
	return params;
}

} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_CACHE_HPP_WKXJ6U9KS */
