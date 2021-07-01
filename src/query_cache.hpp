#ifndef FAER_QUERY_CACHE_HPP_N5X2ZKKCS
#define FAER_QUERY_CACHE_HPP_N5X2ZKKCS
#include "faer/internal/prologue.hpp"

namespace fae {
namespace FAER_ABI_VERSION {
struct CacheSize {
	int l1, l2, l3;
};

auto cache_size_impl() noexcept -> CacheSize;

inline auto cache_size() noexcept -> CacheSize {
	static const auto caches = (cache_size_impl)();
	return caches;
}
} // namespace FAER_ABI_VERSION
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_QUERY_CACHE_HPP_N5X2ZKKCS */
