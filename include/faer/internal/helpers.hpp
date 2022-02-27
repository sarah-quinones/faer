#ifndef FAER_HELPERS_HPP_EA0N2BWSS
#define FAER_HELPERS_HPP_EA0N2BWSS

#include "faer/internal/simd.hpp"
#include "faer/internal/enums.hpp"
#include <veg/type_traits/invocable.hpp>

#ifdef __clang__
#define FAER_NO_UNROLL _Pragma("nounroll")
#define FAER_MINSIZE [[clang::minsize]]
#else
#define FAER_NO_UNROLL
#define FAER_MINSIZE
#endif

#define FAER_CACHE_LINE_BYTES (64)

namespace fae {
namespace _detail {
template <Order>
struct MemAccess;

template <>
struct MemAccess<Order::COLMAJOR> {
	template <typename T>
	static constexpr auto fn(T* ptr, usize i, usize j, isize stride) noexcept -> T* {
		return ptr + (isize(i) + isize(j) * stride);
	}
};
template <>
struct MemAccess<Order::ROWMAJOR> {
	template <typename T>
	static constexpr auto fn(T* ptr, usize i, usize j, isize stride) noexcept -> T* {
		return ptr + (isize(j) + isize(i) * stride);
	}
};
template <typename T>
auto offset_to_align(T* ptr, usize align) noexcept -> usize {
	using UPtr = std::uintptr_t;
	UPtr mask = align - 1;
	return (((UPtr(ptr) + mask) & ~mask) - UPtr(ptr)) / sizeof(T);
}

constexpr auto min2(usize a, usize b) noexcept -> usize {
	return a < b ? a : b;
}
constexpr auto max2(usize a, usize b) noexcept -> usize {
	return a > b ? a : b;
}
constexpr auto round_up(usize n, usize k) noexcept -> usize {
	return (n + (k - 1)) / k * k;
}
constexpr auto round_down(usize n, usize k) noexcept -> usize {
	return n / k * k;
}
template <typename Fn, usize... Is>
VEG_INLINE void
unroll_c_impl(veg::meta::index_sequence<Is...> /*tag*/, Fn fn) noexcept(VEG_CONCEPT(nothrow_fn_mut<Fn, void, usize>)) {
	VEG_EVAL_ALL(fn.template operator()<Is>());
}

template <usize N, typename Fn>
VEG_INLINE void unroll_c(Fn fn) noexcept(VEG_CONCEPT(nothrow_fn_mut<Fn, void, usize>)) {
	_detail::unroll_c_impl(veg::meta::make_index_sequence<N>{}, VEG_FWD(fn));
}

template <typename Fn, usize... Is>
VEG_INLINE void
unroll_impl(veg::meta::index_sequence<Is...> /*tag*/, Fn fn) noexcept(VEG_CONCEPT(nothrow_fn_mut<Fn, void, usize>)) {
	VEG_EVAL_ALL(fn(Is));
}

template <usize N, typename Fn>
VEG_INLINE void unroll(Fn fn) noexcept(VEG_CONCEPT(nothrow_fn_mut<Fn, void, usize>)) {
	_detail::unroll_impl(veg::meta::make_index_sequence<N>{}, VEG_FWD(fn));
}

inline void clobber_memory() {
	asm volatile("" : : : "memory"); // NOLINT(hicpp-no-assembler)
}
template <typename T>
VEG_INLINE auto incr(T* ptr, isize byte_stride) noexcept -> T* {
	using VoidPtr = void*;
	using CharPtr = char*;
	using TPtr = T*;
	return TPtr(VoidPtr(CharPtr(ptr) + byte_stride));
}
constexpr auto count_trailing_zeros(unsigned n) noexcept -> usize {
	return usize(__builtin_ctz(n));
}
constexpr auto count_trailing_zeros(unsigned long n) noexcept -> usize {
	return usize(__builtin_ctzl(n));
}
constexpr auto count_trailing_zeros(unsigned long long n) noexcept -> usize {
	return usize(__builtin_ctzll(n));
}
constexpr auto max_u(usize a, usize b) noexcept -> usize {
	return (a > b) ? a : b;
}
constexpr auto min_u(usize a, usize b) noexcept -> usize {
	return (a < b) ? a : b;
}
} // namespace _detail
} // namespace fae

#endif /* end of include guard FAER_HELPERS_HPP_EA0N2BWSS */
