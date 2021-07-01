#ifndef FAER_CURSOR_HPP_L57LVSDVS
#define FAER_CURSOR_HPP_L57LVSDVS

#include "faer/internal/expr/core.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

template <typename T>
struct ContiguousCursor {
	T* ptr;

	using Element = T;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		constexpr auto _n = usize(N);
		FAER_STATIC_ASSERT_DEV((_n & (_n - 1)) == 0);
		FAER_STATIC_ASSERT_DEV(A == advance_e::inner);

		ptr = ptr::incr(ptr, N);
	}
};

template <typename T>
struct OuterStrideCursor {
	using type = T;
	T* ptr;
	T* inner_begin;
	i64 const outer_stride;
	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		constexpr auto _n = usize(N);
		FAER_STATIC_ASSERT_DEV((_n & (_n - 1)) == 0);
		FAER_STATIC_ASSERT_DEV(A == advance_e::inner or A == advance_e::outer);

		FAER_IF(A == advance_e::inner) { ptr += N; }
		FAER_IF(A == advance_e::outer) {
			FAER_STATIC_ASSERT_DEV(N == 1 || A != advance_e::outer);
			inner_begin += outer_stride;
			ptr = inner_begin;
		}
	}
};

template <typename T>
struct InnerStridedCursor {
	T* ptr;
	T* inner_begin;
	i64 const inner_stride;
	i64 const outer_stride;

	using type = T;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		FAER_STATIC_ASSERT_DEV(N == 1);
		FAER_STATIC_ASSERT_DEV(A == advance_e::inner or A == advance_e::outer);

		FAER_IF(A == advance_e::inner) { ptr += inner_stride; }
		FAER_IF(A == advance_e::outer) {
			inner_begin += outer_stride;
			ptr = inner_begin;
		}
	}
};

template <typename T>
struct BlockCursor {
	T* ptr;
	T* inner_begin;
	T* block_ptr;
	T* block_inner_begin;
	i64 const inner_stride;
	i64 const outer_stride;

  static constexpr i64 block_size = 1;

	using type = T;

	template <i64 N, advance_e A>
	HEDLEY_ALWAYS_INLINE VEG_CPP14(constexpr) void advance() noexcept {
		FAER_STATIC_ASSERT_DEV(N == 1);

		FAER_IF(A == advance_e::inner) { ptr += inner_stride; }
		FAER_IF(A == advance_e::outer) {
			inner_begin += outer_stride;
			ptr = inner_begin;
		}
		FAER_IF(A == advance_e::inner_block) {
			block_ptr += block_size * inner_stride;
			inner_begin = block_ptr;
			ptr = block_ptr;
		}
		FAER_IF(A == advance_e::outer_block) {
			block_inner_begin += block_size * outer_stride;
			block_ptr = block_inner_begin;
			inner_begin = block_ptr;
			ptr = block_ptr;
		}
	}
};

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_CURSOR_HPP_L57LVSDVS */
