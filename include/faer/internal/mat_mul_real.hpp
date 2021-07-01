#ifndef FAER_MAT_MUL_REAL_HPP_DLTIJ62RS
#define FAER_MAT_MUL_REAL_HPP_DLTIJ62RS

#include <veg/internal/typedefs.hpp>
#include "faer/internal/prologue.hpp"

static_assert(static_cast<unsigned char>(-1) == 255, ".");
static_assert(sizeof(float) == 4, ".");
static_assert(sizeof(double) == 8, ".");

namespace fae {
using veg::i64;
using f32 = float;
using f64 = double;

namespace FAER_ABI_VERSION {

template <typename T>
struct eigen_matmul;

template <>
struct eigen_matmul<f32> {
	static void apply(
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
			i64 depth);
};

template <>
struct eigen_matmul<f64> {
	static void apply(
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
			i64 depth);
};

} // namespace FAER_ABI_VERSION
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_MAT_MUL_REAL_HPP_DLTIJ62RS */
