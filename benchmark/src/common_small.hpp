#ifndef FAER_COMMON_SMALL_HPP_NPHNOIWXS
#define FAER_COMMON_SMALL_HPP_NPHNOIWXS

#include <veg/util/assert.hpp>

#include <Eigen/Core>
#include <blaze/math/StaticMatrix.h>

#include <veg/slice.hpp>

#include <faer/internal/mat_mul_real.hpp>
#include <faer/internal/mat_mul_small.hpp>
#include <faer/internal/simd.hpp>
#include "common.hpp"

template <int N, int M, int K, int DO, int LO, int RO>
static void small_faery(benchmark::State& s) {
	alignas(64) Eigen::Matrix<T, N, M, DO> dst(N, M);
	alignas(64) Eigen::Matrix<T, N, K, LO> lhs(N, K);
	alignas(64) Eigen::Matrix<T, K, M, RO> rhs(K, M);
	no_opt(dst, lhs, rhs);

	set_rand<0>(lhs);
	set_rand<1>(rhs);
	set_rand<2>(dst);

	for (auto _ : s) {
		fae::internal::_matmul_smol::matmul_c(
				1.0,
				1,
				dst.data(),
				dst.outerStride(),
				lhs.data(),
				(lhs.IsRowMajor ? lhs.outerStride() : lhs.innerStride()) == 1,
				lhs.IsRowMajor ? lhs.innerStride() : lhs.outerStride(),
				lhs.IsRowMajor ? lhs.outerStride() : lhs.innerStride(),
				rhs.data(),
				rhs.IsRowMajor ? rhs.innerStride() : rhs.outerStride(),
				rhs.IsRowMajor ? rhs.outerStride() : rhs.innerStride(),
				N,
				M,
				K);
		benchmark::ClobberMemory();
	}
}

template <int N, int M, int K, int DO, int LO, int RO>
static void small_eigen(benchmark::State& s) {
	alignas(64) Eigen::Matrix<T, N, M, DO> dst(N, M);
	alignas(64) Eigen::Matrix<T, N, K, LO> lhs(N, K);
	alignas(64) Eigen::Matrix<T, K, M, RO> rhs(K, M);
	no_opt(dst, lhs, rhs);

	set_rand<0>(lhs);
	set_rand<1>(rhs);
	set_rand<2>(dst);

	for (auto _ : s) {
		dst.noalias() += lhs * rhs;
		benchmark::ClobberMemory();
	}
}

constexpr auto C = Eigen::ColMajor;
constexpr auto R = Eigen::RowMajor;

constexpr auto blaze_so(int eigen_so) -> bool {
	return eigen_so == R ? blaze::rowMajor : blaze::columnMajor;
}

template <typename Mat>
constexpr auto eigen_map(Mat& bl) -> veg::meta::conditional_t<
		Mat::storageOrder == blaze::rowMajor,
		Eigen::Map<Eigen::Matrix<typename Mat::ElementType, -1, -1, Eigen::RowMajor>>,
		Eigen::Map<Eigen::Matrix<typename Mat::ElementType, -1, -1, Eigen::ColMajor>>> {
	return {
			bl.data(),
			long(bl.rows()),
			long(bl.columns()),
	};
}

template <int N, int M, int K, int DO, int LO, int RO>
static void small_blaze(benchmark::State& s) {
	alignas(64) blaze::StaticMatrix<T, N, M, blaze_so(DO), blaze::unaligned, blaze::unpadded> dst;
	alignas(64) blaze::StaticMatrix<T, N, K, blaze_so(LO), blaze::unaligned, blaze::unpadded> lhs;
	alignas(64) blaze::StaticMatrix<T, K, M, blaze_so(RO), blaze::unaligned, blaze::unpadded> rhs;
	no_opt(dst, lhs, rhs);

	set_rand<0>(eigen_map(lhs));
	set_rand<1>(eigen_map(rhs));
	set_rand<2>(eigen_map(dst));

	for (auto _ : s) {

		dst += blaze::noalias(lhs * rhs);
		benchmark::ClobberMemory();
	}
}

#define ALL_BM                                                                                                         \
	BM(3, 1, 4);    /* NOLINT */                                                                                         \

#endif /* end of include guard FAER_COMMON_SMALL_HPP_NPHNOIWXS */
