// c++ -O3 -DNDEBUG -march=native -Iinclude -Iexternal/veg/include -Iexternal/simde

#include <veg/util/assert.hpp>

#include <Eigen/Core>
#include <blaze/math/CustomMatrix.h>
#include <benchmark/benchmark.h>

#include <veg/slice.hpp>

#include <faer/internal/mat_mul_real.hpp>
#include <faer/internal/mat_mul_small.hpp>
#include <faer/internal/simd.hpp>
#include <faer/internal/prologue.hpp>

using fae::usize;

template <typename Dst, typename Lhs, typename Rhs>
void matmul(Dst& dst, Lhs const& lhs, Rhs& rhs) {
	fae::FAER_ABI_VERSION::eigen_matmul<typename Dst::Scalar>::apply(
			dst.data(),
			dst.outerStride(),
			dst.innerStride(),
			dst.IsRowMajor == 0,
			lhs.data(),
			lhs.outerStride(),
			lhs.IsRowMajor == 0,
			rhs.data(),
			rhs.outerStride(),
			rhs.IsRowMajor == 0,
			1.0,
			dst.rows(),
			dst.cols(),
			lhs.cols());
}
#include <faer/internal/epilogue.hpp>
#include "common.hpp"

static void large_faery(benchmark::State& s) {
	i64 n = s.range(0);

	CMatrix<T> dst(n, n);
	CMatrix<T> lhs(n, n);
	CMatrix<T> rhs(n, n);
	no_opt(dst, lhs, rhs);

	set_rand<0>(lhs);
	set_rand<1>(rhs);
	set_rand<2>(dst);

	for (auto _ : s) {
		matmul(dst, lhs, rhs);
		// std::terminate();
		benchmark::ClobberMemory();
	}
}

static void large_eigen(benchmark::State& s) {
	i64 n = s.range(0);

	CMatrix<T> dst(n, n);
	CMatrix<T> lhs(n, n);
	CMatrix<T> rhs(n, n);
	no_opt(dst, lhs, rhs);

	set_rand<0>(lhs);
	set_rand<1>(rhs);
	set_rand<2>(dst);

	for (auto _ : s) {
		dst.noalias() += lhs * rhs;
		benchmark::ClobberMemory();
	}
}

static void large_blaze(benchmark::State& s) {
	i64 n = s.range(0);

	CMatrix<T> dst(n, n);
	CMatrix<T> lhs(n, n);
	CMatrix<T> rhs(n, n);
	no_opt(dst, lhs, rhs);

	set_rand<0>(lhs);
	set_rand<1>(rhs);
	set_rand<2>(dst);

	auto _dst =
			blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>(dst.data(), usize(dst.rows()), usize(dst.cols()));
	auto _lhs =
			blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>(lhs.data(), usize(lhs.rows()), usize(lhs.cols()));
	auto _rhs =
			blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>(rhs.data(), usize(rhs.rows()), usize(rhs.cols()));

	for (auto _ : s) {
		_dst = blaze::noalias(_lhs * _rhs);
		benchmark::ClobberMemory();
	}
}

void large_args(benchmark::internal::Benchmark* b) {
	for (i64 n = 4; n < 1025; n *= 2) {
		b->Arg(n);
		b->Arg(n + 1);
		b->Arg(n + 2);
		b->Arg(n + 3);
	}
}

BENCH(large,->Apply(large_args));
