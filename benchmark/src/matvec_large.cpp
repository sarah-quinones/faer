#include <faer/internal/matvec_large.hpp>
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <veg/vec.hpp>
#include <iostream>

using namespace fae;

template <typename T>
auto random_mat(usize m, usize n) {
	auto mat = Eigen::Matrix<T, -1, -1>(long(m), long(n));
	std::srand(0); // NOLINT
	mat.setRandom();
	return mat;
}
template <typename T>
auto random_vec(usize k) {
	auto mat = Eigen::Matrix<T, -1, 1>(long(k));
	std::srand(0); // NOLINT
	mat.setRandom();
	return mat;
}

constexpr usize N = 4;

usize m = 4911;
usize k = 4219;

template <typename T>
void bm_eigen(benchmark::State& s) {

	auto lhs = random_mat<T>(m, k);
	auto rhs = random_vec<T>(k);
	Eigen::Matrix<T, -1, 1> dest(m);
	dest.setZero();

	for (auto _ : s) {
		dest.tail(m - 1).noalias() += 1.0 * lhs.bottomRows(m - 1) * rhs;
	}
	veg::dbg(dest.norm());
}

template <typename T>
void bm_faer_(benchmark::State& s) {
	auto lhs = random_mat<T>(m, k);
	auto rhs = random_vec<T>(k);
	Eigen::Matrix<T, -1, 1> dest(m);
	dest.setZero();

	for (auto _ : s) {
		_detail::matvec_large_vectorized<Order::COLMAJOR, N>( //
				m - 1,
				k,
				dest.tail(m - 1).data(),
				lhs.bottomRows(m - 1).data(),
				rhs.data(),
				lhs.outerStride());
	}
	veg::dbg(dest.norm());
}

BENCHMARK_TEMPLATE(bm_faer_, f64);
BENCHMARK_TEMPLATE(bm_eigen, f64);
BENCHMARK_MAIN();
