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

constexpr usize N = 8;

usize m = 4053;
usize k = 4096;

template <typename T>
void bm_eigen(benchmark::State& s) {

	auto lhs = random_mat<T>(m, k);
	auto rhs = random_vec<T>(k);
	Eigen::Matrix<T, -1, 1> dest(m);
	dest.setZero();

	for (auto _ : s) {
		dest.noalias() += 2.0 * lhs * rhs;
	}
	std::cout << dest.norm() << '\n';
}

template <typename T>
void bm_faer_(benchmark::State& s) {
	auto lhs = random_mat<T>(m, k);
	auto rhs = random_vec<T>(k);
	Eigen::Matrix<T, -1, 1> dest(m);
	dest.setZero();

	for (auto _ : s) {
		_detail::matvec_large_vectorized<Order::COLMAJOR, N>( //
				m,
				k,
				dest.data(),
				lhs.data(),
				rhs.data(),
				lhs.outerStride(),
				2);
	}
	std::cout << dest.norm() << '\n';
}

BENCHMARK_TEMPLATE(bm_eigen, f64);
BENCHMARK_TEMPLATE(bm_faer_, f64);
BENCHMARK_MAIN();
