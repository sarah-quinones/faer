#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <faer/internal/matmul_large.hpp>
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
constexpr usize N = 4;
constexpr usize MR = 12;
constexpr usize NR = 4;

usize m = _detail::round_up(4096, MR);
usize n = _detail::round_up(4096, NR);
usize k = 4096;

template <typename T>
void bm_eigen(benchmark::State& s) {

	auto lhs = random_mat<T>(m, k);
	auto rhs = random_mat<T>(k, n);
	Eigen::Matrix<T, -1, -1> dest(m, n);
	dest.setZero();

	for (auto _ : s) {
		dest.noalias() += lhs * rhs;
	}
	std::cout << dest.norm() << '\n';
}

template <typename T>
void bm_faer_(benchmark::State& s) {
	auto lhs = random_mat<T>(m, k);
	auto rhs = random_mat<T>(k, n);
	Eigen::Matrix<T, -1, -1> dest(m, n);
	dest.setZero();

	auto _stack = veg::Vec<unsigned char>{};
	_stack.resize_for_overwrite(_detail::matmul_large_vectorized_req<N, MR, NR>(veg::Tag<T>{}, m, n, k).alloc_req());
	auto stack = veg::dynstack::DynStackMut{veg::from_slice_mut, _stack.as_mut()};

	for (auto _ : s) {
		_detail::matmul_large_vectorized<Order::COLMAJOR, Order::COLMAJOR, N, MR, NR>( //
				m,
				n,
				k,
				dest.data(),
				lhs.data(),
				rhs.data(),
				dest.outerStride(),
				lhs.outerStride(),
				rhs.outerStride(),
				stack);
	}
	std::cout << dest.norm() << '\n';
}

BENCHMARK_TEMPLATE(bm_faer_, f64);
BENCHMARK_TEMPLATE(bm_eigen, f64);
BENCHMARK_MAIN();
