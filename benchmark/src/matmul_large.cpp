#include <benchmark/benchmark.h>
#include <veg/vec.hpp>
#include <Eigen/Core>
#include <faer/internal/matmul_large.hpp>
#include <iostream>

using namespace fae;

template <typename T>
auto random_mat(usize m, usize n) {
	auto mat = Eigen::Matrix<T, -1, -1>(long(m), long(n));
  int i = 0;
  for (usize c = 0; c < m * n; ++c) {
    mat.data()[c] = i;
    ++i;
  }
	return mat;
}
constexpr usize N = 8;
constexpr usize MR = 3 * N;
constexpr usize NR = 4;

template <typename T>
void bm_eigen(benchmark::State& s) {
	usize m = usize(s.range(0));
	usize n = usize(s.range(1));
	usize k = usize(s.range(2));

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
	usize m = usize(s.range(0));
	usize n = usize(s.range(1));
	usize k = usize(s.range(2));
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
				1,
				stack);
	}
	std::cout << dest.norm() << '\n';
}

void make_args(benchmark::internal::Benchmark* b) noexcept {
	using _detail::round_up;
	b->Args({round_up(4096, MR), round_up(1024, MR), 4096});
	b->Args({round_up(1024, MR), round_up(4096, MR), 4096});
}

BENCHMARK_TEMPLATE(bm_eigen, f64)->Apply(make_args);
BENCHMARK_TEMPLATE(bm_faer_, f64)->Apply(make_args);
BENCHMARK_MAIN();
