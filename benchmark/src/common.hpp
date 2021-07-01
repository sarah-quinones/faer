#ifndef FAER_COMMON_HPP_1MZNR4IJS
#define FAER_COMMON_HPP_1MZNR4IJS

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <faer/internal/simd.hpp>
#include <faer/internal/prologue.hpp>

using fae::i64;
using T = double;

template <typename T>
using CMatrix = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
template <typename T>
using RMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

template <i64 I>
auto rnd_mat(i64 i, i64 j) -> Eigen::Map<CMatrix<T> const> {
	static auto const m = [] {
		CMatrix<T> out(2000, 2000);
		out.setRandom();
		return out;
	}();
	VEG_ASSERT(m.size() >= i * j);
	return Eigen::Map<CMatrix<T> const>{m.data(), i, j};
}

template <i64 I, typename T>
void set_rand(T&& mat) {
	mat = rnd_mat<I>(mat.rows(), mat.cols());
}

template <typename... Ts>
void no_opt(Ts&&... args) {
	using int_arr = int[];
	void(int_arr{0, (benchmark::DoNotOptimize(VEG_FWD(args).data()), 0)...});
}

static void _(benchmark::State& s) {
	for (auto _ : s) {
	}
}

#define BENCH(Name, ...)                                                                                               \
	BENCHMARK(Name##_faery) __VA_ARGS__;                                                                                 \
	BENCHMARK(Name##_eigen) __VA_ARGS__;                                                                                 \
	BENCHMARK(Name##_blaze) __VA_ARGS__;                                                                                 \
	BENCHMARK(_)
#define BENCH_TPL(Name, ...)                                                                                           \
	BENCHMARK_TEMPLATE(Name##_faery, __VA_ARGS__);                                                                       \
	BENCHMARK_TEMPLATE(Name##_eigen, __VA_ARGS__);                                                                       \
	BENCHMARK_TEMPLATE(Name##_blaze, __VA_ARGS__);                                                                       \
	BENCHMARK(_)

#include <faer/internal/epilogue.hpp>
#endif /* end of include guard FAER_COMMON_HPP_1MZNR4IJS */
