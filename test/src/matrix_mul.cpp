#include "faer/internal/helpers.hpp"
#include "faer/internal/matmul_large.hpp"
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <doctest.h>
#include <fmt/ostream.h>

#include <faer/internal/simd.hpp>
#include <faer/internal/cache.hpp>
#include <veg/util/dbg.hpp>
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
constexpr usize MR = 8;
constexpr usize NR = 6;

TEST_CASE("test") {
	usize m = _detail::round_up(31, MR);
	usize n = _detail::round_up(41, NR);
	usize k = 55;
	auto lhs = random_mat<f64>(m, k);
	auto rhs = random_mat<f64>(k, n);
	Eigen::Matrix<f64, -1, -1> dest(m, n);
	{
		Eigen::Matrix<f64, -1, -1> dest_reference(m, n);
		dest_reference = lhs * rhs;
		std::cout << dest_reference.norm() << '\n';
	}

	{
		dest.setZero();

		auto _stack = veg::Vec<unsigned char>{};
		_stack.resize_for_overwrite(_detail::matmul_large_vectorized_req<N, MR, NR>(veg::Tag<f64>{}, m, n, k).alloc_req());
		auto stack = veg::dynstack::DynStackMut{veg::from_slice_mut, _stack.as_mut()};

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
		std::cout << dest.norm() << '\n';
	}
}
