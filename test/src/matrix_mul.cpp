#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <doctest.h>
#include <fmt/ostream.h>

#include <faer/internal/mat_mul_real.hpp>
#include <faer/internal/mat_mul_small.hpp>
#include <faer/internal/prologue.hpp>

template <typename T>
using CMatrix = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
template <typename T>
using RMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

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

template <typename Dst, typename Lhs, typename Rhs>
void matmul_small(Dst& dst, Lhs const& lhs, Rhs& rhs) {
	fae::internal::_matmul_smol::matmul_c(
			typename Dst::Scalar(1.0),
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
			lhs.rows(),
			rhs.cols(),
			lhs.cols());
}

template <typename T, template <typename> class DstFn, template <typename> class LhsFn, template <typename> class RhsFn>
struct Wrapper {
	using Type = T;
	using Dst = DstFn<T>;
	using Lhs = LhsFn<T>;
	using Rhs = RhsFn<T>;
};

DOCTEST_TEST_CASE_TEMPLATE(
		"matrix multiplication small",
		W,

		Wrapper<fae::f32, CMatrix, CMatrix, CMatrix>,
    Wrapper<fae::f32, CMatrix, CMatrix, RMatrix>,
    Wrapper<fae::f32, CMatrix, RMatrix, CMatrix>,
    Wrapper<fae::f32, CMatrix, RMatrix, RMatrix>,

		Wrapper<fae::f64, CMatrix, CMatrix, CMatrix>,
    Wrapper<fae::f64, CMatrix, CMatrix, RMatrix>,
		Wrapper<fae::f64, CMatrix, RMatrix, CMatrix>,
		Wrapper<fae::f64, CMatrix, RMatrix, RMatrix>

) {

	for (int i = 1; i < 32; ++i) {
		for (int j = 1; j < 32; ++j) {
			for (int k = 1; k < 32; ++k) {
				typename W::Dst dst(i, j);
				typename W::Dst dst2(i, j);
				typename W::Lhs lhs(i, k);
				typename W::Rhs rhs(k, j);

				lhs.setRandom();
				rhs.setRandom();
				dst.setRandom();
				dst2 = dst;

				matmul_small(dst, lhs, rhs);
				dst2.noalias() += lhs * rhs;

				DOCTEST_REQUIRE((dst - dst2).norm() <= std::sqrt(std::numeric_limits<typename W::Type>::epsilon()));
			}
		}
	}

	for (int n = 1; n < 129; ++n) {
		typename W::Dst dst(n, n);
		typename W::Dst dst2(n, n);
		typename W::Lhs lhs(n, n);
		typename W::Rhs rhs(n, n);

		lhs.setRandom();
		rhs.setRandom();
		dst.setRandom();
		dst2 = dst;

		matmul_small(dst, lhs, rhs);
		dst2.noalias() += lhs * rhs;

		DOCTEST_REQUIRE((dst - dst2).norm() <= std::sqrt(std::numeric_limits<typename W::Type>::epsilon()));
	}
}

DOCTEST_TEST_CASE_TEMPLATE(
		"matrix multiplication",
		W,

		Wrapper<fae::f32, CMatrix, CMatrix, CMatrix>,
		Wrapper<fae::f32, CMatrix, CMatrix, RMatrix>,
		Wrapper<fae::f32, CMatrix, RMatrix, CMatrix>,
		Wrapper<fae::f32, CMatrix, RMatrix, RMatrix>,
		Wrapper<fae::f32, RMatrix, CMatrix, CMatrix>,
		Wrapper<fae::f32, RMatrix, CMatrix, RMatrix>,
		Wrapper<fae::f32, RMatrix, RMatrix, CMatrix>,
		Wrapper<fae::f32, RMatrix, RMatrix, RMatrix>,

		Wrapper<fae::f64, CMatrix, CMatrix, CMatrix>,
		Wrapper<fae::f64, CMatrix, CMatrix, RMatrix>,
		Wrapper<fae::f64, CMatrix, RMatrix, CMatrix>,
		Wrapper<fae::f64, CMatrix, RMatrix, RMatrix>,
		Wrapper<fae::f64, RMatrix, CMatrix, CMatrix>,
		Wrapper<fae::f64, RMatrix, CMatrix, RMatrix>,
		Wrapper<fae::f64, RMatrix, RMatrix, CMatrix>,
		Wrapper<fae::f64, RMatrix, RMatrix, RMatrix>

) {

	auto test = [](int n) {
		typename W::Dst dst(n, n);
		typename W::Dst dst2(n, n);
		typename W::Lhs lhs(n, n);
		typename W::Rhs rhs(n, n);

		lhs.setRandom();
		rhs.setRandom();
		dst.setRandom();
		dst2 = dst;

		matmul(dst, lhs, rhs);

		dst2.noalias() += lhs * rhs;
		DOCTEST_REQUIRE((dst - dst2).norm() <= std::sqrt(std::numeric_limits<typename W::Type>::epsilon()));
	};
	for (int n = 1; n < 129; ++n) {
		test(n);
	}
	for (int n = 1020; n < 1025; ++n) {
		test(n);
	}

	for (int i = 1; i < 32; ++i) {
		for (int j = 1; j < 32; ++j) {
			for (int k = 1; k < 32; ++k) {
				typename W::Dst dst(i, j);
				typename W::Dst dst2(i, j);
				typename W::Lhs lhs(i, k);
				typename W::Rhs rhs(k, j);

				lhs.setRandom();
				rhs.setRandom();
				dst.setRandom();
				dst2 = dst;

				matmul(dst, lhs, rhs);

				dst2.noalias() += lhs * rhs;
				DOCTEST_REQUIRE((dst - dst2).norm() <= std::sqrt(std::numeric_limits<typename W::Type>::epsilon()));
			}
		}
	}
}

#include <faer/internal/epilogue.hpp>
