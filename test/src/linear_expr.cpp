#include <faer/span.hpp>
#include <faer/ops/cwise_arithmetic.hpp>
#include <Eigen/Core>
#include <fmt/ranges.h>

using namespace veg::literals;

template <typename R>
auto format_rng(R&& r) -> std::string {
	char const* sep = "";
	std::string out = "{";
	for (auto&& e : VEG_FWD(r)) {
		out += fmt::format("{}{}", sep, VEG_FWD(e));
		sep = ", ";
	}
	out += "}";
	return out;
}

struct NonNumber {
	auto operator=(NonNumber) = delete;
	friend auto operator-(NonNumber const&, NonNumber const&) -> NonNumber;
};

auto main() -> int {
	double in0[] = {1.0, 2.0, 3.0};
	double in1[] = {10.5, 20.25, 30.33};
	double in2[] = {123.132145, 5123.03131, -14.1519};

	double out0[3];
	double out1[3];

	{
		using InMap = Eigen::Map<Eigen::Matrix<double, 3, 1> const>;
		using OutMap = Eigen::Map<Eigen::Matrix<double, 3, 1>>;

		auto v0 = InMap{in0};
		auto v1 = InMap{in1};
		auto v2 = InMap{in2};

		OutMap{out0} = v0 + v1 - v2;
	}

	{
		auto span = fae::span_col_major;

		auto v0 = span(in0, 3_c, 1_v);
		auto v1 = span(in1, 3_c, 1_c);
		auto v2 = span(in2, 3_c, 1_c);

		span(out1, 3_c, 1_c).noalias_assign(v0 + v1 - v2);
	}

	fmt::print("{}\n", format_rng(out0));
	fmt::print("{}\n", format_rng(out1));
}
