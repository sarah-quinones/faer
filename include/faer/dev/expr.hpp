#include "veg/type_traits/core.hpp"
#include <veg/util/assert.hpp>
#include <fmt/format.h>
#include <string>
#include <map>
#include <variant>
#include <memory>

namespace fae {
namespace dev {

enum struct binop {
	add = 2,
	mul = 3,
};

struct abstract_expr {
	abstract_expr() = default;
	abstract_expr(abstract_expr const&) = delete;
	abstract_expr(abstract_expr&&) = delete;
	auto operator=(abstract_expr const&) -> abstract_expr& = delete;
	auto operator=(abstract_expr&&) -> abstract_expr& = delete;

	[[nodiscard]] virtual auto clone() const
			-> std::unique_ptr<abstract_expr> = 0;
	[[nodiscard]] virtual auto to_string() const -> std::string = 0;

	virtual ~abstract_expr() = default;
};

struct atomic_expr : abstract_expr {
	std::string name;

	explicit atomic_expr(std::string _name) noexcept : name(VEG_FWD(_name)) {}
	[[nodiscard]] auto clone() const -> std::unique_ptr<abstract_expr> override {
		return std::make_unique<atomic_expr>(this->name);
	}
	[[nodiscard]] auto to_string() const -> std::string override { return name; }
};

struct binary_expr : abstract_expr {
	binop kind;
	std::unique_ptr<abstract_expr> lhs;
	std::unique_ptr<abstract_expr> rhs;

	binary_expr(
			binop _kind,
			std::unique_ptr<abstract_expr> _lhs,
			std::unique_ptr<abstract_expr> _rhs) noexcept
			: kind{_kind}, lhs{VEG_FWD(_lhs)}, rhs{VEG_FWD(_rhs)} {}

	[[nodiscard]] auto clone() const -> std::unique_ptr<abstract_expr> override {
		VEG_ASSERT_ALL_OF(lhs, rhs);
		return std::make_unique<binary_expr>(
				this->kind, this->lhs->clone(), this->rhs->clone());
	}
	[[nodiscard]] auto to_string() const -> std::string override {
		VEG_ASSERT_ALL_OF(lhs, rhs);

		char binop_symbol{};
		switch (kind) {
		case binop::add:
			binop_symbol = '+';
			break;
		case binop::mul:
			binop_symbol = '*';
			break;
		default:
			VEG_ASSERT(false);
		}

		return fmt::format(
				"({} {} {})", lhs->to_string(), binop_symbol, rhs->to_string());
	}
};

struct expr {
	std::unique_ptr<abstract_expr> value;

	explicit expr(abstract_expr const& _value) : value{_value.clone()} {}
	explicit expr(int i) : expr(atomic_expr(fmt::format("{}", i))) {}

	~expr() = default;
	expr() : expr(0) {}
	expr(expr&& other) = default;
	auto operator=(expr&& other) -> expr& = default;

	expr(expr const& other)
			: value{(VEG_ASSERT(other.value), other.value->clone())} {}
	auto operator=(expr const& other) -> expr& {
		VEG_ASSERT(other.value);
		if (this != &other) {
			value = other.value->clone();
		}
		return *this;
	}

	friend auto operator*(expr const& lhs, expr const& rhs) -> expr {
		VEG_ASSERT_ALL_OF(lhs.value, rhs.value);
		return expr(
				binary_expr(binop::mul, lhs.value->clone(), rhs.value->clone()));
	}
	friend auto operator+(expr const& lhs, expr const& rhs) -> expr {
		VEG_ASSERT_ALL_OF(lhs.value, rhs.value);
		return expr(
				binary_expr(binop::add, lhs.value->clone(), rhs.value->clone()));
	}
	friend auto operator+=(expr& lhs, expr const& rhs) -> expr& {
		VEG_ASSERT_ALL_OF(lhs.value, rhs.value);
		lhs = lhs + rhs;
		return lhs;
	}

	auto operator->() const noexcept -> abstract_expr const* {
		return this->value.get();
	}
};

} // namespace dev
} // namespace fae

template <>
struct fmt::formatter<fae::dev::expr> : fmt::formatter<fmt::string_view> {
	template <typename Output_It>
	auto format(
			fae::dev::expr const& value,
			fmt::basic_format_context<Output_It, char>& fc) {

		if (value.value) {
			return fmt::format_to(fc.out(), value.value->to_string());
		} else {
			return fmt::format_to(fc.out(), "{empty}");
		}
	}
};
