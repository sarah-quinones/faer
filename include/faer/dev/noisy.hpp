#include <fmt/core.h>
#include <veg/util/assert.hpp>
#include <veg/type_traits/constructible.hpp>
#include <veg/type_traits/assignable.hpp>
#include <veg/type_traits/tags.hpp>
#include <veg/memory/address.hpp>
#include <atomic>
#include <cstdio>

namespace fae {
namespace dev {
using veg::mem::addressof;

template <typename T>
using deref_expr = decltype(VEG_DECLVAL(T).operator->());
VEG_DEF_CONCEPT(typename T, derefable, VEG_CONCEPT(detected<deref_expr, T>));

template <typename T>
struct noisy {
	inline static std::atomic<std::FILE*>
			file_ptr // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
			{nullptr};

	T value;

	~noisy() = default;
	noisy() = default;
	VEG_TEMPLATE(
			(typename... Args),
			requires(VEG_CONCEPT(constructible<T, Args...>)),
			noisy, // NOLINT(hicpp-explicit-conversions)
			(... args, Args&&))
	noexcept(VEG_CONCEPT(nothrow_constructible<T, Args...>))
			: value(VEG_FWD(args)...) {}

	noisy(noisy&& other) noexcept(VEG_CONCEPT(nothrow_move_constructible<T>))
			: value(VEG_FWD(other.value)) {
		if (noisy::file_ptr != nullptr) {
			fmt::print(
					noisy::file_ptr,
					"move constructed to address {} from address {} with value {}\n",
					static_cast<void const*>(this),
					static_cast<void const*>((addressof)(other)),
					value);
		}
	}

	auto
	operator=(noisy&& other) noexcept(VEG_CONCEPT(nothrow_move_assignable<T>))
			-> noisy& {
		if (this != (addressof)(other)) {
			if (noisy::file_ptr != nullptr) {
				fmt::print(
						noisy::file_ptr,
						"move assigned "
						"to {} "
						"from {} "
						"with addresses {} and {}\n",
						value,
						other.value,
						static_cast<void const*>(this),
						static_cast<void const*>((addressof)(other)));
			}
		} else {
			if (noisy::file_ptr != nullptr) {
				fmt::print(
						noisy::file_ptr,
						"self move assignment at address {} with value {}\n",
						static_cast<void const*>(this),
						value);
			}
		}
		value = other.value;
		return *this;
	}

	noisy(noisy const& other) noexcept(VEG_CONCEPT(nothrow_copy_constructible<T>))
			: value(other.value) {
		if (noisy::file_ptr != nullptr) {
			fmt::print(
					noisy::file_ptr,
					"copy constructed to address {} from address {} with value {}\n",
					static_cast<void const*>(this),
					static_cast<void const*>((addressof)(other)),
					value);
		}
	}

	auto operator=(noisy const& other) noexcept(
			VEG_CONCEPT(nothrow_copy_assignable<T>)) -> noisy& {
		if (this != (addressof)(other)) {
			if (noisy::file_ptr != nullptr) {
				fmt::print(
						noisy::file_ptr,
						"copy assigned "
						"to {} "
						"from {} "
						"with addresses {} and {}\n",
						value,
						other.value,
						static_cast<void const*>(this),
						static_cast<void const*>((addressof)(other)));
			}
		} else {
			if (noisy::file_ptr != nullptr) {
				fmt::print(
						noisy::file_ptr,
						"self copy assignment at address {} with value {}\n",
						static_cast<void const*>(this),
						value);
			}
		}
		value = other.value;
		return *this;
	}

	friend auto operator+(noisy const& lhs, noisy const& rhs) -> noisy {
		if (noisy::file_ptr != nullptr) {
			fmt::print(
					noisy::file_ptr,
					"adding {} and {} "
					"at addresses {} and {}\n",
					lhs.value,
					rhs.value,
					static_cast<void const*>((addressof)(lhs)),
					static_cast<void const*>((addressof)(rhs)));
		}
		return lhs.value + rhs.value;
	}
	friend auto operator*(noisy const& lhs, noisy const& rhs) -> noisy {
		if (noisy::file_ptr != nullptr) {
			fmt::print(
					noisy::file_ptr,
					"multiplying {} and {} "
					"at addresses {} and {}\n",
					lhs.value,
					rhs.value,
					static_cast<void const*>((addressof)(lhs)),
					static_cast<void const*>((addressof)(rhs)));
		}
		return lhs.value * rhs.value;
	}
	friend auto operator+=(noisy& lhs, noisy const& rhs) -> noisy& {
		if (noisy::file_ptr != nullptr) {
			fmt::print(
					noisy::file_ptr,
					"add-assigning to {} from {} "
					"at addresses {} and {}\n",
					lhs.value,
					rhs.value,
					static_cast<void const*>((addressof)(lhs)),
					static_cast<void const*>((addressof)(rhs)));
		}
		VEG_ASSERT(false);
		lhs.value += rhs.value;
		return lhs;
	}

	auto operator->() const noexcept -> typename veg::meta::conditional_t<
									 VEG_CONCEPT_MACRO(fae::dev, derefable<T const&>),
									 veg::meta::meta_apply<deref_expr, T const&>,
									 veg::meta::type_identity<T const*>>::type {
		if constexpr (VEG_CONCEPT_MACRO(fae::dev, derefable<T const&>)) {
			return value.operator->();
		} else {
			return (addressof)(value.get());
		}
	}
};

} // namespace dev
} // namespace fae
