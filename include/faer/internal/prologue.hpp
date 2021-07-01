#ifdef FAER_PROLOGUE
#error "missing epilogue"
#endif
#define FAER_PROLOGUE
#include <veg/internal/prologue.hpp>

#define FAER_STATIC_ASSERT_DEV(...) static_assert(__VA_ARGS__, ".")

#define FAER_ABI_VERSION v0
#define FAER_IS_NOEXCEPT(...) noexcept(__VA_ARGS__)
#define FAER_DECLTYPE_RET(...)                                                                                         \
	->decltype(auto) { return (__VA_ARGS__); } VEG_NOM_SEMICOLON

#if __cplusplus >= 201703L
#define FAER_IF(Cond) if constexpr (Cond)
#else
#define FAER_IF(Cond) if (Cond)
#endif
