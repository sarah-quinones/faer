#ifndef FAER_CORE_HPP_XGZGCO7PS
#define FAER_CORE_HPP_XGZGCO7PS

#include "faer/internal/simd.hpp"
#include "faer/internal/prologue.hpp"

namespace fae {
namespace internal {
namespace expr {

enum struct advance_e { inner, outer, inner_block, outer_block };
enum struct expr_kind_e { reader, sum, neg, vec_prod, scal_mul_left, scal_mul_right, other };

struct One {};

} // namespace expr
} // namespace internal
} // namespace fae

#include "faer/internal/epilogue.hpp"
#endif /* end of include guard FAER_CORE_HPP_XGZGCO7PS */
