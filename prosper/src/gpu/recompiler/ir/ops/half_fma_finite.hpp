// Exact bounded finite half FMA; no floating-point intermediate is rounded.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <array>
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
class Builder;
struct HalfFmaFiniteResult {
    Value bits;
    Value eligible;
};
HalfFmaFiniteResult build_half_fma_finite(Builder& ir, Value a, Value b, Value c);
// Returns raw half bits and eligibility, in that order. No guest folder gains a new domain.
std::array<Known, 2> eval_half_fma_finite(Known a, Known b, Known c);
std::array<uint32_t, 2> emit_half_fma_finite(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t d);
}   // namespace prosper::gpu::ir
