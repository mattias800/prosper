// PERMLANE16/X16 values. Register/source authority and final destination predication stay lower.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct PermlaneInputs {
    Value source, selectors_lo, selectors_hi;
    // Appended/resolved at their old expression positions, never eagerly reading old VDST.
    std::function<Value()> active, old_d;
};
Value permlane16_b32(Builder&, const PermlaneInputs&, bool fetch_inactive, bool bound_ctrl);
Value permlanex16_b32(Builder&, const PermlaneInputs&, bool fetch_inactive, bool bound_ctrl);
using PermlaneInputReader = std::function<uint32_t()>;
uint32_t emit_permlane_b32(SpirvCompute&, uint32_t source, uint32_t selectors_lo,
                           uint32_t selectors_hi, bool across_rows, bool fetch_inactive,
                           bool bound_ctrl, const PermlaneInputReader& active,
                           const PermlaneInputReader& old_d);
} // namespace prosper::gpu::ir
