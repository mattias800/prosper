// SMEM case routes relocated mechanically from emit_alu; no IR memory semantics are added.
// Proof definitions remain with their existing owners; these callbacks retain their old call sites.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
struct SmemCaseRoutes : LoweringContext {
    bool allow_smem;
    const ShaderResourceTable* rt;
    bool (*sreg_range_written)(const RegState&, int, uint32_t);
    bool (*sreg_srt_range_tag)(const RegState&, int, uint32_t, uint32_t&);
    std::optional<bool> pcrel(uint32_t n);
    std::optional<bool> resolved(uint32_t n);
    bool memory(uint32_t n);
};
}   // namespace prosper::gpu::ir
