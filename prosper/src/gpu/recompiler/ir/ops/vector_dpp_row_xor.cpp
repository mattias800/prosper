// Preserve the native helper's append order, including its caller's final bounded-zero select.
#include "gpu/recompiler/ir/ops/vector_dpp_row_xor.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <array>
#include <cstdint>

namespace prosper::gpu::ir {
Value dpp_row_xor_source(Builder& ir, Value source, Value active, uint32_t stride) {
    const Value lane = ir.lane_id();
    const Value peer = ir.xor_(lane, ir.c32(stride & 15u));
    const Value rotated = ir.shuffle(source, peer);
    const Value source_active = ir.shuffle(ir.select(active, ir.c32(1), ir.c32(0)), peer);
    const Value valid = ir.ine(source_active, ir.c32(0));
    // The old helper returns SOURCE for an invalid peer before its caller substitutes zero.
    // Keep that intermediate select so the migration preserves the module's exact bytes.
    const Value fallback = ir.select(valid, rotated, source);
    return ir.select(valid, fallback, ir.c32(0));
}

uint32_t emit_dpp_row_xor_source(SpirvCompute& b, uint32_t source, uint32_t active,
                                 uint32_t stride) {
    Block block;
    Builder ir(block);
    const Value word = ir.input(Type::U32), exec = ir.input(Type::Bool);
    const Value d = dpp_row_xor_source(ir, word, exec, stride);
    const std::array<uint32_t, 2> inputs{source, active};
    return emit_spirv(b, block, inputs)[d.index];
}
} // namespace prosper::gpu::ir
