// lowering_context.hpp — the emit_alu state every family's lowering class starts from.
//
// The members are emit_alu's locals under the same names (`b`, `rs`, `in`, `ok`, `allow_wave`, and
// `val` for its operand lambda), so code moved out of emit_alu compiles unchanged inside a class that
// derives from this. A family adds the further locals it reads as its own members.
#pragma once

#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

struct LoweringContext {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;
    bool allow_wave;
    uint32_t val(const Operand& o) { return operand_bits(b, rs, in, o, &ok); }
};

} // namespace prosper::gpu::ir
