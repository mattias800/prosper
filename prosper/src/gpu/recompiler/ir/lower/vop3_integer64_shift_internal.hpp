// The exact former emit_alu locals used by the relocated integer64 shift cases.
#pragma once
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
struct Vop3Integer64ShiftLowering {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;
    decltype(RegState::vreg)& vreg;
    uint32_t val(const Operand& source) { return operand_bits(b, rs, in, source, &ok); }
    void lshl();
    void lshr();
};
}   // namespace prosper::gpu::ir
