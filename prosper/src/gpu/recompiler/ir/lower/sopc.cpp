// sopc.cpp — SOPC lowering: the wave-mask compare routes, then scalar-data compares.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_sopc(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return SopcLowering({b, rs, in, ok, allow_wave}).lower();
}

bool SopcLowering::lower() {
    if (auto routed = route_wave32_mask_compare()) return *routed;
    if (auto routed = route_b64_mask_compare()) return *routed;
    return lower_data();
}

bool SopcLowering::lower_data() {
    uint32_t a = val(in.src[0]), c = val(in.src[1]);
    // Scalar compare -> SCC (read by s_cselect / s_cbranch_scc), and s_bitcmp0/1_b32: their IR
    // definitions (ir/ops/scalar_compare.cpp). eq/lg are bitwise; the ordered compares are signed
    // for i32 (0x02-0x05) and unsigned for u32 (0x08-0x0b).
    if (const ScalarOpDef* def = find_scalar_op(in.fmt, in.opcode)) {
        rs.scc = emit_scalar_op(b, *def, a, c, rs.scc).scc;
        return true;
    }
    if (in.opcode == 0x12 || in.opcode == 0x13) return lower_compare_u64(a, c);
    ok = false;
    return true;
}

// s_cmp_eq/lg_u64 on scalar data pairs. Not in the IR yet: it reads each source's high word.
bool SopcLowering::lower_compare_u64(uint32_t a, uint32_t c) {
    // AMD RDNA2 ISA 12.4: each encoded source names the low half of an SGPR pair.
    // Inline integer/literal sources are extended to 64 bits; Astro compares
    // s[2:3] against inline zero while rebuilding a wave mask.
    auto high = [&](const Operand& o) -> uint32_t {
        if (o.kind == OperandKind::SGPR ||
            (o.kind == OperandKind::Special && o.value >= 106 && o.value < 124)) {
            auto it = rs.sreg.find(o.value + 1);
            if (it != rs.sreg.end()) return it->second;
            ok = false;
            return b.uconst(0);
        }
        if (o.kind == OperandKind::InlineInt)
            return b.uconst(o.value < 0 ? UINT32_MAX : 0);
        // An inline FLOAT in a 64-bit operand supplies the DOUBLE bit pattern
        // (significant bits in the HIGH dword — e.g. 1.0 -> 0x3FF00000_00000000,
        // 1/(2*pi) -> 0x3fc45f30_6dc9c882), which val() cannot model (it returned
        // the f32 pattern as the LOW dword). Never observed live — reject.
        if (o.kind == OperandKind::InlineFloat) { ok = false; return b.uconst(0); }
        if (o.kind == OperandKind::Literal ||
            (o.kind == OperandKind::Special && o.value == 125))
            return b.uconst(0);
        ok = false;
        return b.uconst(0);
    };
    const uint32_t equal = b.land(
        b.ucmp(Op_IEqual, a, c),
        b.ucmp(Op_IEqual, high(in.src[0]), high(in.src[1])));
    rs.scc = in.opcode == 0x12 ? equal : b.logical_not(equal);
    return true;
}

} // namespace prosper::gpu::ir
