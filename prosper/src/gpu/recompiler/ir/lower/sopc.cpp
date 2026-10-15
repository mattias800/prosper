// sopc.cpp — SOPC lowering: s_bitcmp, the wave-mask compare routes, then scalar-data compares.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_sopc(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return SopcLowering({b, rs, in, ok, allow_wave}).lower();
}

bool SopcLowering::lower() {
    // s_bitcmp0/1_b32 (0x0c/0x0d): SCC = bit (src0 >> (src1 & 31)) & 1, negated for bitcmp0.
    // DOLL's scene PS tests feature-flag bits 0..3 of an s_buffer_load'd word with s_cselect
    // chains (#273). VERIFIED(round-trip llvm-mc gfx1010: 0xbf0d8014 -> s_bitcmp1_b32 s20, 0;
    // 0xbf0c8114 -> s_bitcmp0_b32 s20, 1 — NOT the u64 compares an opcode-table guess said).
    if (in.opcode == 0x0c || in.opcode == 0x0d) {
        uint32_t a = val(in.src[0]), c = val(in.src[1]);
        uint32_t sh  = b.ibin(Op_BitwiseAnd, c, b.uconst(31));
        uint32_t bit = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, a, sh), b.uconst(1));
        uint32_t nz  = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        rs.scc = (in.opcode == 0x0d) ? nz : b.bsel(nz, b.bfalse(), b.btrue());
        return true;
    }
    if (auto routed = route_wave32_mask_compare()) return *routed;
    if (auto routed = route_b64_mask_compare()) return *routed;
    return lower_data();
}

bool SopcLowering::lower_data() {
    uint32_t a = val(in.src[0]), c = val(in.src[1]);
    switch (in.opcode) {
        case 0x00: case 0x06: rs.scc = b.ucmp(Op_IEqual, a, c); break;        // s_cmp_eq_i32/u32
        case 0x01: case 0x07: rs.scc = b.ucmp(Op_INotEqual, a, c); break;     // s_cmp_lg_i32/u32
        case 0x02: rs.scc = b.scmp(Op_SGreaterThan, a, c); break;             // s_cmp_gt_i32
        case 0x03: rs.scc = b.scmp(Op_SGreaterThanEqual, a, c); break;        // s_cmp_ge_i32
        case 0x04: rs.scc = b.scmp(Op_SLessThan, a, c); break;                // s_cmp_lt_i32
        case 0x05: rs.scc = b.scmp(Op_SLessThanEqual, a, c); break;           // s_cmp_le_i32
        case 0x08: rs.scc = b.ucmp(Op_UGreaterThan, a, c); break;             // s_cmp_gt_u32
        case 0x09: rs.scc = b.ucmp(Op_UGreaterThanEqual, a, c); break;        // s_cmp_ge_u32
        case 0x0A: rs.scc = b.ucmp(Op_ULessThan, a, c); break;                // s_cmp_lt_u32
        case 0x0B: rs.scc = b.ucmp(Op_ULessThanEqual, a, c); break;           // s_cmp_le_u32
        case 0x12: case 0x13: {                                                // s_cmp_eq/lg_u64
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
            break;
        }
        default: ok = false;
    }
    return true;
}

} // namespace prosper::gpu::ir
