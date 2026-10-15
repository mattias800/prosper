// sopk.cpp — SOPK lowering: the 16-bit-immediate scalar instructions.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

namespace {
inline bool sopk_sets_full_flat_scratch_base(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPK || in.opcode != 0x13 ||
        in.dst.kind != OperandKind::SGPR)
        return false;
    // S_SETREG_B32's SIMM16 is HWREG(id, offset, width-1). Prosper represents guest scratch as a
    // private Function-storage array and intentionally has no physical FLAT_SCR base, so the exact
    // full-half base relocation emitted by GTA V is semantically absorbed by that abstraction.
    // Partial fields and every other hardware register remain fail-visible.
    const uint32_t hwreg = static_cast<uint16_t>(in.simm16);
    const uint32_t id = hwreg & 0x3fu;
    const uint32_t offset = (hwreg >> 6) & 0x1fu;
    const uint32_t width_minus_one = (hwreg >> 11) & 0x1fu;
    return (id == 20u || id == 21u) && offset == 0u && width_minus_one == 31u;
}
} // namespace

bool lower_sopk(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return SopkLowering({b, rs, in, ok, allow_wave}).lower();
}

bool SopkLowering::lower() {
    // 16-bit-immediate scalar ops. The decoder sign-extends simm16 for signed operations;
    // unsigned comparisons use the original 16-bit bit pattern. SOPK comparisons name their
    // scalar source in the encoded SDST field and write only SCC.
    switch (in.opcode) {
        case 0x00:                                  // s_movk_i32
            rs.sreg[in.dst.value] = b.uconst((uint32_t)in.simm16);
            rs.sreg_srt.erase(in.dst.value);
            break;
        case 0x10: {                                // s_mulk_i32 (read-modify-write)
            // D = low32(D * sign_extend(SIMM16)). Unlike s_addk_i32, MULK does not write
            // SCC. Astro Bot uses the exact `s_mulk_i32 vcc_lo, 276` word to turn a
            // scalar table index into a byte offset before s_buffer_load_dwordx4.
            const uint32_t a = val(in.dst);
            rs.sreg[in.dst.value] = b.ibin(
                Op_IMul, a, b.uconst(static_cast<uint32_t>(in.simm16)));
            rs.sreg_srt.erase(in.dst.value);
            break;
        }
        case 0x0F: {                                // s_addk_i32 (read-modify-write)
            // SIMM16 is sign-extended, the destination receives the low 32 bits, and SCC
            // reports signed overflow. This is the immediate form of SOP2 s_add_i32, so use
            // the same two's-complement overflow identity: (~(a^c)) & (a^d), bit 31.
            const uint32_t a = val(in.dst);
            const uint32_t c = b.uconst(static_cast<uint32_t>(in.simm16));
            const uint32_t d = b.ibin(Op_IAdd, a, c);
            const uint32_t overflow = b.ibin(
                Op_BitwiseAnd,
                b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)),
                b.ibin(Op_BitwiseXor, a, d));
            rs.sreg[in.dst.value] = d;
            rs.sreg_srt.erase(in.dst.value);
            rs.scc = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_ShiftRightLogical, overflow, b.uconst(31)),
                b.uconst(0));
            break;
        }
        case 0x03: case 0x04: case 0x05: case 0x06:
        case 0x07: case 0x08: {                      // s_cmpk_{eq,lg,gt,ge,lt,le}_i32
            const uint32_t a = val(in.dst);
            const uint32_t c = b.uconst((uint32_t)in.simm16);
            switch (in.opcode) {
                case 0x03: rs.scc = b.ucmp(Op_IEqual, a, c); break;
                case 0x04: rs.scc = b.ucmp(Op_INotEqual, a, c); break;
                case 0x05: rs.scc = b.scmp(Op_SGreaterThan, a, c); break;
                case 0x06: rs.scc = b.scmp(Op_SGreaterThanEqual, a, c); break;
                case 0x07: rs.scc = b.scmp(Op_SLessThan, a, c); break;
                case 0x08: rs.scc = b.scmp(Op_SLessThanEqual, a, c); break;
            }
            break;
        }
        case 0x09: case 0x0A: case 0x0B:
        case 0x0C: case 0x0D: case 0x0E: {           // s_cmpk_{eq,lg,gt,ge,lt,le}_u32
            const uint32_t a = val(in.dst);
            const uint32_t c = b.uconst((uint32_t)(uint16_t)in.simm16);
            switch (in.opcode) {
                case 0x09: rs.scc = b.ucmp(Op_IEqual, a, c); break;
                case 0x0A: rs.scc = b.ucmp(Op_INotEqual, a, c); break;
                case 0x0B: rs.scc = b.ucmp(Op_UGreaterThan, a, c); break;
                case 0x0C: rs.scc = b.ucmp(Op_UGreaterThanEqual, a, c); break;
                case 0x0D: rs.scc = b.ucmp(Op_ULessThan, a, c); break;
                case 0x0E: rs.scc = b.ucmp(Op_ULessThanEqual, a, c); break;
            }
            break;
        }
        // Scalar wait counters are SOPK on gfx10, NOT SOPP 0x7d. Loads/stores themselves
        // are synchronous SSA operations in SPIR-V, so vmcnt/expcnt/lgkmcnt remain no-ops.
        // A completed vscnt(0), however, is also the guest's publication point: GTA V writes
        // scan data, waits for those stores, then writes a ready flag that other workgroups
        // poll. Preserve that device-wide UniformMemory ordering with a release barrier.
        // VERIFIED(round-trip llvm-mc gfx1010):
        // vscnt=0xBBFD0000 (op 0x17), vmcnt=0xBC7D0000 (0x18), expcnt=0xBCFD0000 (0x19),
        // lgkmcnt=0xBD7D0000 (0x1A).
        case 0x17:
            if (in.dst.kind == OperandKind::SGPR && in.dst.value == 125 &&
                in.simm16 == 0)
                b.device_uniform_release_barrier();
            break;
        case 0x18: case 0x19: case 0x1A: break;
        case 0x13:                                // s_setreg_b32
            // The encoded SDST field is the SOURCE SGPR. Do not read it: this admission
            // discards only the physical FLAT_SCR relocation which Prosper's private
            // scratch model does not expose. Unsupported/dynamic scratch accesses still
            // reject independently in the FLAT emitter.
            if (!b.is_compute || !sopk_sets_full_flat_scratch_base(in)) ok = false;
            break;
        default: ok = false;
    }
    return true;
}

} // namespace prosper::gpu::ir
