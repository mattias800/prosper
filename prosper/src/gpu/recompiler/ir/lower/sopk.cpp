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
    if (const ScalarOpDef* def = find_scalar_op(in.fmt, in.opcode)) return lower_data(*def);
    switch (in.opcode) {
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

// s_movk_i32, s_addk_i32, s_mulk_i32 and s_cmpk_*: their IR definitions (ir/ops/scalar_alu.cpp,
// ir/ops/scalar_compare.cpp). s_addk is the immediate form of s_add_i32 and shares its definition.
bool SopkLowering::lower_data(const ScalarOpDef& def) {
    const uint32_t s0 = def.operands == ScalarOperands::SimmSext ? 0 : val(in.dst);
    const uint32_t s1 = b.uconst(scalar_op_simm16(def, in));
    const ScalarOpIds result = emit_scalar_op(b, def, s0, s1, rs.scc);
    if (result.d) {
        rs.sreg[in.dst.value] = result.d;
        rs.sreg_srt.erase(in.dst.value);
    }
    if (result.scc) rs.scc = result.scc;
    return true;
}

} // namespace prosper::gpu::ir
