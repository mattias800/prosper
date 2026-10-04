#include "gpu/recompiler/rdna2_movrels.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <map>

namespace prosper::gpu {

bool emit_s_movrels_b32(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    // S_MOVRELS_B32 sDST, sSRC (SOP1 0x2E): dst = SGPR[src0# + M0] (relative-indexed SGPR read).
    // RDNA2 ISA Doc 70648: "addr = SRC0 address; addr += M0; D = SGPR[addr]".
    //
    // Both src0 and dst must be SGPRs (kind == SGPR, 0..105). Non-SGPR sources/destinations refuse.
    // Untracked M0 refuses fail-visibly (matches v_movrels_b32). A candidate SGPR the shader never
    // wrote reads as 0 -- operand_bits' placeholder for an unwritten SGPR, the same convention the
    // v_movrels_b32 lowering uses for unwritten VGPRs (whose hardware value is undefined). A
    // candidate that is not representable (an entry-M0 token, a saved mask half) refuses the whole
    // instruction instead.
    //
    // CONFIDENCE: HIGH for the indexed read itself (ISA pseudocode: "addr = SRC0 address;
    // addr += M0; D = SGPR[addr]").
    // CONFIDENCE: LOW for indices past s105: the ISA text quoted above does not say what such an
    // index reads, so the constant fold refuses them and the dynamic select only spans s[base..105]
    // rather than inventing a value.
    if (in.src[0].kind != OperandKind::SGPR || in.dst.kind != OperandKind::SGPR ||
        in.src[0].value > 105 || in.dst.value > 105) {
        ok = false;
        return true;
    }

    auto m0it = rs.sreg.find(124);
    if (m0it == rs.sreg.end()) {
        ok = false;
        return true;
    }

    const int base = in.src[0].value;

    // Optimization: If M0 is a compile-time constant literal, fold directly to single register read.
    uint32_t const_m0 = 0;
    if (b.uconst_literal(m0it->second, &const_m0)) {
        const int target_reg = base + static_cast<int>(const_m0);
        if (target_reg < 0 || target_reg > 105) {
            ok = false;
            return true;
        }
        bool cand_ok = true;
        const uint32_t val =
            operand_bits(b, rs, in, Operand{OperandKind::SGPR, target_reg}, &cand_ok);
        if (!cand_ok) {
            ok = false;
            return true;
        }
        rs.sreg[in.dst.value] = val;
        auto tag = rs.sreg_srt.find(target_reg);
        if (tag != rs.sreg_srt.end()) {
            rs.sreg_srt[in.dst.value] = tag->second;
        } else {
            rs.sreg_srt.erase(in.dst.value);
        }
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value);
        rs.sreg_entry_m0.erase(in.dst.value);
        rs.sreg_written.insert(in.dst.value);
        return true;
    }

    // Dynamic M0: evaluate every representable candidate register in [base, 105].
    // If any candidate register in [base, 105] is unrepresentable (e.g. entry-M0 token or saved mask),
    // refuse fail-visibly rather than silently fabricating 0.
    uint32_t acc = b.uconst(0);
    for (int r = base; r <= 105; ++r) {
        bool cand_ok = true;
        const uint32_t cand_val = operand_bits(b, rs, in, Operand{OperandKind::SGPR, r}, &cand_ok);
        if (!cand_ok) {
            ok = false;
            return true;
        }
        acc = b.sel(b.ucmp(Op_IEqual, m0it->second, b.uconst(static_cast<uint32_t>(r - base))),
                    cand_val, acc);
    }

    rs.sreg[in.dst.value] = acc;
    rs.sreg_srt.erase(in.dst.value);
    rs.sreg_bool.erase(in.dst.value);
    rs.sreg_bool_narrowed.erase(in.dst.value);
    rs.sreg_bool_b32.erase(in.dst.value);
    rs.sreg_entry_m0.erase(in.dst.value);
    rs.sreg_written.insert(in.dst.value);
    return true;
}

}  // namespace prosper::gpu
