// Remaining VOP1 direct-emission routes.
#include "gpu/recompiler/ir/lower/vop1_case_routes.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/lower/vop1_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_output_modifiers.hpp"
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_float_source.hpp"
#include "gpu/recompiler/ir/lower/vop1_dpp_source_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop1_unary_f16_lowering.hpp"
#include <cstdint>
#include <optional>

namespace prosper::gpu::ir {
bool Vop1CaseRoutes::lower() {
    uint32_t a = val(in.src[0]);
    uint32_t old_d = vreg_old(b, rs, in.dst.value);
    uint32_t dpp_active = 0;
    if (auto routed = ir::lower_vop1_dpp_source(b, rs, in, ok, a, dpp_active)) return *routed;
    ir::lower_vop1_float_source(b, in, a);
    ir::select_vop1_conversion_source(b, in, a);
    ir::select_vop1_integer_source(b, in, a);
    if (auto routed = early_routes(a)) return *routed;
    uint32_t& d = vreg[in.dst.value];
    if (auto routed = sdwa_routes(a, d, old_d)) return *routed;
    if (auto routed = ir::lower_vop1_unary_f16(b, rs, in, a, d, old_d, ok)) return *routed;
    if (auto routed = ir::lower_vop1_conversion_routes(b, rs, in, a, d, old_d, ok)) return *routed;
    const bool integer_handled = ir::lower_vop1_integer(b, in, a, d, ok);
    const bool float_handled = ir::lower_vop1_unary_float(b, in, a, d, ok);
    const bool conversion_handled = ir::lower_vop1_conversion_values(b, rs, in, a, d, old_d, ok);
    if (auto routed = residual_routes(d, integer_handled, float_handled, conversion_handled))
        return *routed;
    ir::lower_vop1_output_modifiers(b, in, d, ok);
    if (ok) {
        if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
        predicate_write(b, rs, in.dst.value, old_d);
    }
    return true;
}

std::optional<bool> Vop1CaseRoutes::early_routes(uint32_t& a) {
    if (in.opcode == 0x02) {   // v_readfirstlane_b32: SGPR dst = value of the lowest active lane
        if (refuse_arbitrary_scalar_publication(b, rs, in)) {
            ok = false;
            return true;
        }
        // Cross-lane broadcast. Our per-lane scalar model has no cross-lane reduction, so we use
        // THIS lane's value. SPECULATIVE(confidence: med): exact only when src0 is wave-uniform —
        // which is the standard use (reading a uniformly-computed VGPR into an SGPR, e.g. the
        // integer-divide reciprocal in the game's shaders). Writes an SGPR, not a VGPR.
        rs.sreg[in.dst.value] = a;
        rs.sreg_srt.erase(in.dst.value);
        retire_scalar_data_word_mask(rs, in.dst.value);
        // #3596: record that this scalar is NOT wave-uniform. The value above is one lane's,
        // so any later proof that reasons "it is in an SGPR, therefore it broadcasts" is
        // wrong about it. Tainting the SSA value rather than the register makes the taint
        // survive scalar copies for free.
        rs.lane_local_scalars.insert(a);
        return true;
    }
    if (in.opcode == kVop1OpcodeMovreldB32) {   // VGPR[VDST + M0] = SRC0
        // RDNA2 ISA sec. 6.6. M0 is a runtime unsigned VGPR offset. Represent the indexed
        // write as selects over every statically referenced destination at or above VDST;
        // an out-of-range destination is unobservable because no later instruction names it.
        // This preserves exact loop-carried SSA while avoiding an architectural 256-word
        // register array for shaders that never use relative source addressing.
        auto m0 = rs.sreg.find(124);
        if (m0 == rs.sreg.end()) {
            ok = false;
            return true;
        }
        for (int reg = in.dst.value; reg <= rs.max_vgpr; ++reg) {
            const uint32_t old = vreg_old(b, rs, reg);
            rs.vreg[reg] = b.sel(
                b.ucmp(Op_IEqual, m0->second, b.uconst(static_cast<uint32_t>(reg - in.dst.value))),
                a, old);
            predicate_write(b, rs, reg, old);
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop1CaseRoutes::sdwa_routes(uint32_t& a, uint32_t& d, uint32_t& old_d) {
    // WORD-select v_mov_b32_sdwa (#273): extract the selected 16-bit source half and insert it
    // into the selected dest half, preserving the other (the f16 half-move; decode accepted
    // only dst WORD_0/1 + PRESERVE with src DWORD/WORD_0/WORD_1).
    if (in.opcode == 0x01 && in.sdwa_dst_sel != 6) {
        uint32_t v = a;
        if (in.sdwa_src0_sel == 5) v = b.ibin(Op_ShiftRightLogical, a, b.uconst(16));
        uint32_t v16 = b.ibin(Op_BitwiseAnd, v, b.uconst(0xFFFFu));
        d = (in.sdwa_dst_sel == 4)
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), v16)
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, v16, b.uconst(16)));
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    // BYTE- or WORD-select v_mov_b32_sdwa (#273 — DOLL's title post PSes unpack a packed
    // dword: `v_mov_b32_sdwa v6, v15 src0_sel:BYTE_0`; #2013 — Sonic Racing: CrossWorlds'
    // `v_mov_b32_sdwa v0, v0 src0_sel:WORD_0`): dst is the whole dword (UNUSED_PAD), so the
    // result is the selected byte/word extended to 32 bits. S0_SEXT selects which extension:
    // sign for `v_mov_b32_sdwa v14, sext(v8) src0_sel:WORD_0`, zero otherwise. These are two
    // different operations on the same opcode, so the bit is read rather than assumed — a
    // zero-extending lowering of the sext form silently produces a large positive value
    // where the guest computed a negative one.
    if (in.opcode == 0x01 && in.sdwa_dst_sel == 6 && in.sdwa_src0_sel <= 5) {
        const uint32_t offset =
            in.sdwa_src0_sel <= 3 ? 8u * in.sdwa_src0_sel : 16u * (in.sdwa_src0_sel - 4u);
        const uint32_t width = in.sdwa_src0_sel <= 3 ? 8u : 16u;
        d = in.sdwa_src0_sext ? b.bfe_s(a, b.uconst(offset), b.uconst(width))
                              : b.bfe_u(a, b.uconst(offset), b.uconst(width));
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop1CaseRoutes::residual_routes(uint32_t& d, const bool& integer_handled,
                                                    const bool& float_handled,
                                                    const bool& conversion_handled) {
    switch (in.opcode) {
        case 0x00:
            return true;   // v_nop — no-op (writes nothing; common
            // scheduling/hazard filler in real shaders)

        case 0x43: {   // v_movrels_b32: dst = VGPR[src0# + M0] (relative-indexed VGPR read)
            // M0 is written by plain scalar ALU (s_mov/s_or m0, … decode dst as SGPR 124), so
            // its per-invocation value lives in rs.sreg[124]. The source register NUMBER is
            // src0# + M0 — a runtime value — so lower to a bounded select chain over every
            // tracked VGPR at src0#+k (k = M0 candidate): dst = Σ sel(m0==k, vreg[src0+k]).
            // Matches the hardware contract for all in-range M0 (reading an unwritten VGPR is
            // undefined on HW too — those candidates read our 0 placeholder). An UNTRACKED M0
            // rejects, never silently indexes 0. VERIFIED(round-trip llvm-mc gfx1010:
            // 0x7e408706 → v_movrels_b32_e32 v32, v6; DOLL UI/skinned VS #273).
            auto m0it = rs.sreg.find(124);
            if (m0it == rs.sreg.end()) {
                ok = false;
                break;
            }
            uint32_t acc = b.uconst(0);
            const int base = in.src[0].value;
            for (const auto& kv : vreg) {
                if (kv.first < base || !kv.second) continue;   // (!kv.second: the dst slot the
                // enclosing `vreg[dst]` may have
                // default-inserted — no value yet)
                acc = b.sel(b.ucmp(Op_IEqual, m0it->second, b.uconst((uint32_t)(kv.first - base))),
                            kv.second, acc);
            }
            d = acc;
            break;
        }
        default:
            if (!integer_handled && !float_handled && !conversion_handled) ok = false;
    }
    return std::nullopt;
}

}   // namespace prosper::gpu::ir
