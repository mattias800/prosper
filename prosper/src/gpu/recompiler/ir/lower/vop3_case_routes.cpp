#include "gpu/recompiler/ir/lower/vop3_case_routes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/lower/native_half_fma_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/recompiler/ir/lower/vop3_sad_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_permute_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad64_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/ir/lower/vop3_unary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_subword_to_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bitfield_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer64_shift_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_align_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mask_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_half_integer_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

namespace prosper::gpu::ir {
void Vop3CaseRoutes::lower() {
    if (ir::lower_vop3_pack16(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_minmax_half(b, rs, in, ok, allow_wave, old_d, fresult)) {
    } else if (in.opcode == 0x34B) {
        ir::lower_native_half_fma(b, rs, in, ok, allow_wave, old_d, std::cref(val),
                                  std::cref(fresult));
    } else if (ir::lower_vop3_integer16(b, rs, in, ok, allow_wave, old_d)) {
    } else if (ir::lower_vop3_mad_float(b, rs, in, ok, allow_wave, std::cref(fv),
                                        std::cref(fresult))) {
    } else if (in.opcode == 0x14B) {
        // GLSL Fma intent; Vulkan precision remains implementation-dependent (#5225).
        vreg[in.dst.value] = fresult(ir::emit_fused_mad_float(b, std::cref(fv)));
    } else if (ir::lower_vop3_mad64(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_carry_out(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_multiply(b, rs, in, ok, allow_wave)) {
    } else if (in.opcode == 0x157 &&
               ir::lower_vop3_minmax_float(b, rs, in, ok, allow_wave, std::cref(fv),
                                           std::cref(fresult))) {
    } else if (ir::lower_vop3_minmax(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_minmax_float(b, rs, in, ok, allow_wave, std::cref(fv),
                                           std::cref(fresult))) {
    } else if (ir::lower_vop3_pack_norm(b, rs, in, ok, allow_wave, std::cref(fv))) {
    } else if (ir::lower_vop3_pack_saturate(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_ldexp(b, rs, in, ok, allow_wave, std::cref(fv), std::cref(fresult))) {
    } else if (ir::lower_vop3_cube_axis(b, rs, in, ok, allow_wave, std::cref(fv),
                                        std::cref(fresult))) {
    } else
        cube_and_integer();
}

void Vop3CaseRoutes::cube_and_integer() {
    if (in.opcode == 0x145 || in.opcode == 0x146) {
        // Cubemap coordinate ops (#273 — DOLL's title post PSes' reflection-probe math):
        // v_cubeid_f32 (0x144) face id, v_cubesc_f32 (0x145) S numerator, v_cubetc_f32
        // (0x146) T numerator, v_cubema_f32 (0x147) 2*major-axis. Src order (x, y, z);
        // face-major selection per the AMD ISA / GL cubemap convention:
        //   |z|>=|x| && |z|>=|y| : id = z<0?5:4  sc = z<0?-x:x  tc = -y      ma = 2z
        //   else |y|>=|x|        : id = y<0?3:2  sc = x         tc = z<0.. = y<0?-z...
        //   (see per-op emission below)
        // VERIFIED(round-trip llvm-mc gfx1030: 0xd544-0xd547). Execution-tested (kernel in
        // test_rdna2_to_spirv) against the GL major-axis table. CONFIDENCE: MED.
        uint32_t x = fv(0), y = fv(1), z = fv(2);
        uint32_t ax = b.fext1(Glsl_FAbs, x), ay = b.fext1(Glsl_FAbs, y), az = b.fext1(Glsl_FAbs, z);
        uint32_t zmaj = b.land(b.fcmp(Op_FOrdGreaterThanEqual, az, ax),
                               b.fcmp(Op_FOrdGreaterThanEqual, az, ay));
        uint32_t ymaj = b.fcmp(Op_FOrdGreaterThanEqual, ay, ax);      // (only used when !zmaj)
        uint32_t zneg = b.fcmp(Op_FOrdLessThan, z, b.uconst(0));
        uint32_t yneg = b.fcmp(Op_FOrdLessThan, y, b.uconst(0));
        uint32_t xneg = b.fcmp(Op_FOrdLessThan, x, b.uconst(0));
        auto fneg = [&](uint32_t v) { return b.fbin(Op_FSub, b.uconst(0), v); };
        uint32_t r2;
        switch (in.opcode) {
            case 0x145: {   // sc: z-major: z<0 ? -x : x; y-major: x; x-major: x<0 ? z : -z
                uint32_t scz = b.sel(zneg, fneg(x), x);
                uint32_t scx = b.sel(xneg, z, fneg(z));
                r2 = b.sel(zmaj, scz, b.sel(ymaj, x, scx));
                break;
            }
            case 0x146: {   // tc: z-major: -y; y-major: y<0 ? -z : z; x-major: -y
                uint32_t tcy = b.sel(yneg, fneg(z), z);
                r2 = b.sel(zmaj, fneg(y), b.sel(ymaj, tcy, fneg(y)));
                break;
            }
            default: std::abort();   // Guard above owns only SC and TC.
        }
        vreg[in.dst.value] = fresult(r2);
    } else if (ir::lower_vop3_mad24(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_sad(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_div_fixup(b, rs, in, ok, allow_wave, fv, fresult)) {
    } else if (ir::lower_vop3_bitfield(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_align(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_permute(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_integer64_shift(b, rs, in, ok)) {
    } else if (ir::lower_vop3_bit_utilities(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_binary_integer(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_unary_integer(b, rs, in, ok, allow_wave)) {
    } else if (ir::lower_vop3_half_integer(b, rs, in, ok, allow_wave, old_d)) {
    } else if (ir::lower_vop3_subword_to_float(b, rs, in, ok)) {
    } else if (ir::lower_vop3_ternary_integer(b, rs, in, ok, allow_wave)) {
        // The common modifier and predicated destination stages still follow.
    } else if (ir::lower_vop3_carry_in_out(b, rs, in, ok, allow_wave)) {
    } else
        mbcnt_values();
}

void Vop3CaseRoutes::mbcnt_values() {
    if ((in.opcode == 0x365 || in.opcode == 0x366) && b.ngg_logical_lane &&
        in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
        // A proven no-GS passthrough executes only the logical vertex producer; Vulkan's
        // flattened vertex/instance invocation is the corresponding guest ES lane.  For
        // the canonical all-ones MBCNT pair, LOW contributes min(lane, 32) and HIGH
        // contributes max(lane - 32, 0). General masks remain fail-closed because a vertex
        // invocation has no peer-lane mask state from which to reconstruct them.
        const uint32_t lane = b.guest_lane_id();
        const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
        const uint32_t count = in.opcode == 0x365
                                   ? b.sel(high, b.uconst(32), lane)
                                   : b.sel(high, b.ibin(Op_ISub, lane, b.uconst(32)), b.uconst(0));
        vreg[in.dst.value] = b.ibin(Op_IAdd, val(in.src[1]), count);
    } else if ((in.opcode == 0x365 || in.opcode == 0x366) && b.ngg_one_lane) {
        // NGG is deliberately lowered as one guest lane per Vulkan vertex invocation. No
        // lane precedes that invocation, so either half of MBCNT contributes zero and leaves
        // its accumulator unchanged. This is the scalar counterpart of recompile_vertex's
        // existing s3=1 (one ES vertex, no GS primitive) ABI model.
        vreg[in.dst.value] = val(in.src[1]);
    } else if ((in.opcode == 0x365 || in.opcode == 0x366) && (b.is_compute || b.is_fragment) &&
               !(in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) &&
               (mbcnt_has_intrinsic_numeric_source(in) ||
                mbcnt_has_scalar_numeric_source(rs, in.src[0]))) {
        // A numeric source word is local DATA. Scanning peer invocations would instead
        // count their execution population (and could discard helper/inactive bits).
        // Preserve the existing all-ones lane-ID fast path below; other numeric masks
        // use the ISA's physical prefix directly, without `allow_wave` or barriers.
        const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
        if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
            in.src_neg[2] || in.clamp || in.omod || opsel || in.has_dpp || in.has_sdwa) {
            b.stage_reject_pc = in.pc;
            b.stage_reject_reason = "mbcnt-numeric-modifiers";
            ok = false;
        } else {
            vreg[in.dst.value] =
                b.numeric_mbcnt(val(in.src[0]), val(in.src[1]), in.opcode == 0x365);
        }
    } else if ((in.opcode == 0x365 || in.opcode == 0x366) && allow_wave &&
               !(in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) &&
               (b.is_compute || b.is_fragment)) {
        // v_mbcnt_lo/hi_u32_b32 (cross-lane): dst = src1 + count of lanes below this one whose mask
        // bit (src0) is set, in the low/high 32. The per-lane "mask bit" comes from src0: EXEC
        // (126/127) -> this lane's exec bool; inline -1 (all-ones) -> always set (mbcnt = lane
        // index, the common "get my lane id" idiom, e.g. shader 037); inline 0 -> never; an SGPR
        // pair -> that saved mask's bool. A general computed 32-bit mask VALUE isn't representable
        // per-lane, so reject that. Portable compute uses LDS+barriers at barrier-uniform sites;
        // exact-size compute subgroups and fragments use a native exclusive subgroup sum.
        const uint32_t active = mbcnt_source_bit(b, rs, in.src[0], in.opcode == 0x366);
        if (active) {
            const uint32_t acc = val(in.src[1]);
            const uint32_t lo = in.opcode == 0x365 ? b.btrue() : b.bfalse();
            vreg[in.dst.value] =
                b.is_fragment ? b.fragment_mbcnt(active, acc, in.opcode == 0x365)
                              : (b.native_subgroup_size ? b.native_compute_mbcnt(active, acc, lo)
                                                        : b.mbcnt(active, acc, in.opcode == 0x365));
        } else
            ok = false;
    } else
        mbcnt_lane_id();
}

void Vop3CaseRoutes::mbcnt_lane_id() {
    if ((in.opcode == 0x365 || in.opcode == 0x366) && in.src[0].kind == OperandKind::InlineInt &&
        in.src[0].value == -1 && (b.is_compute || b.is_fragment)) {
        // An ALL-ONES mask makes MBCNT a pure function of this lane's OWN index: the number
        // of set bits below lane L in an all-ones 64-bit mask is L, so LO contributes
        // min(L, 32) and HI contributes max(L - 32, 0). No other lane's state is read, so
        // this form needs neither `allow_wave` nor a subgroup/LDS reduction and stays exact
        // under divergent control flow — it is the ubiquitous "what is my lane id" idiom,
        // including when `allow_wave` is true: a reduction counts participating invocations,
        // whereas this literal mask counts physical bit positions, including inactive ones.
        // and it is the scalar counterpart of the ngg_logical_lane branch above.
        // (Sonic Racing: CrossWorlds' compute post chain emits `d7650001,000100c1` =
        // `v_mbcnt_lo_u32_b32 v1, -1, 0` inside a structured region, where the compute
        // structurizer sets wave_ok=false and the branch above therefore never runs; #2013.)
        const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
        const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
        const uint32_t count = in.opcode == 0x365
                                   ? b.sel(high, b.uconst(32), lane)
                                   : b.sel(high, b.ibin(Op_ISub, lane, b.uconst(32)), b.uconst(0));
        vreg[in.dst.value] = b.ibin(Op_IAdd, val(in.src[1]), count);
    } else
        mbcnt_rejection();
}

void Vop3CaseRoutes::mbcnt_rejection() {
    if (in.opcode == 0x365 || in.opcode == 0x366) {
        // Cross-lane V_MBCNT that none of the four lowerings above covers. RDNA2 ISA
        // (doc 70648) defines it as
        //     ThreadMask = (1 << lane) - 1
        //     dst = countbits(src0 & ThreadMask[31:0 | 63:32]) + src1
        // -- a prefix population count over the lanes of THIS wave. Every arm above needs a
        // lane set to count over, and gets one a different way:
        //   * the all-ones arms need only THIS lane's index, because the count of set bits
        //     below lane L in an all-ones mask is L;
        //   * `ngg_one_lane`'s wave is one lane, so no lane precedes and either half
        //     contributes zero for ANY mask;
        //   * compute and fragment reduce over a real peer set -- LDS + barriers, or a
        //     subgroup whose exact width the module declares.
        //
        // A generic vertex shell has none of those. Its guest lane id is MODELLED as the
        // flattened vertex/instance invocation index (`guest_lane_id()` in
        // rdna2_to_spirv_internal.hpp), and nothing in Vulkan promises that the invocations
        // sharing a subgroup are the wave_size consecutive flattened indices that model
        // calls one guest wave. A subgroup reduction here would therefore count over a
        // DIFFERENT set of lanes and return a well-formed wrong slot index, which is the
        // silent-miscompile shape this file exists to avoid -- so it stays fail-visible.
        //
        // What changes here is only that the failure NAMES itself. The reject line's `mode`
        // otherwise reports `unresolved-operand`, whose own contract at the reject site is
        // "the lowering exists and the descriptor is the defect" (#2412) -- the opposite of
        // the truth for a stage-unsupported cross-lane op, and a census reads it as a
        // resource-table problem. #3135.
        const char* form = "other-mask";
        if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1)
            form = "all-ones";
        else if (in.src[0].value == 126 || in.src[0].value == 127)
            form = "exec";
        else if (in.src[0].value == 106 || in.src[0].value == 107)
            form = "vcc";
        else if (in.src[0].kind == OperandKind::SGPR)
            form = "sgpr-mask";
        char reason[192];
        if (b.is_vertex && b.vertex_general_mask_mbcnt_pc != UINT32_MAX &&
            std::strcmp(form, "all-ones") == 0)
            // The instruction that failed is lowerable on its own; what disqualified the
            // whole-program flattened-lane model is a general-mask MBCNT further down. Name
            // THAT pc, or the next reader spends the investigation at this one.
            std::snprintf(reason, sizeof reason,
                          "mbcnt-cross-lane stage=vertex form=all-ones wave=none "
                          "disqualified-by-general-mask-pc=%u",
                          b.vertex_general_mask_mbcnt_pc);
        else
            std::snprintf(reason, sizeof reason, "mbcnt-cross-lane stage=%s form=%s wave=none",
                          b.is_vertex ? "vertex" : (b.is_fragment ? "fragment" : "compute"), form);
        b.stage_reject_pc = in.pc;
        b.stage_reject_reason = reason;
        ok = false;
    } else
        float_tail();
}

void Vop3CaseRoutes::float_tail() {
    if (ir::lower_vop3_rtz_pack(b, rs, in, ok, allow_wave, fv)) {
    } else if (ir::lower_vop3_float_integer(b, rs, in, ok)) {
    } else if (ir::lower_vop3_float_values(b, rs, in, ok, allow_wave, fv, fresult)) {
    } else if (ir::lower_vop3_legacy_mul(b, rs, in, ok, allow_wave, fv, fresult)) {
    } else if (ir::lower_vop3_cndmask(b, rs, in, ok, allow_wave, std::cref(fv))) {
    } else if (ir::lower_vop3_mac_float(b, rs, in, ok, allow_wave, std::cref(fv),
                                        std::cref(fresult), old_d)) {
    } else
        ok = false;
}

}   // namespace prosper::gpu::ir
