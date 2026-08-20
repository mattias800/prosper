#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_alu_support.hpp — helpers shared by rdna2_emit_alu.cpp and rdna2_to_spirv.cpp.
// INTERNAL to src/gpu/recompiler/.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {


// A scalar inline integer used by a B64 mask operation is sign-extended to 64 bits.  Return the bit
// belonging to this emulated hardware lane without ever issuing an undefined >=32 SPIR-V shift.
// Astro's reduction tails use 15, 3, and 1 for the final 4/2/1 active lanes.
inline uint32_t inline_int_mask_bit(SpirvCompute& b, int value) {
    if (value == -1) return b.btrue();
    if (value == 0) return b.bfalse();
    // A Vulkan vertex invocation is the single guest lane retained by the NGG approximation. It is
    // lane zero, so an inline B64 mask contributes exactly its low bit; vertex modules do not declare
    // LocalInvocationIndex, and trying to use that compute-only ID emitted invalid SPIR-V ID zero.
    if (b.ngg_one_lane)
        return (static_cast<uint32_t>(value) & 1u) ? b.btrue() : b.bfalse();
    if (b.is_vertex) return 0; // an ordinary VS has no proven lane identity for a partial wave mask
    const uint32_t lane_id = b.is_fragment ? b.subgroup_local_id() : b.linear_localid;
    const uint32_t lane = b.ibin(Op_BitwiseAnd, lane_id,
                                  b.uconst(b.wave_size - 1));
    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
    const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
    const uint32_t word = b.sel(high, b.uconst(value < 0 ? UINT32_MAX : 0u),
                                b.uconst(static_cast<uint32_t>(value)));
    return b.ucmp(Op_INotEqual,
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
        b.uconst(0));
}

// V_MBCNT_HI positions its 32-bit S0 at lanes 32..63 (ISA ops 869/870: HI tests S0 bit i against
// ThreadMask[32+i]) — S0 is an independent 32-bit value, NOT the high dword of a 64-bit-extended
// operand. Lanes < 32 never contribute to the HI window, so their bit is 0.
inline uint32_t inline_int_mask_bit_hi(SpirvCompute& b, int value) {
    if (value == 0) return b.bfalse();
    if (b.ngg_one_lane) return b.bfalse(); // the modeled NGG lane is lane zero, never in the HI window
    if (b.is_vertex) return 0;
    const uint32_t lane_id = b.is_fragment ? b.subgroup_local_id() : b.linear_localid;
    const uint32_t lane = b.ibin(Op_BitwiseAnd, lane_id, b.uconst(b.wave_size - 1));
    const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
    const uint32_t bit  = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));   // lane-32 for lanes >= 32
    const uint32_t isset = b.ucmp(Op_INotEqual,
        b.ibin(Op_BitwiseAnd,
               b.ibin(Op_ShiftRightLogical, b.uconst(static_cast<uint32_t>(value)), bit),
               b.uconst(1)),
        b.uconst(0));
    return b.bsel(high, isset, b.bfalse());
}

// Per-invocation bit of a 64-bit mask consumed by V_MBCNT. The HI instruction names the odd SGPR
// of an aligned scalar pair (for example s7 for s[6:7]), while our bool-domain mask is keyed by the
// pair's low register. The LOW instruction names that root directly; the HIGH instruction names the
// following register, so only source-1 can represent its mask bits. An exact HIGH key would instead
// describe source:source+1 and silently read the wrong physical dword. VCC/EXEC are represented
// directly rather than through the scalar-data register file —
// but only in the canonical pairing (LO with the low half, HI with the high half); a cross-pairing
// (e.g. v_mbcnt_lo with exec_hi) reads OTHER lanes' bits, which the per-invocation model cannot
// represent, so it returns the 0 reject sentinel.
inline uint32_t mbcnt_source_bit(SpirvCompute& b, const RegState& rs, const Operand& source,
                          bool high_half) {
    if (source.value == 126 || source.value == 127)
        return (source.value == (high_half ? 127 : 126)) ? rs.exec : 0;
    if (source.value == 106 || source.value == 107)
        return (source.value == (high_half ? 107 : 106)) ? rs.vcc : 0;
    if (source.kind == OperandKind::InlineInt)
        return high_half ? inline_int_mask_bit_hi(b, source.value)
                         : inline_int_mask_bit(b, source.value);
    if (source.kind != OperandKind::SGPR) return 0;
    const int root = high_half ? source.value - 1 : source.value;
    if (root < 0) return 0;
    const auto found = rs.sreg_bool.find(root);
    return found == rs.sreg_bool.end() ? 0 : found->second;
}

// S_CSELECT_B32 recycling VCC_LO as ordinary scalar scratch. GTA V selects inline constants; the
// stock Unreal volumetric-fog kernel selects tracked SGPRs and then CHAINS through VCC_LO itself
// (#2741: The Plucky Squire 0x3015fd0000 pc217/220/223 is a four-wide cascade -- an SGPR-only
// widening fixes pc217 and re-rejects at pc220, whose second source decodes as Special 106).
//
// This is a SHAPE predicate over operand KINDS only. Whether a non-constant source actually HOLDS
// scalar data is decided elsewhere and deliberately not duplicated here: the whole-stream pre-pass
// answers it with source_is_scalar_word(), and emit_alu's operand_bits() rejects any source whose
// dword is not representable in the per-invocation model. Acceptance additionally requires the
// separate complete_scalar_pair / vcc_b32_low_only_pcs proof at each PC -- whether the untouched
// VCC_HI word may be discarded or must form a complete scalar pair is never a property of the
// select itself.
//
// VCC_HI (Special 107) is deliberately NOT admitted as a source. Where the low-only proof applies
// it has just declared that word dead, and reading it back as data at native subgroup 64 would
// materialize a ballot half the proof said nobody can observe. Two packet-drift guards in
// test_rdna2_to_spirv.cpp exist to keep that fail-visible, and admitting Special 106..125 wholesale
// breaks both.
inline bool is_wave64_vcc_lo_scalar_cselect(const Rdna2Inst& in) {
    const auto scalar_source_kind = [](const Operand& o) {
        return o.kind == OperandKind::InlineInt || o.kind == OperandKind::InlineFloat ||
               o.kind == OperandKind::SGPR ||
               (o.kind == OperandKind::Special && o.value == 106);
    };
    return in.fmt == Rdna2Format::SOP2 && in.opcode == kSop2OpcodeCselectB32 &&
        in.dst.kind == OperandKind::SGPR && in.dst.value == 106 &&
        scalar_source_kind(in.src[0]) && scalar_source_kind(in.src[1]);
}

// IMAGE_GET_LOD currently models only the ordinary FP32 sampled-image form. Keep the unsupported
// Table 100 control families separate so each can be mutation-tested, while production and the
// table-less coverage classifier consume one shared predicate and cannot drift apart.
inline bool mimg_get_lod_has_address_controls(const Rdna2Inst& in) {
    return in.mimg_nsa != 0u || in.mimg_unorm || in.mimg_a16;
}
inline bool mimg_get_lod_has_cache_controls(const Rdna2Inst& in) {
    return in.mimg_dlc || in.mimg_glc || in.mimg_slc;
}
inline bool mimg_get_lod_has_result_controls(const Rdna2Inst& in) {
    return in.mimg_r128 || in.mimg_tfe || in.mimg_lwe || in.mimg_d16;
}
inline bool mimg_get_lod_has_unmodeled_controls(const Rdna2Inst& in) {
    return mimg_get_lod_has_address_controls(in) ||
           mimg_get_lod_has_cache_controls(in) ||
           mimg_get_lod_has_result_controls(in) ||
           in.mimg_reserved;
}

// Resolve an operand to its raw 32-bit value (bits). Float ops bitcast these to float.
// `ok`: cleared when the operand's VALUE is not representable — a Special operand read as ALU DATA
// (VCC/EXEC live as per-lane bools in rs.vcc/rs.exec; their 32-bit wave-mask value does not exist
// in the per-invocation model, and untracked M0/ttmp aren't modeled). SGPR_NULL (field 125) is 0;
// SCC (field 253) is a scalar 0/1 and therefore is representable exactly. Previously other
// untracked Specials silently read as 0 and the shader computed
// garbage (#134); now it rejects, matching the SDWA/DPP reject-rather-than-miscompute discipline.
inline uint32_t operand_bits(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, const Operand& o, bool* ok = nullptr) {
    switch (o.kind) {
        case OperandKind::VGPR: {
            // A scalar-spill vgpr (v_writelane slots) has no per-lane data value — reject, don't read 0.
            if (rs.vgpr_lane_slots.count(o.value) || rs.vgpr_lane_mask_slots.count(o.value)) {
                if (ok) *ok = false; return b.uconst(0);
            }
            auto it = rs.vreg.find(o.value); return it == rs.vreg.end() ? b.uconst(0) : it->second; }
        case OperandKind::SGPR: {
            auto it = rs.sreg.find(o.value);
            if (it != rs.sreg.end()) return it->second;
            auto input = rs.sreg_input.find(o.value);
            if (input != rs.sreg_input.end()) return input->second;
            // A proven Wave32 B32 mask occupies one complete physical SGPR. When Vulkan supplies
            // one exact native subgroup per guest wave, its ballot is therefore the architectural
            // scalar dword. GTA V's terrain kernel writes s20 with an explicit VOPC destination,
            // then consumes s20 as ordinary DATA at pc67. The dispatcher MUST analysis filters the
            // Bool lifetime at every block entry, while record_scalar_write ends it on any scalar
            // overwrite, so this cannot resurrect a stale mask after an SGPR is recycled.
            if (b.is_compute && b.wave_size == 32 && b.native_subgroup_size == 32 &&
                rs.sreg_bool_b32.contains(o.value)) {
                auto mask = rs.sreg_bool.find(o.value);
                if (mask != rs.sreg_bool.end()) {
                    const uint32_t word = b.native_wave_ballot_half(mask->second, 0);
                    if (word) return word;
                }
                if (ok) *ok = false;
                return b.uconst(0);
            }
            // A persisted B64 wave mask has no ordinary scalar dword unless the transfer proof
            // retained that word explicitly. Reject a data-domain read of either physical half;
            // returning the generic unwritten-SGPR zero here would silently replace live ballot
            // bits after a CFG-dispatcher reload. Bool-domain consumers bypass operand_bits.
            const auto is_b64_mask_base = [&](int base) {
                return rs.sreg_bool.contains(base) &&
                       !rs.sreg_bool_b32.contains(base);
            };
            int mask_base = -1;
            if (is_b64_mask_base(o.value)) mask_base = o.value;
            else if (o.value > 0 && is_b64_mask_base(o.value - 1))
                mask_base = o.value - 1;
            if (mask_base >= 0) {
                // The Wave64 MUST analysis in the CFG dispatcher is the lifetime tag that proves
                // this physical word still names the saved B64 predicate rather than recycled
                // scalar scratch. An exact fragment-Wave64 contract or native compute subgroup can
                // materialize that word directly; portable compute dispatchers handle the same
                // FFBH shape in their synchronized common phase below.
                if (b.wave_size == 64 &&
                    (b.is_fragment ||
                     (b.is_compute && b.native_subgroup_size == 64))) {
                    const uint32_t half = b.is_fragment
                        ? b.fragment_wave_ballot_half(
                              rs.sreg_bool.at(mask_base),
                              static_cast<uint32_t>(o.value - mask_base))
                        : b.native_wave_ballot_half(
                              rs.sreg_bool.at(mask_base),
                              static_cast<uint32_t>(o.value - mask_base));
                    if (half) return half;
                }
                if (ok) *ok = false;
            }
            return b.uconst(0);
        }
        case OperandKind::InlineInt:   return b.uconst((uint32_t)o.value);
        case OperandKind::InlineFloat: return b.uconst(fbits(inline_float_value((uint32_t)o.value)));
        case OperandKind::Literal:     return b.uconst(in.literal);
        case OperandKind::Special: {
            if (o.value == 125) return b.uconst(0);   // SGPR_NULL: the one Special whose data value IS 0
            if (o.value == 253) {                     // SCC scalar source
                // rs.scc == 0 marks SCC poisoned by a 64-bit mask op (its SCC is a cross-lane
                // reduction this model cannot form) — reject rather than read a stale condition.
                if (!rs.scc) { if (ok) *ok = false; return b.uconst(0); }
                return b.sel(rs.scc, b.uconst(1), b.uconst(0));
            }
            // VCC_LO/HI (106/107), ttmp0..15 (108..123) and M0 (124) double as plain scalar SCRATCH
            // in compiled code — the NGG preamble does `s_bfe_u32 vcc_lo, s3, ...` then reads vcc
            // back as data, and DOLL's skinned VS round-trips M0 (`s_mov m0, s4 … s_mov s36, m0`,
            // #273). A scalar write lands in rs.sreg[o.value] (the DST field decodes as SGPR); read
            // it back from there. Only an UNTRACKED read (a VOPC-produced mask, EXEC, SCC, or a
            // never-written scratch) has no representable per-invocation data value.
            if (o.value >= 106 && o.value <= 124) {
                auto it = rs.sreg.find(o.value);
                if (it != rs.sreg.end()) return it->second;
            }
            // One Vulkan vertex invocation models one live lane of a virtual guest wave (lane 0),
            // which is also the contract used by vertex MBCNT and BFE-to-mask lowering. A compiler
            // generated NGG prolog may temporarily consume VCC_LO as scalar DATA after a VOPC; in
            // this shell its complete representable dword is therefore exactly {bit0=vcc}. VCC_HI
            // is zero. Compute/fragment keep their real multi-lane masks and must not take this path.
            if (!b.is_compute && !b.is_fragment && (o.value == 106 || o.value == 107))
                return o.value == 106 ? b.sel(rs.vcc, b.uconst(1), b.uconst(0)) : b.uconst(0);
            // VCC/EXEC read as scalar DATA, materialised exactly (#2420/#2481). GTA V computes a
            // descriptor-table pointer from VCC_LO in graphics shaders, and a Wave64 compute kernel
            // negates EXEC as an integer pair (`s_sub_u32 ...,exec_lo`; `s_subb_u32 ...,exec_hi`). A
            // per-invocation bool has no complete 32-bit value for either sequence.
            //
            // `subgroupBallot` supplies those bits EXACTLY, and only under one condition: its .x/.y
            // are lanes 0..31 / 32..63 of THIS SUBGROUP, so it equals the guest wave mask only when
            // the subgroup IS the guest wave. Narrower and it would report half a mask as though it
            // were whole -- a wrong descriptor address, i.e. silent wrong rendering. So the exact-size
            // contract is required, and anything else keeps rejecting exactly as before.
            //
            // This is the final path before the operand is rejected, so it can only turn a previous
            // reject into working code; no shader that already compiled changes.
            const bool reads_vcc = o.value == 106 || o.value == 107;
            const bool reads_exec = o.value == 126 || o.value == 127;
            const uint32_t mask = reads_vcc ? rs.vcc : reads_exec ? rs.exec : 0;
            if (mask && b.native_subgroup_size && b.native_subgroup_size == b.wave_size) {
                const uint32_t half =
                    b.native_wave_ballot_half(mask,
                        (o.value == 106 || o.value == 126) ? 0u : 1u);
                if (half) return half;
            }
            if (ok) *ok = false;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[scalar-data-reject] pc=%u special=s%d tracked=%d mask=%d\n",
                             in.pc, o.value, rs.sreg.contains(o.value),
                             rs.sreg_bool.contains(o.value));
            return b.uconst(0);
        }
        default:
            if (ok) *ok = false;
            return b.uconst(0);
    }
}

// DS_APPEND/DS_CONSUME use M0 as two different architectural layouts. For LDS, M0[15:0]
// is the byte base. For GDS, M0[31:16] is the 16-bit byte base and M0[15:0] is the
// allocation size. Astro Bot supplies M0=0x0c600020 and resets the resulting counters at
// 0xc70..0xc7c; treating the low size field as the base instead placed them at 0x30..0x3c,
// where they accumulated without bound (#1742). The final 16-bit mask is part of both address
// domains and keeps base+offset arithmetic wrapping at the 64 KiB share boundary.
inline uint32_t ds_append_consume_index(SpirvCompute& b, uint32_t m0, uint32_t literal, bool gds) {
    const uint32_t base = gds
        ? b.ibin(Op_ShiftRightLogical, m0, b.uconst(16))
        : b.ibin(Op_BitwiseAnd, m0, b.uconst(0xffffu));
    const uint32_t byte_addr = b.ibin(
        Op_BitwiseAnd, b.ibin(Op_IAdd, base, b.uconst(literal)), b.uconst(0xffffu));
    return b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
}

// Exact in-place integer-add DPP row reduction emitted by GTA V compute shaders. Keeping the
// register/control contract shared between the ordinary emitter and CFG dispatcher prevents one
// path from silently widening beyond the reviewed full-mask, BC0 VDST/SRC0/SRC1 shape.
inline bool is_inplace_vadd_nc_u32_dpp_row_shr(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == 0x25 && in.has_dpp &&
        !in.dpp_bound_ctrl && in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
        in.dpp_row_mask == 0xfu && in.dpp_bank_mask == 0xfu &&
        in.dst.kind == OperandKind::VGPR && in.src[0].kind == OperandKind::VGPR &&
        in.src[1].kind == OperandKind::VGPR && in.dst.value == in.src[0].value &&
        in.dst.value == in.src[1].value;
}

enum class DppRowRor8Op : uint32_t {
    None = 0,
    MovB32 = 1,
    MinF32 = 2,
    MaxF32 = 3,
};

// The XOR stride a DPP control applies to the lane id, or 0 if the control is not in this family.
//
// ROW_ROR:8 (0x128) and ROW_XMASK:n (0x160..0x16f) are one family: ROW_XMASK:n is XOR n by
// definition, and XOR 8 is exactly (row_lane - 8) modulo 16, so ROW_ROR:8 IS ROW_XMASK:8. The
// emitter already lowered the 8 case through an XOR; this only lets the other strides reach it.
//
// XMASK:0 is included: its stride is the identity, which is a legal permutation. It was once excluded
// because a no-op is hard to tell from a decode error BY ITS RESULT -- a statement about diagnosis,
// not about whether the guest may issue the instruction.
// Membership and stride are SEPARATE answers. Folding them into one return made 0 mean both "stride
// zero" and "not in this family", which excluded ROW_XMASK:0 -- a legal member whose stride happens to
// be the identity. That is a representation artifact, not a semantic reason to refuse a guest
// instruction, so the caller asks the two questions independently.
inline bool dpp_row_xor_ctrl(uint32_t dpp_ctrl, uint32_t* stride = nullptr) {
    uint32_t s = 0;
    if (dpp_ctrl == 0x128u) s = 8u;
    else if (dpp_ctrl >= 0x160u && dpp_ctrl <= 0x16fu) s = dpp_ctrl - 0x160u;
    else return false;
    if (stride) *stride = s;
    return true;
}

// Exact bounded row-rotate family emitted by GTA V's screen-space compute passes. The decoder has
// already proved FI=0/no source modifiers; repeat every retained control field here so the ordinary
// emitter and CFG dispatcher share one fail-closed contract. VOP1 MOV has only the permuted SRC0;
// the two VOP2 float operations combine that value with the destination lane's unpermuted SRC1.
inline DppRowRor8Op dpp_row_ror8_op(const Rdna2Inst& in) {
    if (!in.has_dpp || !in.dpp_bound_ctrl || !dpp_row_xor_ctrl(in.dpp_ctrl) ||
        in.dpp_row_mask != 0xfu || in.dpp_bank_mask != 0xfu ||
        in.dst.kind != OperandKind::VGPR || in.src[0].kind != OperandKind::VGPR)
        return DppRowRor8Op::None;
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x01)
        return DppRowRor8Op::MovB32;
    if (in.fmt != Rdna2Format::VOP2 || in.src[1].kind != OperandKind::VGPR)
        return DppRowRor8Op::None;
    if (in.opcode == 0x0f) return DppRowRor8Op::MinF32;
    if (in.opcode == 0x10) return DppRowRor8Op::MaxF32;
    return DppRowRor8Op::None;
}

// Exact identity-QUAD_PERM tail of the same reduction. No value crosses lanes; ROW_MASK selects
// architectural DPP16 rows 1 and 3, while the other rows preserve VDST.
inline bool is_vadd_nc_u32_dpp_partial_row(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == 0x25 && in.has_dpp &&
        !in.dpp_bound_ctrl && in.dpp_ctrl == 0xe4u && in.dpp_row_mask == 0xau &&
        in.dpp_bank_mask == 0xfu && in.dst.kind == OperandKind::VGPR &&
        in.src[0].kind == OperandKind::VGPR && in.src[1].kind == OperandKind::VGPR &&
        in.dst.value == in.src[0].value && in.dst.value != in.src[1].value;
}

}  // namespace prosper::gpu
