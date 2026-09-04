// rdna2_emit_alu.cpp — emit_alu: the RDNA2 instruction-family translator, split out of
// rdna2_to_spirv.cpp. Shared state lives in gpu/recompiler/rdna2_to_spirv_internal.hpp.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/texture/bc_decode.hpp"   // guest_texture_is_uploaded_array (#325)
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
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
#include "gpu/recompiler/rdna2_alu_support.hpp"

namespace prosper::gpu {


namespace {


// The f16 bit pattern an inline float constant supplies in a 16-bit operand position (ISA Table 10
// lists per-width encodings: "0.5 ... half: 0x3800" etc.). Only 1/(2*pi) (code 248, 0x3118) differs
// from rounding the f32 value — the f32 table entry 0.15915494 would round to a different last bit
// than the documented operand, so 16-bit consumers must use these bits, not the f32 constant.
uint32_t inline_float_f16_bits(int code) {
    switch (code) {
        case 240: return 0x3800u; case 241: return 0xB800u;   // +/-0.5
        case 242: return 0x3C00u; case 243: return 0xBC00u;   // +/-1.0
        case 244: return 0x4000u; case 245: return 0xC000u;   // +/-2.0
        case 246: return 0x4400u; case 247: return 0xC400u;   // +/-4.0
        case 248: return 0x3118u;                             // 1/(2*pi)
        default:  return 0;
    }
}

// The 32-bit dword an inline constant places on the source bus when the consuming operand is
// 16 bits wide. Hardware materializes the constant at the OPERAND's width and leaves the rest of
// the dword ZERO, so the packed operand is simply that dword: an inline FLOAT contributes its
// documented f16 encoding in [15:0] and 0 in [31:16] — it is NOT duplicated into the high half —
// while an inline INT is its sign-extended two's-complement dword, which is why `-1` legitimately
// reads 0xffff in BOTH halves but `1` reads 0 in the high one. A half select (VOP3P OPSEL /
// OPSEL_HI, VOP3 OPSEL) therefore applies to an inline constant exactly as it does to a register:
// it is NEVER a no-op. Returns false when `o` is not an inline constant.
//
// VERIFIED (llvm-mc gfx1030 literal folding, both directions — #2119):
//   `v_pk_add_f16 v0, 0x00003c00, v1` canonicalizes to `v_pk_add_f16 v0, 1.0, v1`, while
//   `0x3c003c00` stays a 32-bit literal. The fold is value-preserving, so inline `1.0` IS
//   0x00003c00 in a packed f16 operand. The integer side rules out "the assembler simply never
//   folds packed literals": 0x00000001 folds to `1` and 0xffffffff to `-1`, but 0x00010001 does
//   not fold. Identical on gfx1010/1030/1100/1200 — there is NO gfx10-vs-gfx11 split here, and
//   #2119 was filed supposing one. The ISA is consistent with this and states no replication
//   rule: VOP3P pseudocode simply reads S0[31:16] for the high result, and the inline-constant
//   table gives a half-precision constant as a 16-bit pattern ("half: 0x3118"). CONFIDENCE: HIGH.
//
// Three f16 paths used to hardcode the LOW half here and call the select a no-op; kernels
// 32r14a/b/c in test_rdna2_to_spirv pin the high-half value each one now reads.
bool inline_16bit_operand_dword(const Operand& o, uint32_t& out) {
    if (o.kind == OperandKind::InlineFloat) {
        out = inline_float_f16_bits(o.value);
        return true;
    }
    if (o.kind == OperandKind::InlineInt) {
        out = static_cast<uint32_t>(o.value);
        return true;
    }
    return false;
}

}  // namespace

// Predicate a just-computed VGPR write against EXEC: under a narrowed mask, inactive lanes keep their
// prior value. A no-op when EXEC is full (the straight-line common case), so nothing is perturbed.
inline void predicate_write(SpirvCompute& b, RegState& rs, int idx, uint32_t old_val) {
    if (rs.exec_narrowed) rs.vreg[idx] = b.sel(rs.exec, rs.vreg[idx], old_val);
    // A VGPR can be recycled after serving as a v_writelane scalar-spill array. Any ordinary
    // per-lane write starts a new register lifetime, so later ALU/EXP reads must see that value
    // rather than rejecting it as a stale cross-lane spill. Blasphemous 2 does exactly this after
    // an image_sample overwrites the shader's early scalar-spill v11 (#652).
    if (rs.vgpr_lane_slots.count(idx) || rs.vgpr_lane_mask_slots.count(idx))
        rs.invalidated_vgpr_lane_slots.insert(idx);
    rs.vgpr_lane_slots.erase(idx);
    rs.vgpr_lane_mask_slots.erase(idx);
}
inline uint32_t vreg_old(SpirvCompute& b, RegState& rs, int idx) {
    auto it = rs.vreg.find(idx); return it == rs.vreg.end() ? b.uconst(0) : it->second;
}

inline bool sreg_range_written(const RegState& rs, int base, uint32_t words) {
    for (uint32_t word = 0; word < words; ++word)
        if (rs.sreg_written.count(base + static_cast<int>(word))) return true;
    return false;
}

// An SRT tag describes the complete hardware descriptor, not merely its base SGPR. Accept it only
// while every word still carries the same provenance; any partial overwrite makes the descriptor
// unrepresentable even when the base word itself was untouched.
inline bool sreg_srt_range_tag(const RegState& rs, int base, uint32_t words, uint32_t& tag) {
    auto first = rs.sreg_srt.find(base);
    if (first == rs.sreg_srt.end()) return false;
    tag = first->second;
    for (uint32_t word = 1; word < words; ++word) {
        auto it = rs.sreg_srt.find(base + static_cast<int>(word));
        if (it == rs.sreg_srt.end() || it->second != tag) return false;
    }
    return true;
}

// A DIRECT descriptor staged into `base` by copying it word-for-word out of entry-time user data
// (#1773). Succeeds only for a FAITHFUL WHOLE-DESCRIPTOR move: every word `base + i` must alias
// exactly `origin + i` for one origin. A partial, permuted or partly-recomputed copy is a descriptor
// the shader assembled itself, which this must not resolve -- binding the wrong resource renders
// silently wrong texels, which is strictly worse than declining the draw.
//
// The load-bearing condition is applied where the alias is ESTABLISHED, not here: the source must
// still have been entry-time user data at the moment of the copy (`record_scalar_write`). It is
// deliberately NOT re-checked at consumption, because a copy captures bits -- a later write to the
// SOURCE register cannot change what the DESTINATION already holds. Re-checking it would reject the
// exact shape #1773 documents: Earthion stages s[9:16] into s[20:27] and then immediately reuses
// s[12:19] for the second descriptor, so five of the origin words are overwritten before the sample.
//
// Two invariants carry that instead, and both live elsewhere: a write to the DESTINATION expires
// the alias (`record_scalar_write`), and every control-flow join meets the two edges' claims
// (`merge_ud_alias`). Neither is visible here, which is why they are named here.
inline bool sreg_range_ud_alias(const RegState& rs, int base, uint32_t words, int& origin) {
    auto first = rs.sreg_ud_alias.find(base);
    if (first == rs.sreg_ud_alias.end()) return false;
    origin = first->second;
    for (uint32_t word = 0; word < words; ++word) {
        auto it = rs.sreg_ud_alias.find(base + static_cast<int>(word));
        if (it == rs.sreg_ud_alias.end() ||
            it->second != origin + static_cast<int>(word)) return false;
    }
    return true;
}

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

namespace {
}

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
}

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

} // namespace

// A 64-bit mask write architecturally overwrites the destination SGPR pair ("D = ..." in every
// ISA mask-op description): any tracked data-domain SSA value or descriptor-provenance tag for
// those registers is stale afterwards. Erase both so a later data read / SRSRC resolution cannot
// alias the pre-mask-op value (the mask itself lives in sreg_bool / rs.vcc / rs.exec).
inline void mask_write_clobbers_pair(RegState& rs, int dst) {
    rs.sreg.erase(dst); rs.sreg.erase(dst + 1);
    rs.sreg_srt.erase(dst); rs.sreg_srt.erase(dst + 1);
}

// The VOP1 f16 unary family's operation, given its already-unpacked f16 operand as an f32 value.
// Both the plain e32 form and the SDWA word-select form lower through this one function, so the two
// encodings of one opcode cannot drift apart — the SDWA path previously carried its own three-op
// ternary, which is exactly how a family gains a member in one form and not the other.
// The domain is `vop1_is_f16_unary` (rdna2_decode.hpp), shared with the decoder's SDWA admission and
// asserted below rather than assumed: a caller that widens its gate without widening this switch
// would otherwise get `v_cos_f16` silently. Opcodes verified there by llvm-mc round-trip; trig input
// is in REVOLUTIONS, exactly as for the f32 forms. Returns 0 (never a valid SPIR-V id) for anything
// outside the family, which the callers treat as an unsupported instruction. CONFIDENCE: HIGH.
uint32_t emit_f16_unary(SpirvCompute& b, uint32_t opcode, uint32_t x) {
    const auto turns = [&] { return b.uconst(fbits(6.28318530717958647692f)); };
    switch (opcode) {
        case 0x54: return b.frcp(x);                        // v_rcp_f16
        case 0x55: return b.fext1(Glsl_Sqrt, x);            // v_sqrt_f16
        case 0x56: return b.fext1(Glsl_InverseSqrt, x);     // v_rsq_f16
        case 0x57: return b.fext1(Glsl_Log2, x);            // v_log_f16
        case 0x58: return b.fext1(Glsl_Exp2, x);            // v_exp_f16
        case 0x5B: return b.fext1(Glsl_Floor, x);           // v_floor_f16
        case 0x5C: return b.fext1(Glsl_Ceil, x);            // v_ceil_f16
        case 0x5D: return b.fext1(Glsl_Trunc, x);           // v_trunc_f16
        case 0x5E: return b.fext1(Glsl_RoundEven, x);       // v_rndne_f16
        case 0x5F: return b.fext1(Glsl_Fract, x);           // v_fract_f16
        case 0x60: return b.fext1(Glsl_Sin, b.fbin(Op_FMul, x, turns()));  // v_sin_f16
        case 0x61: return b.fext1(Glsl_Cos, b.fbin(Op_FMul, x, turns()));  // v_cos_f16
        default:   return 0;                                // outside the family — fail visibly
    }
}

// Emit one ALU instruction (VOP1/2/C/3 or SOP1/2) into `b`, updating `rs`. Returns true if `in` is an
// ALU format handled here; sets ok=false if it is an ALU op this stage doesn't support yet. Non-ALU
// formats (EXP/memory/...) return false so the stage-specific caller can handle them.
static bool instruction_reads_scc_as_scalar_data(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOP2:
            return in.opcode == kSop2OpcodeAddcU32 || in.opcode == 0x05u ||
                   in.opcode == kSop2OpcodeCselectB32 ||
                   in.opcode == kSop2OpcodeCselectB64;
        case Rdna2Format::SOP1:
            return in.opcode == kSop1OpcodeCmovB32 ||
                   in.opcode == kSop1OpcodeCmovB64;
        case Rdna2Format::SOPK:
            return in.opcode == kSopkOpcodeCmovkI32;
        default:
            return false;
    }
}

bool emit_alu(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_exec_update,
              const std::unordered_set<uint32_t>* safe_execz, bool allow_smem,
              const ShaderResourceTable* rt, bool allow_wave) {
    if (b.is_fragment && instruction_reads_scc_as_scalar_data(in))
        b.mark_fragment_wave_scalar_use(rs.scc);
    auto& vreg = rs.vreg; uint32_t& vcc = rs.vcc;
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    auto write_vop3b_carry_output = [&](uint32_t carry_mask) {
        if (in.sdst.kind == OperandKind::SGPR && b.allow_b32_masks &&
            (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
            // A Wave32 carry destination is one physical word. Preserve an independently-live
            // adjacent SGPR and register the new lifetime in the B32 mask domain used by CFG joins.
            rs.sreg_bool[in.sdst.value] = carry_mask;
            rs.sreg_bool_narrowed[in.sdst.value] = true;
            rs.sreg_bool_b32.insert(in.sdst.value);
            rs.sreg.erase(in.sdst.value);
            rs.sreg_srt.erase(in.sdst.value);
            if (in.sdst.value == 106) rs.vcc = carry_mask;
        } else if (in.sdst.value == 106 || in.sdst.value == 107) {
            rs.vcc = carry_mask;
        } else if (in.sdst.kind == OperandKind::SGPR) {
            rs.sreg_bool[in.sdst.value] = carry_mask;
        } else {
            ok = false;
        }
    };
    // SDWA/DPP forms carry a sub-dword select or cross-lane control word we don't model. The decoder
    // flags them (and gets their length right); reject here rather than compute with a wrong operand.
    if (in.has_modifier) { ok = false; return true; }
    switch (in.fmt) {
        case Rdna2Format::SOP1: {
            // s_quadmask_b64 D, S (#2412): D[i] = S[4i]|S[4i+1]|S[4i+2]|S[4i+3] for i in 0..15, upper
            // bits zero. Each output bit ORs a QUAD of lanes, so a per-invocation bool cannot form it.
            // `s_quadmask_b64 vcc, vcc` (0xbeea2d6a, confirmed with llvm-mc) is the first reject in 20
            // of GTA V's 59 failing shaders.
            //
            // A ballot gives the source mask exactly; the quadmask is then integer arithmetic over it.
            // BOTH representations prosper keeps for a mask register are produced -- the scalar DATA
            // value in sreg and the per-lane bool -- so it cannot matter which a later instruction
            // consumes; writing one and leaving the other stale would be silent wrong rendering.
            //
            // Wave64 only, and each stage supplies its exact-wave contract its own way: compute carries
            // native_subgroup_size, the fragment stage DECLARES fragment_required_subgroup_size. An
            // earlier revision gated on native_subgroup_size alone and therefore never fired for the
            // graphics shaders that need it -- dead code that ctest could not see, caught only by
            // measuring the failing-shader count. Vertex has neither mechanism and still rejects.
            if (in.opcode == 0x2d && b.wave_size == 64 &&
                (b.is_fragment || (b.native_subgroup_size == b.wave_size))) {
                uint32_t src_mask = 0;
                if (in.src[0].value == 106 || in.src[0].value == 107) src_mask = rs.vcc;
                else if (in.src[0].value == 126 || in.src[0].value == 127) src_mask = rs.exec;
                else if (in.src[0].kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(in.src[0].value);
                    if (it != rs.sreg_bool.end()) src_mask = it->second;
                }
                const uint32_t lo = src_mask ? (b.is_fragment
                        ? b.fragment_wave_ballot_half(src_mask, 0)
                        : b.native_wave_ballot_half(src_mask, 0)) : 0;
                const uint32_t hi = src_mask ? (b.is_fragment
                        ? b.fragment_wave_ballot_half(src_mask, 1)
                        : b.native_wave_ballot_half(src_mask, 1)) : 0;
                if (lo && hi) {
                    uint32_t value = b.uconst(0);
                    for (uint32_t i = 0; i < 16; ++i) {
                        const uint32_t word = i < 8 ? lo : hi;
                        const uint32_t shift = (i < 8 ? i : i - 8) * 4u;
                        const uint32_t nibble = b.ibin(
                            Op_BitwiseAnd,
                            b.ibin(Op_ShiftRightLogical, word, b.uconst(shift)), b.uconst(0xFu));
                        value = b.ibin(Op_BitwiseOr, value,
                                       b.sel(b.ucmp(Op_INotEqual, nibble, b.uconst(0)),
                                             b.uconst(1u << i), b.uconst(0)));
                    }
                    const int dst = in.dst.value;
                    rs.sreg[dst] = value;
                    rs.sreg[dst + 1] = b.uconst(0);
                    rs.sreg_srt.erase(dst); rs.sreg_srt.erase(dst + 1);
                    const uint32_t lane = b.ibin(Op_BitwiseAnd, b.subgroup_local_id(),
                                                 b.uconst(b.wave_size - 1));
                    const uint32_t my_bit = b.ibin(
                        Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, value, lane), b.uconst(1u));
                    const uint32_t as_bool = b.land(
                        b.ucmp(Op_ULessThan, lane, b.uconst(16u)),
                        b.ucmp(Op_INotEqual, my_bit, b.uconst(0)));
                    if (dst == 126 || dst == 127) { rs.exec = as_bool; rs.exec_narrowed = true; }
                    else if (dst == 106 || dst == 107) rs.vcc = as_bool;
                    else rs.sreg_bool[dst] = as_bool;
                    rs.scc = b.ucmp(Op_INotEqual, value, b.uconst(0));
                    return true;
                }
            }
            if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 &&
                in.opcode == kSop1OpcodeFf1I32B64) { // s_ff1_i32_b64
                // RDNA2 scans the complete 64-bit source from its least-significant bit and
                // returns the first set index, or 0xffffffff for an empty mask.  The captured GTA
                // compute sites feed EXEC, VCC, or a VOPC-saved mask pair into this scalar result.
                // A required native Wave64 subgroup makes native_wave_first_active architectural;
                // a 32-wide or unknown subgroup would silently lose bits 32..63.
                const auto data_word_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                const int source = in.src[0].value;
                uint32_t mask = 0;
                // EXEC is architectural mask state, even when structured-loop or dispatcher state
                // bookkeeping also leaves synthetic scalar placeholders for physical s126/s127.
                // Unlike VCC and saved SGPR pairs, those words cannot carry competing scalar data.
                if (in.src[0].kind == OperandKind::Special && source == 126) {
                    mask = rs.exec;
                } else {
                    const bool proven_saved_mask =
                        b.structured_wave64_mask_reduction_pcs.contains(in.pc);
                    const bool competing_data = !proven_saved_mask &&
                        (data_word_present(source) || data_word_present(source + 1));
                    if (!competing_data && source == 106) {
                        mask = rs.vcc;
                    } else if (!competing_data && in.src[0].kind == OperandKind::SGPR &&
                               !rs.sreg_bool_b32.contains(source)) {
                        const auto saved = rs.sreg_bool.find(source);
                        if (saved != rs.sreg_bool.end()) mask = saved->second;
                    }
                }
                if (!mask || in.dst.value == 126 || in.dst.value == 127) {
                    ok = false;
                    return true;
                }

                const uint32_t result = b.native_wave_first_active(mask);
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);

                // S_FF1 writes ordinary 32-bit scalar DATA.  End every complete-mask lifetime
                // overlapping that physical dword, including a pair rooted one register earlier;
                // otherwise a later implicit predicate consumer could observe the pre-S_FF1 mask.
                const auto erase_mask_alias = [&](int base) {
                    rs.sreg_bool.erase(base);
                    rs.sreg_bool_narrowed.erase(base);
                    rs.sreg_bool_b32.erase(base);
                };
                erase_mask_alias(in.dst.value);
                if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
                if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
                // S_FF1 does not modify SCC.
                return true;
            }
            if (b.is_compute && b.wave_size == 32 && b.native_subgroup_size == 32 &&
                in.opcode == 0x13) { // s_ff1_i32_b32
                // RDNA2 returns the first set bit from the low end, or 0xffffffff for an empty
                // source. In Wave32, VCC_LO/EXEC_LO and one-word saved masks are complete scalar
                // masks. The live Astro traversal kernel uses this to select the first hit lane.
                // A required 32-wide Vulkan subgroup makes the reduction architectural rather than
                // an implementation-width approximation.
                const auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                uint32_t mask = 0;
                if (in.src[0].value == 126 && !data_value_present(126)) {
                    mask = rs.exec;
                } else if (in.src[0].value == 106 && !data_value_present(106)) {
                    mask = rs.vcc;
                } else if ((in.src[0].kind == OperandKind::SGPR ||
                            in.src[0].kind == OperandKind::Special) &&
                           rs.sreg_bool_b32.contains(in.src[0].value) &&
                           !data_value_present(in.src[0].value)) {
                    auto saved = rs.sreg_bool.find(in.src[0].value);
                    if (saved != rs.sreg_bool.end()) mask = saved->second;
                }
                if (!mask || in.dst.value == 126 || in.dst.value == 127) {
                    ok = false;
                    return true;
                }

                const uint32_t result = b.native_wave_first_active(mask);
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value);
                if (in.dst.value == 106) {
                    // VCC_LO is also a physical scalar dword. Preserve its architectural mask view
                    // for any implicit VCC consumer while publishing the same bits as scalar data.
                    const uint32_t lane = b.subgroup_local_id();
                    const uint32_t bit = b.ibin(
                        Op_BitwiseAnd,
                        b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
                    rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                }
                // S_FF1 does not modify SCC.
                return true;
            }
            if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
                in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
                (in.src[0].value == 106 || in.src[0].value == 107) &&
                (in.dst.value - 126) == (in.src[0].value - 106)) {
                // Wave64 can update one EXEC dword without touching the other. Preserve that
                // distinction per invocation for the common VCC_LO->EXEC_LO / VCC_HI->EXEC_HI
                // copy. A cross-half copy would need another guest lane's predicate and remains
                // unsupported; the matching-half form maps directly to this lane's VCC bit.
                const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                const uint32_t in_half = in.dst.value == 126
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                rs.exec = b.bsel(in_half, rs.vcc, rs.exec);
                rs.exec_narrowed = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
                in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
                in.src[0].kind == OperandKind::InlineInt) {
                // The companion immediate form restores or clears one EXEC dword. Select the
                // addressed 32-lane half and preserve the other half exactly; this covers the
                // compiler's `s_mov_b32 exec_lo, -1` reconvergence after a low-half mask.
                const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                const uint32_t in_half = in.dst.value == 126
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                const uint32_t value = in.dst.value == 126
                    ? inline_int_mask_bit(b, in.src[0].value)
                    : inline_int_mask_bit_hi(b, in.src[0].value);
                rs.exec = b.bsel(in_half, value, rs.exec);
                rs.exec_narrowed = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            // Wave32 shaders use the low 32-bit halves of EXEC/VCC for the same save/copy/restore
            // idioms that wave64 shaders express with s_mov_b64.  A wave mask is one bool in this
            // per-invocation model, so preserve that bool domain when either source is an
            // unambiguous low-half mask or EXEC_LO is the destination.  VCC_LO remains available as
            // ordinary scalar scratch; when that scalar dword is copied back to EXEC_LO, select this
            // invocation's bit from the complete Wave32 value.  High-half moves remain unsupported:
            // without the guest wave mode they may name another lane rather than this invocation's bit.
            if (b.allow_b32_masks && in.opcode == 0x03) {   // s_mov_b32 (mask-domain form)
                const auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                const bool src_exec_lo = in.src[0].value == 126;
                const bool src_vcc_lo = in.src[0].value == 106 &&
                    !data_value_present(106);
                const auto saved = (in.src[0].kind == OperandKind::SGPR ||
                                    in.src[0].kind == OperandKind::Special)
                    ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
                const bool src_saved_mask = saved != rs.sreg_bool.end() &&
                    rs.sreg_bool_b32.contains(in.src[0].value) &&
                    !data_value_present(in.src[0].value);
                const bool dst_exec_lo = in.dst.value == 126;
                uint32_t scalar_mask_data = 0;
                bool src_scalar_data = false;
                if (dst_exec_lo &&
                    (in.src[0].kind == OperandKind::SGPR ||
                     (in.src[0].kind == OperandKind::Special &&
                      in.src[0].value >= 106 && in.src[0].value <= 124))) {
                    auto current = rs.sreg.find(in.src[0].value);
                    if (current != rs.sreg.end()) {
                        scalar_mask_data = current->second;
                        src_scalar_data = true;
                    } else {
                        auto input = rs.sreg_input.find(in.src[0].value);
                        if (input != rs.sreg_input.end()) {
                            scalar_mask_data = input->second;
                            src_scalar_data = true;
                        }
                    }
                }
                const bool mask_move = src_exec_lo || src_vcc_lo || src_saved_mask ||
                    src_scalar_data ||
                    (dst_exec_lo && in.src[0].kind == OperandKind::InlineInt);
                if (mask_move) {
                    uint32_t mask = 0;
                    bool narrowed = true;
                    if (src_exec_lo) {
                        mask = rs.exec;
                        narrowed = rs.exec_narrowed;
                    } else if (src_vcc_lo) {
                        mask = rs.vcc;
                        auto state = rs.sreg_bool_narrowed.find(106);
                        narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
                    } else if (src_saved_mask) {
                        mask = saved->second;
                        auto state = rs.sreg_bool_narrowed.find(in.src[0].value);
                        narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
                    } else if (src_scalar_data) {
                        const uint32_t lane = b.ibin(
                            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                        const uint32_t bit = b.ibin(
                            Op_BitwiseAnd,
                            b.ibin(Op_ShiftRightLogical, scalar_mask_data, lane),
                            b.uconst(1));
                        mask = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                    } else if (in.src[0].kind == OperandKind::InlineInt) {
                        // Every inline B32 dword is a valid Wave32 EXEC mask, not only 0/-1.
                        // Select this invocation's architectural bit exactly; GTA V uses 1 here
                        // to isolate lane zero before a scalar load.
                        mask = inline_int_mask_bit(b, in.src[0].value);
                        narrowed = in.src[0].value != -1;
                    }
                    if (!mask || in.dst.value == 127) {
                        ok = false;
                        return true;
                    }

                    if (dst_exec_lo) {
                        rs.exec = mask;
                        rs.exec_narrowed = narrowed;
                    } else if (in.dst.value == 106) {
                        rs.vcc = mask;
                        rs.sreg_bool[106] = mask;
                        rs.sreg_bool_narrowed[106] = narrowed;
                        rs.sreg_bool_b32.insert(106);
                    } else {
                        rs.sreg_bool[in.dst.value] = mask;
                        rs.sreg_bool_narrowed[in.dst.value] = narrowed;
                        rs.sreg_bool_b32.insert(in.dst.value);
                    }
                    // A B32 move overwrites only the addressed physical word.  Remove stale scalar
                    // data/descriptor provenance for that word while leaving its neighbor intact.
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    return true;
                }
            }
            // `s_mov_b32 exec_lo, -1` while EXEC is still the full unnarrowed wave mask, at ANY
            // wave width and in any stage.
            //
            // Everything above needs to know the wave width, because "write all-ones to the low
            // EXEC dword" means the whole mask in Wave32 and only half of it in Wave64 -- and the
            // GRAPHICS wave size is not plumbed into the recompiler, so the vertex shell fails
            // closed (rdna2_to_spirv.cpp: `b.allow_b32_masks = b.ngg_one_lane`, an exception for
            // one byte-exact captured wrapper rather than a property of any stage).
            //
            // This case does not need it. `exec_narrowed == false` is prosper's invariant for "every
            // lane is on" (the field's own definition, and what the s_wqm_b32 path below and the
            // fragment export path in rdna2_to_spirv.cpp both already rely on). Setting the low
            // dword to all-ones from that state leaves every lane on under EITHER width: in Wave32
            // it rewrites the whole mask with what it already held, and in Wave64 it rewrites the
            // low half with what it already held and does not touch the high half. So the exact
            // lowering is a no-op, derived rather than approximated -- no lane mapping, no wave
            // width, no peer lane.
            //
            // It is the compiler's standard program prologue, which is why it gates whole titles
            // rather than odd shaders: Yakuza Kiwami (PPSA31334) emits `bfa00003` (s_branch) then
            // `befe03c1` (this instruction) at pc=3 in 19 of the 20 distinct programs a boot
            // reaches, and every draw was skipped for it.
            //
            // Deliberately NOT extended to the narrowed case. Restoring all-lanes from a NARROWED
            // EXEC is the Wave32 reconvergence idiom, and reading it that way in a Wave64 program
            // would silently re-enable the wrong lanes -- that one does need the wave size, and
            // stays fail-closed. CONFIDENCE: HIGH.
            if (in.opcode == kSop1OpcodeMovB32 && in.dst.value == 126 && !rs.exec_narrowed &&
                in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                rs.exec = b.btrue();
                rs.exec_narrowed = false;
                rs.sreg.erase(126);
                rs.sreg_srt.erase(126);
                return true;
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                sop1_opcode_is_emitted_saveexec_b32(in.opcode)) {
                // s_and_saveexec_b32 / s_orn2_saveexec_b32 / s_andn1_saveexec_b32
                // Save the previous EXEC_LO into one physical SGPR, then narrow EXEC_LO by the
                // one-word source mask. Astro uses VCC_HI as the saved-mask destination and restores
                // it later with s_mov_b32. The ANDN1 form selects the complementary old-EXEC arm.
                // Both halves are independent complete values in Wave32.
                auto source_mask = [&]() -> uint32_t {
                    if (in.src[0].value == 126) return rs.exec;
                    if ((in.src[0].kind == OperandKind::SGPR ||
                         in.src[0].kind == OperandKind::Special) &&
                        rs.sreg_bool_b32.contains(in.src[0].value)) {
                        auto it = rs.sreg_bool.find(in.src[0].value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (in.src[0].value == 106) return rs.vcc;
                    }
                    if (in.src[0].kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, in.src[0].value);
                    return 0;
                };
                const uint32_t mask = source_mask();
                if (!mask || in.dst.value == 126 || in.dst.value == 127) {
                    ok = false; return true;
                }
                const uint32_t old_exec = rs.exec;
                rs.sreg_bool[in.dst.value] = old_exec;
                rs.sreg_bool_narrowed[in.dst.value] = rs.exec_narrowed;
                rs.sreg_bool_b32.insert(in.dst.value);
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                if (in.dst.value == 106) rs.vcc = old_exec;
                rs.exec = in.opcode == kSop1OpcodeAndSaveexecB32 ? b.land(old_exec, mask)
                        : in.opcode == kSop1OpcodeOrn2SaveexecB32
                            ? b.lor(mask, b.logical_not(old_exec))
                                            : b.land(old_exec, b.logical_not(mask));
                rs.exec_narrowed = true;
                rs.scc = b.is_fragment ? b.fragment_wave_any(rs.exec) : 0;
                return true;
            }
            if (b.allow_b32_masks && in.opcode == 0x09) {   // s_wqm_b32
                // Wave32 uses the low half of EXEC for the same whole-quad-mode idiom as
                // s_wqm_b64.  In the per-invocation fragment model helper lanes are implicit, so
                // widening is an identity on the tracked bool.  Keep this in the mask domain and
                // reject either high half; treating EXEC_LO as scalar data drops Astro's material
                // fragments before their first sample.
                const auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                const bool src_exec_lo = in.src[0].value == 126;
                const bool src_vcc_lo = in.src[0].value == 106 &&
                    !data_value_present(106);
                const auto saved = in.src[0].kind == OperandKind::SGPR
                    ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
                const bool src_saved_mask = saved != rs.sreg_bool.end() &&
                    !data_value_present(in.src[0].value);
                uint32_t mask = 0;
                if (src_exec_lo) mask = rs.exec;
                else if (src_vcc_lo) mask = rs.vcc;
                else if (src_saved_mask) mask = saved->second;
                else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1)
                    mask = b.btrue();
                else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == 0)
                    mask = b.bfalse();
                if (!mask || in.dst.value == 127 || in.dst.value == 107) {
                    ok = false;
                    return true;
                }

                // ISA SCC is a reduction over the whole resulting wave mask.  That value cannot be
                // recovered from one invocation, so poison it exactly like s_wqm_b64 does.
                rs.scc = 0;
                if (in.dst.value == 126) {
                    if (src_exec_lo) { /* exec <- wqm(exec): tracked identity */ }
                    else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                        rs.exec = b.btrue();
                        rs.exec_narrowed = false;
                    } else {
                        rs.exec = mask;
                        rs.exec_narrowed = true;
                    }
                } else if (in.dst.value == 106) {
                    rs.vcc = mask;
                    rs.sreg_bool[106] = mask;
                    rs.sreg_bool_narrowed[106] = true;
                    rs.sreg_bool_b32.insert(106);
                } else {
                    rs.sreg_bool[in.dst.value] = mask;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                }
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                in.opcode == 0x07) {   // s_not_b32 (mask-domain form)
                // Wave32 shader compilers also invert a saved one-word mask in an ordinary SGPR
                // before combining it back into VCC. Astro's world-map kernel does exactly this:
                //   s_mov_b32 s20, exec_lo
                //   ...
                //   s_not_b32 vcc_lo, s20
                //   s_nor_b32 vcc_lo, vcc_lo, vcc_hi
                // The old special case below only recognized `s_not_b32 vcc_lo,vcc_lo`; routing the
                // saved-SGPR form through scalar DATA either rejected (there are no uint bits for a
                // per-lane mask) or left a stale VCC lifetime at a dispatcher boundary. Accept only
                // an unambiguous complete Wave32 mask source with no competing scalar-data value.
                auto data_value_present = [&](int reg) {
                    return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                };
                uint32_t source_mask = 0;
                if (in.src[0].value == 126) {
                    source_mask = rs.exec;
                } else if ((in.src[0].kind == OperandKind::SGPR ||
                            in.src[0].kind == OperandKind::Special) &&
                           rs.sreg_bool_b32.contains(in.src[0].value) &&
                           !data_value_present(in.src[0].value)) {
                    auto found = rs.sreg_bool.find(in.src[0].value);
                    if (found != rs.sreg_bool.end()) source_mask = found->second;
                    else if (in.src[0].value == 106) source_mask = rs.vcc;
                }
                if (source_mask && in.dst.value != 127) {
                    const uint32_t result = b.logical_not(source_mask);
                    if (in.dst.value == 126) {
                        rs.exec = result;
                        rs.exec_narrowed = true;
                    } else {
                        rs.sreg_bool[in.dst.value] = result;
                        // Complementing an arbitrary mask may activate any previously-clear lane;
                        // it is not a proof that the result is the full wave.
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg_bool_b32.insert(in.dst.value);
                        if (in.dst.value == 106) rs.vcc = result;
                    }
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-wave reduction.
                    return true;
                }
            }
            if (in.opcode == kSop1OpcodeBcnt1I32B64) {
                // s_bcnt1_i32_b64: popcount the complete scalar pair
                // A VOPC may write an arbitrary SGPR pair as a wave mask.  In the portable vertex
                // shell that pair contains the one represented guest lane, so its exact population
                // count is 0/1.  Ordinary data pairs retain full uint semantics and use two native
                // OpBitCount operations.  GTA's compute shaders also count EXEC, VCC, and saved
                // VOPC masks.  Those are exact only when one native subgroup is the complete guest
                // Wave64; a narrower or unknown host width would silently discard guest lanes.
                // EXEC is tracked separately from scalar DATA.  Until the scalar count can be
                // expanded into architectural EXEC bits, reject either EXEC destination for every
                // source domain rather than only for the native-wave reduction below.
                if (in.dst.value == 126 || in.dst.value == 127) {
                    ok = false;
                    return true;
                }
                uint32_t result = 0;
                bool reduced_wave_mask = false;
                auto mask = rs.sreg_bool.find(in.src[0].value);
                if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64) {
                    const auto data_word_present = [&](int reg) {
                        return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
                    };
                    const int source = in.src[0].value;
                    uint32_t source_mask = 0;
                    // Match S_FF1's EXEC rule above: canonical special:126 names architectural
                    // EXEC, while structured state may also carry synthetic scalar placeholders
                    // for the same physical words.  VCC and saved SGPR pairs can genuinely be
                    // recycled as scalar data and therefore retain their ambiguity checks.
                    if (in.src[0].kind == OperandKind::Special && source == 126) {
                        source_mask = rs.exec;
                    } else {
                        const bool proven_saved_mask =
                            b.structured_wave64_mask_reduction_pcs.contains(in.pc);
                        const bool competing_data = !proven_saved_mask &&
                            (data_word_present(source) || data_word_present(source + 1));
                        if (!competing_data && source == 106) {
                            source_mask = rs.vcc;
                        } else if (!competing_data && in.src[0].kind == OperandKind::SGPR &&
                                   !rs.sreg_bool_b32.contains(source) &&
                                   mask != rs.sreg_bool.end()) {
                            source_mask = mask->second;
                        }
                    }
                    if (source_mask) {
                        result = b.native_wave_popcount(source_mask);
                        reduced_wave_mask = true;
                    }
                }
                if (!reduced_wave_mask && !b.is_compute && !b.is_fragment && b.ngg_one_lane &&
                           mask != rs.sreg_bool.end()) {
                    result = b.sel(mask->second, b.uconst(1), b.uconst(0));
                } else if (!reduced_wave_mask) {
                    // A Boolean-domain pair is a whole wave mask, not two ordinary scalar dwords.
                    // Only the exact native-wave and one-lane NGG projections above can reduce it.
                    if (mask != rs.sreg_bool.end()) { ok = false; return true; }
                    auto scalar_half = [&](int reg, uint32_t& value) {
                        auto current = rs.sreg.find(reg);
                        if (current != rs.sreg.end()) { value = current->second; return true; }
                        auto input = rs.sreg_input.find(reg);
                        if (input != rs.sreg_input.end()) { value = input->second; return true; }
                        return false;
                    };
                    uint32_t lo = 0, hi = 0;
                    if (in.src[0].kind == OperandKind::SGPR) {
                        if (!scalar_half(in.src[0].value, lo) ||
                            !scalar_half(in.src[0].value + 1, hi)) {
                            ok = false; return true;
                        }
                    } else if (in.src[0].kind == OperandKind::InlineInt) {
                        lo = b.uconst(static_cast<uint32_t>(in.src[0].value));
                        hi = b.uconst(in.src[0].value < 0 ? UINT32_MAX : 0u);
                    } else if (in.src[0].kind == OperandKind::Literal) {
                        lo = b.uconst(in.literal); hi = b.uconst(0);
                    } else {
                        ok = false; return true;
                    }
                    result = b.ibin(Op_IAdd, b.iun(Op_BitCount, lo), b.iun(Op_BitCount, hi));
                }
                // The result is ordinary scalar DATA regardless of whether its source was a wave
                // mask or a scalar pair.  A one-dword write can overlap either half of a saved B64
                // mask, so terminate both possible Boolean-domain aliases.  VCC has the same
                // split-register hazard and must not retain an older implicit predicate.
                const auto erase_mask_alias = [&](int base) {
                    rs.sreg_bool.erase(base);
                    rs.sreg_bool_narrowed.erase(base);
                    rs.sreg_bool_b32.erase(base);
                };
                erase_mask_alias(in.dst.value);
                if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
                if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                // A B32 write to VCC_LO also replaces virtual lane zero's architectural mask bit.
                // VCC_HI cannot affect that lane.  Other stages keep their multi-lane masks out of
                // the scalar domain and therefore never take this update.
                if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                    rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                    rs.sreg_bool_narrowed[106] = true;
                }
                return true;
            }
            // 64-bit per-lane MASK ops (EXEC / VCC / saved masks). In our per-invocation model a wave
            // mask is a single bool for this lane. EXEC=SGPR 126/127, VCC=106/107; a saved mask lives
            // in sreg_bool. These implement divergent control flow (if/endif via saveexec + restore).
            if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a ||
                (in.opcode >= 0x24 && in.opcode <= 0x2b) ||
                in.opcode == 0x37 || in.opcode == 0x38) {
                // ISA: every op here EXCEPT s_mov_b64 writes SCC=(result!=0) — a cross-lane
                // reduction the per-invocation model cannot form. POISON the tracked SCC (SSA id 0)
                // so a later consumer (s_cselect/s_addc/SCC-source/scc-branch emission) rejects
                // fail-visibly instead of silently evaluating the value an OLDER s_cmp produced.
                // Any real SCC writer re-arms it. Branches the mask_test/waterfall linearizers
                // claim never read rs.scc, so the exercised adjacent shapes are unaffected.
                if (in.opcode != 0x04) rs.scc = 0;
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto src_mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;    // VCC
                    if (o.value == 126 || o.value == 127) return rs.exec;   // EXEC
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;   // not a recognizable mask
                };
                if (in.opcode == 0x0a) {                    // s_wqm_b64: whole-quad-mode mask widen
                    // WQM widens a lane mask to whole 2x2 quads so derivative/sample helper lanes stay
                    // active. In our per-invocation scalar SPIR-V model each lane is one bool and helper
                    // lanes are implicit in the fragment stage, so WQM-widening a mask is the IDENTITY.
                    // VERIFIED(round-trip llvm-mc gfx1030): SOP1 op 0x0a. Common PS preamble around
                    // image_sample (real game shaders 26-29,39), where it is `s_wqm_b64 exec,exec`.
                    // exec_narrowed handling mirrors s_mov_b64 (0x04): the exec<-exec self case is a true
                    // no-op (leave exec AND its narrowed flag untouched — so a later forward s_cbranch_execz
                    // is not spuriously rejected); replacing exec with a *different* mask may narrow it, so
                    // set exec_narrowed conservatively (else inactive-lane writes would escape predication).
                    // SCC = (mask != 0) is a CROSS-lane reduction our per-lane model can't form —
                    // rs.scc was POISONED above so later SCC consumers reject instead of misreading.
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    if (is_exec(in.dst)) {
                        if (is_exec(in.src[0])) { /* exec <- wqm(exec): identity; exec & narrowed unchanged */ }
                        else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;      // exec = all lanes on
                        } else { rs.exec = m; rs.exec_narrowed = true; }        // replaced by a (maybe narrowed) mask
                    } else { rs.sreg_bool[in.dst.value] = m; rs.sreg_bool_narrowed[in.dst.value] = true;
                             mask_write_clobbers_pair(rs, in.dst.value); }  // conservative: WQM widens
                    return true;
                }
                // Narrowed-state carried alongside a saved mask: restoring EXEC from a mask that was saved
                // while EXEC was all-on must clear exec_narrowed (else it stays stuck true past an if/endif
                // — which e.g. breaks a loop that must re-enter the header with full EXEC).
                auto saved_narrowed = [&](const Operand& o) -> bool {
                    // Key by operand VALUE (VCC=106/107 and saved SGPR pairs alike; VCC decodes as Special,
                    // not SGPR, so don't gate on kind). The flag is kept in sync with the mask at EVERY
                    // writer (v_cmp/saveexec/s_mov/s_cselect all update it), so a lookup is accurate.
                    // Unknown provenance -> conservatively narrowed (over-narrowing is a safe no-op).
                    auto it = rs.sreg_bool_narrowed.find(o.value);
                    return it != rs.sreg_bool_narrowed.end() ? it->second : true;
                };
                if (in.opcode == 0x08) {                    // s_not_b64
                    const uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    const uint32_t inverted = b.logical_not(m);
                    if (is_exec(in.dst)) {
                        rs.exec = inverted;
                        rs.exec_narrowed = true;
                    } else if (in.dst.value == 106 || in.dst.value == 107) {
                        rs.vcc = inverted;
                        rs.sreg_bool[in.dst.value] = inverted;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        mask_write_clobbers_pair(rs, in.dst.value);
                    } else {
                        rs.sreg_bool[in.dst.value] = inverted;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        mask_write_clobbers_pair(rs, in.dst.value);
                    }
                    return true;
                }
                if (in.opcode == 0x04) {                    // s_mov_b64
                    if (is_exec(in.dst)) {                  // set/restore EXEC
                        if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                            rs.exec = b.btrue(); rs.exec_narrowed = false;   // exec = all lanes on
                        } else {
                            uint32_t m = src_mask(in.src[0]);
                            // A fragment compiler may restore EXEC from two ordinary scalar words,
                            // notably after round-tripping a saved ballot pair through fixed
                            // V_WRITELANE/V_READLANE slots. Reconstruct this invocation's mask bit
                            // from the exact physical half. subgroup_local_id records the required
                            // guest-Wave64 contract; a missing half or another stage remains
                            // fail-visible instead of treating an arbitrary scalar as a Bool alias.
                            if (!m && b.is_fragment && b.wave_size == 64 &&
                                (in.src[0].kind == OperandKind::SGPR ||
                                 (in.src[0].kind == OperandKind::Special &&
                                  in.src[0].value >= 106 && in.src[0].value <= 123))) {
                                auto scalar_word = [&](int reg) -> uint32_t {
                                    const auto current = rs.sreg.find(reg);
                                    if (current != rs.sreg.end()) return current->second;
                                    const auto input = rs.sreg_input.find(reg);
                                    return input != rs.sreg_input.end() ? input->second : 0;
                                };
                                const uint32_t lo = scalar_word(in.src[0].value);
                                const uint32_t hi = scalar_word(in.src[0].value + 1);
                                if (lo && hi) {
                                    const uint32_t lane = b.ibin(
                                        Op_BitwiseAnd, b.subgroup_local_id(), b.uconst(63));
                                    const uint32_t word = b.sel(
                                        b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)), hi, lo);
                                    const uint32_t bit = b.ibin(
                                        Op_BitwiseAnd, lane, b.uconst(31));
                                    m = b.ucmp(
                                        Op_INotEqual,
                                        b.ibin(Op_BitwiseAnd,
                                               b.ibin(Op_ShiftRightLogical, word, bit),
                                               b.uconst(1)),
                                        b.uconst(0));
                                }
                            }
                            if (!m) ok = false;
                            else { rs.exec = m; rs.exec_narrowed = saved_narrowed(in.src[0]); }
                        }
                    } else {                                // s_mov_b64 sDST, <mask-or-data> : save a mask / copy a pair
                        uint32_t m = src_mask(in.src[0]);
                        if (m) { rs.sreg_bool[in.dst.value] = m;
                                 rs.sreg_bool_narrowed[in.dst.value] = is_exec(in.src[0]) ? rs.exec_narrowed : saved_narrowed(in.src[0]);
                                 // A move INTO VCC is a VCC write (DOLL's scalar-indexed unroll does
                                 // `s_mov_b64 vcc, s[4:5]` before its vccz break, #273): the branch reads
                                 // rs.vcc, so it must be updated too — sreg_bool[106] alone left the
                                 // branch on a stale VCC. (The SOP2 mask ops already special-case 106.)
                                 if (in.dst.value == 106) rs.vcc = m; }
                        // s_mov_b64 ALSO moves a plain 64-bit VALUE (a descriptor pair, a scratch pair,
                        // an inline constant) — compilers use it to shuffle T#/V# halves and constants,
                        // not just wave masks. Copy the data SSA + descriptor provenance (sreg_srt) for
                        // both halves so a later ALU read / buffer op resolves; an untracked source half
                        // clears the stale dest entry rather than aliasing old data. Mask save (above)
                        // and data copy coexist: reads pick their own domain. CONFIDENCE: HIGH.
                        bool data_copied = false;
                        if (in.src[0].kind == OperandKind::SGPR ||
                            (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 && in.src[0].value <= 123)) {
                            bool complete_pair = true;
                            for (int k = 0; k < 2; k++) {
                                int s = in.src[0].value + k, dr = in.dst.value + k;
                                auto it = rs.sreg.find(s);
                                if (it != rs.sreg.end()) {
                                    rs.sreg[dr] = it->second;
                                } else {
                                    auto input = rs.sreg_input.find(s);
                                    if (input != rs.sreg_input.end())
                                        rs.sreg[dr] = input->second;
                                    else {
                                        rs.sreg.erase(dr);
                                        complete_pair = false;
                                    }
                                }
                                auto jt = rs.sreg_srt.find(s);
                                if (jt != rs.sreg_srt.end()) rs.sreg_srt[dr] = jt->second; else rs.sreg_srt.erase(dr);
                            }
                            // Preserve the established unwritten-SGPR-as-zero behavior for an
                            // ordinary data move. A Bool alias needs both real words, however; an
                            // incomplete pair must fall through to exact ballot materialization.
                            data_copied = complete_pair || !m;
                        } else if (in.src[0].kind == OperandKind::InlineInt) {   // 64-bit sign-extended constant
                            rs.sreg[in.dst.value]     = b.uconst((uint32_t)in.src[0].value);
                            rs.sreg[in.dst.value + 1] = b.uconst(in.src[0].value < 0 ? 0xFFFFFFFFu : 0u);
                            rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                            data_copied = true;
                        } else if (in.src[0].kind == OperandKind::Literal) {
                            // 32-bit literal on a B64 move: zero-extended (the f64 high-placement rule is
                            // for double operands only; integer/untyped b64 literals zero-extend).
                            // CONFIDENCE: MED (matches llvm-mc's value validation for s_mov_b64 literals).
                            rs.sreg[in.dst.value]     = b.uconst(in.literal);
                            rs.sreg[in.dst.value + 1] = b.uconst(0);
                            rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                            data_copied = true;
                        }
                        // One exact native subgroup per guest Wave64 makes the Bool mask's two
                        // architectural SGPR words available as subgroup-ballot .x/.y. Preserve
                        // both representations for S_MOV_B64. GTA's BVH loop enters with
                        // `vcc=exec` and reaches its backedge with VCC rebuilt by v_readlane; this
                        // materialization gives that mixed-domain join one common scalar form.
                        if (m && !data_copied && b.is_compute && b.wave_size == 64 &&
                            b.native_subgroup_size == 64) {
                            rs.sreg[in.dst.value] = b.native_wave_ballot_half(m, 0);
                            rs.sreg[in.dst.value + 1] = b.native_wave_ballot_half(m, 1);
                            rs.sreg_srt.erase(in.dst.value);
                            rs.sreg_srt.erase(in.dst.value + 1);
                            data_copied = true;
                        }
                        if (data_copied && !m) {   // data-only move: drop any stale saved-mask alias
                            rs.sreg_bool.erase(in.dst.value); rs.sreg_bool_narrowed.erase(in.dst.value);
                        }
                        // Mask-only source (EXEC/a saved bool with no data half): the pair was still
                        // architecturally overwritten — stale data/descriptor entries must not survive.
                        if (m && !data_copied) mask_write_clobbers_pair(rs, in.dst.value);
                        if (!m && !data_copied) ok = false;
                    }
                } else {                                    // s_*_saveexec_b64 sDST, src
                    // All ten SAVEEXEC forms read OLD EXEC and S0 before writing either destination.
                    // Resolve S0 first because D may itself be VCC (Plucky's exact
                    // `s_orn2_saveexec_b64 vcc, exec`); publishing the saved mask early would change
                    // a VCC source into OLD EXEC and silently compute a different operation.
                    const uint32_t old_exec = rs.exec;
                    const bool old_exec_narrowed = rs.exec_narrowed;
                    uint32_t m = src_mask(in.src[0]);
                    if (!m) { ok = false; return true; }
                    rs.sreg_bool[in.dst.value] = old_exec;  // D = OLD_EXEC
                    rs.sreg_bool_narrowed[in.dst.value] = old_exec_narrowed;
                    if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = old_exec;
                    mask_write_clobbers_pair(rs, in.dst.value);
                    const uint32_t ne = b.logical_not(old_exec);
                    const uint32_t nm = b.logical_not(m);
                    const uint32_t xor_em = b.bsel(old_exec, nm, m);
                    rs.exec = in.opcode == 0x24 ? b.land(old_exec, m)       // AND
                            : in.opcode == 0x25 ? b.lor(old_exec, m)        // OR
                            : in.opcode == 0x26 ? xor_em                    // XOR
                            : in.opcode == 0x27 ? b.land(m, ne)             // ANDN2: S0 & ~EXEC
                            : in.opcode == 0x28 ? b.lor(m, ne)              // ORN2: S0 | ~EXEC
                            : in.opcode == 0x29 ? b.lor(ne, nm)             // NAND
                            : in.opcode == 0x2a ? b.land(ne, nm)            // NOR
                            : in.opcode == 0x2b ? b.logical_not(xor_em)     // XNOR
                            : in.opcode == 0x37 ? b.land(old_exec, nm)      // GFX10 ANDN1_SAVEEXEC
                                                : b.lor(old_exec, nm);      // GFX10 ORN1_SAVEEXEC
                    // Every non-trivial mask expression may be narrower than full EXEC. Two self
                    // identities are exactly all-on and must clear the flag or a later forward
                    // EXECZ guard is conservatively (and incorrectly) rejected.
                    const bool source_is_exec = in.src[0].value == 126 || in.src[0].value == 127;
                    rs.exec_narrowed = !((in.opcode == 0x28 || in.opcode == 0x2b) && source_is_exec);
                    // RE-ARM SCC for the FRAGMENT stage (#2418). The poison above is the right default
                    // — SCC here is `(result != 0)` over the whole 64-lane mask, which one lane's bool
                    // cannot express — but the fragment stage has an exact instrument for precisely
                    // this reduction, and this file already uses it for the same purpose after v_cmpx
                    // narrows EXEC (`rs.scc = b.is_fragment ? b.fragment_wave_any(rs.exec) : 0;`).
                    // `fragment_wave_any`'s contract is "exact scalar wave vote for fragment control
                    // flow AND MASK REDUCTIONS".
                    //
                    // Without it, the poisoned SCC makes every later consumer reject — `s_cselect_b32`
                    // at the SOP2 switch (`if (!rs.scc) { ok = false; }`), `s_addc`, and SCC branch
                    // emission — which is a large share of GTA V's 23,386 skipped draws, since
                    // saveexec-then-`s_cbranch_scc0` is an ordinary shape in its pixel shaders.
                    //
                    // GATED ON `reads_scc`, and that gate is the whole point. `fragment_wave_any` sets
                    // `fragment_required_subgroup_size`, so an unconditional re-arm would give every
                    // shader that merely saves EXEC a wave64 requirement it does not need — and that
                    // shader is then GATED on a 32-wide device where it works today. On a native-wave64
                    // host that regression is invisible, which is exactly why it is guarded here rather
                    // than discovered later on other hardware.
                    //
                    // This is a REDUCE in #2410's taxonomy: the value becomes a guest scalar consumed
                    // as data, so forcing wave64 where it IS used is correct and must not be relaxed.
                    // Compute and vertex keep the poison — compute has its own synchronized guest-wave
                    // reduction paths, and the vertex stage has no equivalent exact vote at all.
                    if (b.is_fragment && rs.reads_scc) rs.scc = b.fragment_wave_any(rs.exec);
                }
                return true;
            }
            if ((b.is_compute || b.is_fragment) && in.opcode == 0x07 &&
                (in.dst.value == 106 || in.dst.value == 107) &&
                in.src[0].value == in.dst.value &&
                !rs.sreg.contains(in.src[0].value)) {
                // s_not_b32 on one physical VCC half updates only those 32 guest lanes. Large
                // Wave64 compute kernels use `v_cmp ... vcc; s_not_b32 vcc_lo,vcc_lo;
                // s_mov_b64 exec,vcc` to select the complement for lanes 0..31 while retaining the
                // original predicate for lanes 32..63. The per-invocation Bool already represents
                // this lane's VCC bit, so select the inversion only in the addressed half.
                uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(),
                                       b.uconst(b.wave_size - 1));
                const uint32_t in_half = in.dst.value == 106
                    ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
                    : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                if (!rs.vcc) { ok = false; return true; }
                rs.vcc = b.bsel(in_half, b.logical_not(rs.vcc), rs.vcc);
                rs.sreg_bool[in.dst.value] = rs.vcc;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                if (b.allow_b32_masks) rs.sreg_bool_b32.insert(in.dst.value);
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-half reduction.
                return true;
            }
            if (in.opcode == 0x1f) {   // s_getpc_b64
                // Accepted ONLY when the pcrel pre-pass FOLDED an embedded-table load from this
                // shader — the pair then only feeds that folded chain. Otherwise the PC would flow
                // into unmodeled address math: keep rejecting.
                if (rs.mubuf_pcrel_tables.empty() && rs.smem_pcrel_tables.empty() &&
                    rs.mtbuf_pcrel_tables.empty()) {
                    ok = false; return true;
                }
                for (int k = 0; k < 2; k++) {
                    rs.sreg.erase(in.dst.value + k); rs.sreg_srt.erase(in.dst.value + k);
                }
                return true;
            }
            if (in.opcode == 0x10 && b.ngg_one_lane) { // s_bcnt1_i32_b64
                // NGG's final primitive packing counts active bits in a saved wave mask. A Vulkan
                // vertex invocation models one guest lane, so the population count of that lane's
                // Boolean mask is exactly 0 or 1. Keep the result in the ordinary scalar-data domain
                // for the following integer packing arithmetic.
                uint32_t mask = 0;
                if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
                else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
                else if (in.src[0].kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(in.src[0].value);
                    if (it != rs.sreg_bool.end()) mask = it->second;
                }
                if (!mask) { ok = false; return true; }
                rs.sreg[in.dst.value] = b.sel(mask, b.uconst(1), b.uconst(0));
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x14 && b.ngg_one_lane) { // s_ff1_i32_b64
                // RDNA2 returns the bit index of the first set bit, or -1 for an empty mask.
                // The exact Astro NGG projection represents guest lane zero only, so a tracked
                // mask has either that bit set (result 0) or no bits set (result 0xffffffff).
                uint32_t mask = 0;
                if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
                else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
                else if (in.src[0].kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(in.src[0].value);
                    if (it != rs.sreg_bool.end()) mask = it->second;
                }
                if (!mask) { ok = false; return true; }
                rs.sreg[in.dst.value] = b.sel(mask, b.uconst(0), b.uconst(0xffffffffu));
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x15 || in.opcode == 0x16) {
                // s_flbit_i32_b32 / s_flbit_i32_b64: the count of leading zeros in the UNSIGNED
                // 32/64-bit source, or -1 when the source is zero. Unlike s_ff1/s_bcnt above, the
                // operand here is ordinary scalar DATA, not a wave mask — Sonic Racing:
                // CrossWorlds' downsample kernel does `s_flbit_i32_b64 vcc_lo, s[14:15]` and
                // immediately `s_sub_i32 vcc_lo, 64, vcc_lo`, i.e. it is computing the index of the
                // highest set bit (a log2 / mip count), then feeds that to v_ldexp_f32 (#2013).
                // VERIFIED(round-trip llvm-mc gfx1030): the live word `beea160e` is
                // `s_flbit_i32_b64 vcc_lo, s[14:15]`, and 0x15 is the same operation over one word.
                // The signed forms 0x17/0x18 (s_flbit_i32 / _i32_i64, which count leading SIGN
                // bits) are a different operation and keep rejecting until a title exercises them.
                if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
                auto scalar_word = [&](int reg, uint32_t& value) {
                    auto current = rs.sreg.find(reg);
                    if (current != rs.sreg.end()) { value = current->second; return true; }
                    auto input = rs.sreg_input.find(reg);
                    if (input != rs.sreg_input.end()) { value = input->second; return true; }
                    return false;
                };
                const bool wide = in.opcode == 0x16;
                uint32_t low = 0, high = b.uconst(0);
                const Operand& source = in.src[0];
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special &&
                     source.value >= 106 && source.value < 124)) {
                    if (!scalar_word(source.value, low)) { ok = false; return true; }
                    if (wide && !scalar_word(source.value + 1, high)) { ok = false; return true; }
                } else if (source.kind == OperandKind::InlineInt) {
                    low = b.uconst(static_cast<uint32_t>(source.value));
                    if (wide) high = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                } else if (source.kind == OperandKind::Literal) {
                    low = b.uconst(in.literal);
                } else {
                    ok = false; return true;
                }
                // FindUMsb is undefined at zero, so never hand it a zero: OR in bit 0 (which cannot
                // change the highest set bit of a non-zero value) and select the zero answer after.
                auto leading_zeros = [&](uint32_t word) {
                    return b.ibin(Op_ISub, b.uconst(31),
                                  b.find_umsb(b.ibin(Op_BitwiseOr, word, b.uconst(1))));
                };
                const uint32_t none = b.uconst(0xffffffffu);
                uint32_t result;
                if (!wide) {
                    result = b.sel(b.ucmp(Op_INotEqual, low, b.uconst(0)),
                                   leading_zeros(low), none);
                } else {
                    result = b.sel(
                        b.ucmp(Op_INotEqual, high, b.uconst(0)), leading_zeros(high),
                        b.sel(b.ucmp(Op_INotEqual, low, b.uconst(0)),
                              b.ibin(Op_IAdd, b.uconst(32), leading_zeros(low)), none));
                }
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value);
                // The live destination is VCC_LO used purely as scalar data. Overwriting the
                // physical register destroys whatever predicate the pair held, so drop the tracked
                // mask instead of leaving a stale one: a later consumer of VCC then fails visibly
                // rather than branching on a value the guest has already overwritten.
                if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
                return true;
            }
            if (in.opcode == 0x1b || in.opcode == 0x1d) {
                // s_bitset{0,1}_b32 is an in-place scalar read/modify/write. The encoded source is
                // the bit index while SDST supplies both the old value and destination. Astro's
                // world-map kernel uses `s_bitset1_b32 s5,31` in a resource-table address path;
                // Plucky's post-Desk transition uses `s_bitset0_b32 vcc_hi,31`. Opcode 0x1c is the
                // distinct B64 clear form and must remain fail-visible until both words are modeled.
                if (in.dst.value == 126 || in.dst.value == 127) {
                    ok = false; return true;
                }
                uint32_t old_value = 0;
                auto current = rs.sreg.find(in.dst.value);
                if (current != rs.sreg.end()) {
                    old_value = current->second;
                } else {
                    auto input = rs.sreg_input.find(in.dst.value);
                    if (input == rs.sreg_input.end()) { ok = false; return true; }
                    old_value = input->second;
                }
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd, val(in.src[0]), b.uconst(31));
                if (!ok) return true;
                const uint32_t mask = b.ibin(
                    Op_ShiftLeftLogical, b.uconst(1), bit);
                rs.sreg[in.dst.value] = in.opcode == 0x1b
                    ? b.ibin(Op_BitwiseAnd, old_value, b.iun(Op_Not, mask))
                    : b.ibin(Op_BitwiseOr, old_value, mask);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value);
                return true;
            }
            if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
                in.opcode == kSop1OpcodeBrevB32) {
                const auto scalar_data_operand = [&](const Operand& source) {
                    switch (source.kind) {
                        case OperandKind::InlineInt:
                        case OperandKind::InlineFloat:
                        case OperandKind::Literal:
                            return true;
                        case OperandKind::SGPR:
                            return rs.sreg.contains(source.value) ||
                                rs.sreg_input.contains(source.value);
                        case OperandKind::Special:
                            if (source.value == 125) return true; // SGPR_NULL
                            if (source.value == 253) return rs.scc != 0;
                            return source.value >= 106 && source.value <= 124 &&
                                rs.sreg.contains(source.value);
                        default:
                            return false;
                    }
                };
                if (!scalar_data_operand(in.src[0])) { ok = false; return true; }
                // A scalar BREV result written to physical EXEC_LO is still the complete Wave32
                // execution mask. GTA V reverses inline -16 (0xfffffff0) into 0x0fffffff, enabling
                // lanes 0..27. Select the exact result bit per invocation; S_BREV_B32 leaves SCC.
                const uint32_t result = b.iun(Op_BitReverse, val(in.src[0]));
                if (!ok) return true;
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
                rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.exec_narrowed = true;
                rs.sreg.erase(126);
                rs.sreg_srt.erase(126);
                return true;
            }
            // A 32-bit scalar DATA write into an EXEC half would leave the live per-lane mask
            // (rs.exec) stale — hardware updates EXEC (and EXECZ) immediately. No exercised title
            // writes EXEC halves via b32 scalar ops (wave64 compilers use the b64 forms), so
            // reject rather than model it. VCC_LO/HI (106/107) and M0 stay accepted as the
            // documented data-scratch round-trip (NGG preamble s_bfe_u32 vcc_lo, DOLL M0 moves).
            if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
            // #3133, the M0 entry-value round trip. Compiled code borrows M0 as scratch: it saves
            // the driver's entry value, overwrites M0 with the LDS base it wants, uses it, and puts
            // the original back.
            //
            //     s_mov_b32 s0, m0 | s_movk_i32 m0, 0 | ds_... | s_mov_b32 m0, s0
            //
            // Stray's `0x300c010000` -- a full-screen fragment pass -- does exactly that twice, and
            // the SAVE was an unresolved operand, so the whole stage failed and every draw bound to
            // it was discarded.
            //
            // Both halves are modelled as an OPAQUE TOKEN, never as a number. The save marks the
            // destination as holding entry-M0 and leaves it with no `sreg` value, so arithmetic on
            // it, a V_WRITELANE scalar source, an LDS index -- every consumer that is not the
            // restore -- still rejects exactly as an untracked read does. The restore consumes the
            // token and returns M0 to UNTRACKED, which is the state the movrel / ADDTID /
            // ds_append guards require and check for. No value is invented at any point, so there
            // is no "is zero the right constant" question and no way for a fabricated word to leak
            // into the data domain through a copy.
            //
            // The DESTINATION is gated as tightly as the source. `in.dst.value` is the raw SOP1 DST
            // field (`sgpr(w >> 16)`, 0..127), so it also names VCC_LO/HI (106/107), ttmp0..15
            // (108..123), M0 itself (124) and SGPR_NULL (125). Only an ordinary SGPR may hold the
            // token: those Special words are read back through the `OperandKind::Special` arm of
            // `operand_bits`, which never consults `sreg_entry_m0`, so tokenising one of them would
            // be worse than doing nothing. `s_mov_b32 vcc_lo, m0` is the case that bites -- the arm
            // erases `rs.sreg[106]` but not `rs.vcc`, and a later data read of VCC_LO then
            // materialises the STALE BALLOT as if it were entry-M0. Anything above s105 falls
            // through to the ordinary path below, where an untracked M0 source rejects loudly, i.e.
            // exactly the behaviour that predates this arm. (EXEC, 126/127, already rejected above.)
            if (in.opcode == 0x03 && in.dst.value <= 105 &&
                in.src[0].kind == OperandKind::Special &&
                in.src[0].value == 124 && !rs.sreg.contains(124)) {
                rs.sreg.erase(in.dst.value);
                rs.sreg_entry_m0.insert(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value);
                rs.sreg_written.insert(in.dst.value);
                return true;
            }
            if (in.opcode == 0x03 && in.dst.value == 124 &&
                in.src[0].kind == OperandKind::SGPR &&
                rs.sreg_entry_m0.contains(in.src[0].value) &&
                !rs.sreg.contains(in.src[0].value)) {
                rs.sreg.erase(124);                 // M0 is untracked again, as it was at entry
                rs.sreg_entry_m0.erase(124);
                return true;
            }
            // A write of anything else ends this register's entry-M0 lifetime.
            rs.sreg_entry_m0.erase(in.dst.value);
            uint32_t a = val(in.src[0]); uint32_t& d = rs.sreg[in.dst.value];
            // An ordinary scalar-data write starts a new lifetime for this physical word.  Do not
            // let an earlier Wave32 mask save alias that new value in later mask-domain moves.
            rs.sreg_bool.erase(in.dst.value);
            rs.sreg_bool_narrowed.erase(in.dst.value);
            rs.sreg_bool_b32.erase(in.dst.value);
            switch (in.opcode) {
                case 0x03: {                                // s_mov_b32
                    d = a;
                    // Descriptor provenance is part of the scalar value. Shader compilers commonly
                    // load several V#s into separate SGPR ranges and copy the selected four words into
                    // one reused SRSRC range before each MUBUF. Keeping that range's OLD tag makes all
                    // subsequent buffer loads resolve to the first descriptor (DOLL scene VS: the four
                    // transform rows became one row, degenerating every triangle).
                    auto tag = rs.sreg_srt.find(in.src[0].value);
                    if ((in.src[0].kind == OperandKind::SGPR ||
                         (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 && in.src[0].value <= 123)) &&
                        tag != rs.sreg_srt.end())
                        rs.sreg_srt[in.dst.value] = tag->second;
                    else
                        rs.sreg_srt.erase(in.dst.value);
                    break;
                }
                case 0x07:                                  // s_not_b32 changes the descriptor word
                    // ISA op 7: "D = ~S0; SCC = (D != 0)". (s_brev_b32 below has NO SCC write.)
                    d = b.iun(Op_Not, a); rs.sreg_srt.erase(in.dst.value);
                    rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0)); break;
                case kSop1OpcodeBrevB32:                     // s_brev_b32
                    d = b.iun(Op_BitReverse, a); rs.sreg_srt.erase(in.dst.value); break;
                case 0x34: {                                // s_abs_i32
                    // Two's-complement absolute value.  OpISub deliberately preserves the ISA's
                    // INT_MIN -> INT_MIN wraparound; SCC is set iff the resulting bits are nonzero.
                    const uint32_t negative = b.scmp(Op_SLessThan, a, b.uconst(0));
                    d = b.sel(negative, b.ibin(Op_ISub, b.uconst(0), a), a);
                    rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0));
                    rs.sreg_srt.erase(in.dst.value);
                    break;
                }
                default: ok = false;
            }
            return true;
        }
        case Rdna2Format::SOP2: {
            const bool gtav_wave32_vcchi_scalar_packet =
                allows_compute_scalar_vcc_bridge(b) && b.native_subgroup_size == 32 &&
                is_gtav_wave32_vcchi_scalar_packet(in);
            if (b.is_compute && b.wave_size == 64 && in.opcode == 0x0a &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == 106) {
                // A B32 write leaves Wave64 VCC_HI untouched, so the resulting mixed physical pair
                // can remain in the architectural mask domain only when the dispatcher proves that
                // sibling is scalar data at this exact PC. Otherwise admit only a whole-stream
                // low-only proof and poison VCC until its later complete replacement.
                const bool path_local_scalar_pair =
                    rs.scalar_presence_has_no_placeholders && rs.sreg.contains(107);
                const bool complete_scalar_pair =
                    path_local_scalar_pair || b.vcc_b32_scalar_pair_pcs.contains(in.pc);
                if (!is_wave64_vcc_lo_scalar_cselect(in) ||
                    (!complete_scalar_pair &&
                     !b.vcc_b32_low_only_pcs.contains(in.pc)) || !rs.scc) {
                    ok = false; return true;
                }
                const uint32_t selected = b.sel(
                    rs.scc, val(in.src[0]), val(in.src[1]));
                if (!ok) return true;
                rs.sreg[106] = selected;
                rs.sreg_srt.erase(106);
                rs.sreg_bool.erase(106);
                rs.sreg_bool_narrowed.erase(106);
                rs.sreg_bool_b32.erase(106);
                if (complete_scalar_pair) {
                    auto high = rs.sreg.find(107);
                    if (high == rs.sreg.end()) { ok = false; return true; }
                    const uint32_t lane = b.ibin(
                        Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                    const uint32_t word = b.sel(
                        b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                        high->second, selected);
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                    rs.vcc = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd,
                               b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                        b.uconst(0));
                    rs.sreg_bool[106] = rs.vcc;
                    rs.sreg_bool_narrowed[106] = true;
                } else {
                    rs.vcc = 0;
                }
                return true;
            }
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                in.opcode == 0x0a &&
                (in.dst.value == 126 ||
                 in.src[0].value == 126 || in.src[1].value == 126 ||
                 (((in.src[0].kind == OperandKind::SGPR ||
                    in.src[0].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[0].value))) ||
                 (((in.src[1].kind == OperandKind::SGPR ||
                    in.src[1].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[1].value))))) {
                // Wave32 s_cselect_b32 can select complete one-word wave masks. Astro's material
                // setup uses `s_cselect_b32 vcc_hi, exec_lo, 0` after a scalar mode comparison,
                // then consumes that VCC_HI mask with VOP3 cndmask instructions. Keep the select in
                // the Bool domain; routing either operand through scalar bits would lose lane state.
                auto mask = [&](const Operand& source) -> uint32_t {
                    if (source.value == 126) return rs.exec;
                    if ((source.value == 106 || source.value == 107) &&
                        rs.sreg_bool_b32.contains(source.value)) {
                        auto it = rs.sreg_bool.find(source.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (source.value == 106) return rs.vcc;
                    }
                    if (source.kind == OperandKind::SGPR &&
                        rs.sreg_bool_b32.contains(source.value)) {
                        auto it = rs.sreg_bool.find(source.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (source.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, source.value);
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1 || !rs.scc || in.dst.value == 127) {
                    ok = false; return true;
                }
                const uint32_t result = b.bsel(rs.scc, m0, m1);
                if (in.dst.value == 126) {
                    rs.exec = result;
                    rs.exec_narrowed = true;
                } else {
                    rs.sreg_bool[in.dst.value] = result;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) rs.vcc = result;
                }
                return true;
            }
            if (b.is_compute && is_scalar_cselect_b32_to_vcc_lo(in)) {
                // The mask-domain cselect above remains first: this bridge is only for the distinct
                // scalar-data lifetime seen in GTA V. Preserve the full selected dword in `sreg`,
                // publish its bit for this guest lane as VCC, and leave SCC unchanged as required by
                // S_CSELECT_B32. Wave64 cannot represent both dwords this way and stays rejected.
                if (!allows_compute_scalar_vcc_bridge(b)) { ok = false; return true; }
                if (!rs.scc) { ok = false; return true; }
                const uint32_t selected = b.sel(rs.scc, val(in.src[0]), val(in.src[1]));
                if (!ok) return true;
                rs.sreg[106] = selected;
                rs.sreg_srt.erase(106);
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, selected, lane), b.uconst(1));
                rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.sreg_bool[106] = rs.vcc;
                rs.sreg_bool_narrowed[106] = true;
                rs.sreg_bool_b32.insert(106);
                return true;
            }
            auto wave32_mask_operand = [&](const Operand& source) {
                // Inline integer/literal operands are representable in both domains. Classify
                // them as masks so a pure mask operation preserves its Bool lifetime; only a
                // genuinely mixed mask/raw-data operation should take the scalar route below.
                if (source.kind == OperandKind::InlineInt ||
                    source.kind == OperandKind::Literal)
                    return true;
                if (source.kind == OperandKind::Special && source.value == 126)
                    return true; // EXEC_LO
                return (source.kind == OperandKind::SGPR ||
                        source.kind == OperandKind::Special) &&
                    rs.sreg_bool_b32.contains(source.value);
            };
            auto wave32_live_mask_operand = [&](const Operand& source) {
                if (source.kind == OperandKind::Special && source.value == 126)
                    return true; // EXEC_LO
                return (source.kind == OperandKind::SGPR ||
                        source.kind == OperandKind::Special) &&
                    rs.sreg_bool_b32.contains(source.value);
            };
            auto exact_wave32_scalar_dword = [&](const Operand& source) {
                switch (source.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::SGPR:
                        return rs.sreg.contains(source.value) ||
                            rs.sreg_input.contains(source.value) ||
                            (b.is_compute && b.wave_size == 32 &&
                             b.native_subgroup_size == 32 &&
                             rs.sreg_bool_b32.contains(source.value) &&
                             rs.sreg_bool.contains(source.value));
                    case OperandKind::Special:
                        if (source.value == 125) return true; // SGPR_NULL
                        if (source.value == 253) return rs.scc != 0;
                        if (source.value >= 106 && source.value <= 124)
                            return rs.sreg.contains(source.value);
                        return b.is_compute && b.wave_size == 32 &&
                            b.native_subgroup_size == 32 &&
                            (source.value == 126 || source.value == 127) && rs.exec;
                    default:
                        return false;
                }
            };
            auto scalar_data_operand = [&](const Operand& source) {
                switch (source.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::SGPR:
                        return rs.sreg.contains(source.value) ||
                            rs.sreg_input.contains(source.value);
                    case OperandKind::Special:
                        if (source.value == 125) return true; // SGPR_NULL
                        if (source.value == 253) return rs.scc != 0;
                        return source.value >= 106 && source.value <= 124 &&
                            rs.sreg.contains(source.value);
                    default:
                        return false;
                }
            };
            auto logical_u32 = [&](uint32_t a, uint32_t c) {
                return in.opcode == kSop2OpcodeAndB32 ? b.ibin(Op_BitwiseAnd, a, c)
                     : in.opcode == kSop2OpcodeOrB32 ? b.ibin(Op_BitwiseOr, a, c)
                     : in.opcode == kSop2OpcodeXorB32 ? b.ibin(Op_BitwiseXor, a, c)
                     : in.opcode == kSop2OpcodeAndn2B32
                           ? b.ibin(Op_BitwiseAnd, a, b.iun(Op_Not, c))
                     : in.opcode == kSop2OpcodeOrn2B32
                           ? b.ibin(Op_BitwiseOr, a, b.iun(Op_Not, c))
                     : in.opcode == kSop2OpcodeNandB32
                           ? b.iun(Op_Not, b.ibin(Op_BitwiseAnd, a, c))
                     : in.opcode == kSop2OpcodeNorB32
                           ? b.iun(Op_Not, b.ibin(Op_BitwiseOr, a, c))
                           : b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c));
            };
            auto bitfield_mask_u32 = [&](uint32_t width, uint32_t offset) {
                width = b.ibin(Op_BitwiseAnd, width, b.uconst(31));
                offset = b.ibin(Op_BitwiseAnd, offset, b.uconst(31));
                const uint32_t ones = b.ibin(
                    Op_ISub,
                    b.ibin(Op_ShiftLeftLogical, b.uconst(1), width), b.uconst(1));
                return b.ibin(Op_ShiftLeftLogical, ones, offset);
            };
            if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
                sop2_is_b32_logical(in.opcode) &&
                scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1])) {
                // A physical EXEC_LO write always changes the wave mask, even when both inputs are
                // ordinary scalar DATA rather than Bool-domain masks. GTA V writes 0x80 into
                // VCC_LO as scalar scratch, then ORs it with literal 0x100 to select lanes 7 and 8.
                // Both complete dwords are available uniformly, so compute the architectural u32,
                // publish exact SCC, and select this invocation's Wave32 bit without a subgroup op.
                const uint32_t result = logical_u32(val(in.src[0]), val(in.src[1]));
                if (!ok) return true;
                rs.scc = b.ucmp(Op_INotEqual, result, b.uconst(0));
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
                rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.exec_narrowed = true;
                rs.sreg.erase(126);
                rs.sreg_srt.erase(126);
                return true;
            }
            if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
                in.opcode == kSop2OpcodeBfmB32 &&
                scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1])) {
                // BFM writes an ordinary 32-bit result, but physical EXEC_LO still consumes that
                // dword as the complete Wave32 mask. GTA V uses width=16, offset=16 to enable the
                // upper half of a traversal wave. Both operands are uniform scalar DATA, so select
                // the resulting architectural bit for this invocation; S_BFM_B32 leaves SCC alone.
                const uint32_t result = bitfield_mask_u32(
                    val(in.src[0]), val(in.src[1]));
                if (!ok) return true;
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
                rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.exec_narrowed = true;
                rs.sreg.erase(126);
                rs.sreg_srt.erase(126);
                return true;
            }
            const bool mixed_wave32_mask_scalar_data =
                b.is_compute && b.wave_size == 32 && b.native_subgroup_size == 32 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value <= 105 &&
                wave32_mask_operand(in.src[0]) != wave32_mask_operand(in.src[1]) &&
                exact_wave32_scalar_dword(in.src[0]) &&
                exact_wave32_scalar_dword(in.src[1]);
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
                !gtav_wave32_vcchi_scalar_packet &&
                !mixed_wave32_mask_scalar_data &&
                sop2_is_b32_logical(in.opcode) &&
                (in.dst.value == 126 ||
                 wave32_live_mask_operand(in.src[0]) ||
                 wave32_live_mask_operand(in.src[1]))) {
                // Wave32 B32 logical operations are one-word wave-mask operations, parallel to the
                // B64 family immediately below. The live Astro material uses
                //   s_andn2_b32 s64, s64, vcc_hi
                // after an explicit VOPC write to VCC_HI. Resolve each operand in the per-lane Bool
                // domain; treating VCC_HI as scalar bits is unrepresentable in this model.
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 126) return rs.exec;
                    if ((o.value == 106 || o.value == 107) &&
                        rs.sreg_bool_b32.contains(o.value)) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                        if (o.value == 106) return rs.vcc;
                    }
                    if (o.kind == OperandKind::SGPR && rs.sreg_bool_b32.contains(o.value)) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    if (o.kind == OperandKind::Literal)
                        return inline_int_mask_bit(
                            b, static_cast<int32_t>(in.literal));
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1 || in.dst.value == 127) { ok = false; return true; }
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);
                const uint32_t r = in.opcode == kSop2OpcodeAndB32 ? b.land(m0, m1)
                                 : in.opcode == kSop2OpcodeOrB32 ? b.lor(m0, m1)
                                 : in.opcode == kSop2OpcodeXorB32 ? x
                                 : in.opcode == kSop2OpcodeAndn2B32 ? b.land(m0, n1)
                                 : in.opcode == kSop2OpcodeOrn2B32 ? b.lor(m0, n1)
                                 : in.opcode == kSop2OpcodeNandB32 ? b.lor(n0, n1)
                                 : in.opcode == kSop2OpcodeNorB32 ? b.land(n0, n1)
                                 : b.logical_not(x);
                // SCC=(result!=0) is a guest-wave reduction. A fragment with proven Wave32 can
                // request one exact 32-lane Vulkan subgroup and vote here; compute retains the
                // existing poison unless its dispatcher supplies a synchronized reduction.
                rs.scc = b.is_fragment ? b.fragment_wave_any(r) : 0;
                if (in.dst.value == 126) {
                    rs.exec = r;
                    rs.exec_narrowed = true;
                } else {
                    rs.sreg_bool[in.dst.value] = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    rs.sreg_bool_b32.insert(in.dst.value);
                    rs.sreg.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) rs.vcc = r;
                }
                return true;
            }
            if (in.opcode >= 0x32 && in.opcode <= 0x34) {
                // GFX10 scalar halfword pack family.  LL selects src0.lo/src1.lo, LH selects
                // src0.lo/src1.hi, and HH selects src0.hi/src1.hi.  These are pure scalar DATA
                // operations (no SCC write); NGG uses LL to pack two wave population counts that
                // temporarily live in VCC_LO/HI.
                const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                if (!ok) return true;
                const uint32_t lo = in.opcode == 0x34
                    ? b.ibin(Op_ShiftRightLogical, a, b.uconst(16))
                    : b.ibin(Op_BitwiseAnd, a, b.uconst(0xffff));
                const uint32_t hi = in.opcode == 0x32
                    ? b.ibin(Op_ShiftLeftLogical,
                             b.ibin(Op_BitwiseAnd, c, b.uconst(0xffff)), b.uconst(16))
                    : b.ibin(Op_BitwiseAnd, c, b.uconst(0xffff0000));
                const uint32_t result = b.ibin(Op_BitwiseOr, lo, hi);
                rs.sreg[in.dst.value] = result;
                rs.sreg_srt.erase(in.dst.value);
                if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                    rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                    rs.sreg_bool_narrowed[106] = true;
                }
                return true;
            }
            if (in.opcode == kSop2OpcodeBfmB64) { // s_bfm_b64
                if (b.is_vertex && !b.ngg_one_lane) { ok = false; return true; }
                // Wave masks are represented as one bool per SPIR-V invocation.  Constructing the
                // architectural 64-bit integer and then splitting it would lose that domain, so test
                // the current lane directly: bits [offset, offset+width) are set, truncated at bit 63.
                // This is also exact for width==0 (the ISA expression produces an empty mask).
                const uint32_t width = b.ibin(Op_BitwiseAnd, val(in.src[0]), b.uconst(63));
                const uint32_t offset = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
                // Vertex NGG is the same one-lane approximation as the inline-mask helpers: its
                // represented guest lane is lane zero and it has no compute LocalInvocationIndex.
                const uint32_t lane = b.ngg_one_lane ? b.uconst(0) :
                    b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1));
                const uint32_t at_or_after = b.ucmp(Op_UGreaterThanEqual, lane, offset);
                const uint32_t within_width = b.ucmp(
                    Op_ULessThan, b.ibin(Op_ISub, lane, offset), width);
                const uint32_t r = b.land(at_or_after, within_width);
                if (in.dst.value == 126 || in.dst.value == 127) {
                    rs.exec = r;
                    rs.exec_narrowed = true;
                } else if (in.dst.value == 106 || in.dst.value == 107) {
                    rs.vcc = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    mask_write_clobbers_pair(rs, in.dst.value);
                } else {
                    rs.sreg_bool[in.dst.value] = r;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                    mask_write_clobbers_pair(rs, in.dst.value);
                }
                return true;
            }
            if (in.opcode == 0x0b) {   // s_cselect_b64: SCC ? src0 : src1 (mask or scalar-pair domain)
                // Operands/dest are wave masks (EXEC/VCC/saved/inline), NOT uint bits — resolve like the
                // SOP1 mask ops and select in the bool domain. Keep this path first so established
                // mask lifetimes retain byte-identical behavior even when Function-backed scalar
                // placeholders coexist in a CFG dispatcher. (s_cselect_b32 stays in the uint path.)
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if ((o.value == 106 || o.value == 107) && rs.vcc) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;   // not a recognizable mask
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (m0 && m1) {
                    if (!rs.scc) { ok = false; return true; }   // SCC poisoned by a mask op
                    uint32_t r = b.bsel(rs.scc, m0, m1);
                    if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                    else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                                                                           mask_write_clobbers_pair(rs, in.dst.value); }
                    else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true;   // conservative flag
                           mask_write_clobbers_pair(rs, in.dst.value); }
                    return true;
                }

                // Ordinary scalar-data pairs are selected word-for-word. Unlike EXEC, VCC has a
                // dual role: its two physical scalar dwords remain readable as data, while vector
                // instructions consume the bit for this invocation. Materialize both views from
                // the same selected pair so native and portable compute paths agree exactly.
                struct ScalarPair {
                    uint32_t lo = 0, hi = 0;
                    bool have_lo = false, have_hi = false;
                };
                auto scalar_word = [&](int reg, uint32_t& value) {
                    auto current = rs.sreg.find(reg);
                    if (current != rs.sreg.end()) { value = current->second; return true; }
                    auto input = rs.sreg_input.find(reg);
                    if (input != rs.sreg_input.end()) { value = input->second; return true; }
                    return false;
                };
                auto scalar_pair = [&](const Operand& source) {
                    ScalarPair pair;
                    if (source.kind == OperandKind::SGPR ||
                        (source.kind == OperandKind::Special &&
                         source.value >= 106 && source.value < 124)) {
                        pair.have_lo = scalar_word(source.value, pair.lo);
                        pair.have_hi = scalar_word(source.value + 1, pair.hi);
                    } else if (source.kind == OperandKind::InlineInt) {
                        pair.lo = b.uconst(static_cast<uint32_t>(source.value));
                        pair.hi = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                        pair.have_lo = pair.have_hi = true;
                    } else if (source.kind == OperandKind::Literal) {
                        pair.lo = b.uconst(in.literal);
                        pair.hi = b.uconst(0);
                        pair.have_lo = pair.have_hi = true;
                    }
                    return pair;
                };
                if (is_exec(in.dst) || !rs.scc) { ok = false; return true; }
                const ScalarPair p0 = scalar_pair(in.src[0]);
                const ScalarPair p1 = scalar_pair(in.src[1]);
                const bool complete = p0.have_lo && p0.have_hi && p1.have_lo && p1.have_hi;
                const bool low_only = p0.have_lo && p1.have_lo &&
                    b.is_compute && b.wave_size == 64 &&
                    b.cselect_b64_low_only_pcs.contains(in.pc);
                if (!complete && !low_only) { ok = false; return true; }

                const uint32_t selected_lo = b.sel(rs.scc, p0.lo, p1.lo);
                const uint32_t selected_hi = complete
                    ? b.sel(rs.scc, p0.hi, p1.hi) : b.uconst(0);
                rs.sreg[in.dst.value] = selected_lo;
                if (complete) rs.sreg[in.dst.value + 1] = selected_hi;
                else rs.sreg.erase(in.dst.value + 1);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value + 1);

                // A scalar pair write ends every overlapping saved-mask/B32 lifetime. The complete
                // VCC form immediately installs its newly-derived predicate below; the GTA low-only
                // form deliberately leaves VCC unavailable because the CFG proof says no implicit
                // mask consumer can observe it.
                auto erase_mask_alias = [&](int base) {
                    rs.sreg_bool.erase(base);
                    rs.sreg_bool_narrowed.erase(base);
                    rs.sreg_bool_b32.erase(base);
                };
                if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
                erase_mask_alias(in.dst.value);
                erase_mask_alias(in.dst.value + 1);
                if (in.dst.value == 106) {
                    if (complete) {
                        uint32_t lane = b.ibin(
                            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
                        const uint32_t high = b.ucmp(
                            Op_UGreaterThanEqual, lane, b.uconst(32));
                        const uint32_t word = b.sel(high, selected_hi, selected_lo);
                        const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                        rs.vcc = b.ucmp(
                            Op_INotEqual,
                            b.ibin(Op_BitwiseAnd,
                                   b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                            b.uconst(0));
                        rs.sreg_bool[106] = rs.vcc;
                        rs.sreg_bool_narrowed[106] = true;
                    } else {
                        rs.vcc = 0;
                    }
                }
                return true;
            }
            if (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 || in.opcode == 0x15 ||
                in.opcode == 0x17 || in.opcode == 0x19 || in.opcode == 0x1b || in.opcode == 0x1d) {
                // 64-bit wave-mask LOGICAL ops on per-lane bools: AND/OR/XOR/ANDN2 plus the
                // complementary ORN2/NAND/NOR/XNOR family. Used for lane-mask arithmetic around
                // divergent control flow / ballot. SCC=(result!=0) is a cross-lane reduction we can't
                // form per-lane — POISON rs.scc so a later consumer rejects instead of silently
                // reading an older s_cmp's value (the adjacent scc-branch shapes are claimed by the
                // mask_test/waterfall linearizers and never read rs.scc). Same mask-resolution as
                // s_cselect_b64.
                auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
                auto mask = [&](const Operand& o) -> uint32_t {
                    if ((o.value == 106 || o.value == 107) && rs.vcc) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second; }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    // A native Wave64 subgroup can project an ordinary scalar pair into the same
                    // per-invocation Bool representation used for wave masks: select this guest
                    // lane's 32-bit half, then extract its bit. GTA copies EXEC_LO/HI ballots into
                    // scalar scratch and intersects that pair with VCC at pc1467. Both halves must
                    // exist; narrower/unknown subgroup modes remain fail-visible.
                    if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 &&
                        (o.kind == OperandKind::SGPR ||
                         (o.kind == OperandKind::Special &&
                          (o.value == 106 || o.value == 107)))) {
                        auto scalar_word = [&](int reg, uint32_t& value) {
                            auto current = rs.sreg.find(reg);
                            if (current != rs.sreg.end()) {
                                value = current->second;
                                return true;
                            }
                            auto input = rs.sreg_input.find(reg);
                            if (input != rs.sreg_input.end()) {
                                value = input->second;
                                return true;
                            }
                            return false;
                        };
                        uint32_t lo = 0, hi = 0;
                        if (scalar_word(o.value, lo) && scalar_word(o.value + 1, hi)) {
                            // Invert the ballot with the same subgroup-local index that assigned
                            // its bits; this needs no LocalInvocationIndex ordering assumption.
                            const uint32_t lane = b.ibin(
                                Op_BitwiseAnd, b.subgroup_local_id(), b.uconst(63));
                            const uint32_t word = b.sel(
                                b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)), hi, lo);
                            const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                            return b.ucmp(
                                Op_INotEqual,
                                b.ibin(Op_BitwiseAnd,
                                       b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                                b.uconst(0));
                        }
                    }
                    return 0;
                };
                uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (!m0 || !m1) { ok = false; return true; }
                rs.scc = 0;   // poison: hardware SCC=(result!=0) is unrepresentable per-lane
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);               // xor = m0 ? !m1 : m1
                uint32_t r = in.opcode == 0x0f ? b.land(m0, m1)      // and
                           : in.opcode == 0x11 ? b.lor(m0, m1)       // or
                           : in.opcode == 0x13 ? x                    // xor
                           : in.opcode == 0x15 ? b.land(m0, n1)      // andn2
                           : in.opcode == 0x17 ? b.lor(m0, n1)       // orn2
                           : in.opcode == 0x19 ? b.lor(n0, n1)       // nand
                           : in.opcode == 0x1b ? b.land(n0, n1)      // nor
                           : b.logical_not(x);                        // xnor
                if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
                else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                                                                       mask_write_clobbers_pair(rs, in.dst.value); }
                else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                       mask_write_clobbers_pair(rs, in.dst.value); }
                // One exact native subgroup per guest Wave64 also makes the logical result's
                // architectural SGPR pair available. GTA joins an early scalar EXEC ballot with
                // a later S_ANDN2_B64 survivor mask and then compares s[56:57] as u64. Preserve
                // both views from the same Bool; portable/unknown subgroup modes remain mask-only.
                if (!is_exec(in.dst) && b.is_compute && b.wave_size == 64 &&
                    b.native_subgroup_size == 64) {
                    rs.sreg[in.dst.value] = b.native_wave_ballot_half(r, 0);
                    rs.sreg[in.dst.value + 1] = b.native_wave_ballot_half(r, 1);
                    rs.sreg_srt.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value + 1);
                }
                return true;
            }
            // The NGG wave-packing s_lshr_b64 form (dst = EXEC) sets EXEC to the count of active
            // vertices/primitives in the wave. A per-invocation SPIR-V shader has no wave to pack,
            // so leave EXEC full. Handle that special form before the ordinary scalar-pair shift.
            if (in.opcode == 0x21) {
                if (in.dst.value == 126 || in.dst.value == 127) {
                    rs.scc = 0; // poison: hardware SCC=(result!=0) over the packed mask is cross-lane
                    return true;
                }
            }
            if (in.opcode == 0x1f || in.opcode == 0x21) {
                // Ordinary s_lshl/lshr_b64 operates on a complete scalar-data pair. Generated
                // compute shaders frequently borrow VCC_LO/HI for 64-bit address arithmetic; that
                // lifetime coexists with VCC's per-lane predicate view, so retain both exactly.
                auto scalar_word = [&](int reg, uint32_t& value) {
                    auto current = rs.sreg.find(reg);
                    if (current != rs.sreg.end()) { value = current->second; return true; }
                    auto input = rs.sreg_input.find(reg);
                    if (input != rs.sreg_input.end()) { value = input->second; return true; }
                    return false;
                };
                uint32_t source_lo = 0, source_hi = 0;
                const Operand& source = in.src[0];
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special &&
                     source.value >= 106 && source.value < 124)) {
                    if (!scalar_word(source.value, source_lo) ||
                        !scalar_word(source.value + 1, source_hi)) {
                        ok = false; return true;
                    }
                } else if (source.kind == OperandKind::InlineInt) {
                    source_lo = b.uconst(static_cast<uint32_t>(source.value));
                    source_hi = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                } else if (source.kind == OperandKind::Literal) {
                    source_lo = b.uconst(in.literal);
                    source_hi = b.uconst(0);
                } else {
                    ok = false; return true;
                }
                const uint32_t amount = b.ibin(
                    Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
                if (!ok) return true;
                const uint32_t result = b.u64_shift(
                    in.opcode == 0x1f ? Op_ShiftLeftLogical : Op_ShiftRightLogical,
                    b.u64_from_lohi(source_lo, source_hi), amount);
                const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
                rs.scc = b.ucmp(
                    Op_INotEqual, b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
                const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
                const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
                if (writes_exec || writes_vcc) {
                    uint32_t lane = b.ibin(
                        Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
                    const uint32_t mask_bit = b.u64_bit(result, lane);
                    if (writes_exec) {
                        rs.exec = mask_bit;
                        rs.exec_narrowed = true;
                        rs.sreg.erase(in.dst.value);
                        rs.sreg.erase(in.dst.value + 1);
                    } else {
                        rs.vcc = mask_bit;
                        rs.sreg_bool[in.dst.value] = mask_bit;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg[in.dst.value] = lo;
                        rs.sreg[in.dst.value + 1] = hi;
                    }
                } else {
                    rs.sreg[in.dst.value] = lo;
                    rs.sreg[in.dst.value + 1] = hi;
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool.erase(in.dst.value + 1);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value + 1);
                }
                rs.sreg_bool_b32.erase(in.dst.value);
                rs.sreg_bool_b32.erase(in.dst.value + 1);
                rs.sreg_srt.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value + 1);
                return true;
            }
            if (in.opcode == kSop2OpcodeBfeU64) { // s_bfe_u64
                // BFE's sources are scalar DATA even when its destination is the architectural
                // EXEC/VCC mask pair. Keep the complete uniform source pair long enough to extract
                // the bit belonging to this emulated lane; treating the destination as ordinary
                // SGPR data leaves rs.exec/rs.vcc stale, while rejecting EXEC drops valid NGG
                // merged-stage prologues. A Vulkan vertex invocation represents the one live guest
                // lane for that vertex; compute/fragment shells retain their real wave lane.
                auto high = [&]() -> uint32_t {
                    const Operand& source = in.src[0];
                    if (source.kind == OperandKind::SGPR) {
                        auto value = rs.sreg.find(source.value + 1);
                        if (value != rs.sreg.end()) return value->second;
                        auto input = rs.sreg_input.find(source.value + 1);
                        return input == rs.sreg_input.end() ? b.uconst(0) : input->second;
                    }
                    if (source.kind == OperandKind::Special &&
                        source.value >= 106 && source.value < 124) {
                        auto value = rs.sreg.find(source.value + 1);
                        if (value != rs.sreg.end()) return value->second;
                        ok = false; return b.uconst(0);
                    }
                    if (source.kind == OperandKind::InlineInt)
                        return b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                    if (source.kind == OperandKind::Literal) return b.uconst(0);
                    ok = false; return b.uconst(0);
                };
                const uint32_t source_lo = val(in.src[0]);
                const uint32_t source_hi = high();
                const uint32_t control = val(in.src[1]);
                if (!ok) return true;
                const uint32_t offset = b.ibin(Op_BitwiseAnd, control, b.uconst(0x3f));
                const uint32_t width = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, control, b.uconst(16)),
                    b.uconst(0x7f));
                const uint32_t result = b.bfe_u64(
                    b.u64_from_lohi(source_lo, source_hi), offset, width);
                const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
                const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
                if (writes_exec || writes_vcc) {
                    uint32_t lane = b.guest_lane_id();
                    lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                    const uint32_t mask_bit = b.u64_bit(result, lane);
                    rs.scc = 0;   // poison: SCC=(complete mask != 0) is a guest-wave reduction
                    if (writes_exec) {
                        rs.exec = mask_bit;
                        rs.exec_narrowed = true;
                    } else {
                        rs.vcc = mask_bit;
                        rs.sreg_bool[in.dst.value] = mask_bit;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                    }
                    mask_write_clobbers_pair(rs, in.dst.value);
                } else {
                    const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
                    rs.sreg[in.dst.value] = lo;
                    rs.sreg[in.dst.value + 1] = hi;
                    rs.scc = b.ucmp(Op_INotEqual,
                                    b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
                    rs.sreg_srt.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value + 1);
                }
                return true;
            }
            if ((in.dst.value == 106 || in.dst.value == 107) &&
                in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0) {
                // 32-bit logical writes to VCC_LO/HI bridge scalar DATA masks into the architectural
                // lane mask. NGG prologs use this to merge a scalar bitset with a VOPC result before
                // a v_cndmask. We cannot materialize the complete VCC dword in a per-invocation
                // module, but its bit for this guest lane is exact. Preserve the untouched VCC half;
                // unlike a B64 mask write, a B32 write changes only LO or HI.
                auto has_scalar_data = [&](const Operand& source) {
                    switch (source.kind) {
                        case OperandKind::SGPR:
                        case OperandKind::InlineInt:
                        case OperandKind::InlineFloat:
                        case OperandKind::Literal:
                            return true;
                        case OperandKind::Special:
                            if (source.value == 125) return true;       // SGPR_NULL
                            if (source.value == 253) return rs.scc != 0; // SCC as scalar 0/1
                            return source.value >= 106 && source.value <= 124 &&
                                   rs.sreg.count(source.value) != 0;
                        default:
                            return false;
                    }
                };
                const bool has_proven_scalar_sources =
                    has_scalar_data(in.src[0]) && has_scalar_data(in.src[1]) &&
                    (!b.is_compute || b.wave_size != 64 ||
                     rs.scalar_presence_has_no_placeholders ||
                     b.vcc_b32_scalar_result_pcs.contains(in.pc));
                if ((!b.is_fragment && !b.is_compute) || gtav_wave32_vcchi_scalar_packet ||
                    has_proven_scalar_sources) {
                    // The vertex shell is a complete one-lane virtual wave, so its scalar VCC
                    // dwords are representable: LO starts as {bit0=vcc}, HI as zero, and later B32
                    // operations may use either as ordinary scratch. In Wave32 compute, VCC_HI is
                    // entirely outside the architectural mask and is therefore always scalar
                    // scratch; an exact 32-lane subgroup also lets operand_bits materialize an
                    // EXEC_LO/VCC_LO source through one complete ballot. Other fragment/compute
                    // forms retain a full dword only when BOTH inputs already have scalar-data
                    // representations. The captured Plucky Squire tonemap shader, for example,
                    // computes `s_and_b32 vcc_lo, loop_index, 3` and immediately compares VCC_LO as
                    // a uint. True VOPC/mask inputs have no scalar representation and continue into
                    // the wave-mask path below.
                    const uint32_t result = logical_u32(val(in.src[0]), val(in.src[1]));
                    if (!ok) return true;
                    rs.sreg[in.dst.value] = result;
                    rs.sreg_srt.erase(in.dst.value);
                    rs.scc = b.ucmp(Op_INotEqual, result, b.uconst(0));
                    const bool writes_hi = in.dst.value == 107;
                    // A whole-CFG proof can establish that this physical VCC word is scalar data
                    // only: the untouched sibling dies before any mask-domain read. Do not ask for
                    // a guest lane merely to manufacture a predicate that no later instruction can
                    // observe. Vertex emission has a separate one-lane virtual-wave contract, so
                    // this applies only to fragment and compute stages.
                    const bool scalar_low_only = (b.is_fragment || b.is_compute) &&
                        b.vcc_b32_low_only_pcs.contains(in.pc);
                    if (b.wave_size != 32 && scalar_low_only) {
                        rs.vcc = 0;
                        rs.sreg_bool.erase(106);
                        rs.sreg_bool_narrowed.erase(106);
                        rs.sreg_bool_b32.erase(106);
                        rs.sreg_bool.erase(107);
                        rs.sreg_bool_narrowed.erase(107);
                        rs.sreg_bool_b32.erase(107);
                        return true;
                    }
                    uint32_t lane = b.guest_lane_id();
                    lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                    const uint32_t in_written_half = writes_hi
                        ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
                        : b.ucmp(Op_ULessThan, lane, b.uconst(32));
                    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                    const uint32_t result_bit = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd,
                               b.ibin(Op_ShiftRightLogical, result, bit), b.uconst(1)),
                        b.uconst(0));
                    if (b.wave_size == 32) {
                        // A complete scalar write covers every architectural bit of VCC_LO in
                        // Wave32, so it establishes a fresh predicate without needing the old VCC
                        // lifetime. VCC_HI is ordinary scratch outside the 32-lane mask and cannot
                        // affect the implicit VCC condition at all. This distinction matters for
                        // generated kernels that load scalar data into LO, derive a flag in HI, and
                        // compare both as uints before creating their next vector predicate.
                        if (writes_hi) {
                            rs.sreg_bool.erase(107);
                            rs.sreg_bool_narrowed.erase(107);
                            rs.sreg_bool_b32.erase(107);
                        } else {
                            rs.vcc = result_bit;
                            rs.sreg_bool[106] = result_bit;
                            rs.sreg_bool_narrowed[106] = true;
                        }
                        return true;
                    }
                    if (b.vcc_b32_scalar_pair_pcs.contains(in.pc)) {
                        const int sibling = writes_hi ? 106 : 107;
                        auto other = rs.sreg.find(sibling);
                        if (other == rs.sreg.end()) { ok = false; return true; }
                        const uint32_t other_bit = b.ucmp(
                            Op_INotEqual,
                            b.ibin(Op_BitwiseAnd,
                                   b.ibin(Op_ShiftRightLogical, other->second, bit),
                                   b.uconst(1)),
                            b.uconst(0));
                        rs.vcc = b.bsel(in_written_half, result_bit, other_bit);
                    } else if (scalar_low_only) {
                        // The whole-CFG proof established that the untouched high word cannot be
                        // observed before a complete pair replacement. This instruction therefore
                        // starts a scalar-only VCC_LO lifetime; retaining the old Bool predicate
                        // would give later dispatcher blocks a competing stale mask domain.
                        rs.vcc = 0;
                        rs.sreg_bool.erase(106);
                        rs.sreg_bool_narrowed.erase(106);
                        rs.sreg_bool_b32.erase(106);
                    } else {
                        if (!rs.vcc) { ok = false; return true; }
                        rs.vcc = b.bsel(in_written_half, result_bit, rs.vcc);
                    }
                    if (!scalar_low_only) {
                        rs.sreg_bool[in.dst.value] = rs.vcc;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                    }
                    return true;
                }

                uint32_t lane = b.guest_lane_id();
                lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
                const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                const bool writes_hi = in.dst.value == 107;
                const uint32_t in_written_half = writes_hi
                    ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
                    : b.ucmp(Op_ULessThan, lane, b.uconst(32));
                auto mask_bit = [&](const Operand& source) -> uint32_t {
                    if (source.kind == OperandKind::Special &&
                        (source.value == 106 || source.value == 107)) {
                        // Reading the same VCC half maps directly to this invocation's predicate.
                        // Cross-half bit movement needs another guest lane and remains unsupported.
                        if (source.value != in.dst.value) { ok = false; return b.bfalse(); }
                        return rs.vcc;
                    }
                    const uint32_t raw = val(source);
                    return b.ucmp(Op_INotEqual,
                                  b.ibin(Op_BitwiseAnd,
                                         b.ibin(Op_ShiftRightLogical, raw, bit), b.uconst(1)),
                                  b.uconst(0));
                };
                const uint32_t m0 = mask_bit(in.src[0]);
                const uint32_t m1 = mask_bit(in.src[1]);
                if (!ok) return true;
                const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
                const uint32_t x = b.bsel(m0, n1, m1);
                const uint32_t result = in.opcode == 0x0e ? b.land(m0, m1)
                                      : in.opcode == 0x10 ? b.lor(m0, m1)
                                      : in.opcode == 0x12 ? x
                                      : in.opcode == 0x14 ? b.land(m0, n1)
                                      : in.opcode == 0x16 ? b.lor(m0, n1)
                                      : in.opcode == 0x18 ? b.lor(n0, n1)
                                      : in.opcode == 0x1a ? b.land(n0, n1)
                                      : b.logical_not(x);
                if (!rs.vcc) { ok = false; return true; }
                rs.vcc = b.bsel(in_written_half, result, rs.vcc);
                rs.sreg_bool[in.dst.value] = rs.vcc;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                rs.scc = 0; // SCC=(complete written dword != 0) is a guest-wave reduction
                return true;
            }
            // 32-bit scalar DATA writes into EXEC halves are rejected (see the SOP1 uint path).
            if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t& d = rs.sreg[in.dst.value];
            auto scc_nz = [&](uint32_t v){ rs.scc = b.ucmp(Op_INotEqual, v, b.uconst(0)); };  // SCC = (result != 0)
            switch (in.opcode) {
                case 0x00: d = b.ibin(Op_IAdd, a, c); rs.scc = b.ucmp(Op_ULessThan, d, a); break;  // s_add_u32 (SCC=carry)
                case 0x01: d = b.ibin(Op_ISub, a, c); rs.scc = b.ucmp(Op_ULessThan, a, c); break;  // s_sub_u32 (SCC=borrow)
                // Signed add/sub: two's-complement result is bit-identical to the unsigned op; SCC is
                // signed OVERFLOW. add ovf = operands same sign AND result sign differs: (~(a^c))&(a^d);
                // sub ovf = operands differ in sign AND result sign differs from a: (a^c)&(a^d). Bit 31 = ovf.
                case 0x02: { d = b.ibin(Op_IAdd, a, c);                                              // s_add_i32
                             uint32_t o = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)),
                                                               b.ibin(Op_BitwiseXor, a, d));
                             rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
                case 0x03: { d = b.ibin(Op_ISub, a, c);                                              // s_sub_i32
                             uint32_t o = b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, a, c),
                                                               b.ibin(Op_BitwiseXor, a, d));
                             rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
                case 0x04: {   // s_addc_u32: dst = src0 + src1 + SCC(carry-in); SCC = carry-out. The high
                               // half of a 64-bit add (pairs with s_add_u32's SCC). Round-trip llvm-mc
                               // gfx1010: 0x82252580 → s_addc_u32 s37, 0, s37. CONFIDENCE: HIGH.
                    if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op: carry-in unknown
                    uint32_t cin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
                    uint32_t s1 = b.ibin(Op_IAdd, a, c);
                    uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+c
                    d = b.ibin(Op_IAdd, s1, cin);
                    uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);        // wrap in +cin
                    rs.scc = b.bsel(k1, b.btrue(), k2); break;
                }
                case 0x05: {   // s_subb_u32: dst = src0 - src1 - SCC(borrow-in); SCC = borrow-out.
                    if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op: borrow-in unknown
                    const uint32_t bin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
                    const uint32_t difference = b.ibin(Op_ISub, a, c);
                    const uint32_t first_borrow = b.ucmp(Op_ULessThan, a, c);
                    d = b.ibin(Op_ISub, difference, bin);
                    const uint32_t second_borrow = b.ucmp(Op_ULessThan, difference, bin);
                    rs.scc = b.bsel(first_borrow, b.btrue(), second_borrow); break;
                }
                // s_min/max (0x06 min_i32, 0x07 min_u32, 0x08 max_i32, 0x09 max_u32): SCC = "src0 was
                // selected", STRICT in both directions per the ISA pseudocode (S_MIN: SCC = S0 < S1;
                // S_MAX_I32: "D.i = (S0.i > S1.i) ? S0.i : S1.i; SCC = (S0.i > S1.i)", doc 70648
                // sec 12.1 ops 6-9) — on a tie SCC = 0 for both min and max. An earlier change (#397)
                // flipped max to non-strict `>=` quoting a pseudocode line that is not in the actual
                // document; the 2026-07 ISA audit (#879) re-derived the strict form from the PDF.
                // D is unaffected either way (the tie value is identical). Round-trip llvm-mc
                // gfx1010: 0x83000201/0x83800201/0x84000201/0x84800201. CONFIDENCE: HIGH.
                case 0x06: rs.scc = b.scmp(Op_SLessThan, a, c);    d = b.sext2(Glsl_SMin, a, c); break;
                case 0x07: rs.scc = b.ucmp(Op_ULessThan, a, c);    d = b.uext2(Glsl_UMin, a, c); break;
                case 0x08: rs.scc = b.scmp(Op_SGreaterThan, a, c); d = b.sext2(Glsl_SMax, a, c); break;
                case 0x09: rs.scc = b.ucmp(Op_UGreaterThan, a, c); d = b.uext2(Glsl_UMax, a, c); break;
                case 0x0A:   // s_cselect_b32: SCC ? src0 : src1 (reads SCC, writes none)
                    if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op
                    d = b.sel(rs.scc, a, c); break;
                case 0x0E: d = b.ibin(Op_BitwiseAnd, a, c); scc_nz(d); break;   // s_and_b32
                case 0x10: d = b.ibin(Op_BitwiseOr,  a, c); scc_nz(d); break;   // s_or_b32
                case 0x12: d = b.ibin(Op_BitwiseXor, a, c); scc_nz(d); break;   // s_xor_b32
                case 0x14: d = b.ibin(Op_BitwiseAnd, a, b.iun(Op_Not, c));       // s_andn2_b32
                           scc_nz(d); break;
                case 0x16: d = b.ibin(Op_BitwiseOr, a, b.iun(Op_Not, c));        // s_orn2_b32
                           scc_nz(d); break;
                case 0x18: d = b.iun(Op_Not, b.ibin(Op_BitwiseAnd, a, c));       // s_nand_b32
                           scc_nz(d); break;
                case 0x1A: d = b.iun(Op_Not, b.ibin(Op_BitwiseOr, a, c));        // s_nor_b32
                           scc_nz(d); break;
                case 0x1C: d = b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c));       // s_xnor_b32
                           scc_nz(d); break;
                case 0x1E: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshl_b32
                             d = b.ibin(Op_ShiftLeftLogical, a, sh); scc_nz(d); break; }  // dst = src0 << (src1 & 31)
                case kSop2OpcodeBfmB32:
                    d = bitfield_mask_u32(a, c); break; // no SCC write
                case 0x20: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshr_b32
                             d = b.ibin(Op_ShiftRightLogical, a, sh); scc_nz(d); break; }  // dst = src0 >> (src1 & 31)
                case 0x22: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_ashr_i32
                             d = b.sbin(Op_ShiftRightArithmetic, a, sh); scc_nz(d); break; }  // dst = src0 >>a (src1 & 31)
                case 0x26: d = b.ibin(Op_IMul, a, c); break;         // s_mul_i32 (low 32 bits; no SCC)
                case 0x2E: case 0x2F: case 0x30: case 0x31: {
                    // s_lshl{1,2,3,4}_add_u32 = (src0<<N)+src1; SCC is unsigned carry-out of the
                    // FULL 64-bit sum (RDNA2 ISA ops 46-49: ((u64)S0<<N)+S1 >= 2^32). If any of
                    // S0's top N bits are set, the shift alone overflowed; otherwise overflow is
                    // exactly the ordinary low-dword addition wrap (D < S1).
                    const uint32_t shift = in.opcode - 0x2Du;
                    uint32_t shifted = b.ibin(Op_ShiftLeftLogical, a, b.uconst(shift));
                    d = b.ibin(Op_IAdd, shifted, c);
                    uint32_t shifted_out = b.ucmp(Op_INotEqual,
                        b.ibin(Op_ShiftRightLogical, a, b.uconst(32u - shift)), b.uconst(0));
                    uint32_t wrapped = b.ucmp(Op_ULessThan, d, c);
                    rs.scc = b.bsel(shifted_out, b.btrue(), wrapped); break;
                }
                case kSop2OpcodePackLlB32B16: { // s_pack_ll_b32_b16: D={S1[15:0],S0[15:0]}
                    const uint32_t lo = b.ibin(Op_BitwiseAnd, a, b.uconst(0xFFFFu));
                    const uint32_t hi = b.ibin(Op_ShiftLeftLogical,
                        b.ibin(Op_BitwiseAnd, c, b.uconst(0xFFFFu)), b.uconst(16));
                    d = b.ibin(Op_BitwiseOr, lo, hi);
                    break;
                }
                case 0x35: d = b.umul_hi(a, c); break;               // s_mul_hi_u32 (high 32 bits; no SCC)
                case 0x36: d = b.smul_hi(a, c); break;               // s_mul_hi_i32 (high 32 bits; no SCC).
                                                                     // gfx10 SOP2 opcode is 0x36 (llvm-mc:
                                                                     // 0x9b000201>>23&0x7f); 0x37 was WRONG
                                                                     // (invalid encoding), so the handler was
                                                                     // dead and s_mul_hi_i32 got rejected. #462
                case 0x27: {                                         // s_bfe_u32: offset=src1[4:0], width=src1[22:16]
                    uint32_t off = b.ibin(Op_BitwiseAnd, c, b.uconst(0x1f));
                    uint32_t width = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, c, b.uconst(16)), b.uconst(0x7f));
                    d = b.bfe_u(a, off, width); scc_nz(d); break;
                }
                default: ok = false;
            }
            // Every modeled SOP2 operation changes the scalar bits (including cselect unless both
            // inputs happen to be identical). A previous descriptor tag on the destination is stale;
            // retaining it can bind an unrelated later SRSRC to the old buffer.
            if (ok) {
                rs.sreg_srt.erase(in.dst.value);
                if (in.opcode == kSop2OpcodeBfeU64)
                    rs.sreg_srt.erase(in.dst.value + 1);
            }
            return true;
        }
        case Rdna2Format::VOP3P: {
            if (in.opcode <= 0x0Du) {
                // The packed 16-bit INTEGER family. Sonic Racing: CrossWorlds' post chain rejects on
                // v_pk_add_u16 (#2013); the rest of the family is the same lowering behind the same
                // half-select wrapper. Each result half k is computed from source halves selected by
                // OPSEL[src] (low result) / OPSEL_HI[src] (high result), and BOTH halves are always
                // written — unlike the scalar 16-bit VOP3 ops, a packed op has no preserved half.
                //   0x00 v_pk_mad_i16      D.i16[k] = S0*S1 + S2
                //   0x01 v_pk_mul_lo_u16   D.u16[k] = low16(S0 * S1)
                //   0x02/0x03 v_pk_add_i16 / v_pk_sub_i16
                //   0x04/0x05/0x06 v_pk_lshlrev_b16 / v_pk_lshrrev_b16 / v_pk_ashrrev_i16
                //                  (REVERSED operands: D = S1 <op> (S0 & 0xf))
                //   0x07/0x08 v_pk_max_i16 / v_pk_min_i16
                //   0x09 v_pk_mad_u16      (identical 16-bit bit pattern to v_pk_mad_i16)
                //   0x0a/0x0b v_pk_add_u16 / v_pk_sub_u16 (identical bit result to the i16 forms)
                //   0x0c/0x0d v_pk_max_u16 / v_pk_min_u16
                // VERIFIED(round-trip llvm-mc gfx1030, every opcode above); the live encoding
                // `cc0a0002,18020504` disassembles to `v_pk_add_u16 v2, v4, v2`.
                // CONFIDENCE: HIGH on the operations themselves.
                //
                // NEG_LO/NEG_HI on a packed INTEGER source: applied as a flip of bit 15 of the
                // selected 16-bit half — physically the same shared sign-negate the packed f16 path
                // above already models with fneg, sitting between the operand read and the ALU and
                // indifferent to how the ALU then interprets the bits. CONFIDENCE: MED — the
                // derivation below is LIVE evidence, and no published statement confirms it.
                //
                // DO NOT RAISE THIS TO HIGH ON THE LLVM CITATION. It was tried and withdrawn. Two
                // things were offered as "published support" and neither holds: (1) LLVM's AMDGPU
                // modifier reference says of neg_lo/neg_hi "This modifier is valid for
                // floating-point operands only" — that is a RESTRICTION, and if anything it cuts
                // AGAINST applying the bit to an integer opcode, which is exactly the unpublished
                // step this MED is hedging; (2) the claim that LLVM folds `xor 0x80008000` into
                // neg_lo/neg_hi for integer packed ops traces to llvm-project PR #130234, which was
                // merged and REVERTED the same day and covered the dot-product family, not
                // v_pk_add_u16/v_pk_max_i16. Under the charter's evidence hierarchy that is a single
                // secondary implementation — tier 4, hypothesis only.
                //
                // What actually carries the reading is the live derivation, and it is strong. All
                // FOUR sites in this title that set the bits use the same literal
                // `0x00007fff` and set NEG_LO[0] and NEG_HI[0] together, with OPSEL/OPSEL_HI
                // selecting that literal's LOW half:
                //   * `v_pk_add_u16 v4, -(0x7fff), v2 op_sel:[0,1] op_sel_hi:[0,0]` produces the
                //     base of a gather whose per-lane offsets are then +0/+2 (a 2x2 quad), so the
                //     base must be (x-1, y-1);
                //   * `v_pk_max_i16 v4, -(0x7fff), v3` sits beside `v_min_i16 2, …` and
                //     `v_pk_min_i16 1, …`, i.e. an offset clamped into a small range around zero.
                // Bit-15 flip gives 0xffff = -1, which satisfies both. The alternative reading,
                // two's-complement negation, gives -32767 — which would make that `v_pk_max_i16` a
                // no-op sitting next to a `min` of +1, and no compiler emits that. If a later title
                // contradicts this, the discriminating input is any negated source whose selected
                // half is not 0x7fff.
                const uint32_t old_d = vreg_old(b, rs, in.dst.value);
                // A 16-bit operand position carries the 16-bit encoding of an inline constant, not
                // the f32 pattern val() would materialize — the same rule the scalar 16-bit VOP3
                // family follows. The half select still applies to the sign-extended 32-bit value.
                auto u16src = [&](int k, bool high_result) {
                    const Operand& operand = in.src[k];
                    const uint8_t selectors = high_result ? in.vop3p_opsel_hi : in.vop3p_opsel;
                    const uint32_t half = (selectors >> k) & 1u;
                    uint32_t v, inline_dword;
                    if (inline_16bit_operand_dword(operand, inline_dword))
                        v = b.uconst((inline_dword >> (16u * half)) & 0xFFFFu);
                    else
                        v = b.bfe_u(val(operand), b.uconst(16u * half), b.uconst(16));
                    const bool negate = high_result ? ((in.vop3p_neg_hi >> k) & 1u) != 0
                                                    : in.src_neg[k];
                    return negate ? b.ibin(Op_BitwiseXor, v, b.uconst(0x8000u)) : v;
                };
                auto s16 = [&](uint32_t v) { return b.bfe_s(v, b.uconst(0), b.uconst(16)); };
                auto operation = [&](bool high) {
                    const uint32_t s0 = u16src(0, high), s1 = u16src(1, high);
                    uint32_t r;
                    switch (in.opcode) {
                        case 0x00: case 0x09:
                            r = b.ibin(Op_IAdd, b.ibin(Op_IMul, s0, s1), u16src(2, high)); break;
                        case 0x01:  r = b.ibin(Op_IMul, s0, s1); break;
                        case 0x02: case 0x0A: r = b.ibin(Op_IAdd, s0, s1); break;
                        case 0x03: case 0x0B: r = b.ibin(Op_ISub, s0, s1); break;
                        case 0x04:  r = b.ibin(Op_ShiftLeftLogical, s1,
                                               b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu))); break;
                        case 0x05:  r = b.ibin(Op_ShiftRightLogical, s1,
                                               b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu))); break;
                        case 0x06:  r = b.sbin(Op_ShiftRightArithmetic, s16(s1),
                                               b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu))); break;
                        case 0x07:  r = b.sext2(Glsl_SMax, s16(s0), s16(s1)); break;
                        case 0x08:  r = b.sext2(Glsl_SMin, s16(s0), s16(s1)); break;
                        case 0x0C:  r = b.uext2(Glsl_UMax, s0, s1); break;
                        default:    r = b.uext2(Glsl_UMin, s0, s1); break;     // 0x0d v_pk_min_u16
                    }
                    return b.ibin(Op_BitwiseAnd, r, b.uconst(0xFFFFu));
                };
                rs.vreg[in.dst.value] = b.ibin(
                    Op_BitwiseOr, operation(false),
                    b.ibin(Op_ShiftLeftLogical, operation(true), b.uconst(16)));
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            if (in.opcode >= 0x0Eu && in.opcode <= 0x12u) {  // v_pk_fma/add/mul/min/max_f16
                const uint32_t old_d = vreg_old(b, rs, in.dst.value);
                auto half = [&](int source, bool high_result) -> uint32_t {
                    uint32_t v = val(in.src[source]);
                    const uint8_t selectors = high_result ? in.vop3p_opsel_hi : in.vop3p_opsel;
                    const uint32_t sel = (selectors >> source) & 1u;
                    // An inline constant is the raw dword its width produces (see
                    // inline_16bit_operand_dword): a float contributes its f16 encoding in the LOW
                    // half and ZERO in the high one — exact even for 1/(2*pi) — and an int its
                    // sign-extended two's-complement bits, so inline 1 reads as the f16 denormal
                    // 0x0001, NOT 1.0. Either way the half select applies; it is not a no-op. This
                    // path used to replicate the float into both halves (#2119, kernel 32r14a).
                    uint32_t inline_dword;
                    if (inline_16bit_operand_dword(in.src[source], inline_dword)) {
                        v = b.unpack_half(b.uconst(inline_dword), sel);
                    } else {
                        v = b.unpack_half(v, sel);
                    }
                    const bool negate = high_result
                        ? ((in.vop3p_neg_hi >> source) & 1u) != 0
                        : in.src_neg[source];
                    return negate ? b.fneg(v) : v;
                };
                // 0x0e v_pk_fma_f16 is the only three-source form; min/max return the OTHER operand
                // when exactly one input is NaN (ISA 12.7), which is GLSL NMin/NMax, not FMin/FMax.
                // VERIFIED(round-trip llvm-mc gfx1030) for 0x0e/0x11/0x12.
                auto operation = [&](bool high) {
                    uint32_t r;
                    switch (in.opcode) {
                        case 0x0E: r = b.fbin(Op_FAdd,
                                              b.fbin(Op_FMul, half(0, high), half(1, high)),
                                              half(2, high)); break;
                        case 0x0F: r = b.fbin(Op_FAdd, half(0, high), half(1, high)); break;
                        case 0x10: r = b.fbin(Op_FMul, half(0, high), half(1, high)); break;
                        case 0x11: r = b.fext2(Glsl_NMin, half(0, high), half(1, high)); break;
                        default:   r = b.fext2(Glsl_NMax, half(0, high), half(1, high)); break;
                    }
                    if (in.clamp) r = b.clamp01(r);
                    return r;
                };
                const uint32_t lo = b.pack_half_lo(operation(false));
                const uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(operation(true)), b.uconst(16));
                rs.vreg[in.dst.value] = b.ibin(Op_BitwiseOr, lo, hi);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // Mixed-precision FMA family, trivial form only (all sources full f32 — the decoder set
            // has_modifier for any opsel/neg/clamp bits, rejected above). v_fma_mix_f32 (0x20):
            // d = s0*s1+s2. v_fma_mixlo/hi_f16 (0x21/0x22): the f32 result converts to f16 into the
            // LOW/HIGH half of d, PRESERVING the other half (DOLL's box-blur PS packs its result
            // this way, #273). VERIFIED(round-trip llvm-mc gfx1010: 0xcc210000 0x041600f2 ->
            // v_fma_mixlo_f16 v0, 1.0, v0, v5 — the live blur bytes).
            if (in.opcode != 0x20 && in.opcode != 0x21 && in.opcode != 0x22) { ok = false; return true; }
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // Per-source mix resolve (#273): OPSEL_HI[k] -> f16 half (OPSEL[k]: 0=lo,1=hi) converted
            // to f32; else full f32. An inline constant read at f16 width is the dword its width
            // produces (see inline_16bit_operand_dword), so OPSEL selects a half of THAT — a float
            // reads 0.0h from the high half, not the constant again. This used to treat the select
            // as a no-op for inline floats (#2119, kernel 32r14b).
            // NEG_HI = abs, NEG = negate (abs first, hardware order).
            auto mixv = [&](int k) -> uint32_t {
                uint32_t v = val(in.src[k]);
                const bool half = (in.vop3p_opsel_hi >> k) & 1u;
                if (half) {
                    const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
                    uint32_t inline_dword;
                    if (inline_16bit_operand_dword(in.src[k], inline_dword))
                        v = b.unpack_half(b.uconst(inline_dword), sel);
                    else
                        v = b.unpack_half(v, sel);
                }
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fneg(v);
                return v;
            };
            uint32_t r = b.fbin(Op_FAdd, b.fbin(Op_FMul, mixv(0), mixv(1)), mixv(2));
            if (in.clamp) r = b.clamp01(r);
            uint32_t& d = rs.vreg[in.dst.value];
            if (in.opcode == 0x20) d = r;
            else if (in.opcode == 0x21)
                d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), b.pack_half_lo(r));
            else
                d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                           b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(r), b.uconst(16)));
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::SOPC: {
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
            // Scalar compare -> SCC (read by s_cselect / s_cbranch_scc). eq/lg are bitwise (sign-agnostic);
            // the ordered compares are signed for i32 (0x02-0x05), unsigned for u32 (0x08-0x0b).
            // Wave32 compilers also compare the physical VCC_LO dword with zero. The mask has no
            // per-invocation integer representation, but EQ/LG and the bounded unsigned forms below
            // reduce exactly to none/any across the guest wave.
            if (allow_wave && b.is_compute && b.wave_size == 32) {
                auto zero = [](const Operand& operand) {
                    return operand.kind == OperandKind::InlineInt && operand.value == 0;
                };
                auto mask = [&](const Operand& operand) -> uint32_t {
                    if (operand.kind != OperandKind::SGPR &&
                        operand.kind != OperandKind::Special) return 0;
                    auto found = rs.sreg_bool.find(operand.value);
                    return found == rs.sreg_bool.end() ? 0 : found->second;
                };
                const uint32_t first_mask = mask(in.src[0]);
                const uint32_t second_mask = mask(in.src[1]);
                const bool mask_first = first_mask && zero(in.src[1]) &&
                    (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x08 ||
                     in.opcode == 0x0b);
                const bool mask_second = second_mask && zero(in.src[0]) &&
                    (in.opcode == 0x06 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     in.opcode == 0x0a);
                if (mask_first || mask_second) {
                    const uint32_t any = b.native_subgroup_size
                        ? b.native_wave_any(mask_first ? first_mask : second_mask)
                        : b.guest_wave_any(mask_first ? first_mask : second_mask);
                    const bool invert = in.opcode == 0x06 ||
                        (mask_first ? in.opcode == 0x0b : in.opcode == 0x09);
                    rs.scc = invert ? b.logical_not(any) : any;
                    return true;
                }
            }
            //
            // A B64 compare may consume EXEC, VCC, or a saved wave mask. Those values intentionally
            // have no scalar-data representation: one SPIR-V bool represents this invocation's bit.
            // At a wave-uniform fragment site, reduce the per-lane mismatch across the enforced
            // 64-lane subgroup. This is the exact SCC result of s_cmp_eq/lg_u64 and, unlike reading
            // VCC_LO as uint data, also preserves masks built by vector comparisons and saveexec.
            if ((in.opcode == 0x12 || in.opcode == 0x13) && allow_wave && b.is_fragment) {
                auto mask = [&](const Operand& o) -> uint32_t {
                    if (o.value == 106 || o.value == 107) return rs.vcc;
                    if (o.value == 126 || o.value == 127) return rs.exec;
                    if (o.kind == OperandKind::SGPR) {
                        auto it = rs.sreg_bool.find(o.value);
                        if (it != rs.sreg_bool.end()) return it->second;
                    }
                    if (o.kind == OperandKind::InlineInt)
                        return inline_int_mask_bit(b, o.value);
                    return 0;
                };
                const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
                if (m0 && m1) {
                    const uint32_t mismatch = b.bsel(m0, b.logical_not(m1), m1);
                    const uint32_t different = b.fragment_wave_any(mismatch);
                    rs.scc = in.opcode == 0x12 ? b.logical_not(different) : different;
                    return true;
                }
            }
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
        case Rdna2Format::SOPK: {
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
        case Rdna2Format::VOP1: {
            uint32_t a = val(in.src[0]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            uint32_t dpp_active = 0;
            // DPP16 quad_perm on src0 (#273): fragment shaders reconstruct the selected quad lane
            // from derivatives. Compute shaders use the exact subgroup quad-swap operation for the
            // three XOR permutations emitted by Astro Bot's blur kernels: horizontal (0xb1), vertical
            // (0x4e), and diagonal (0x1b). Quad boundaries are architectural and remain exact even
            // when the host subgroup width differs from the guest wave width.
            if (in.has_dpp) {
                const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
                uint32_t row_xor = 0;
                const bool row_ror8 = dpp_row_xor_ctrl(in.dpp_ctrl, &row_xor);
                if (row_ror8) {
                    // Direct shuffle is valid only when one native subgroup is exactly one guest
                    // wave. Portable/default-subgroup compute is routed through synchronized CFG
                    // scratch below. FI=0 makes an EXEC-inactive rotated source read as zero; a
                    // ROW_ROR source is always in-range, so BOUND_CTRL does not decide this case.
                    if (!b.is_compute || dpp_row_ror8_op(in) != DppRowRor8Op::MovB32 ||
                        !b.native_subgroup_size) {
                        ok = false; return true;
                    }
                    uint32_t valid_source = 0;
                    const uint32_t rotated =
                        b.subgroup_row_xor(a, rs.exec, row_xor, &valid_source);
                    a = b.sel(valid_source, rotated, b.uconst(0));
                } else if (row_shr) {
                    // The portable NGG vertex shell represents the one live guest lane as lane 0.
                    // Every non-zero row-right shift therefore addresses a lane before the start of
                    // its row; BOUND_CTRL=1 supplies the architectural zero.  An unbounded access
                    // retains the prior destination value and cannot be represented here, so reject.
                    if (b.is_fragment || (!b.is_compute && !in.dpp_bound_ctrl) ||
                        (b.is_compute && in.opcode != 0x01)) {
                        ok = false; return true;
                    }
                    if (b.is_compute) {
                        b.mark_subgroup_min16();
                        const uint32_t shift = in.dpp_ctrl - 0x110u;
                        const uint32_t lane = b.subgroup_local_id();
                        const uint32_t row_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(15));
                        dpp_active = b.ucmp(Op_UGreaterThanEqual, row_lane, b.uconst(shift));
                        const uint32_t source_lane = b.sel(
                            dpp_active, b.ibin(Op_ISub, lane, b.uconst(shift)), lane);
                        const uint32_t shuffled = b.subgroup_shuffle(a, source_lane);
                        a = in.dpp_bound_ctrl ? b.sel(dpp_active, shuffled, b.uconst(0))
                                              : shuffled;
                    } else {
                        a = b.uconst(0);
                    }
                } else {
                if (b.is_fragment) {
                    if (in.opcode != 0x01) { ok = false; return true; }
                    a = b.dpp_quad(a, in.dpp_ctrl);
                } else if (b.is_compute) {
                    // DPP transforms SRC0 before the VOP1 operation. v_mov and the numeric
                    // v_cvt_u32_f32 used by UE light-grid kernels are lane-pure afterward.
                    if (in.opcode != 0x01 && in.opcode != 0x07) { ok = false; return true; }
                    a = b.subgroup_quad_permute(a, in.dpp_ctrl);
                } else {
                    ok = false; return true;
                }
                }
            }
            // SDWA float source modifiers on src0 (abs then neg). v_cvt_f32_f16 applies them after
            // selecting and unpacking the requested half below; applying them to the packed u32 as
            // though it were an f32 changes both the value and the selected sign bit.
            if (in.opcode != 0x0B) {
                if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
                if (in.src_neg[0]) a = b.fbin(Op_FSub, b.uconst(0), a);
            }
            if ((in.opcode == 0x05 || in.opcode == 0x06) && in.sdwa_src0_sel <= 5) {
                const uint32_t bits = in.sdwa_src0_sel <= 3 ? 8u : 16u;
                const uint32_t offset = in.sdwa_src0_sel <= 3
                    ? 8u * in.sdwa_src0_sel : 16u * (in.sdwa_src0_sel - 4u);
                a = ((in.words[1] >> 19) & 1u)
                    ? b.bfe_s(a, b.uconst(offset), b.uconst(bits))
                    : b.bfe_u(a, b.uconst(offset), b.uconst(bits));
            }
            // SDWA BREV selects and zero-extends its byte/word before reversing the resulting
            // dword. The decoder admits only this unmodified full-destination subset, so neither
            // sign extension nor destination preservation can reach this path.
            if (in.opcode == 0x38 && in.has_sdwa && in.sdwa_src0_sel <= 5) {
                const uint32_t bits = in.sdwa_src0_sel <= 3 ? 8u : 16u;
                const uint32_t offset = in.sdwa_src0_sel <= 3
                    ? 8u * in.sdwa_src0_sel : 16u * (in.sdwa_src0_sel - 4u);
                a = b.bfe_u(a, b.uconst(offset), b.uconst(bits));
            }
            if (in.opcode == 0x02) {   // v_readfirstlane_b32: SGPR dst = value of the lowest active lane
                // Cross-lane broadcast. Our per-lane scalar model has no cross-lane reduction, so we use
                // THIS lane's value. SPECULATIVE(confidence: med): exact only when src0 is wave-uniform —
                // which is the standard use (reading a uniformly-computed VGPR into an SGPR, e.g. the
                // integer-divide reciprocal in the game's shaders). Writes an SGPR, not a VGPR.
                rs.sreg[in.dst.value] = a;
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == kVop1OpcodeMovreldB32) { // VGPR[VDST + M0] = SRC0
                // RDNA2 ISA sec. 6.6. M0 is a runtime unsigned VGPR offset. Represent the indexed
                // write as selects over every statically referenced destination at or above VDST;
                // an out-of-range destination is unobservable because no later instruction names it.
                // This preserves exact loop-carried SSA while avoiding an architectural 256-word
                // register array for shaders that never use relative source addressing.
                auto m0 = rs.sreg.find(124);
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                for (int reg = in.dst.value; reg <= rs.max_vgpr; ++reg) {
                    const uint32_t old = vreg_old(b, rs, reg);
                    rs.vreg[reg] = b.sel(
                        b.ucmp(Op_IEqual, m0->second,
                               b.uconst(static_cast<uint32_t>(reg - in.dst.value))),
                        a, old);
                    predicate_write(b, rs, reg, old);
                }
                return true;
            }
            uint32_t& d = vreg[in.dst.value];
            // WORD-select v_mov_b32_sdwa (#273): extract the selected 16-bit source half and insert it
            // into the selected dest half, preserving the other (the f16 half-move; decode accepted
            // only dst WORD_0/1 + PRESERVE with src DWORD/WORD_0/WORD_1).
            if (in.opcode == 0x01 && in.sdwa_dst_sel != 6) {
                uint32_t v = a;
                if (in.sdwa_src0_sel == 5)      v = b.ibin(Op_ShiftRightLogical, a, b.uconst(16));
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
                const uint32_t offset = in.sdwa_src0_sel <= 3
                    ? 8u * in.sdwa_src0_sel : 16u * (in.sdwa_src0_sel - 4u);
                const uint32_t width = in.sdwa_src0_sel <= 3 ? 8u : 16u;
                d = in.sdwa_src0_sext ? b.bfe_s(a, b.uconst(offset), b.uconst(width))
                                      : b.bfe_u(a, b.uconst(offset), b.uconst(width));
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // Packed unary f16 SDWA: select one source half, compute in f32, round back to f16, and
            // insert while preserving the opposite destination half. The admitted opcode set is the
            // whole VOP1 f16 unary family (see emit_f16_unary), matching the plain e32 lowering
            // below; the decoder gates the operand shape (WORD dst + UNUSED_PRESERVE, WORD/DWORD
            // source, no sext/neg/abs/clamp/omod).
            if (vop1_is_f16_unary(in.opcode) && in.sdwa_dst_sel != 6) {
                uint32_t raw = a;   // inline constants: f16-width encoding, not the f32 pattern
                if (in.src[0].kind == OperandKind::InlineFloat)
                    raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                                   << (in.sdwa_src0_sel == 5 ? 16 : 0));
                else if (in.src[0].kind == OperandKind::InlineInt)
                    raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                uint32_t x = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
                uint32_t result = emit_f16_unary(b, in.opcode, x);
                if (!result) { ok = false; return true; }
                uint32_t r16 = b.pack_half_lo(result);
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // PLAIN-form unary f16 transcendentals/rounders (no SDWA word select). The SDWA WORD
            // form of this family is handled above; the e32 form reaches here and used to reject —
            // Sonic Racing: CrossWorlds' post chain hits `v_rcp_f16_e32` (0x7e1ea910) three times
            // per boot (#2013). Every op in this family reads S0.f16 and writes D.f16_lo, so they
            // share one lowering: unpack the half, compute in f32, round once back to f16.
            // Bits [31:16] are PRESERVED (the gfx10 16-bit-VALU contract); only the SDWA
            // DWORD+UNUSED_PAD encoding zero-fills, which the decoder marks with has_sdwa.
            // VERIFIED(round-trip llvm-mc gfx1030, VOP1): 0x54 v_rcp_f16, 0x55 v_sqrt_f16,
            // 0x56 v_rsq_f16, 0x57 v_log_f16, 0x58 v_exp_f16, 0x5B v_floor_f16, 0x5C v_ceil_f16,
            // 0x5D v_trunc_f16, 0x5E v_rndne_f16, 0x5F v_fract_f16, 0x60 v_sin_f16, 0x61 v_cos_f16.
            // Trig input is in REVOLUTIONS, exactly as for the f32 forms. CONFIDENCE: HIGH.
            // PLAIN e32 ONLY (`!has_sdwa`). A DWORD-select SDWA form of one of these carries
            // modifiers this lowering does not model, and each would be dropped SILENTLY rather
            // than fail-visibly: the decoder's "trivial-or-modifier-only" admission at
            // rdna2_decode.cpp:64 gates neither src0 NEG/ABS nor DST_UNUSED, the generic VOP1
            // abs/neg at the top of this case applies them in the f32 domain to the packed dword
            // (wrong half for a 16-bit read), and UNUSED_SEXT would need sign extension rather than
            // the zero fill `has_sdwa` implies. CLAMP/OMOD likewise apply in the 16-bit domain.
            // Every live encoding this title uses is plain `_e32` (`7e1ea910`, `7e08a504`), so
            // rejecting the SDWA forms costs nothing and is strictly no worse than the whole-opcode
            // reject they got before. (Raised in review of PR #2067.)
            if (vop1_is_f16_unary(in.opcode) &&
                !in.has_sdwa && in.sdwa_dst_sel == 6 && !in.clamp && !in.omod) {
                uint32_t raw = a;   // inline constants: f16-width encoding, not the f32 pattern
                if (in.src[0].kind == OperandKind::InlineFloat)
                    raw = b.uconst(inline_float_f16_bits(in.src[0].value));
                else if (in.src[0].kind == OperandKind::InlineInt)
                    raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                // Plain e32 has no source modifiers and no half select; the SDWA forms that would
                // carry them are rejected by the `!has_sdwa` gate above.
                const uint32_t x = b.unpack_half(raw, 0);
                const uint32_t result = emit_f16_unary(b, in.opcode, x);
                if (!result) { ok = false; return true; }
                const uint32_t r16 = b.pack_half_lo(result);
                d = b.ibin(Op_BitwiseOr,
                           b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // 16-bit integer <-> f16 converts (plain e32 form). CrossWorlds' post chain hits
            // v_cvt_u16_f16 at ten sites per boot (#2013); the other three are the same family and
            // the same wrapper. All four read S0's low 16 bits (WORD_1 under an SDWA source select)
            // and write D's low 16 bits, PRESERVING bits [31:16].
            //   0x50 v_cvt_f16_u16   D.f16 = f16(S0.u16)
            //   0x51 v_cvt_f16_i16   D.f16 = f16(S0.i16)
            //   0x52 v_cvt_u16_f16   D.u16 = u16(S0.f16), truncated toward zero and CLAMPED
            //   0x53 v_cvt_i16_f16   D.i16 = i16(S0.f16), truncated toward zero and CLAMPED
            // The 16-bit clamp is explicit: cvt_f2u/cvt_f2i saturate at the 32-bit boundary, which
            // is the wrong boundary here, and a bare mask would WRAP an out-of-range value instead
            // of saturating it. VERIFIED(round-trip llvm-mc gfx1030, VOP1 0x50-0x53; the live
            // encoding 7e08a504 is `v_cvt_u16_f16_e32 v4, v4`). CONFIDENCE: HIGH.
            // PLAIN e32 ONLY — see the `!has_sdwa` note on the transcendental block above.
            // The 16-bit convert family, in the plain e32 form (no SDWA control word at all) or in
            // the WORD-destination SDWA form the decoder admits: dst WORD_0/1 + UNUSED_PRESERVE
            // from a full-DWORD source with no modifier. A DWORD-destination SDWA form still
            // rejects, because its control word can carry src0 NEG/ABS that the generic VOP1 path
            // would apply in the f32 domain to the packed dword — the wrong half for an op that
            // reads bits[15:0].
            const bool convert16_word_sdwa =
                in.has_sdwa && in.sdwa_dst_sel != 6 && in.sdwa_src0_sel >= 4 && in.sdwa_src0_sel <= 6;
            if (in.opcode >= 0x50 && in.opcode <= 0x53 &&
                (!in.has_sdwa || convert16_word_sdwa) && !in.clamp && !in.omod) {
                const bool from_float = in.opcode >= 0x52;
                uint32_t raw = a;   // inline constants: 16-bit-width encoding, not the f32 pattern
                if (in.src[0].kind == OperandKind::InlineFloat)
                    raw = b.uconst(inline_float_f16_bits(in.src[0].value));
                else if (in.src[0].kind == OperandKind::InlineInt)
                    raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                else if (in.sdwa_src0_sel == 5)
                    // SDWA src0_sel WORD_1: the 16-bit operand is bits [31:16]. DWORD and WORD_0
                    // both name [15:0], which is where the reads below already look.
                    raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
                uint32_t r16;
                if (from_float) {
                    // Clamping the INTEGER result is correct and sufficient here, and an outer
                    // float-domain clamp would be redundant work on a hot path. `cvt_f2u` and
                    // `cvt_f2i` are not thin OpConvertFToU/OpConvertFToS wrappers: they are
                    // prosper's own saturating helpers (#135/#686, defined above in this file).
                    // Both select NaN to 0 BEFORE converting, bound the operand so the conversion
                    // itself is always defined, and saturate at the 32-bit rail. So every f16 a
                    // source can hold is already mapped to a defined 32-bit value, and the only
                    // thing left to do is narrow that rail from 32 to 16 bits.
                    const uint32_t x = b.unpack_half(raw, 0);
                    r16 = in.opcode == 0x52
                              ? b.uext2(Glsl_UMin, b.cvt_f2u(x), b.uconst(0xFFFFu))
                              : b.sext2(Glsl_SMin,
                                        b.sext2(Glsl_SMax, b.cvt_f2i(x), b.uconst(0xFFFF8000u)),
                                        b.uconst(0x7FFFu));
                } else {
                    const uint32_t word = b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
                    r16 = b.pack_half_lo(in.opcode == 0x50
                                             ? b.cvt_u2f(word)
                                             : b.cvt_i2f(b.bfe_s(word, b.uconst(0), b.uconst(16))));
                }
                r16 = b.ibin(Op_BitwiseAnd, r16, b.uconst(0xFFFFu));
                // dst_sel 6 (plain e32) and 4 (WORD_0) both write bits [15:0]; 5 writes [31:16].
                // Either way the opposite half is preserved.
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // v_cvt_f16_f32_sdwa inserts the converted half into the selected destination word and
            // preserves the other word (unlike the plain form, whose result occupies the low half).
            if (in.opcode == 0x0A && in.sdwa_dst_sel != 6) {
                uint32_t r16 = b.pack_half_lo(a);
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // v_cvt_i32_f32_sdwa / v_cvt_u32_f32_sdwa convert the full source, then insert the
            // selected low/high result word while preserving the other destination word. The
            // decoder accepts only this exact WORD + UNUSED_PRESERVE + DWORD-source subset. The
            // two opcodes differ only in the signedness of the 32-bit conversion; the inserted
            // half is its low 16 bits either way, so they share this lowering.
            if ((in.opcode == 0x07 || in.opcode == 0x08) && in.sdwa_dst_sel != 6) {
                const uint32_t result = in.opcode == 0x07 ? b.cvt_f2u(a) : b.cvt_f2i(a);
                const uint32_t word = b.ibin(Op_BitwiseAnd, result, b.uconst(0xFFFFu));
                d = in.sdwa_dst_sel == 5
                    ? b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, word, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), word);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            switch (in.opcode) {
                case 0x00: return true;                              // v_nop — no-op (writes nothing; common
                                                                     // scheduling/hazard filler in real shaders)
                case 0x01: d = a; break;                              // v_mov_b32
                case 0x05: d = b.cvt_i2f(a); break;                   // v_cvt_f32_i32
                case 0x06: d = b.cvt_u2f(a); break;                   // v_cvt_f32_u32
                case 0x07: d = b.cvt_f2u(a); break;                   // v_cvt_u32_f32
                case 0x08: d = b.cvt_f2i(a); break;                   // v_cvt_i32_f32
                // f16<->f32 converts (#273 — DOLL's title post PSes carry f16 intermediates):
                // v_cvt_f16_f32 packs into the LOW half (high bits zero); v_cvt_f32_f16 unpacks the
                // low half. VERIFIED(round-trip llvm-mc gfx1030: 0x0a/0x0b).
                case 0x0A:
                    // Plain-form 16-bit results write D.f16_lo and PRESERVE bits [31:16] (the gfx10
                    // 16-bit-VALU contract; zero-fill is only the SDWA DWORD+UNUSED_PAD behavior,
                    // which the decoder marks with has_sdwa — the accepted cvt SDWA subsets all use
                    // WORD dsts and take the preserve branch above, so has_sdwa here means PAD).
                    d = in.has_sdwa
                        ? b.pack_half_lo(a)
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                                 b.pack_half_lo(a));
                    break;                                          // v_cvt_f16_f32
                case 0x0B: {                                        // v_cvt_f32_f16 (SDWA may select high half)
                    // An inline constant in a 16-bit operand position supplies its f16-width
                    // encoding (float) or raw two's-complement bits (int) — NOT the f32 pattern
                    // val() materializes (unpacking THAT low half turned inline 1.0 into 0.0).
                    uint32_t raw = a;
                    if (in.src[0].kind == OperandKind::InlineFloat)
                        raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                                       << (in.sdwa_src0_sel == 5 ? 16 : 0));
                    else if (in.src[0].kind == OperandKind::InlineInt)
                        raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
                    d = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
                    if (in.src_abs[0]) d = b.fext1(Glsl_FAbs, d);
                    if (in.src_neg[0]) d = b.fneg(d);
                    break;
                }
                case 0x0C: {                                        // v_cvt_rpi_i32_f32
                    // AMD RDNA2 ISA: D.i = (int)floor(S0.f + 0.5), i.e. nearest integer with
                    // halfway cases rounded toward +infinity (not ceil for every fraction).
                    // The guide does not restate NaN/out-of-range handling for this specialized
                    // form, so use the adjacent f32->i32 conversion's deterministic convention:
                    // NaN -> 0 and saturation at INT_MIN/INT_MAX. cvt_f2i also keeps the SPIR-V
                    // conversion operand representable on every backend.
                    //
                    // Only GTA V's exact plain e32 form is established here. SDWA/DPP and source
                    // or output modifiers remain fail-visible instead of being silently ignored.
                    // CONFIDENCE: MED (rounding formula HIGH; exceptional-value policy follows the
                    // established adjacent architectural conversion because this opcode omits it).
                    if (in.has_sdwa || in.has_dpp || in.src_abs[0] || in.src_neg[0] ||
                        in.clamp || in.omod) {
                        ok = false;
                        break;
                    }
                    // Do not literally add 0.5 in f32 before floor: the float immediately below
                    // 0.5 would double-round to 1.0. Split off the fractional part, then apply the
                    // >= tie rule; for integral large magnitudes the fraction is already zero.
                    const uint32_t lower = b.fext1(Glsl_Floor, a);
                    const uint32_t fraction = b.fbin(Op_FSub, a, lower);
                    const uint32_t round_up = b.fcmp(
                        Op_FOrdGreaterThanEqual, fraction, b.uconst(fbits(0.5f)));
                    const uint32_t rounded = b.fbin(
                        Op_FAdd, lower,
                        b.sel(round_up, b.uconst(fbits(1.0f)), b.uconst(fbits(0.0f))));
                    d = b.cvt_f2i(rounded);
                    break;
                }
                case 0x0D:                                          // v_cvt_flr_i32_f32
                    // AMD RDNA2 ISA: floor the f32 value before converting to signed i32. This is
                    // observably different from v_cvt_i32_f32's truncation for negative fractions.
                    d = b.cvt_f2i(b.fext1(Glsl_Floor, a));
                    break;
                // v_cvt_off_f32_i4: sign-extend the low 4-bit integer and scale by 1/16.
                // AMD RDNA2 ISA: "4-bit signed int to 32-bit float"; LLVM's intrinsic
                // contract specifies result = 0.0625f * src_i4. This is the only opcode
                // that blocked The Messenger's 1024x32 grading-LUT producer (#527).
                case 0x0E:
                    d = b.fbin(Op_FMul,
                               b.cvt_i2f(b.bfe_s(a, b.uconst(0), b.uconst(4))),
                               b.uconst(fbits(0.0625f)));
                    break;
                // AMD RDNA2 ISA: select one unsigned byte from the source dword and convert it
                // directly to f32. The opcode number selects BYTE_0 through BYTE_3.
                case 0x11: case 0x12: case 0x13: case 0x14:
                    d = b.cvt_u2f(b.bfe_u(a,
                                          b.uconst(8u * (in.opcode - 0x11u)),
                                          b.uconst(8)));
                    break;
                case 0x20: d = b.fext1(Glsl_Fract, a); break;         // v_fract_f32
                case 0x21: d = b.fext1(Glsl_Trunc, a); break;         // v_trunc_f32
                case 0x22: d = b.fext1(Glsl_Ceil, a); break;          // v_ceil_f32
                case 0x23: d = b.fext1(Glsl_RoundEven, a); break;     // v_rndne_f32 (round to nearest even)
                case 0x24: d = b.fext1(Glsl_Floor, a); break;         // v_floor_f32
                case 0x25: d = b.fext1(Glsl_Exp2, a); break;          // v_exp_f32 (2^x)
                case 0x27: d = b.fext1(Glsl_Log2, a); break;          // v_log_f32 (log2)
                case 0x2A: d = b.frcp(a); break;                      // v_rcp_f32
                case 0x2B: d = b.frcp(a); break;                      // v_rcp_iflag_f32 (~= v_rcp_f32)
                case 0x2E: d = b.fext1(Glsl_InverseSqrt, a); break;   // v_rsq_f32
                case 0x33: d = b.fext1(Glsl_Sqrt, a); break;          // v_sqrt_f32
                // v_sin_f32 (0x35) / v_cos_f32 (0x36): the RDNA trig input is in REVOLUTIONS (units of
                // 2π radians) — the compiler pre-multiplies by 1/2π (0.15915494) before these, so
                // d = sin/cos(2π·src). VERIFIED(round-trip llvm-mc gfx1010: 0x7e026b02/0x7e026d02 →
                // v_sin/cos_f32; DOLL's dither PS does exactly `v_mul 0.15915494, x` → `v_cos`).
                // CONFIDENCE: HIGH (RDNA2 ISA V_SIN_F32: D = sin(S0 * 2π)).
                case 0x35: d = b.fext1(Glsl_Sin, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f)))); break;
                case 0x36: d = b.fext1(Glsl_Cos, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f)))); break;
                case 0x37: d = b.iun(Op_Not, a); break;               // v_not_b32
                case 0x38: d = b.iun(Op_BitReverse, a); break;        // v_bfrev_b32
                case kVop1OpcodeFfbhU32: {                            // v_ffbh_u32 (plain e32)
                    // AMD RDNA2 ISA: count the zeroes preceding the first set bit from the MSB;
                    // return -1 when no bit is set. GLSL.std.450 FindUMsb instead returns the SET
                    // BIT INDEX (and all bits set at zero). OR bit zero in before calling it (which
                    // leaves every nonzero input's highest bit unchanged), convert index -> count,
                    // then explicitly select the architectural zero sentinel. GTA V uses the exact in-place
                    // e32 packets 7e087304 / 7e047302 in its compute culling kernels.
                    //
                    // Keep SDWA out of this admission. The decoder's generic DWORD SDWA path can
                    // carry NEG/ABS, which the VOP1 prologue above applies in the f32 domain; that
                    // is not an integer source modifier and would silently change FFBH's operand.
                    // DPP reaches and rejects in the stage-specific source-transform block above.
                    if (in.has_sdwa || in.has_dpp) { ok = false; break; }
                    d = b.ffbh_u32(a);
                    break;
                }
                case kVop1OpcodeFfblB32: {                            // v_ffbl_b32 (plain e32)
                    // GTA V's Wave32 terrain dispatcher reaches the exact 0x7e047515 packet at
                    // pc106. Like FFBH above this integer scan has no valid SDWA/DPP admission:
                    // accepting either would let the common VOP1 modifier path reinterpret the
                    // source as float before scanning its bits.
                    if (in.has_sdwa || in.has_dpp) { ok = false; break; }
                    d = b.ffbl_b32(a);
                    break;
                }
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
                    if (m0it == rs.sreg.end()) { ok = false; break; }
                    uint32_t acc = b.uconst(0);
                    const int base = in.src[0].value;
                    for (const auto& kv : vreg) {
                        if (kv.first < base || !kv.second) continue;   // (!kv.second: the dst slot the
                                                                       // enclosing `vreg[dst]` may have
                                                                       // default-inserted — no value yet)
                        acc = b.sel(b.ucmp(Op_IEqual, m0it->second, b.uconst((uint32_t)(kv.first - base))),
                                    kv.second, acc);
                    }
                    d = acc; break;
                }
                default: ok = false;
            }
            // SDWA output modifiers: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result
            // opcodes only (mirrors the VOP2/VOP3 fresult path; DOLL VS: `v_exp_f32_sdwa … clamp`).
            // A modifier on a non-float-result op would silently drop — reject loudly instead.
            if (ok && (in.omod || in.clamp)) switch (in.opcode) {
                case 0x05: case 0x06: case 0x0B: case 0x0E: case 0x11: case 0x12: case 0x13: case 0x14:
                case 0x20: case 0x21: case 0x22: case 0x23: case 0x24:
                case 0x25: case 0x27: case 0x2A: case 0x2B: case 0x2E: case 0x33: case 0x35: case 0x36:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.clamp01(d);
                    break;
                default: ok = false; break;
            }
            if (ok) {
                if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
            }
            return true;
        }
        case Rdna2Format::VOP2: {
            uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t old_d = vreg_old(b, rs, in.dst.value);
            uint32_t dpp_active = 0;
            // The non-dispatch GTA V cohort uses the exact {1,2,4,8} ROW_SHR reduction ladder.
            // FI=0 makes an EXEC-inactive source invalid, BC0 preserves VDST at the row
            // edge/invalid source, and the destination write remains independently
            // EXEC-predicated. Other shift amounts remain limited to the event-isolated CFG
            // dispatcher below.
            const bool compute_inplace_add_row_shr = b.is_compute &&
                is_inplace_vadd_nc_u32_dpp_row_shr(in) &&
                (in.dpp_ctrl == 0x111u || in.dpp_ctrl == 0x112u ||
                 in.dpp_ctrl == 0x114u || in.dpp_ctrl == 0x118u);
            if (compute_inplace_add_row_shr) {
                uint32_t valid_source = 0;
                const uint32_t shifted = b.subgroup_row_shr(
                    a, rs.exec, in.dpp_ctrl - 0x110u, &valid_source);
                const uint32_t result = b.ibin(Op_IAdd, a, shifted);
                vreg[in.dst.value] = b.sel(valid_source, result, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            if (b.is_compute && is_vadd_nc_u32_dpp_partial_row(in)) {
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1u));
                const uint32_t row = b.ibin(
                    Op_ShiftRightLogical, lane, b.uconst(4));
                const uint32_t row_bit = b.ibin(
                    Op_ShiftLeftLogical, b.uconst(1), row);
                const uint32_t row_selected = b.ucmp(
                    Op_INotEqual,
                    b.ibin(Op_BitwiseAnd, row_bit, b.uconst(in.dpp_row_mask)),
                    b.uconst(0));
                const uint32_t result = b.ibin(Op_IAdd, a, c);
                vreg[in.dst.value] = b.sel(row_selected, result, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            // DPP16 quad_perm on src0 (#273): fragment FLOAT ops and compute quad swaps.  Bounded
            // ROW_SHR in the proven one-live-lane NGG projection supplies zero for lane 0.
            if (in.has_dpp) {
                const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
                uint32_t row_xor = 0;
                const bool row_ror8 = dpp_row_xor_ctrl(in.dpp_ctrl, &row_xor);
                const bool fop = in.opcode == 0x03 || in.opcode == 0x04 || in.opcode == 0x05 ||
                                 in.opcode == 0x08 || in.opcode == 0x0F || in.opcode == 0x10 ||
                                 in.opcode == 0x1F || in.opcode == 0x2B;
                if (row_ror8) {
                    // The decoder admits only GTA V's full-mask, BC1, FI0 MIN/MAX packets here.
                    // Direct subgroup shuffle is valid only when one native subgroup is exactly one
                    // guest wave; portable/default-subgroup compute uses CFG scratch below.
                    if (!b.is_compute || dpp_row_ror8_op(in) == DppRowRor8Op::None ||
                        !b.native_subgroup_size) {
                        ok = false; return true;
                    }
                    uint32_t valid_source = 0;
                    const uint32_t rotated =
                        b.subgroup_row_xor(a, rs.exec, row_xor, &valid_source);
                    a = b.sel(valid_source, rotated, b.uconst(0));
                } else if (row_shr) {
                    if (b.is_vertex) {
                        if (!b.ngg_one_lane || in.opcode != 0x25 || !in.dpp_bound_ctrl) {
                            ok = false; return true;
                        }
                        a = b.uconst(0);
                    } else if (b.is_fragment) {
                        // Unbounded ROW_SHR is a pure SOURCE transform: DPP16 rewrites SRC0 only,
                        // and an out-of-row or EXEC-inactive source disables the instruction,
                        // preserving the old destination even when VDST and the two sources are
                        // distinct. Both halves are opcode-independent, so every VOP2 whose only
                        // architectural result is VDST lowers identically — Astro's material mask
                        // uses v_or_b32, Syberia's Forward+ light scalarization uses the
                        // v_min_u32 ROW_SHR:{1,2,4,8} wave reduction, and a per-opcode allow-list
                        // silently drops every draw using any other one. The carry trio
                        // (v_add_co_ci/v_sub_co_ci/v_subrev_co_ci) ALSO writes VCC, and a disabled
                        // lane would have to preserve its old VCC bit too; the epilogue below only
                        // restores VDST, so keep exactly that shape fail-visible.
                        //
                        // The generalization additionally requires `allow_wave`, which is exactly
                        // "this emitter guarantees the whole guest wave is observable here". The
                        // structured fragment shell passes it (every scalar branch and loop
                        // condition there is subgroup-uniform, so all lanes reach this PC
                        // together); the per-invocation CFG dispatcher passes false, because
                        // adjacent lanes can be parked at DIFFERENT static DPP instructions and a
                        // plain row shuffle would consume a neighbour published by another one.
                        // That dispatcher lowers the exact V_MIN_U32 reduction itself, with a
                        // static event tag carried through the identical shuffle. V_OR_B32 keeps
                        // its historical unconditional admission so no stage that compiles today
                        // starts failing; widening the dispatcher's coverage is separate work.
                        const bool writes_carry_out = in.opcode == 0x28 ||
                                                      in.opcode == 0x29 || in.opcode == 0x2A;
                        const bool wave_visible = allow_wave || in.opcode == 0x1c;
                        if (writes_carry_out || !wave_visible || in.dpp_bound_ctrl) {
                            ok = false; return true;
                        }
                        a = b.subgroup_row_shr(
                            a, rs.exec, in.dpp_ctrl - 0x110u, &dpp_active);
                    } else {
                        if (!b.is_compute || !fop) { ok = false; return true; }
                        // ROW_SHR:N reads SRC0 from lane-N inside each architectural 16-lane row.
                        // BOUND_CTRL=1 substitutes zero before the row; BOUND_CTRL=0 disables the
                        // instruction for those lanes, preserving the old destination exactly.
                        b.mark_subgroup_min16();
                        const uint32_t shift = in.dpp_ctrl - 0x110u;
                        const uint32_t lane = b.subgroup_local_id();
                        const uint32_t row_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(15));
                        dpp_active = b.ucmp(Op_UGreaterThanEqual, row_lane, b.uconst(shift));
                        // Never issue a shuffle with an out-of-range unsigned lane after subtraction;
                        // inactive lanes address themselves and are then zeroed or masked off.
                        const uint32_t source_lane = b.sel(
                            dpp_active, b.ibin(Op_ISub, lane, b.uconst(shift)), lane);
                        const uint32_t shuffled = b.subgroup_shuffle(a, source_lane);
                        a = in.dpp_bound_ctrl ? b.sel(dpp_active, shuffled, b.uconst(0))
                                              : shuffled;
                    }
                } else {
                    if (!fop) { ok = false; return true; }
                    if (b.is_fragment) {
                        a = b.dpp_quad(a, in.dpp_ctrl);
                    } else if (b.is_compute) {
                        // DPP transforms src0 only; src1 remains in this lane. A general shuffle covers
                        // all 256 architectural quad_perm tables, including broadcasts and duplicates.
                        a = b.subgroup_quad_permute(a, in.dpp_ctrl);
                    } else {
                        ok = false; return true;
                    }
                }
            }
            // SDWA float source modifiers (only ever set on float ops by the assembler): abs then neg.
            // Packed-f16 ops apply these after selecting/unpacking the half below.
            const bool packed_f16 = in.opcode == 0x32 || in.opcode == 0x33 || in.opcode == 0x35 ||
                                    in.opcode == 0x39 || in.opcode == 0x3A;
            const bool integer_sdwa = in.has_sdwa &&
                (in.opcode == 0x0B || (in.opcode >= 0x11 && in.opcode <= 0x14) ||
                 in.opcode == 0x16 || in.opcode == 0x18 ||
                 (in.opcode >= 0x1A && in.opcode <= 0x1E) ||
                 (in.opcode >= 0x25 && in.opcode <= 0x2A));
            if (integer_sdwa) {
                // SEXT sign-extends the selected field instead of zero-extending it (#2013's
                // `v_add_nc_u32_sdwa v4, 8, sext(v5) src1_sel:WORD_0`). The decoder only sets the
                // flag alongside a real sub-dword select, so a DWORD source never reaches bfe_s.
                auto select = [&](uint32_t raw, uint8_t sel, bool sext) {
                    if (sel > 5u) return raw;
                    const uint32_t offset = sel <= 3u ? 8u * sel : 16u * (sel - 4u);
                    const uint32_t width = sel <= 3u ? 8u : 16u;
                    return sext ? b.bfe_s(raw, b.uconst(offset), b.uconst(width))
                                : b.bfe_u(raw, b.uconst(offset), b.uconst(width));
                };
                a = select(a, in.sdwa_src0_sel, in.sdwa_src0_sext);
                c = select(c, in.sdwa_src1_sel, in.sdwa_src1_sext);
            } else if (!packed_f16) {
                if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
                if (in.src_neg[0]) a = b.fneg(a);
                if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
                if (in.src_neg[1]) c = b.fneg(c);
            }
            uint32_t& d = vreg[in.dst.value];
            switch (in.opcode) {
                case 0x01: {                                         // v_cndmask_b32: dst = vcc ? src1 : src0
                    if (!vcc) { ok = false; return true; }
                    if (in.sdwa_dst_sel == 6) { d = b.sel(vcc, c, a); break; }
                    auto word = [&](uint32_t raw, uint8_t sel) {
                        if (sel == 5) raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
                        return b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
                    };
                    uint32_t selected = b.sel(vcc, word(c, in.sdwa_src1_sel), word(a, in.sdwa_src0_sel));
                    d = in.sdwa_dst_sel == 5
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, selected, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), selected);
                    break;
                }
                case 0x03: d = b.fbin(Op_FAdd, a, c); break;          // v_add_f32
                case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
                case 0x05: d = b.fbin(Op_FSub, c, a); break;          // v_subrev_f32 (src1 - src0; e32 form of
                                                                      // VOP3 0x105 — round-trip llvm-mc gfx1010 0x0a020702)
                case 0x08: d = b.fbin(Op_FMul, a, c); break;          // v_mul_f32
                case 0x0B: {                                        // v_mul_u32_u24
                    // Only the low 24 bits of each source participate; the result is the low
                    // 32 bits of the unsigned product (AMD RDNA2 ISA 11.6).
                    const uint32_t mask = b.uconst(0x00FFFFFFu);
                    d = b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, a, mask),
                                          b.ibin(Op_BitwiseAnd, c, mask));
                    break;
                }
                // v_min/v_max: hardware returns the OTHER operand when exactly one input is NaN
                // (ISA 12.7 ops 15/16: "if (S0 == NaN) D = S1; ..."), which is GLSL NMin/NMax.
                // Plain FMin/FMax are NaN-UNDEFINED — llvmpipe propagates a second-operand NaN, so
                // the ubiquitous max(min(x,K),lo) clamp idiom could yield NaN pixels where hardware
                // yields the bound.
                case 0x0F: d = b.fext2(Glsl_NMin, a, c); break;       // v_min_f32
                case 0x10: d = b.fext2(Glsl_NMax, a, c); break;       // v_max_f32
                case 0x11: d = b.sext2(Glsl_SMin, a, c); break;       // v_min_i32
                case 0x12: d = b.sext2(Glsl_SMax, a, c); break;       // v_max_i32
                case 0x13: d = b.uext2(Glsl_UMin, a, c); break;       // v_min_u32
                case 0x14: d = b.uext2(Glsl_UMax, a, c); break;       // v_max_u32
                case 0x16: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_lshrrev_b32
                             d = b.ibin(Op_ShiftRightLogical, c, sh); break; }       // dst = src1 >> (src0 & 31)
                case 0x18: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_ashrrev_i32
                             d = b.sbin(Op_ShiftRightArithmetic, c, sh); break; }    // dst = src1 >>a (src0 & 31)
                case 0x1A: { uint32_t sh = b.ibin(Op_BitwiseAnd, a, b.uconst(31));   // v_lshlrev_b32
                             d = b.ibin(Op_ShiftLeftLogical, c, sh); break; }        // dst = src1 << (src0 & 31)
                case 0x1B: d = b.ibin(Op_BitwiseAnd, a, c); break;    // v_and_b32
                case 0x1C: d = b.ibin(Op_BitwiseOr,  a, c); break;    // v_or_b32
                case 0x1D: d = b.ibin(Op_BitwiseXor, a, c); break;    // v_xor_b32
                case 0x1E: d = b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)); break; // v_xnor_b32
                case 0x25: d = b.ibin(Op_IAdd, a, c); break;          // v_add_nc_u32
                case 0x26: d = b.ibin(Op_ISub, a, c); break;          // v_sub_nc_u32
                case 0x27: d = b.ibin(Op_ISub, c, a); break;          // v_subrev_nc_u32 (reverse: src1 - src0)
                // Carry ops (VOP2 e32 form): carry-in + carry-out are VCC. v_add_co_ci(0x28)/
                // v_sub_co_ci(0x29)/v_subrev_co_ci(0x2a). Mirrors the VOP3B 0x128/129/12A logic with VCC.
                case 0x28: case 0x29: case 0x2A: {
                    if (!vcc) { ok = false; return true; }
                    uint32_t cin = b.sel(vcc, b.uconst(1), b.uconst(0));
                    uint32_t carry;
                    if (in.opcode == 0x28) {                          // (a + c) + cin
                        uint32_t s1 = b.ibin(Op_IAdd, a, c); uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);
                        d = b.ibin(Op_IAdd, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);
                        carry = b.bsel(k1, b.btrue(), k2);
                    } else {                                          // (x - y) - cin  (subrev swaps)
                        uint32_t x = in.opcode == 0x29 ? a : c, y = in.opcode == 0x29 ? c : a;
                        uint32_t s1 = b.ibin(Op_ISub, x, y); uint32_t k1 = b.ucmp(Op_ULessThan, x, y);
                        d = b.ibin(Op_ISub, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, s1, cin);
                        carry = b.bsel(k1, b.btrue(), k2);
                    }
                    // Carry-out masks follow ISA 3.9 like compare results: an EXEC-inactive lane's
                    // VCC bit is written 0, not its raw carry (wave votes must not see phantom bits).
                    vcc = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
                    break;
                }
                // v_mac_f32 (0x1f) / v_fmac_f32 (0x2b): dst = src0*src1 + dst (accumulate into the dest).
                // mac vs fmac differ only in fused rounding — immaterial here. old_d = the dst accumulator.
                // NOTE(opcode ID): op 0x1f is v_mac_f32 on the PS5's ISA. v_mac_f32 was REMOVED on desktop
                // RDNA2/gfx1030 (where 0x1f is invalid and llvm-mc reads canonical v_dot2c at 0x02), but the
                // PS5 GPU retains the RDNA1/gfx1010 encoding — VERIFIED by round-tripping the scene VS's
                // actual op-0x1f word 0x3e261221 through `llvm-mc -mcpu=gfx1010` → `v_mac_f32_e32`. The VOP3
                // (e64) form of this same op is 0x11f, handled in the VOP3 switch below.
                case 0x1F: case 0x2B: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), old_d); break;
                // The four mul-add-with-literal-K ops (K = in.literal). madmk/fmamk = src0*K + src1;
                // madak/fmaak = src0*src1 + K. (mad vs fma differ only in fused rounding — immaterial here.)
                case 0x20: case 0x2C: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, b.uconst(in.literal)), c); break;  // v_madmk / v_fmamk
                case 0x21: case 0x2D: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), b.uconst(in.literal)); break;  // v_madak / v_fmaak
                case 0x2F: d = b.pack_half2x16_rtz(a, c); break;      // v_cvt_pkrtz_f16_f32 (e32 form): RTZ clamp (#452)
                case 0x3C: {                                         // v_pk_fmac_f16: packed dst += src0*src1
                    auto fmac_half = [&](uint32_t half) {
                        return b.fbin(Op_FAdd,
                                      b.fbin(Op_FMul, b.unpack_half(a, half), b.unpack_half(c, half)),
                                      b.unpack_half(old_d, half));
                    };
                    uint32_t lo = b.pack_half_lo(fmac_half(0));
                    uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(fmac_half(1)), b.uconst(16));
                    d = b.ibin(Op_BitwiseOr, lo, hi);
                    break;
                }
                case 0x32: case 0x33: case 0x35: case 0x39: case 0x3A: { // v_add/sub/mul/max/min_f16
                    // (SDWA WORD_1 = high 16; DWORD/WORD_0 = low 16 — an f16 op reads bits[15:0]);
                    // f16xf16 products are exact in f32, so multiply in f32 and round once to f16.
                    // The 16-bit result inserts into the selected dest half PRESERVING the other
                    // (dst_sel WORD_1 for the SDWA pack idiom; DWORD/WORD_0 = the plain e32 form's
                    // "write [15:0], preserve [31:16]" gfx10 f16-VOP2 contract). #273 (DOLL box-blur).
                    auto source = [&](uint32_t raw, const Operand& operand, uint8_t sel, int k) {
                        // An inline constant in a 16-bit operand carries its f16-width encoding
                        // (float — exact even for 1/(2*pi)) or raw two's-complement bits (int:
                        // inline 1 is the f16 denormal 0x0001, never 1.0). Register/scalar operands
                        // carry packed halves and must be selected before abs/neg.
                        uint32_t v = operand.kind == OperandKind::InlineFloat
                                       ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
                                   : operand.kind == OperandKind::InlineInt
                                       ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)),
                                                       sel == 5 ? 1 : 0)
                                       : b.unpack_half(raw, sel == 5 ? 1 : 0);
                        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                        if (in.src_neg[k]) v = b.fneg(v);
                        return v;
                    };
                    uint32_t x = source(a, in.src[0], in.sdwa_src0_sel, 0);
                    uint32_t y = source(c, in.src[1], in.sdwa_src1_sel, 1);
                    uint32_t p = in.opcode == 0x32 ? b.fbin(Op_FAdd, x, y)
                               : in.opcode == 0x33 ? b.fbin(Op_FSub, x, y)
                               : in.opcode == 0x35 ? b.fbin(Op_FMul, x, y)
                               : in.opcode == 0x39 ? b.fext2(Glsl_NMax, x, y)   // NaN -> other operand
                                                   : b.fext2(Glsl_NMin, x, y);
                    if (in.clamp) p = b.clamp01(p);
                    uint32_t r16 = b.pack_half_lo(p);
                    // dst_sel==6 covers TWO encodings the decoder now distinguishes: the SDWA
                    // DWORD+UNUSED_PAD form (zero-fill, has_sdwa) and the plain e32 form, whose
                    // gfx10 contract writes [15:0] and PRESERVES [31:16] (the comment above always
                    // said so; the code zero-filled both, corrupting live packed high halves).
                    d = in.sdwa_dst_sel == 6
                        ? (in.has_sdwa
                               ? r16
                               : b.ibin(Op_BitwiseOr,
                                        b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16))
                      : (in.sdwa_dst_sel == 5)
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
                    break;
                }
                default: ok = false;
            }
            // SDWA sub-dword DESTINATION for the integer VOP2 ops (#2013). The operation itself ran
            // on the (already sub-dword-selected) sources above and produced a full dword; DST_SEL
            // says which byte/word of the register receives its low bits, and DST_UNUSED says what
            // happens to the rest — UNUSED_PAD (0) zero-fills it, UNUSED_PRESERVE (2) keeps the old
            // register bits. (Sonic Racing: CrossWorlds' `v_and_b32_sdwa v2, 6, v0 dst_sel:WORD_0
            // dst_unused:UNUSED_PRESERVE`; the decoder admits only PAD/PRESERVE, so UNUSED_SEXT
            // still rejects.) Without this the whole result landed in bits [31:0], silently
            // clobbering the half the guest asked to preserve.
            if (ok && integer_sdwa && in.sdwa_dst_sel != 6) {
                const uint32_t width  = in.sdwa_dst_sel <= 3 ? 8u : 16u;
                const uint32_t offset = in.sdwa_dst_sel <= 3 ? 8u * in.sdwa_dst_sel
                                                             : 16u * (in.sdwa_dst_sel - 4u);
                const uint32_t field_mask = ((1u << width) - 1u) << offset;
                const uint32_t placed = b.ibin(
                    Op_ShiftLeftLogical,
                    b.ibin(Op_BitwiseAnd, d, b.uconst((1u << width) - 1u)), b.uconst(offset));
                d = in.sdwa_dst_unused == 2u
                    ? b.ibin(Op_BitwiseOr,
                             b.ibin(Op_BitwiseAnd, old_d, b.uconst(~field_mask)), placed)
                    : placed;
            }
            // SDWA output modifier: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result opcodes
            // only (int ops never carry omod). Mirrors the VOP3 fresult path; a no-op when omod/clamp unset.
            if (ok && (in.omod || in.clamp) && !packed_f16) switch (in.opcode) {
                case 0x03: case 0x04: case 0x05: case 0x08: case 0x0F: case 0x10:
                case 0x1F: case 0x2B: case 0x20: case 0x2C: case 0x21: case 0x2D:
                    if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
                    else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
                    else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
                    if (in.clamp) d = b.clamp01(d);
                    break;
                // A non-float-result opcode carrying a modifier (e.g. an INTEGER SDWA op with CLAMP =
                // integer saturation) is not modeled by the float-domain omod/clamp above, so applying
                // nothing would SILENTLY drop the saturation and emit a valid-but-wrong shader. Reject
                // loudly instead — the same fail-visibly-over-miscompile discipline as the forward-if
                // clamp (#129/#174). The guard means this only fires for a modifier-carrying op.
                default: ok = false; break;
            }
            if (ok) {
                if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
                predicate_write(b, rs, in.dst.value, old_d);
            }
            return true;
        }
        case Rdna2Format::VOPC: {                                     // v_cmp_* -> VCC; v_cmpx_* also -> EXEC
            const auto scalar_data_operand = [&](const Operand& source) {
                switch (source.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::SGPR:
                        return rs.sreg.contains(source.value) ||
                            rs.sreg_input.contains(source.value);
                    case OperandKind::Special:
                        if (source.value == 125) return true; // SGPR_NULL
                        if (source.value == 253) return rs.scc != 0;
                        return source.value >= 106 && source.value <= 124 &&
                            rs.sreg.contains(source.value);
                    default:
                        return false;
                }
            };
            const uint32_t ra = val(in.src[0]), rc = val(in.src[1]);  // raw bits (f16 compares re-derive)
            uint32_t a = ra, c = rc;
            uint32_t op = in.opcode;
            bool is_cmpx = vopc_is_cmpx(op);
            if (is_cmpx && !allow_exec_update) { ok = false; return true; }
            uint32_t eff = is_cmpx ? op - 0x10 : op;
            const bool integer64_compare =
                (eff >= 0xA1u && eff <= 0xA6u) ||
                (eff >= 0xE1u && eff <= 0xE6u);
            auto scalar_integer64_operand = [&](const Operand& source) {
                if (!scalar_data_operand(source)) return false;
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special && source.value >= 106 &&
                     source.value <= 123)) {
                    Operand high = source;
                    ++high.value;
                    return scalar_data_operand(high);
                }
                // Inline integers sign-extend, while literals and SGPR_NULL zero-extend.  Each
                // implied high half is a compile-time scalar.  Inline floats are rejected later
                // by integer64_halves and therefore cannot publish a proof from this path.
                return source.kind != OperandKind::InlineFloat;
            };
            const bool compare_wave_uniform = !is_cmpx && !rs.exec_narrowed &&
                (integer64_compare
                    ? scalar_integer64_operand(in.src[0]) &&
                      scalar_integer64_operand(in.src[1])
                    : scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1]));
            // Integer compares have no ABS/NEG source semantics. A malformed e64 packet carrying
            // those bits must not be lowered as an unmodified integer operation.
            if (integer64_compare &&
                (in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1])) {
                ok = false;
                return true;
            }
            // Float source modifiers (abs then neg — hardware order), set only on FLOAT compares by the
            // assembler (VOP3-encoded e64 or SDWA forms; e.g. DOLL's `v_cmp_gt_f32_sdwa vcc, |v5|, s4`).
            if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
            if (in.src_neg[0]) a = b.fneg(a);
            if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
            if (in.src_neg[1]) c = b.fneg(c);
            // v_cmpx_* shares each type's compare set at base+0x10. It writes EXEC in addition to
            // VCC. Map to the base compare, then narrow; vopc_is_cmpx covers the f32/f64,
            // i32/i64, and u32/u64 windows.
            const bool integer_compare =
                (eff >= 0x81u && eff <= 0x86u) ||
                (eff >= 0x89u && eff <= 0x8Eu) ||
                (eff >= 0xA9u && eff <= 0xAEu) ||
                (eff >= 0xC1u && eff <= 0xC6u);
            if (integer_compare) {
                auto sdwa_integer = [&](uint32_t raw, uint8_t sel) {
                    if (sel <= 3u) return b.bfe_u(raw, b.uconst(8u * sel), b.uconst(8));
                    if (sel <= 5u) return b.bfe_u(raw, b.uconst(16u * (sel - 4u)), b.uconst(16));
                    return raw;
                };
                a = sdwa_integer(ra, in.sdwa_src0_sel);
                c = sdwa_integer(rc, in.sdwa_src1_sel);
            }
            struct Integer64Halves { uint32_t lo = 0, hi = 0; };
            auto integer64_halves = [&](const Operand& operand, uint32_t lo) {
                Integer64Halves result{lo, b.uconst(0)};
                if (operand.kind == OperandKind::VGPR || operand.kind == OperandKind::SGPR ||
                    (operand.kind == OperandKind::Special && operand.value >= 106 &&
                     operand.value <= 123)) {
                    Operand high = operand;
                    ++high.value;
                    result.hi = val(high);
                } else if (operand.kind == OperandKind::InlineInt) {
                    result.hi = b.uconst(operand.value < 0 ? 0xffffffffu : 0u);
                } else if (operand.kind == OperandKind::Literal ||
                           (operand.kind == OperandKind::Special && operand.value == 125)) {
                    // Integer/untyped 32-bit literals and SGPR_NULL zero-extend in a B64 source.
                    result.hi = b.uconst(0);
                } else {
                    // In particular, an inline floating-point constant is a 64-bit DOUBLE operand
                    // in a 64-bit compare, not an integer pair assembled from its f32 bit pattern.
                    ok = false;
                }
                return result;
            };
            Integer64Halves a64, c64;
            if (integer64_compare) {
                a64 = integer64_halves(in.src[0], ra);
                c64 = integer64_halves(in.src[1], rc);
            }
            uint32_t cmp = 0;
            switch (eff) {
                case 0x00: cmp = b.bfalse(); break;                              // v_cmp_f_f32
                case 0x01: cmp = b.fcmp(Op_FOrdLessThan, a, c); break;         // v_cmp_lt_f32
                case 0x02: cmp = b.fcmp(Op_FOrdEqual, a, c); break;            // v_cmp_eq_f32
                case 0x03: cmp = b.fcmp(Op_FOrdLessThanEqual, a, c); break;    // v_cmp_le_f32
                case 0x04: cmp = b.fcmp(Op_FOrdGreaterThan, a, c); break;      // v_cmp_gt_f32
                case 0x05: cmp = b.fcmp(Op_FOrdNotEqual, a, c); break;         // v_cmp_lg_f32
                case 0x06: cmp = b.fcmp(Op_FOrdGreaterThanEqual, a, c); break; // v_cmp_ge_f32
                case 0x07: {                                                    // v_cmp_o_f32
                    const uint32_t a_ordered = b.fcmp(Op_FOrdEqual, a, a);
                    const uint32_t c_ordered = b.fcmp(Op_FOrdEqual, c, c);
                    cmp = b.land(a_ordered, c_ordered);
                    break;
                }
                case 0x08: {                                                    // v_cmp_u_f32
                    const uint32_t a_nan = b.fcmp(Op_FUnordNotEqual, a, a);
                    const uint32_t c_nan = b.fcmp(Op_FUnordNotEqual, c, c);
                    cmp = b.lor(a_nan, c_nan);
                    break;
                }
                // NaN-inclusive f32 compares (the "n"-prefix set is the unordered negation of 0x1-0x6):
                case 0x09: cmp = b.fcmp(Op_FUnordLessThan, a, c); break;       // v_cmp_nge_f32 = !(a>=b)
                case 0x0A: cmp = b.fcmp(Op_FUnordEqual, a, c); break;          // v_cmp_nlg_f32 = !(a!=b)
                case 0x0B: cmp = b.fcmp(Op_FUnordLessThanEqual, a, c); break;  // v_cmp_ngt_f32 = !(a>b)
                case 0x0C: cmp = b.fcmp(Op_FUnordGreaterThan, a, c); break;    // v_cmp_nle_f32 = !(a<=b)
                case 0x0D: cmp = b.fcmp(Op_FUnordNotEqual, a, c); break;       // v_cmp_neq_f32 = !(a==b)
                case 0x0E: cmp = b.fcmp(Op_FUnordGreaterThanEqual, a, c); break;// v_cmp_nlt_f32 = !(a<b)
                case 0x0F: cmp = b.btrue(); break;                              // v_cmp_tru_f32
                case 0x81: cmp = b.scmp(Op_SLessThan, a, c); break;            // v_cmp_lt_i32
                case 0x82: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_i32
                case 0x83: cmp = b.scmp(Op_SLessThanEqual, a, c); break;       // v_cmp_le_i32
                case 0x84: cmp = b.scmp(Op_SGreaterThan, a, c); break;         // v_cmp_gt_i32
                case 0x85: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_i32 (sign-agnostic)
                case 0x86: cmp = b.scmp(Op_SGreaterThanEqual, a, c); break;    // v_cmp_ge_i32
                // 16-bit integer compares (i16 0x89-0x8e, u16 0xa9-0xae) read bits [15:0] of each
                // source; the i16 window sign-extends and the u16 window zero-extends, after which
                // the ordinary 32-bit relational operators give the exact 16-bit answer. Both
                // windows carry the same six operations in the same order (lt/eq/le/gt/ne/ge), so
                // one shared body indexes them by the low three bits.
                // VERIFIED(round-trip llvm-mc gfx1030): 0x89 is v_cmp_lt_i16 and 0xab is
                // v_cmp_le_u16 — the encoding Sonic Racing: CrossWorlds emits (#2013).
                case 0x89: case 0x8A: case 0x8B: case 0x8C: case 0x8D: case 0x8E:
                case 0xA9: case 0xAA: case 0xAB: case 0xAC: case 0xAD: case 0xAE: {
                    const bool narrow_signed = eff <= 0x8Eu;
                    const uint32_t x = narrow_signed ? b.bfe_s(a, b.uconst(0), b.uconst(16))
                                                     : b.bfe_u(a, b.uconst(0), b.uconst(16));
                    const uint32_t y = narrow_signed ? b.bfe_s(c, b.uconst(0), b.uconst(16))
                                                     : b.bfe_u(c, b.uconst(0), b.uconst(16));
                    switch (eff & 7u) {
                        case 1: cmp = narrow_signed ? b.scmp(Op_SLessThan, x, y)
                                                    : b.ucmp(Op_ULessThan, x, y); break;
                        case 2: cmp = b.ucmp(Op_IEqual, x, y); break;
                        case 3: cmp = narrow_signed ? b.scmp(Op_SLessThanEqual, x, y)
                                                    : b.ucmp(Op_ULessThanEqual, x, y); break;
                        case 4: cmp = narrow_signed ? b.scmp(Op_SGreaterThan, x, y)
                                                    : b.ucmp(Op_UGreaterThan, x, y); break;
                        case 5: cmp = b.ucmp(Op_INotEqual, x, y); break;
                        default: cmp = narrow_signed ? b.scmp(Op_SGreaterThanEqual, x, y)
                                                     : b.ucmp(Op_UGreaterThanEqual, x, y); break;
                    }
                    break;
                }
                case 0xA1: case 0xA2: case 0xA3: case 0xA4: case 0xA5: case 0xA6: {
                    const uint32_t lhs = b.u64_from_lohi(a64.lo, a64.hi);
                    const uint32_t rhs = b.u64_from_lohi(c64.lo, c64.hi);
                    // Flipping the sign bit maps two's-complement signed order onto unsigned order.
                    // Keep the values in the declared u64 type: the generic `scmp` helper bitcasts
                    // 32-bit operands and therefore cannot be used for a 64-bit scalar.
                    const uint32_t ordered_lhs = b.u64_from_lohi(
                        a64.lo, b.ibin(Op_BitwiseXor, a64.hi, b.uconst(0x80000000u)));
                    const uint32_t ordered_rhs = b.u64_from_lohi(
                        c64.lo, b.ibin(Op_BitwiseXor, c64.hi, b.uconst(0x80000000u)));
                    switch (eff) {
                        case 0xA1: cmp = b.ucmp(Op_ULessThan, ordered_lhs, ordered_rhs); break; // v_cmp_lt_i64
                        case 0xA2: cmp = b.ucmp(Op_IEqual, lhs, rhs); break;               // v_cmp_eq_i64
                        case 0xA3: cmp = b.ucmp(Op_ULessThanEqual, ordered_lhs, ordered_rhs); break; // v_cmp_le_i64
                        case 0xA4: cmp = b.ucmp(Op_UGreaterThan, ordered_lhs, ordered_rhs); break; // v_cmp_gt_i64
                        case 0xA5: cmp = b.ucmp(Op_INotEqual, lhs, rhs); break;            // v_cmp_ne_i64
                        default:   cmp = b.ucmp(Op_UGreaterThanEqual, ordered_lhs, ordered_rhs); break; // v_cmp_ge_i64
                    }
                    break;
                }
                case 0x88: {                                                   // v_cmp_class_f32
                    // CLASS tests the raw IEEE-754 category rather than doing a floating-point
                    // comparison. Keep this entirely in the integer domain so signalling/quiet
                    // NaNs and the sign of zero/NaN survive on every SPIR-V target.
                    uint32_t class_raw = ra;
                    if (in.src_abs[0])
                        class_raw = b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7fffffffu));
                    if (in.src_neg[0])
                        class_raw = b.ibin(Op_BitwiseXor, class_raw, b.uconst(0x80000000u));
                    const uint32_t sign = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x80000000u)), b.uconst(0));
                    const uint32_t exponent =
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7f800000u));
                    const uint32_t mantissa =
                        b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x007fffffu));
                    const uint32_t exponent_zero = b.ucmp(Op_IEqual, exponent, b.uconst(0));
                    const uint32_t exponent_all = b.ucmp(Op_IEqual, exponent, b.uconst(0x7f800000u));
                    const uint32_t mantissa_zero = b.ucmp(Op_IEqual, mantissa, b.uconst(0));
                    const uint32_t quiet_nan = b.ucmp(
                        Op_INotEqual,
                        b.ibin(Op_BitwiseAnd, mantissa, b.uconst(0x00400000u)), b.uconst(0));

                    // AMD's mask order is sNaN, qNaN, -Inf, -normal, -subnormal, -zero,
                    // +zero, +subnormal, +normal, +Inf. Select the input's one-hot class bit,
                    // then test it against SRC1 (Astro's live packet uses 3 = either NaN).
                    const uint32_t nan_class = b.sel(quiet_nan, b.uconst(2), b.uconst(1));
                    const uint32_t inf_class = b.sel(sign, b.uconst(4), b.uconst(512));
                    const uint32_t exp_all_class = b.sel(mantissa_zero, inf_class, nan_class);
                    const uint32_t zero_class = b.sel(sign, b.uconst(32), b.uconst(64));
                    const uint32_t subnormal_class = b.sel(sign, b.uconst(16), b.uconst(128));
                    const uint32_t exp_zero_class = b.sel(mantissa_zero, zero_class, subnormal_class);
                    const uint32_t normal_class = b.sel(sign, b.uconst(8), b.uconst(256));
                    const uint32_t class_bit = b.sel(
                        exponent_all, exp_all_class,
                        b.sel(exponent_zero, exp_zero_class, normal_class));
                    cmp = b.ucmp(Op_INotEqual,
                                 b.ibin(Op_BitwiseAnd, rc, class_bit), b.uconst(0));
                    break;
                }
                case 0xC1: cmp = b.ucmp(Op_ULessThan, a, c); break;            // v_cmp_lt_u32
                case 0xC2: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_u32
                case 0xC3: cmp = b.ucmp(Op_ULessThanEqual, a, c); break;       // v_cmp_le_u32
                case 0xC4: cmp = b.ucmp(Op_UGreaterThan, a, c); break;         // v_cmp_gt_u32
                case 0xC5: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_u32
                case 0xC6: cmp = b.ucmp(Op_UGreaterThanEqual, a, c); break;    // v_cmp_ge_u32
                case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: case 0xE6: {
                    const uint32_t lhs = b.u64_from_lohi(a64.lo, a64.hi);
                    const uint32_t rhs = b.u64_from_lohi(c64.lo, c64.hi);
                    switch (eff) {
                        case 0xE1: cmp = b.ucmp(Op_ULessThan, lhs, rhs); break;             // v_cmp_lt_u64
                        case 0xE2: cmp = b.ucmp(Op_IEqual, lhs, rhs); break;                // v_cmp_eq_u64
                        case 0xE3: cmp = b.ucmp(Op_ULessThanEqual, lhs, rhs); break;        // v_cmp_le_u64
                        case 0xE4: cmp = b.ucmp(Op_UGreaterThan, lhs, rhs); break;          // v_cmp_gt_u64
                        case 0xE5: cmp = b.ucmp(Op_INotEqual, lhs, rhs); break;             // v_cmp_ne_u64
                        default:   cmp = b.ucmp(Op_UGreaterThanEqual, lhs, rhs); break;     // v_cmp_ge_u64
                    }
                    break;
                }
                // f16 compares (0xC8-0xCF; cmpx at +0x10 = 0xD8-0xDF folds here too — DOLL's title
                // post PSes: `v_cmp_lt_f16_sdwa s6, 0, v7`). VERIFIED(round-trip llvm-mc gfx1030:
                // v_cmp_lt/eq/le/gt/lg/ge_f16 = VOPC 0xC9-0xCE). The f16 value lives in the source's
                // LOW half — unpack to f32 and compare there (exact: f16 order-embeds into f32); an
                // inline FLOAT constant is already an f32 value, so it is used directly. abs/neg
                // modifiers are applied after conversion (equivalent, conversion is monotone/exact).
                case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: {
                    auto f16v = [&](int k, uint32_t raw) -> uint32_t {
                        // Inline constants supply their 16-bit operand encoding: floats the f16
                        // pattern (0x3C00 for 1.0), ints raw two's-complement low bits (1 -> the
                        // f16 denormal 0x0001). Reading the raw f32-pattern (float) or the value as
                        // full-width f32 BITS (int) modeled a different number than hardware.
                        uint32_t v = in.src[k].kind == OperandKind::InlineFloat
                                         ? b.unpack_half(b.uconst(inline_float_f16_bits(in.src[k].value)), 0)
                                   : in.src[k].kind == OperandKind::InlineInt
                                         ? b.unpack_half(b.uconst(static_cast<uint32_t>(in.src[k].value)), 0)
                                         : b.unpack_half(raw, in.sdwa_src0_sel == 5u && k == 0 ? 1u :
                                                              in.sdwa_src1_sel == 5u && k == 1 ? 1u : 0u);
                        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                        if (in.src_neg[k]) v = b.fneg(v);
                        return v;
                    };
                    uint32_t ha = f16v(0, ra), hc = f16v(1, rc);
                    switch (eff) {
                        case 0xC9: cmp = b.fcmp(Op_FOrdLessThan, ha, hc); break;          // v_cmp_lt_f16
                        case 0xCA: cmp = b.fcmp(Op_FOrdEqual, ha, hc); break;             // v_cmp_eq_f16
                        case 0xCB: cmp = b.fcmp(Op_FOrdLessThanEqual, ha, hc); break;     // v_cmp_le_f16
                        case 0xCC: cmp = b.fcmp(Op_FOrdGreaterThan, ha, hc); break;       // v_cmp_gt_f16
                        case 0xCD: cmp = b.fcmp(Op_FOrdNotEqual, ha, hc); break;          // v_cmp_lg_f16
                        default:   cmp = b.fcmp(Op_FOrdGreaterThanEqual, ha, hc); break;  // v_cmp_ge_f16
                    }
                    break;
                }
                default: ok = false;
            }
            // A compare result is a per-lane mask. The e64/SDWAB form can target an SGPR pair (SDST)
            // instead of VCC — track it in sreg_bool so a later v_cndmask_b32_e64 / s_cselect can read it.
            // Otherwise it writes VCC; keep VCC's narrowed-state in sync (106/107 = VCC_LO/HI).
            if (ok) {
                if (is_cmpx) {
                    // v_cmpx writes EXEC ONLY on gfx10 (EXEC &= cmp) — it has NO VCC/SGPR destination.
                    // The old shared handler fell into the `else` and set vcc = cmp for cmpx too,
                    // clobbering a VCC value kept live ACROSS the cmpx, so a later v_cndmask/s_cbranch_vccz/
                    // v_add_co_ci reading VCC got the compare mask instead of the real predicate (#464).
                    rs.exec = b.land(rs.exec, cmp); rs.exec_narrowed = true;
                } else {
                    // ISA 3.9: "VCC[n] = EXEC[n] & (test passed for thread n)" — a lane inactive in
                    // EXEC gets its mask bit forced to 0, never the raw test result. Masking at the
                    // write keeps wave reductions (vccz votes, mbcnt publications, v_writelane mask
                    // spills) from seeing phantom bits from inactive lanes. No-op when EXEC is full.
                    const uint32_t masked = rs.exec_narrowed ? b.land(rs.exec, cmp) : cmp;
                    if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
                        rs.sreg_bool[in.dst.value] = masked; rs.sreg_bool_narrowed[in.dst.value] = true;
                        if (b.allow_b32_masks &&
                            (b.is_fragment || (b.is_compute && b.wave_size == 32)))
                            rs.sreg_bool_b32.insert(in.dst.value);
                    } else if (b.allow_b32_masks && in.dst.kind == OperandKind::SGPR &&
                               (in.dst.value == 106 || in.dst.value == 107)) {
                        // Wave32 SDWA/e64 compares can explicitly target either one-word VCC half as
                        // an independent saved mask (Astro world-map PC1060 targets VCC_HI, then
                        // consumes it with s_andn2_b32). Keep that physical destination distinct;
                        // only VCC_LO is also the implicit condition used by vccz/cndmask forms.
                        rs.sreg_bool[in.dst.value] = masked;
                        rs.sreg_bool_narrowed[in.dst.value] = true;
                        rs.sreg_bool_b32.insert(in.dst.value);
                        rs.sreg.erase(in.dst.value);
                        rs.sreg_srt.erase(in.dst.value);
                        if (in.dst.value == 106) vcc = masked;
                    } else {
                        // VOPC replaces the architectural VCC predicate. In Wave32 the complete mask
                        // occupies VCC_LO only; VCC_HI remains available as ordinary scalar scratch
                        // (Astro's sibling traversal keeps a float there across an implicit compare).
                        // Wave64 still replaces both physical words. Retain the low-word mask marker
                        // so a following s_mov_b32 save sees the fresh predicate rather than stale
                        // dispatcher-loaded scalar data.
                        vcc = masked;
                        rs.vcc_wave_uniform = compare_wave_uniform ? masked : 0;
                        if (b.wave_size == 32) {
                            rs.sreg.erase(106);
                            rs.sreg_srt.erase(106);
                        } else {
                            mask_write_clobbers_pair(rs, 106);
                        }
                        rs.sreg_bool_narrowed[106] = true;
                        if (b.wave_size != 32) rs.sreg_bool_narrowed[107] = true;
                        if (b.allow_b32_masks &&
                            (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
                            rs.sreg_bool[106] = masked;
                            rs.sreg_bool_b32.insert(106);
                        }
                    }
                }
            }
            return true;
        }
        case Rdna2Format::VOP3: {
            // SCALAR-SPILL lane slots (#273): v_writelane_b32 (0x361) / v_readlane_b32 (0x360) with a
            // COMPILE-TIME lane index — the pack-scalars-into-a-VGPR's-lanes idiom (DOLL's big post PS
            // spills 19 s_buffer_load results into v36 and reads them back). Per-invocation each
            // (vgpr, lane) is a named wave-uniform scalar. An exact native compute wave also supports
            // a dynamic writelane selector by updating the one matching subgroup invocation in an
            // ordinary VGPR lifetime. The portable NGG vertex shell additionally projects an ordinary
            // VGPR read from any physical lane onto its one represented live lane; this is the same
            // collapsed-wave contract used for MBCNT and Function LDS. Other compute/fragment stages
            // use a native subgroup shuffle for ordinary VGPR reads and therefore support both scalar
            // and inline lane selectors. Neither op is EXEC-predicated on hardware, so no
            // predicate_write.
            // VERIFIED(round-trip llvm-mc gfx1030: 0xd761 v_writelane_b32 / 0xd760 v_readlane_b32).
            if (in.opcode == 0x361) {                                 // v_writelane_b32 vDST, sSRC, lane
                if (b.is_compute && (b.wave_size == 32 || b.wave_size == 64) &&
                    b.native_subgroup_size == b.wave_size &&
                    in.src[1].kind != OperandKind::InlineInt) {
                    // RDNA2 masks the scalar selector to the active wave width and replaces VDST
                    // only in that physical lane. Under the enforced exact-width native-subgroup
                    // contract, SubgroupLocalInvocationId is that architectural lane. The scalar
                    // source and selector are uniform, so this needs no shuffle or barrier. Astro's
                    // Wave32 and GTA V's Wave64 traversal kernels reach this form after readlane or
                    // a mask scan publishes the chosen hit as scalar data.
                    if (in.src[1].kind == OperandKind::VGPR ||
                        rs.vgpr_lane_slots.count(in.dst.value) ||
                        rs.vgpr_lane_mask_slots.count(in.dst.value)) {
                        ok = false; return true;
                    }
                    const uint32_t value = val(in.src[0]);
                    const uint32_t selector = val(in.src[1]);
                    if (!ok) return true;
                    const uint32_t lane = b.subgroup_local_id();
                    const uint32_t selected = b.ucmp(
                        Op_IEqual, lane,
                        b.ibin(Op_BitwiseAnd, selector, b.uconst(b.wave_size - 1u)));
                    rs.vreg[in.dst.value] = b.sel(
                        selected, value, vreg_old(b, rs, in.dst.value));
                    rs.invalidated_vgpr_lane_slots.erase(in.dst.value);
                    rs.vgpr_lane_slots.erase(in.dst.value);
                    rs.vgpr_lane_mask_slots.erase(in.dst.value);
                    return true;
                }
                if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 || in.src[1].value > 63) {
                    ok = false; return true;
                }
                const int lane = in.src[1].value;
                rs.invalidated_vgpr_lane_slots.erase(in.dst.value);
                uint32_t mask_value = 0;
                bool mask_source = false;
                const bool proven_half_alias =
                    b.wave64_mask_writelane_alias_pcs.contains(in.pc);
                if (proven_half_alias) {
                    const auto half = rs.sreg_wave64_mask_half.find(in.src[0].value);
                    if (half == rs.sreg_wave64_mask_half.end()) {
                        ok = false; return true;
                    }
                    mask_value = half->second;
                    mask_source = true;
                } else if (in.src[0].value == 126 || in.src[0].value == 127) {
                    mask_value = rs.exec; mask_source = true;
                } else if ((in.src[0].value == 106 || in.src[0].value == 107) &&
                           !rs.sreg.count(in.src[0].value)) {
                    if (b.is_compute && b.wave_size == 64 &&
                        b.native_subgroup_size == 64) {
                        // A physical VCC word has no runtime mask/data tag. The Wave64 half proof
                        // deliberately seeds only EXEC; until VCC is tied to the mask-domain MUST
                        // analysis, persisting this spill could resurrect a false Bool placeholder.
                        ok = false; return true;
                    }
                    mask_value = rs.vcc; mask_source = true;
                } else if (in.src[0].kind == OperandKind::SGPR &&
                           !(b.is_fragment && b.wave_size == 64 &&
                             (rs.sreg_bool.contains(in.src[0].value) ||
                              (in.src[0].value > 0 &&
                               rs.sreg_bool.contains(in.src[0].value - 1))))) {
                    // A Wave64 fragment V_WRITELANE consumes one physical scalar word, not the
                    // complete Bool alias rooted at the pair's low SGPR. Let val() materialize that
                    // exact ballot half lazily below. Other stages retain the mask-slot model.
                    auto saved = rs.sreg_bool.find(in.src[0].value);
                    if (saved != rs.sreg_bool.end()) { mask_value = saved->second; mask_source = true; }
                }
                rs.vreg.erase(in.dst.value);                            // spill-array lifetime
                if (mask_source) {
                    rs.vgpr_lane_mask_slots[in.dst.value][lane] = mask_value;
                    auto data = rs.vgpr_lane_slots.find(in.dst.value);
                    if (data != rs.vgpr_lane_slots.end()) {
                        data->second.erase(lane);
                        if (data->second.empty()) rs.vgpr_lane_slots.erase(data);
                    }
                } else {
                    rs.vgpr_lane_slots[in.dst.value][lane] = val(in.src[0]);
                    auto masks = rs.vgpr_lane_mask_slots.find(in.dst.value);
                    if (masks != rs.vgpr_lane_mask_slots.end()) {
                        masks->second.erase(lane);
                        if (masks->second.empty()) rs.vgpr_lane_mask_slots.erase(masks);
                    }
                }
                return true;
            }
            if (in.opcode == 0x360) {                                 // v_readlane_b32 sDST, vSRC, lane
                // V_READLANE is an architectural scalar write.  Drop an older dedicated half
                // before selecting this read's source domain; only the exact CFG-proven mask-slot
                // path below may publish a replacement.  record_scalar_write consequently knows
                // that map presence after emission describes this instruction, not stale state.
                expire_wave64_mask_half(rs, in.dst.value);
                if (b.is_fragment && b.wave_size == 32 &&
                    in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 32) {
                    ok = false; return true;
                }
                auto vit = rs.vgpr_lane_slots.find(in.src[0].value);
                auto mit = rs.vgpr_lane_mask_slots.find(in.src[0].value);
                const bool vertex_shell = !b.is_compute && !b.is_fragment;
                if (!vertex_shell && vit == rs.vgpr_lane_slots.end() &&
                    mit == rs.vgpr_lane_mask_slots.end()) {
                    auto source = rs.vreg.find(in.src[0].value);
                    if (source == rs.vreg.end()) { ok = false; return true; }
                    const uint32_t selector = val(in.src[1]);
                    if (!ok) return true;
                    uint32_t result = 0;
                    if (allow_wave && b.is_compute &&
                        b.native_subgroup_size != b.wave_size) {
                        // A workgroup-uniform site can model the guest wave exactly through
                        // scratch even when a Wave64 guest does not fit in the host subgroup.
                        result = b.guest_wave_readlane(source->second, selector);
                    } else {
                        if (b.wave_size == 64) b.mark_subgroup_min64();
                        else b.mark_subgroup_min32();
                        const uint32_t native_lane = b.subgroup_local_id();
                        const uint32_t wave_mask = b.uconst(b.wave_size - 1u);
                        const uint32_t wave_base = b.ibin(
                            Op_BitwiseAnd, native_lane, b.uconst(~(b.wave_size - 1u)));
                        const uint32_t source_lane = b.ibin(
                            Op_BitwiseOr, wave_base,
                            b.ibin(Op_BitwiseAnd, selector, wave_mask));
                        result = b.subgroup_shuffle(source->second, source_lane);
                    }
                    rs.sreg[in.dst.value] = result;
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    if (in.dst.value == 106) {
                        const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                        rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                        rs.sreg_bool_narrowed[106] = true;
                    }
                    return true;
                }
                const bool static_lane = in.src[1].kind == OperandKind::InlineInt &&
                                         in.src[1].value >= 0 && in.src[1].value <= 63;
                const bool dynamic_absent_ngg_peer = b.ngg_one_lane &&
                    in.src[1].kind == OperandKind::Special &&
                    in.src[1].value >= 106 && in.src[1].value <= 124;
                if (!static_lane && !dynamic_absent_ngg_peer) {
                    ok = false; return true;
                }
                const int lane = static_lane ? in.src[1].value : -1;
                if (mit != rs.vgpr_lane_mask_slots.end()) {
                    auto sit = mit->second.find(lane);
                    if (sit != mit->second.end()) {
                        const auto proven_half =
                            b.wave64_mask_readlane_half_for_pc.find(in.pc);
                        if (b.is_compute && b.wave_size == 64 &&
                            b.native_subgroup_size == 64 &&
                            proven_half != b.wave64_mask_readlane_half_for_pc.end()) {
                            // This is one scalar dword plus one Bool spelling of the SAME physical
                            // half, not a complete saved B64 predicate. End every overlapping full
                            // mask alias and retain the Bool only in the dedicated re-spill domain.
                            auto erase_full_mask = [&](int base) {
                                rs.sreg_bool.erase(base);
                                rs.sreg_bool_narrowed.erase(base);
                                rs.sreg_bool_b32.erase(base);
                            };
                            if (in.dst.value > 0) erase_full_mask(in.dst.value - 1);
                            erase_full_mask(in.dst.value);
                            if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
                            rs.sreg_wave64_mask_half[in.dst.value] = sit->second;
                            rs.sreg_wave64_mask_half_index[in.dst.value] = proven_half->second;
                            // Override even an existing scalar placeholder: a physical spill lane
                            // can be allocated in both dispatcher domains, and its uint Function
                            // variable is zero on mask-only paths.
                            rs.sreg[in.dst.value] = b.native_wave_ballot_half(
                                sit->second, proven_half->second);
                            // Two adjacent scalar words become one complete B64 predicate only in
                            // architectural LO/HI order. Combine their per-lane Bool spellings;
                            // they need not originate from the same historical EXEC value.
                            const int base = in.dst.value & ~1;
                            const auto low = rs.sreg_wave64_mask_half.find(base);
                            const auto high = rs.sreg_wave64_mask_half.find(base + 1);
                            const auto low_index =
                                rs.sreg_wave64_mask_half_index.find(base);
                            const auto high_index =
                                rs.sreg_wave64_mask_half_index.find(base + 1);
                            if (low != rs.sreg_wave64_mask_half.end() &&
                                high != rs.sreg_wave64_mask_half.end() &&
                                low_index != rs.sreg_wave64_mask_half_index.end() &&
                                high_index != rs.sreg_wave64_mask_half_index.end() &&
                                low_index->second == 0 && high_index->second == 1) {
                                const uint32_t lane = b.ibin(
                                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                                rs.sreg_bool[base] = b.bsel(
                                    b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                                    high->second, low->second);
                                rs.sreg_bool_narrowed[base] = true;
                            }
                        } else {
                            rs.sreg_wave64_mask_half.erase(in.dst.value);
                            rs.sreg_wave64_mask_half_index.erase(in.dst.value);
                            rs.sreg_bool[in.dst.value] = sit->second;
                            rs.sreg_bool_narrowed[in.dst.value] = true;
                        }
                        // The compute CFG dispatcher can persist a physical spill lane that is
                        // recycled between scalar-data and wave-mask lifetimes. It exposes both
                        // views after a block join; keep both destination domains when present so
                        // the statically typed consumer selects the representation it needs.
                        if (proven_half == b.wave64_mask_readlane_half_for_pc.end() &&
                            vit != rs.vgpr_lane_slots.end()) {
                            auto data = vit->second.find(lane);
                            if (data != vit->second.end()) rs.sreg[in.dst.value] = data->second;
                            else rs.sreg.erase(in.dst.value);
                        } else if (proven_half ==
                                   b.wave64_mask_readlane_half_for_pc.end()) {
                            rs.sreg.erase(in.dst.value);
                        }
                        rs.sreg_srt.erase(in.dst.value);
                        return true;
                    }
                }
                if (vit == rs.vgpr_lane_slots.end()) {
                    if (rs.invalidated_vgpr_lane_slots.count(in.src[0].value)) {
                        ok = false; return true;
                    }
                    // Fragment shaders execute in an enforced 64-lane Vulkan subgroup, so a static
                    // hardware v_readlane maps exactly to a subgroup shuffle. Unlike ordinary VALU,
                    // v_readlane ignores EXEC; do not mask the source or predicate the scalar write.
                    if (b.is_fragment && static_lane) {
                        rs.sreg[in.dst.value] = b.subgroup_shuffle(
                            val(in.src[0]), b.uconst(static_cast<uint32_t>(lane)));
                        rs.sreg_bool.erase(in.dst.value);
                        rs.sreg_bool_narrowed.erase(in.dst.value);
                        rs.sreg_srt.erase(in.dst.value);
                        return true;
                    }
                    // NGG's final wave-packing tail reads peer lanes (Astro Bot uses lanes 63 and 3)
                    // from an ordinary VGPR. A Vulkan vertex invocation models guest lane zero: a
                    // lane-zero read returns this invocation's source, while every absent peer reads
                    // zero. The dynamic form remains exact by selecting on its scalar lane index.
                    if (!b.ngg_one_lane) { ok = false; return true; }
                    const uint32_t source = val(in.src[0]);
                    uint32_t reads_self = 0;
                    if (static_lane) {
                        reads_self = lane == 0 ? b.btrue() : b.bfalse();
                    } else {
                        auto lane_value = rs.sreg.find(in.src[1].value);
                        if (lane_value == rs.sreg.end()) { ok = false; return true; }
                        reads_self = b.ucmp(Op_IEqual, lane_value->second, b.uconst(0));
                    }
                    rs.sreg[in.dst.value] = b.sel(reads_self, source, b.uconst(0));
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value);
                    return true;
                }
                auto sit = vit->second.find(lane);
                if (sit == vit->second.end()) { ok = false; return true; }          // slot never written
                rs.sreg[in.dst.value] = sit->second;   // dst field is the SGPR number (like readfirstlane)
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            if (in.opcode == 0x377 || in.opcode == 0x378) {            // v_permlane{x}16_b32
                // AMD RDNA ISA 12.12: SRC1:SRC2 is a packed table of sixteen four-bit lane
                // selectors. PERMLANE16 gathers within the current 16-lane row; PERMLANEX16 gathers
                // from the paired row (0<->1, 2<->3). These are untyped operations; every ordinary
                // VOP3 float modifier is reserved. The two overloaded OPSEL bits were retained by
                // the decoder as FI and BOUND_CTRL.
                const uint32_t reserved_opsel = (in.words[0] >> 13) & 3u;
                if ((!b.is_compute && !b.is_fragment) || reserved_opsel || in.clamp || in.omod ||
                    in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                    in.src[0].kind != OperandKind::VGPR ||
                    in.src[1].kind == OperandKind::VGPR || in.src[2].kind == OperandKind::VGPR) {
                    ok = false; return true;
                }
                if (in.opcode == 0x378) b.mark_subgroup_min32();
                else b.mark_subgroup_min16();
                uint32_t source_lane = 0;
                const uint32_t shuffled = b.subgroup_permlane16(
                    val(in.src[0]), val(in.src[1]), val(in.src[2]),
                    in.opcode == 0x378, &source_lane);
                uint32_t result = shuffled;
                if (!in.permlane_fetch_inactive) {
                    const uint32_t active_word = b.sel(rs.exec, b.uconst(1), b.uconst(0));
                    const uint32_t fetched_active = b.subgroup_shuffle(active_word, source_lane);
                    const uint32_t source_active = b.ucmp(
                        Op_INotEqual, fetched_active, b.uconst(0));
                    result = b.sel(source_active, shuffled,
                                   in.permlane_bound_ctrl ? b.uconst(0)
                                                          : vreg_old(b, rs, in.dst.value));
                }
                const uint32_t old_d = vreg_old(b, rs, in.dst.value);
                vreg[in.dst.value] = result;
                predicate_write(b, rs, in.dst.value, old_d);
                return true;
            }
            uint32_t old_d = vreg_old(b, rs, in.dst.value);
            // Float source with its VOP3 modifiers applied: abs then a sign-bit-exact negate
            // (hardware order neg(abs(x))). Returns raw bits (fbin/fext re-bitcast), so it's a
            // drop-in for val() in the FLOAT ops only — integer VOP3 ops keep val() (modifiers are
            // float-domain; assemblers don't set them on int ops).
            auto fv = [&](int k) -> uint32_t {
                uint32_t bits = val(in.src[k]);
                if (in.src_abs[k]) bits = b.fext1(Glsl_FAbs, bits);
                if (in.src_neg[k]) bits = b.fneg(bits);
                return bits;
            };
            // Output modifiers on a FLOAT result: OMOD scale (×2/×4/×0.5) then CLAMP saturate to [0,1]
            // (hardware order: clamp(omod(x))). Wrap each float op's result through this. Opcodes
            // that route through it mark the CLAMP bit consumed; a set CLAMP on any opcode that
            // does NOT (integer ops, cndmask, the pack family) means unmodeled INTEGER saturation
            // (ISA 6.5) or an unhandled combination — the dispatch tail rejects it fail-visibly
            // instead of silently dropping the saturation.
            bool clamp_routed = false;
            auto fresult = [&](uint32_t bits) -> uint32_t {
                clamp_routed = true;
                if (in.omod == 1)      bits = b.fbin(Op_FMul, bits, b.uconst(fbits(2.0f)));
                else if (in.omod == 2) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(4.0f)));
                else if (in.omod == 3) bits = b.fbin(Op_FMul, bits, b.uconst(fbits(0.5f)));
                if (in.clamp) bits = b.clamp01(bits);
                return bits;
            };
            if (in.opcode == 0x311) {
                // v_pack_b32_f16: D.u32 = {S1.f16, S0.f16} — S0's selected half becomes bits
                // [15:0] and S1's becomes [31:16]. OPSEL[k] picks which half of each source is
                // read; the destination is a whole dword, so OPSEL[3] has nothing to select.
                // This is the instruction that re-assembles the two halves the 16-bit convert
                // family produces, so it appears immediately behind them (Sonic Racing:
                // CrossWorlds, `d7110020,00020500` = `v_pack_b32_f16 v32, v0, v2`, #2013).
                // VERIFIED(round-trip llvm-mc gfx1030). It is a bit move, so ABS/NEG (which would
                // be f16 sign operations on the selected halves) and CLAMP/OMOD are not modeled
                // and reject rather than being silently dropped.
                if (in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1] ||
                    in.clamp || in.omod) {
                    ok = false;
                } else {
                    auto packed_half = [&](int k) {
                        const Operand& operand = in.src[k];
                        // OPSEL[k] selects a half of an inline constant exactly as it does of a
                        // register — see inline_16bit_operand_dword. Both inline branches used to
                        // ignore it, so `1.0` read 0x3c00 and inline `1` read 1 from the HIGH half
                        // where hardware reads 0 (#2119, kernel 32r14d).
                        const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
                        uint32_t inline_dword;
                        if (inline_16bit_operand_dword(operand, inline_dword))
                            return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
                        return b.bfe_u(val(operand), b.uconst(16u * sel), b.uconst(16));
                    };
                    vreg[in.dst.value] = b.ibin(
                        Op_BitwiseOr, packed_half(0),
                        b.ibin(Op_ShiftLeftLogical, packed_half(1), b.uconst(16)));
                }
            } else if (in.opcode == 0x34B || in.opcode == 0x351 ||
                in.opcode == 0x354 || in.opcode == 0x357) {
                // Scalar f16 VOP3: one selected half from each packed source and one independently
                // selected destination half. The opposite destination half is preserved. The
                // `v_max3_f16` occurs eight times in Syberia's nonzero 960x544 gameplay composite
                // dispatch; llvm-mc gfx1030 identifies opcode 0x354 exactly, and the same round trip
                // names its two siblings 0x351 v_min3_f16 and 0x357 v_med3_f16 — Sonic Racing:
                // CrossWorlds' upscale kernel emits `d7571003,02020af2`
                // = `v_med3_f16 v3, 1.0, v5, 0 op_sel:[0,1,0,0]` (#2013).
                auto f16src = [&](int k) {
                    const Operand& operand = in.src[k];
                    uint32_t raw = val(operand);
                    // Inline constants supply the dword their operand width produces (see
                    // inline_16bit_operand_dword) and OPSEL[k] selects a half of THAT — a float
                    // reads 0.0h from the high half rather than the constant again (#2119,
                    // kernel 32r14c).
                    const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
                    uint32_t inline_dword;
                    uint32_t value = inline_16bit_operand_dword(operand, inline_dword)
                                       ? b.unpack_half(b.uconst(inline_dword), sel)
                                       : b.unpack_half(raw, sel);
                    if (in.src_abs[k]) value = b.fext1(Glsl_FAbs, value);
                    if (in.src_neg[k]) value = b.fneg(value);
                    return value;
                };
                uint32_t result = 0;
                if (in.opcode == 0x34B) {                            // v_fma_f16
                    result = b.fbin(Op_FAdd, b.fbin(Op_FMul, f16src(0), f16src(1)), f16src(2));
                } else if (in.opcode == 0x351) {                     // v_min3_f16
                    // ISA op 849 is a plain nested min — `V_MIN_F16(V_MIN_F16(S0,S1),S2)` — with
                    // NO any-NaN clause of the kind op 855 v_med3_f16 carries. NMin already
                    // matches V_MIN_F16's own one-NaN rule (return the other operand), so the
                    // bare chain is the whole contract. CONFIDENCE: HIGH.
                    result = b.fext2(Glsl_NMin,
                                     b.fext2(Glsl_NMin, f16src(0), f16src(1)), f16src(2));
                } else if (in.opcode == 0x354) {                     // v_max3_f16
                    // ISA op 852, likewise: `V_MAX_F16(V_MAX_F16(S0,S1),S2)`, no NaN clause.
                    result = b.fext2(Glsl_NMax,
                                     b.fext2(Glsl_NMax, f16src(0), f16src(1)), f16src(2));
                } else {                                             // v_med3_f16
                    // ISA op 855 opens with a SEPARATE any-NaN clause, exactly as op 343
                    // v_med3_f32 does (see 0x157 below — same structure, only the width differs):
                    //   if (isNan(S0) || isNan(S1) || isNan(S2)) D.f16 = V_MIN3_F16(S0,S1,S2)
                    // The plain max(min(...)) median formula does NOT reproduce that fallback on
                    // its own, so the clause is emitted explicitly. It is not academic: Sonic
                    // Racing: CrossWorlds' NaN-safe saturate idiom `v_med3_f16 v3, 1.0, v5, 0`
                    // (#2013) with S1 = NaN must yield MIN3(1.0,NaN,0) = 0.0h, where the bare
                    // formula yields 1.0h — black rendered as white. Kernel 32r10n pins it.
                    // The min/max legs use NaN-aware NMin/NMax because V_MIN_F16/V_MAX_F16 return
                    // the OTHER operand when exactly one input is NaN. CONFIDENCE: HIGH.
                    const uint32_t x = f16src(0), y = f16src(1), z = f16src(2);
                    const uint32_t mn = b.fext2(Glsl_NMin, x, y), mx = b.fext2(Glsl_NMax, x, y);
                    const uint32_t med = b.fext2(Glsl_NMax, mn, b.fext2(Glsl_NMin, mx, z));
                    const uint32_t min3 = b.fext2(Glsl_NMin, mn, z);
                    const uint32_t nan_any = b.lor(b.fcmp(Op_FUnordNotEqual, x, x),
                                                   b.lor(b.fcmp(Op_FUnordNotEqual, y, y),
                                                         b.fcmp(Op_FUnordNotEqual, z, z)));
                    result = b.sel(nan_any, min3, med);
                }
                result = fresult(result);
                uint32_t r16 = b.pack_half_lo(result);
                vreg[in.dst.value] = (in.vop3p_opsel & 8u)
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
            } else if ((in.opcode >= 0x303 && in.opcode <= 0x30E && in.opcode != 0x306) ||
                       in.opcode == 0x314 || in.opcode == 0x340 || in.opcode == 0x35E ||
                       in.opcode == 0x352 || in.opcode == 0x353 || in.opcode == 0x355 ||
                       in.opcode == 0x356 || in.opcode == 0x358 || in.opcode == 0x359) {
                // The 16-bit integer VALU family. Sonic Racing: CrossWorlds' post chain rejects on
                // v_lshrrev_b16 at thirty sites and v_add_nc_u16 at twenty-four per boot (#2013);
                // the rest of the family is the same lowering behind the same half-select wrapper.
                // Every one of them selects each source half with OPSEL[k] and the destination half
                // with OPSEL[3], PRESERVING the opposite destination half.
                //   0x303 v_add_nc_u16   D.u16 = S0.u16 + S1.u16
                //   0x304 v_sub_nc_u16   D.u16 = S0.u16 - S1.u16
                //   0x305 v_mul_lo_u16   D.u16 = (S0.u16 * S1.u16) & 0xffff
                //   0x307 v_lshrrev_b16  D.u16 = S1.u16 >> S0.u[3:0]      (REVERSED operands)
                //   0x308 v_ashrrev_i16  D.i16 = S1.i16 >> S0.u[3:0]      (REVERSED operands)
                //   0x309/0x30a/0x30b/0x30c  v_max_u16 / v_max_i16 / v_min_u16 / v_min_i16
                //   0x30d/0x30e  v_add_nc_i16 / v_sub_nc_i16 (identical bit result to the u16 forms)
                //   0x314 v_lshlrev_b16  D.u16 = S1.u16 << S0.u[3:0]      (REVERSED operands)
                //   0x340/0x35e  v_mad_u16 / v_mad_i16   D = S0*S1 + S2 (identical 16-bit result)
                //   0x352/0x353/0x355/0x356/0x358/0x359
                //                v_min3_i16 / v_min3_u16 / v_max3_i16 / v_max3_u16 /
                //                v_med3_i16 / v_med3_u16
                // VERIFIED(round-trip llvm-mc gfx1030, every opcode above; the live encoding
                // `d7074001,00020081` disassembles to `v_lshrrev_b16 v1, 1, v0 op_sel:[0,0,1]`,
                // which writes the HIGH half). 0x306 is not an instruction on gfx10.3, and
                // v_lshlrev_b16 is 0x314 — NOT 0x305, which is the 16-bit multiply.
                // CONFIDENCE: HIGH.
                // ABS/NEG are float-only modifiers with no meaning on a 16-bit integer op; reject
                // rather than silently drop them.
                if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2]) {
                    ok = false;
                } else {
                    auto u16src = [&](int k) {
                        const Operand& operand = in.src[k];
                        // A 16-bit operand position carries the 16-bit encoding of an inline
                        // constant, not the f32 pattern val() would materialize — and OPSEL[k]
                        // selects a half of THAT dword (see inline_16bit_operand_dword). Both
                        // inline branches used to ignore the select, so a positive inline int read
                        // its own value from the HIGH half where hardware reads 0 (#2119, kernel
                        // 32r14e). Negative inline ints were already right by accident: -N is
                        // sign-extended, so both halves are 0xffff.
                        const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
                        uint32_t inline_dword;
                        if (inline_16bit_operand_dword(operand, inline_dword))
                            return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
                        const uint32_t raw = val(operand);
                        return b.ibin(Op_BitwiseAnd,
                                      sel ? b.ibin(Op_ShiftRightLogical, raw, b.uconst(16)) : raw,
                                      b.uconst(0xFFFFu));
                    };
                    // Sign-extend a 16-bit value into the full 32-bit signed domain.
                    auto s16 = [&](uint32_t v) { return b.bfe_s(v, b.uconst(0), b.uconst(16)); };
                    const uint32_t s0 = u16src(0), s1 = u16src(1);
                    // Three-source forms only; u16src(2) is not evaluated for the two-source ops.
                    auto s2 = [&] { return u16src(2); };
                    // Integer median of three, the same formula the u32 form uses.
                    auto med3 = [&](uint32_t x, uint32_t y, uint32_t z, bool is_signed) {
                        const uint32_t lo = is_signed ? (uint32_t)Glsl_SMin : (uint32_t)Glsl_UMin;
                        const uint32_t hi = is_signed ? (uint32_t)Glsl_SMax : (uint32_t)Glsl_UMax;
                        auto mn = [&](uint32_t p, uint32_t q) {
                            return is_signed ? b.sext2(lo, p, q) : b.uext2(lo, p, q); };
                        auto mx = [&](uint32_t p, uint32_t q) {
                            return is_signed ? b.sext2(hi, p, q) : b.uext2(hi, p, q); };
                        return mx(mn(x, y), mx(mn(x, z), mn(y, z)));
                    };
                    uint32_t r16;
                    switch (in.opcode) {
                        case 0x340: case 0x35E:
                            r16 = b.ibin(Op_IAdd, b.ibin(Op_IMul, s0, s1), s2()); break;
                        case 0x353: r16 = b.uext2(Glsl_UMin, b.uext2(Glsl_UMin, s0, s1), s2()); break;
                        case 0x356: r16 = b.uext2(Glsl_UMax, b.uext2(Glsl_UMax, s0, s1), s2()); break;
                        case 0x352: r16 = b.sext2(Glsl_SMin,
                                                  b.sext2(Glsl_SMin, s16(s0), s16(s1)),
                                                  s16(s2())); break;
                        case 0x355: r16 = b.sext2(Glsl_SMax,
                                                  b.sext2(Glsl_SMax, s16(s0), s16(s1)),
                                                  s16(s2())); break;
                        case 0x359: r16 = med3(s0, s1, s2(), false); break;
                        case 0x358: r16 = med3(s16(s0), s16(s1), s16(s2()), true); break;
                        case 0x303: case 0x30D: r16 = b.ibin(Op_IAdd, s0, s1); break;
                        case 0x304: case 0x30E: r16 = b.ibin(Op_ISub, s0, s1); break;
                        case 0x305:             r16 = b.ibin(Op_IMul, s0, s1); break;
                        case 0x309:             r16 = b.uext2(Glsl_UMax, s0, s1); break;
                        case 0x30B:             r16 = b.uext2(Glsl_UMin, s0, s1); break;
                        case 0x30A:             r16 = b.sext2(Glsl_SMax, s16(s0), s16(s1)); break;
                        case 0x30C:             r16 = b.sext2(Glsl_SMin, s16(s0), s16(s1)); break;
                        case 0x314: r16 = b.ibin(Op_ShiftLeftLogical, s1,
                                                 b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu))); break;
                        case 0x307: r16 = b.ibin(Op_ShiftRightLogical, s1,
                                                 b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu))); break;
                        default:    r16 = b.sbin(Op_ShiftRightArithmetic, s16(s1),
                                                 b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu)));
                                    break;                                    // 0x308 v_ashrrev_i16
                    }
                    r16 = b.ibin(Op_BitwiseAnd, r16, b.uconst(0xFFFFu));
                    vreg[in.dst.value] = (in.vop3p_opsel & 8u)
                        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                                 b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                                 r16);
                }
            } else if (in.opcode == 0x14B || in.opcode == 0x141) {    // v_fma_f32 / v_mad_f32 = src0*src1 + src2
                // v_mad_f32 (op 0x141) is a gfx10.1 (Navi) instruction REMOVED in gfx10.3, so llvm-mc
                // -mcpu=gfx1030 rejects it as invalid — but the PS5 shader compiler targets gfx10.1 and
                // emits it (real game shaders 5,26-29: manual attribute interpolation p0+i*p1). Its result
                // (unfused mul-then-add) maps exactly to OpFMul+OpFAdd; v_fma's fused rounding is
                // immaterial here. VERIFIED(round-trip llvm-mc gfx1010, both directions): VOP3 op 0x141.
                uint32_t m = b.fbin(Op_FMul, fv(0), fv(1));
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, m, fv(2)));
            } else if (in.opcode == 0x176) {                          // v_mad_u64_u32
                // AMD RDNA2: {carry,D.u64} = S0.u32*S1.u32 + S2.u64. A literal
                // or positive inline constant used as the 64-bit addend is zero-extended;
                // a negative integer inline constant is sign-extended. Register addends
                // consume the consecutive high register.
                if (in.dst.value >= 255) { ok = false; }
                else {
                    auto high_half = [&](const Operand& operand) -> uint32_t {
                        if (operand.kind == OperandKind::InlineInt)
                            return b.uconst(operand.value < 0 ? 0xFFFFFFFFu : 0u);
                        // An inline FLOAT 64-bit addend supplies the DOUBLE bit pattern (high dword
                        // carries the exponent/mantissa; 1/(2*pi) has a nonzero LOW dword too) —
                        // the previous {f32-bits, 0} model was wrong in both halves. No compiler
                        // emits a float inline as an integer-mad addend: reject, stay fail-visible.
                        if (operand.kind == OperandKind::InlineFloat) { ok = false; return b.uconst(0); }
                        if (operand.kind == OperandKind::Literal)
                            return b.uconst(0);
                        if (operand.kind == OperandKind::Special && operand.value == 125)
                            return b.uconst(0);                       // null pair
                        if (operand.kind == OperandKind::VGPR ||
                            operand.kind == OperandKind::SGPR ||
                            (operand.kind == OperandKind::Special &&
                             operand.value >= 106 && operand.value < 124)) {
                            Operand next = operand;
                            ++next.value;
                            return val(next);
                        }
                        ok = false;
                        return b.uconst(0);
                    };

                    const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                    const uint32_t add_lo = val(in.src[2]), add_hi = high_half(in.src[2]);
                    const uint32_t mul_lo = b.ibin(Op_IMul, a, c);
                    const uint32_t mul_hi = b.umul_hi(a, c);
                    const uint32_t result_lo = b.ibin(Op_IAdd, mul_lo, add_lo);
                    const uint32_t carry_lo = b.ucmp(Op_ULessThan, result_lo, mul_lo);
                    const uint32_t high_sum = b.ibin(Op_IAdd, mul_hi, add_hi);
                    const uint32_t carry_hi0 = b.ucmp(Op_ULessThan, high_sum, mul_hi);
                    const uint32_t carry_word = b.sel(carry_lo, b.uconst(1), b.uconst(0));
                    const uint32_t result_hi = b.ibin(Op_IAdd, high_sum, carry_word);
                    const uint32_t carry_hi1 = b.ucmp(Op_ULessThan, result_hi, high_sum);
                    const uint32_t carry_out = b.lor(carry_hi0, carry_hi1);

                    const int hi_dst = in.dst.value + 1;
                    const uint32_t old_hi = vreg_old(b, rs, hi_dst);
                    vreg[in.dst.value] = result_lo;
                    vreg[hi_dst] = result_hi;
                    predicate_write(b, rs, hi_dst, old_hi);
                    // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, never the
                    // raw carry (wave votes/spills must not see phantom bits from inactive lanes).
                    const uint32_t carry_masked =
                        rs.exec_narrowed ? b.land(rs.exec, carry_out) : carry_out;
                    if (in.sdst.value == 106 || in.sdst.value == 107) rs.vcc = carry_masked;
                    else if (in.sdst.kind == OperandKind::SGPR) rs.sreg_bool[in.sdst.value] = carry_masked;
                    else ok = false;
                }
            } else if (in.opcode == 0x30F || in.opcode == 0x310 || in.opcode == 0x319) {
                // v_add_co_u32 (0x30F) / v_sub_co_u32 (0x310) / v_subrev_co_u32 (0x319): 32-bit add/
                // subtract that writes a carry/borrow-out to the VOP3B sdst mask (VCC or an SGPR pair).
                // No carry-IN — that is the _co_ci_ family (VOP2 0x28-0x2A / VOP3B 0x128-0x12A). Mirrors
                // the VOP2 e32 carry emit (0x28-0x2A above) plus the v_mad_u64_u32 carry-out sdst write.
                // GTA V (PPSA04263) UI/content compute shaders use these; the missing op dropped the whole
                // shader, so its output texture rendered as bare untextured triangles (#1163/#1165).
                // CONFIDENCE: HIGH (RDNA2 ISA 6.5/3.9; coverage test in test_recompile_coverage).
                const uint32_t a = val(in.src[0]), c = val(in.src[1]);
                uint32_t d, carry;
                if (in.opcode == 0x30F) {                     // a + c ; carry = unsigned overflow
                    d = b.ibin(Op_IAdd, a, c);
                    carry = b.ucmp(Op_ULessThan, d, a);
                } else {                                      // sub: a-c ; subrev: c-a ; borrow = x < y
                    const uint32_t x = in.opcode == 0x310 ? a : c;
                    const uint32_t y = in.opcode == 0x310 ? c : a;
                    d = b.ibin(Op_ISub, x, y);
                    carry = b.ucmp(Op_ULessThan, x, y);
                }
                vreg[in.dst.value] = d;
                // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, not its raw carry.
                const uint32_t carry_masked = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
                write_vop3b_carry_output(carry_masked);
            } else if (in.opcode == 0x169) {                          // v_mul_lo_u32
                vreg[in.dst.value] = b.ibin(Op_IMul, val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16a) {                          // v_mul_hi_u32 (high 32 bits)
                vreg[in.dst.value] = b.umul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x16c) {                          // v_mul_hi_i32 (high 32 bits, signed)
                vreg[in.dst.value] = b.smul_hi(val(in.src[0]), val(in.src[1]));
            } else if (in.opcode == 0x157) {                          // v_med3_f32 = median(s0,s1,s2)
                // ISA op 343: "if (isNan(S0) || isNan(S1) || isNan(S2)) D = V_MIN3_F32(S0,S1,S2)".
                // The min/max legs use the NaN-aware NMin/NMax (return-the-other-operand), and the
                // any-NaN case selects min3 explicitly — the plain max(min(...)) formula does not
                // reproduce the documented NaN fallback on its own.
                uint32_t s0 = fv(0), s1 = fv(1), s2 = fv(2);
                uint32_t mn = b.fext2(Glsl_NMin, s0, s1), mx = b.fext2(Glsl_NMax, s0, s1);
                uint32_t med = b.fext2(Glsl_NMax, mn, b.fext2(Glsl_NMin, mx, s2));
                uint32_t min3 = b.fext2(Glsl_NMin, mn, s2);
                uint32_t nan_any = b.lor(b.fcmp(Op_FUnordNotEqual, s0, s0),
                                         b.lor(b.fcmp(Op_FUnordNotEqual, s1, s1),
                                               b.fcmp(Op_FUnordNotEqual, s2, s2)));
                vreg[in.dst.value] = fresult(b.sel(nan_any, min3, med));
            } else if (in.opcode == 0x159) {                          // v_med3_u32
                // Unsigned median of three values: max(min(a,b), min(max(a,b),c)).
                // Astro Bot uses this to clamp a material index into [0,31] before its world-map
                // depth prepass. VERIFIED(llvm-mc gfx1030: VOP3 0x159 = v_med3_u32).
                const uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]), s2 = val(in.src[2]);
                const uint32_t mn = b.uext2(Glsl_UMin, s0, s1);
                const uint32_t mx = b.uext2(Glsl_UMax, s0, s1);
                vreg[in.dst.value] = b.uext2(Glsl_UMax, mn, b.uext2(Glsl_UMin, mx, s2));
            } else if (in.opcode == 0x153 || in.opcode == 0x156) {    // v_min3_u32 / v_max3_u32
                // Unsigned min/max of three values. Sonic Racing: CrossWorlds' post chain rejects on
                // v_max3_u32 four times per boot (#2013); v_min3_u32 is its neighbour in the same
                // ISA family and the same lowering. VERIFIED(round-trip llvm-mc gfx1030: VOP3
                // 0x153 = v_min3_u32, 0x156 = v_max3_u32). CONFIDENCE: HIGH.
                const uint32_t op = in.opcode == 0x153 ? (uint32_t)Glsl_UMin : (uint32_t)Glsl_UMax;
                vreg[in.dst.value] =
                    b.uext2(op, b.uext2(op, val(in.src[0]), val(in.src[1])), val(in.src[2]));
            } else if (in.opcode == 0x151 || in.opcode == 0x154) {    // v_min3_f32 / v_max3_f32
                // min/max of three floats (DOLL's AA-clamp PS). VERIFIED(round-trip llvm-mc gfx1010:
                // VOP3 0x151 = v_min3_f32, 0x154 = v_max3_f32 — 0xd551…/0xd554…). NaN-aware NMin/
                // NMax per the ISA one-NaN rule. CONFIDENCE: HIGH.
                uint32_t op = in.opcode == 0x151 ? (uint32_t)Glsl_NMin : (uint32_t)Glsl_NMax;
                vreg[in.dst.value] = fresult(b.fext2(op, b.fext2(op, fv(0), fv(1)), fv(2)));
            } else if (in.opcode == 0x368 || in.opcode == 0x369) {    // v_cvt_pknorm_{i16,u16}_f32
                // Clamp and normalize two f32 values, then pack src0 in bits[15:0] and src1 in
                // bits[31:16]. Astro's title ship VS uses the unsigned form (exact first word
                // d7690002). AMD specifies round-to-nearest-even for the normalized conversion.
                const bool is_signed = in.opcode == 0x368;
                const float scale = is_signed ? 32767.0f : 65535.0f;
                const uint32_t lo = b.pack_norm(fv(0), 16, is_signed, scale);
                const uint32_t hi = b.pack_norm(fv(1), 16, is_signed, scale);
                vreg[in.dst.value] = b.ibin(
                    Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
            } else if (in.opcode == 0x36A) {                          // v_cvt_pk_u16_u32
                // Pack two u32 into u16 halves with UNSIGNED SATURATION: lo = min(s0,0xFFFF),
                // hi = min(s1,0xFFFF); dst = lo | hi<<16. VERIFIED(round-trip llvm-mc gfx1010:
                // VOP3 0x36a — 0xd76a…). CONFIDENCE: HIGH.
                uint32_t lo = b.uext2(Glsl_UMin, val(in.src[0]), b.uconst(0xFFFFu));
                uint32_t hi = b.uext2(Glsl_UMin, val(in.src[1]), b.uconst(0xFFFFu));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
            } else if (in.opcode == 0x362) {                          // v_ldexp_f32
                // AMD opcode 866: D.f = S0.f * 2**S1.i. The exact GTA V production packet is
                // d7620000,0002030d (`v_ldexp_f32 v0,v13,v1`) with no modifiers. Admit that proven
                // shape only for now: ABS/NEG on the integer exponent and output-modifier denormal
                // behavior need separate contracts, and silently ignoring either would corrupt
                // mip/scale reconstruction. ldexp_f32_bits covers the full unmodified i32 exponent
                // domain without relying on GLSL.std.450 Ldexp's undefined overflow cases.
                // src0 is the FLOAT operand, and ABS/NEG on a float source is the ordinary VOP3
                // modifier every other float op here already applies through fv() -- hardware order
                // ABS then NEG. Admitting it is not a new contract (#3138). The original gate
                // rejected all three sources together, which also refused this well-defined case:
                // Stray's `0x3011560000` is `v_ldexp_f32 v39, |v35|, -2` (`d7620127,00018523`,
                // ABS[2:0]=001), and that ONE instruction failed a 3684-dword fragment stage and
                // discarded 1536 full-screen 3840x2160 draws per routed boot.
                //
                // src1 keeps rejecting, and for a reason the gate's own comment gives: it is the
                // integer EXPONENT, where "absolute value" and "negate" are not float modifiers at
                // all and silently ignoring either would corrupt mip/scale reconstruction. CLAMP and
                // OMOD keep rejecting too -- their denormal behaviour needs its own contract.
                if (in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[1] || in.src_neg[2] ||
                    in.clamp || in.omod) {
                    ok = false;
                } else {
                    vreg[in.dst.value] = b.ldexp_f32_bits(fv(0), val(in.src[1]));
                }
            } else if (in.opcode >= 0x144 && in.opcode <= 0x147) {
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
                    case 0x144: {   // face id: z-major 4/5, y-major 2/3, x-major 0/1 (negative = +1)
                        uint32_t idz = b.sel(zneg, b.uconst(fbits(5.0f)), b.uconst(fbits(4.0f)));
                        uint32_t idy = b.sel(yneg, b.uconst(fbits(3.0f)), b.uconst(fbits(2.0f)));
                        uint32_t idx2 = b.sel(xneg, b.uconst(fbits(1.0f)), b.uconst(fbits(0.0f)));
                        r2 = b.sel(zmaj, idz, b.sel(ymaj, idy, idx2)); break;
                    }
                    case 0x145: {   // sc: z-major: z<0 ? -x : x; y-major: x; x-major: x<0 ? z : -z
                        uint32_t scz = b.sel(zneg, fneg(x), x);
                        uint32_t scx = b.sel(xneg, z, fneg(z));
                        r2 = b.sel(zmaj, scz, b.sel(ymaj, x, scx)); break;
                    }
                    case 0x146: {   // tc: z-major: -y; y-major: y<0 ? -z : z; x-major: -y
                        uint32_t tcy = b.sel(yneg, fneg(z), z);
                        r2 = b.sel(zmaj, fneg(y), b.sel(ymaj, tcy, fneg(y))); break;
                    }
                    default: {      // 0x147 ma: 2 * major axis (signed)
                        uint32_t maj = b.sel(zmaj, z, b.sel(ymaj, y, x));
                        r2 = b.fbin(Op_FMul, b.uconst(fbits(2.0f)), maj); break;
                    }
                }
                vreg[in.dst.value] = fresult(r2);
            } else if (in.opcode == 0x143) {                          // v_mad_u32_u24 = (s0&0xFFFFFF)*(s1&0xFFFFFF)+s2
                uint32_t m24 = b.uconst(0xFFFFFF);
                uint32_t p = b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, val(in.src[0]), m24),
                                              b.ibin(Op_BitwiseAnd, val(in.src[1]), m24));
                vreg[in.dst.value] = b.ibin(Op_IAdd, p, val(in.src[2]));
            } else if (in.opcode == 0x15D) {                          // v_sad_u32 = |s0-s1| (unsigned) + s2
                // RDNA2 ISA (document 70648), V_SAD_U32: D.u32 = abs(S0.u32 - S1.u32) + S2.u32.
                // The absolute difference is the UNSIGNED magnitude, so max-min is exact and cannot
                // wrap (a signed abs would be wrong for operands straddling 0x80000000). Worms
                // Armageddon's PSSL compiler emits the src1=0 accumulate form `v_sad_u32 vN, sM, 0,
                // vN` as its vertex-shader index prologue; its shipped .ags shader assets carry the
                // identical word. VERIFIED(llvm-mc gfx1030: VOP3 0x15d = v_sad_u32).
                // CONFIDENCE: HIGH.
                const uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]);
                const uint32_t diff = b.ibin(Op_ISub, b.uext2(Glsl_UMax, s0, s1),
                                                       b.uext2(Glsl_UMin, s0, s1));
                vreg[in.dst.value] = b.ibin(Op_IAdd, diff, val(in.src[2]));
            } else if (in.opcode == 0x148 || in.opcode == 0x149) {    // v_bfe_u32 / v_bfe_i32
                uint32_t off = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                uint32_t cnt = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
                vreg[in.dst.value] = (in.opcode == 0x148) ? b.bfe_u(val(in.src[0]), off, cnt)
                                                          : b.bfe_s(val(in.src[0]), off, cnt);
            } else if (in.opcode == 0x14A) {                          // v_bfi_b32 = (s0&s1)|(~s0&s2)
                uint32_t s0 = val(in.src[0]);
                uint32_t t1 = b.ibin(Op_BitwiseAnd, s0, val(in.src[1]));
                uint32_t t2 = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, s0), val(in.src[2]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t1, t2);
            } else if (in.opcode == kVop3OpcodeAlignbitB32 ||
                       in.opcode == kVop3OpcodeAlignbyteB32) {
                // RDNA2 defines ALIGNBIT as
                //   D = ({S0,S1} >> S2.u[4:0]) & 0xffffffff
                // and ALIGNBYTE as the same 64-bit join shifted by 8*S2.u[4:0]. S0 is the high
                // dword and S1 the low dword. Keep every emitted SPIR-V shift below 32:
                // OpSelect is not short-circuiting, so even an apparently unused shift by 32 would
                // be undefined. ALIGNBIT's masked bit count always remains in the joined low-dword
                // domain; ALIGNBYTE additionally has the high-dword and zero-fill domains.
                const bool modified = in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                                      in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                                      in.clamp || in.omod;
                if (modified) {
                    ok = false;
                } else {
                    const uint32_t s0 = val(in.src[0]);
                    const uint32_t s1 = val(in.src[1]);
                    const uint32_t count = b.ibin(
                        Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
                    const uint32_t shift = in.opcode == kVop3OpcodeAlignbitB32
                        ? count
                        : b.ibin(Op_BitwiseAnd,
                                 b.ibin(Op_ShiftLeftLogical, count, b.uconst(3)),
                                 b.uconst(31));
                    const uint32_t inv_shift = b.ibin(
                        Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift),
                        b.uconst(31));
                    const uint32_t joined = b.ibin(
                        Op_BitwiseOr, b.ibin(Op_ShiftRightLogical, s1, shift),
                        b.ibin(Op_ShiftLeftLogical, s0, inv_shift));
                    const uint32_t low = b.sel(
                        b.ucmp(Op_IEqual, count, b.uconst(0)), s1, joined);
                    if (in.opcode == kVop3OpcodeAlignbitB32) {
                        vreg[in.dst.value] = low;
                    } else {
                        const uint32_t upper_byte = b.ibin(Op_ISub, count, b.uconst(4));
                        const uint32_t upper_shift = b.ibin(
                            Op_BitwiseAnd,
                            b.ibin(Op_ShiftLeftLogical, upper_byte, b.uconst(3)),
                            b.uconst(31));
                        const uint32_t upper = b.ibin(Op_ShiftRightLogical, s0, upper_shift);
                        vreg[in.dst.value] = b.sel(
                            b.ucmp(Op_ULessThan, count, b.uconst(4)), low,
                            b.sel(b.ucmp(Op_ULessThan, count, b.uconst(8)), upper,
                                  b.uconst(0)));
                    }
                }
            } else if (in.opcode == kVop3OpcodeLshlrevB64) {         // v_lshlrev_b64
                // GTA V constructs a per-lane bit as `1ull << lane` immediately after MBCNT. This
                // bounded admission keeps SRC1 at the exact inline integer 1; general register-pair
                // and floating-inline sources need wider source-span/provenance handling.
                const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
                const bool exact_one = in.src[1].kind == OperandKind::InlineInt &&
                                       in.src[1].value == 1;
                if (in.dst.value < 0 || in.dst.value >= 255 || !exact_one ||
                    in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                    in.clamp || in.omod || opsel) {
                    ok = false;
                } else {
                    const uint32_t amount = b.ibin(
                        Op_BitwiseAnd, val(in.src[0]), b.uconst(63));
                    const uint32_t result = b.u64_shift(
                        Op_ShiftLeftLogical,
                        b.u64_from_lohi(b.uconst(1), b.uconst(0)), amount);
                    const int hi_dst = in.dst.value + 1;
                    const uint32_t old_hi = vreg_old(b, rs, hi_dst);
                    vreg[in.dst.value] = b.u64_lo(result);
                    vreg[hi_dst] = b.u64_hi(result);
                    predicate_write(b, rs, hi_dst, old_hi);
                }
            } else if (in.opcode == kVop3OpcodeLshrrevB64) {         // v_lshrrev_b64
                // GTA V's compact-BVH kernels shift a VGPR pair by either a scalar scratch word or
                // a VGPR lane value. RDNA masks the count to six bits. Keep the pair and modifier
                // admission exact: SRC1 must be an addressable consecutive register pair, and this
                // integer operation has no float modifiers or output modifier spelling.
                const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
                const bool source_pair = in.src[1].kind == OperandKind::VGPR &&
                                         in.src[1].value >= 0 && in.src[1].value < 255;
                if (in.dst.value < 0 || in.dst.value >= 255 || !source_pair ||
                    in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                    in.clamp || in.omod || opsel) {
                    ok = false;
                } else {
                    const uint32_t amount = b.ibin(
                        Op_BitwiseAnd, val(in.src[0]), b.uconst(63));
                    Operand high_source = in.src[1];
                    ++high_source.value;
                    const uint32_t result = b.u64_shift(
                        Op_ShiftRightLogical,
                        b.u64_from_lohi(val(in.src[1]), val(high_source)), amount);
                    const int hi_dst = in.dst.value + 1;
                    const uint32_t old_hi = vreg_old(b, rs, hi_dst);
                    vreg[in.dst.value] = b.u64_lo(result);
                    vreg[hi_dst] = b.u64_hi(result);
                    predicate_write(b, rs, hi_dst, old_hi);
                }
            } else if (in.opcode == 0x363) {                          // v_bfm_b32
                // RDNA2: D.u32 = ((1 << S0[4:0]) - 1) << S1[4:0]. Mask both shift
                // amounts before emitting SPIR-V so neither can reach the undefined >=32 range;
                // in particular, an input width of 32 wraps to zero rather than producing all ones.
                // The instruction has no defined modifier forms. OP_SEL is not retained by the
                // generic decoder for this opcode, so reject its raw field explicitly as well.
                const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
                if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
                    in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
                    in.clamp || in.omod || opsel) {
                    ok = false;
                } else {
                    const uint32_t width = b.ibin(
                        Op_BitwiseAnd, val(in.src[0]), b.uconst(31));
                    const uint32_t offset = b.ibin(
                        Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                    const uint32_t mask = b.ibin(
                        Op_ISub,
                        b.ibin(Op_ShiftLeftLogical, b.uconst(1), width),
                        b.uconst(1));
                    vreg[in.dst.value] = b.ibin(Op_ShiftLeftLogical, mask, offset);
                }
            } else if (in.opcode == 0x364) {                          // v_bcnt_u32_b32
                // AMD RDNA2: D = popcount(S0) + S1. The third VOP3 source field is unused.
                vreg[in.dst.value] = b.ibin(Op_IAdd,
                                             b.iun(Op_BitCount, val(in.src[0])),
                                             val(in.src[1]));
            } else if (in.opcode == kVop3OpcodeAdd3U32) {             // v_add3_u32 = s0+s1+s2
                vreg[in.dst.value] = b.ibin(Op_IAdd, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), val(in.src[2]));
            } else if (in.opcode == 0x346) {                          // v_lshl_add_u32 = (s0<<(s1&31))+s2
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_IAdd, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
            } else if (in.opcode == kVop3OpcodeAndOrB32) {           // v_and_or_b32 = (s0&s1)|s2
                uint32_t t = b.ibin(Op_BitwiseAnd, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
            } else if (in.opcode == 0x372) {                          // v_or3_b32 = s0|s1|s2
                uint32_t t = b.ibin(Op_BitwiseOr, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
            } else if (in.opcode == 0x178) {                          // v_xor3_b32 = s0^s1^s2
                uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_BitwiseXor, t, val(in.src[2]));
            } else if (in.opcode == 0x36F) {                          // v_lshl_or_b32 = (s0<<(s1&31))|s2
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
            } else if (in.opcode == 0x345) {                          // v_xad_u32 = (s0^s1)+s2
                uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
                vreg[in.dst.value] = b.ibin(Op_IAdd, t, val(in.src[2]));
            } else if (in.opcode == 0x347) {                          // v_add_lshl_u32 = (s0+s1)<<(s2&31)
                uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
                vreg[in.dst.value] = b.ibin(Op_ShiftLeftLogical, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), sh);
            } else if (in.opcode == 0x128 || in.opcode == 0x129 || in.opcode == 0x12A) {
                // Add/sub with carry-in + carry-out (VOP3B): v_add_co_ci_u32 (0x128), v_sub_co_ci (0x129),
                // v_subrev_co_ci (0x12A). carry-in = src2 mask (VCC or an SGPR bool); carry-out -> sdst mask.
                // dst = s0 (+/-) s1 (+/-) cin; carryout = unsigned overflow (add) / borrow (sub).
                // Carry-in from an UNTRACKED mask must reject (like cndmask above), not silently
                // default to 0 — a wrong carry produces 64-bit address math off by one in the low
                // word with no diagnostic.
                const Operand& s2 = in.src[2]; uint32_t cin_mask = 0;
                if (s2.value == 106 || s2.value == 107) {
                    auto it = rs.sreg_bool.find(s2.value);
                    if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
                        cin_mask = it->second;
                    else
                        cin_mask = rs.vcc;
                }
                else if (s2.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(s2.value); if (it != rs.sreg_bool.end()) cin_mask = it->second; }
                if (!cin_mask) ok = false;
                else {
                    uint32_t a = val(in.src[0]), c = val(in.src[1]);
                    uint32_t cin = b.sel(cin_mask, b.uconst(1), b.uconst(0));
                    uint32_t res, cout;
                    if (in.opcode == 0x128) {                             // add: (a + b) + cin
                        uint32_t s1 = b.ibin(Op_IAdd, a, c);
                        uint32_t c1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+b
                        res = b.ibin(Op_IAdd, s1, cin);
                        uint32_t c2 = b.ucmp(Op_ULessThan, res, s1);      // wrap in +cin
                        cout = b.bsel(c1, b.btrue(), c2);                 // c1 || c2
                    } else {                                              // sub / subrev: (a - b) - cin (borrow)
                        uint32_t x = in.opcode == 0x129 ? a : c, y = in.opcode == 0x129 ? c : a;  // subrev swaps
                        uint32_t s1 = b.ibin(Op_ISub, x, y);
                        uint32_t b1 = b.ucmp(Op_ULessThan, x, y);         // borrow in x-y
                        res = b.ibin(Op_ISub, s1, cin);
                        uint32_t b2 = b.ucmp(Op_ULessThan, s1, cin);      // borrow in -cin
                        cout = b.bsel(b1, b.btrue(), b2);
                    }
                    vreg[in.dst.value] = res;
                    // carry-out -> sdst mask (VCC or a saved SGPR-pair bool). ISA 3.9: an
                    // EXEC-inactive lane's mask bit is written 0, never its raw carry.
                    const uint32_t cout_masked = rs.exec_narrowed ? b.land(rs.exec, cout) : cout;
                    write_vop3b_carry_output(cout_masked);
                }
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && b.ngg_logical_lane &&
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
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) && allow_wave &&
                       (b.is_compute || b.is_fragment)) {
                // v_mbcnt_lo/hi_u32_b32 (cross-lane): dst = src1 + count of lanes below this one whose mask
                // bit (src0) is set, in the low/high 32. The per-lane "mask bit" comes from src0: EXEC
                // (126/127) -> this lane's exec bool; inline -1 (all-ones) -> always set (mbcnt = lane
                // index, the common "get my lane id" idiom, e.g. shader 037); inline 0 -> never; an SGPR
                // pair -> that saved mask's bool. A general computed 32-bit mask VALUE isn't representable
                // per-lane, so reject that. Portable compute uses LDS+barriers at barrier-uniform sites;
                // exact-size compute subgroups and fragments use a native exclusive subgroup sum.
                const uint32_t active = mbcnt_source_bit(
                    b, rs, in.src[0], in.opcode == 0x366);
                if (active) {
                    const uint32_t acc = val(in.src[1]);
                    const uint32_t lo = in.opcode == 0x365 ? b.btrue() : b.bfalse();
                    vreg[in.dst.value] = b.is_fragment
                        ? b.fragment_mbcnt(active, acc, in.opcode == 0x365)
                        : (b.native_subgroup_size
                            ? b.native_compute_mbcnt(active, acc, lo)
                            : b.mbcnt(active, acc, in.opcode == 0x365));
                }
                else ok = false;
            } else if ((in.opcode == 0x365 || in.opcode == 0x366) &&
                       in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1 &&
                       (b.is_compute || b.is_fragment)) {
                // An ALL-ONES mask makes MBCNT a pure function of this lane's OWN index: the number
                // of set bits below lane L in an all-ones 64-bit mask is L, so LO contributes
                // min(L, 32) and HI contributes max(L - 32, 0). No other lane's state is read, so
                // this form needs neither `allow_wave` nor a subgroup/LDS reduction and stays exact
                // under divergent control flow — it is the ubiquitous "what is my lane id" idiom,
                // and it is the scalar counterpart of the ngg_logical_lane branch above.
                // (Sonic Racing: CrossWorlds' compute post chain emits `d7650001,000100c1` =
                // `v_mbcnt_lo_u32_b32 v1, -1, 0` inside a structured region, where the compute
                // structurizer sets wave_ok=false and the branch above therefore never runs; #2013.)
                const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(),
                                             b.uconst(b.wave_size - 1));
                const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
                const uint32_t count = in.opcode == 0x365
                    ? b.sel(high, b.uconst(32), lane)
                    : b.sel(high, b.ibin(Op_ISub, lane, b.uconst(32)), b.uconst(0));
                vreg[in.dst.value] = b.ibin(Op_IAdd, val(in.src[1]), count);
            } else if (in.opcode == 0x365 || in.opcode == 0x366) {
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
                else if (in.src[0].value == 126 || in.src[0].value == 127) form = "exec";
                else if (in.src[0].value == 106 || in.src[0].value == 107) form = "vcc";
                else if (in.src[0].kind == OperandKind::SGPR) form = "sgpr-mask";
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
                    std::snprintf(reason, sizeof reason,
                                  "mbcnt-cross-lane stage=%s form=%s wave=none",
                                  b.is_vertex ? "vertex"
                                              : (b.is_fragment ? "fragment" : "compute"),
                                  form);
                b.stage_reject_pc = in.pc;
                b.stage_reject_reason = reason;
                ok = false;
            } else if (in.opcode == 0x12F) {                          // v_cvt_pkrtz_f16_f32 = pack(s0->lo, s1->hi)
                vreg[in.dst.value] = b.pack_half2x16_rtz(fv(0), fv(1)); // v_cvt_pkrtz VOP3: RTZ clamp (#452)
            } else if (in.opcode == 0x103) {                          // v_add_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, fv(0), fv(1)));
            } else if (in.opcode == 0x104) {                          // v_sub_f32 (VOP3 form) = s0 - s1
                vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(0), fv(1)));
            } else if (in.opcode == 0x105) {                          // v_subrev_f32 (VOP3 form) = s1 - s0
                vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(1), fv(0)));
            } else if (in.opcode == 0x108) {                          // v_mul_f32 (VOP3 form)
                vreg[in.dst.value] = fresult(b.fbin(Op_FMul, fv(0), fv(1)));
            } else if (in.opcode == 0x10F) {                          // v_min_f32 (VOP3 form; NaN -> other operand)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_NMin, fv(0), fv(1)));
            } else if (in.opcode == 0x110) {                          // v_max_f32 (VOP3 form; NaN -> other operand)
                vreg[in.dst.value] = fresult(b.fext2(Glsl_NMax, fv(0), fv(1)));
            } else if (in.opcode == 0x107) {                          // v_mul_legacy_f32: DX9 multiply —
                // 0 * x == 0 for ALL x including ±Inf/NaN (that guarantee is WHY compilers emit it,
                // e.g. attenuation=0 times 1/dist). Plain IEEE FMul gives NaN for 0*Inf, so emit
                // select(s0==0 || s1==0, 0, s0*s1). ±0.0 both compare equal to 0.0 under FOrdEqual.
                uint32_t s0b = fv(0), s1b = fv(1), zb = b.uconst(0);
                uint32_t anyz = b.lor(b.fcmp(Op_FOrdEqual, s0b, zb), b.fcmp(Op_FOrdEqual, s1b, zb));
                vreg[in.dst.value] = fresult(b.sel(anyz, zb, b.fbin(Op_FMul, s0b, s1b)));
            } else if (in.opcode == 0x101) {                          // v_cndmask_b32_e64: src2_mask ? src1 : src0
                const Operand& s2 = in.src[2]; uint32_t m = 0;        // src2 is an SGPR-pair (or VCC) wave mask
                if (s2.value == 106 || s2.value == 107) {
                    // Wave32 e64/SDWA forms may explicitly select either physical VCC word as an
                    // independent one-dword mask. Prefer that typed lifetime when present. Only
                    // VCC_LO may fall back to the architectural implicit predicate; VCC_HI is a
                    // distinct word in Wave32 and must have its own proven mask lifetime.
                    auto it = rs.sreg_bool.find(s2.value);
                    if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
                        m = it->second;
                    else if (s2.value == 106)
                        m = rs.vcc;
                }
                else if (s2.kind == OperandKind::SGPR) {
                    auto it = rs.sreg_bool.find(s2.value);
                    if (it != rs.sreg_bool.end()) {
                        m = it->second;
                    } else if (b.ngg_one_lane) {
                        // The exact Astro NGG projection represents guest lane zero. Some wrappers
                        // construct a literal/dynamic B64 lane mask in ordinary scalar DATA registers
                        // rather than through a mask-domain instruction (7f5f: s4:s5=0xaaaaaaaa...
                        // before pc3870). For lane zero, consuming that pair as a wave mask is exactly
                        // its low dword's bit zero. Keep arbitrary vertex shaders fail-closed.
                        auto data = rs.sreg.find(s2.value);
                        if (data != rs.sreg.end())
                            m = b.ucmp(Op_INotEqual,
                                b.ibin(Op_BitwiseAnd, data->second, b.uconst(1)),
                                b.uconst(0));
                    }
                }
                // fv (not val): cndmask is float-modifier-capable and compilers emit it with -v/|v|
                // sources (sign-select idioms). Raw val() silently dropped neg/abs — the shader
                // recompiled "successfully" and computed the un-negated value. fv == val when no
                // modifier bits are set.
                if (!m) ok = false; else vreg[in.dst.value] = b.sel(m, fv(1), fv(0));
            } else if (in.opcode == 0x11F) {                          // v_mac_f32_e64 (VOP3 form of VOP2 0x1f)
                // dst = src0*src1 + dst, with the VOP3 float source modifiers (neg/abs via fv) and output
                // modifiers (omod/clamp via fresult). The scene VS emits this e64 form with a `-|v10|`
                // modifier (round-trip: `llvm-mc -mcpu=gfx1010` of 0xd51f020a → v_mac_f32_e64 v10,v28,-|v10|).
                vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, b.fbin(Op_FMul, fv(0), fv(1)), old_d));
            } else ok = false;
            // A set CLAMP bit on an opcode that does not route through fresult means unmodeled
            // INTEGER saturation (ISA 6.5: "for integer operations, it clamps the result to the
            // largest and smallest representable value") or an unhandled pack/select combination
            // (v_cvt_pkrtz, v_cndmask_e64, the VOP3B carry ops) — reject fail-visibly rather than
            // silently emitting the wrapping/unclamped result. OMOD on integer results is
            // architecturally ignored, so only CLAMP gates here.
            if (ok && in.clamp && !clamp_routed) ok = false;
            if (ok) predicate_write(b, rs, in.dst.value, old_d);
            return true;
        }
        case Rdna2Format::SOPP: {
            // Control flow. The only branch we can safely linearize today is the common forward
            // s_cbranch_execz "skip the EXEC-predicated block if no lane is active" idiom. Branches
            // on SCC/VCC (or EXECNZ) can skip code for reasons not represented by EXEC predication,
            // so accepting them as no-ops would silently execute the wrong path.
            switch (in.opcode) {
                // Hints / sync with no effect in our synchronous SSA model — safe no-ops.
                case 0x00:   // s_nop
                case 0x0c:   // s_waitcnt        (no async memory latency to wait on)
                case 0x10:   // s_sendmsg        (NGG GS_ALLOC_REQ etc. — no wave/primitive allocation in
                             //                   our per-invocation model; only meaningful for NGG/GS,
                             //                   which we lower per-invocation, so it's a safe no-op)
                case 0x16:   // s_ttracedata     (sends M0 to the thread-trace stream — a profiling side
                             //                   channel with no architectural effect: it writes no SGPR,
                             //                   VGPR, or memory and does not branch. Prosper models no
                             //                   thread trace, so emitting nothing IS the correct lowering.
                             //                   Observed live in Greak (PPSA02849) vertex shaders, which
                             //                   ship it at pc=2 and were rejected whole. CONFIDENCE: HIGH
                             //                   — round-tripped against llvm-mc gfx1030.)
                case 0x17:   // s_cbranch_cdbgsys (Prosper exposes no attached GPU system debugger, so
                             //                    COND_DBG_SYS is permanently clear and the branch falls through)
                case 0x20:   // s_inst_prefetch  (I-cache hint)
                case 0x21:   // s_clause         (memory-clause scheduling hint)
                case 0x22:   // s_wait_idle
                    break;   // (s_waitcnt_vscnt is SOPK on gfx10, not SOPP 0x7d — see the SOPK case)
                case 0x08:                                          // s_cbranch_execz
                    if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
                    else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
                    break;                                          // forward = no-op (predication covers it)
                case 0x0a:                                          // s_barrier
                    if (b.is_compute || b.ngg_private_lds) b.barrier();
                    else ok = false;                                // barrier only meaningful in compute
                    break;
                case 0x04: case 0x05:                              // s_cbranch_scc0 / scc1
                    // An alpha-test / clip() kill-mask early-out (SCC = "any lane survives", set by a 64-bit
                    // wave-mask op — see mask_test_branches) is a pure wave optimization: per-invocation the
                    // survivor mask narrows EXEC in the block below and the export OpKills the failed lanes,
                    // so the branch LINEARIZES away exactly like a forward s_cbranch_execz. Only these
                    // recognized branches are dropped; any other SCC branch is still a real uniform-if we
                    // can't model straight-line, so it rejects.
                    if (safe_execz && safe_execz->count(in.pc)) break;   // recognized kill-mask branch -> no-op
                    ok = false;
                    break;
                case 0x06: case 0x07:                              // s_cbranch_vccz / vccnz
                    // A byte-exact Astro NGG terminal suffix may skip only trailing PARAM exports
                    // after POS has already been exported. In the one-lane projection emitting those
                    // otherwise-unused values is harmless; the gate proof records only that bounded
                    // branch in safe_execz. General VCC branches remain unsupported below.
                    if (safe_execz && safe_execz->count(in.pc)) break;
                    ok = false;
                    break;
                case 0x09:                                         // s_cbranch_execnz
                    ok = false;
                    break;
                default: ok = false;   // s_branch / s_sendmsg / s_setreg / etc. -> reject
            }
            return true;
        }
        case Rdna2Format::SMEM: {
            // Scalar memory load. Modeled as a load from a single bound constant buffer indexed by the
            // immediate byte offset (>>2 -> dword index); SBASE/descriptor base is folded into the
            // binding. N consecutive dwords -> SDATA..SDATA+N-1. Only the compute path binds the cbuf,
            // so reject in graphics stages (allow_smem=false). Register-offset SMEM not yet handled.
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n = 1;  break;   // s_load_dword     / s_buffer_load_dword
                case 0x1: case 0x9: n = 2;  break;   // s_load_dwordx2   / s_buffer_load_dwordx2
                case 0x2: case 0xA: n = 4;  break;   // s_load_dwordx4   / s_buffer_load_dwordx4
                case 0x3: case 0xB: n = 8;  break;   // s_load_dwordx8   / s_buffer_load_dwordx8
                case 0x4: case 0xC: n = 16; break;   // s_load_dwordx16  / s_buffer_load_dwordx16
                default: ok = false; return true;    // stores / others not yet
            }
            // PC-relative scalar embedded table (#1054): this s_buffer_load consumes a descriptor
            // built from s_getpc_b64 and a bounded table carried by the shader blob. Resolve it before
            // the external-resource gate, exactly like the established MUBUF form. The byte offset is
            // still the real tracked scalar value; an untracked SOFFSET remains fail-visible.
            auto pcrel = rs.smem_pcrel_tables.find(in.pc);
            if (pcrel != rs.smem_pcrel_tables.end()) {
                uint32_t soff = 0;
                const bool soff_null = in.src[1].kind == OperandKind::Special &&
                                       in.src[1].value == 125;
                if (soff_null) {
                    soff = b.uconst(0);
                } else if (in.src[1].kind == OperandKind::SGPR ||
                           (in.src[1].kind == OperandKind::Special &&
                            in.src[1].value >= 106 && in.src[1].value <= 123)) {
                    auto tracked = rs.sreg.find(in.src[1].value);
                    if (tracked == rs.sreg.end()) { ok = false; return true; }
                    soff = tracked->second;
                } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
                    soff = b.uconst(static_cast<uint32_t>(in.src[1].value));
                } else {
                    ok = false; return true;
                }
                uint32_t idx = b.ibin(Op_ShiftRightLogical,
                    b.ibin(Op_IAdd, soff, b.uconst(in.literal)), b.uconst(2));
                for (uint32_t k = 0; k < n; ++k) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    uint32_t value = b.uconst(0); // bounded V# OOB contract
                    for (uint32_t t = 0; t < pcrel->second.size(); ++t)
                        value = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)),
                                      b.uconst(pcrel->second[t]), value);
                    rs.sreg[in.dst.value + static_cast<int>(k)] = value;
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                return true;
            }
            if (!allow_smem) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[smem-reject] pc=%u reason=graphics-disabled op=0x%x\n",
                            in.pc, in.opcode);
                ok = false; return true;
            }
            // GTA V 0x413ce6000 partitions the active wave by a read-first-lane selector, multiplies
            // that selector by the outer V# stride, then reads one four-dword inner V# at pc153.
            // The front half has proved, for this exact dispatch, that every post-MUL SOFFSET is
            // either the sole reachable in-bounds record (480 bytes, record 4) or wholly outside the
            // 600-byte scalar-buffer range. Materialize the architectural per-dword SMEM result here:
            // selected record words for 480, zero otherwise. The complete-dispatch authority bit is
            // deliberately required in addition to the serialized marker shape.
            const ShaderResource* selected_sbuffer =
                rt && in.opcode == kSmemOpcodeBufferLoadDwordX4
                    ? rt->by_fetch_pc(in.pc) : nullptr;
            const bool exact_gta5_selected_sbuffer_site =
                in.pc == 153u && in.fmt == Rdna2Format::SMEM && in.len_dwords == 2u &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == 8 &&
                in.src[0].kind == OperandKind::SGPR && in.src[0].value == 4 &&
                in.src[1].kind == OperandKind::Special && in.src[1].value == 106 &&
                in.literal == 8u && in.words[0] == 0xf4280202u &&
                in.words[1] == 0xd4000008u;
            if (selected_sbuffer &&
                is_gta5_selected_sbuffer_marker_candidate(*selected_sbuffer)) {
                if (!b.gta5_selected_sbuffer_dispatch_validated ||
                    !is_gta5_selected_sbuffer_descriptor(*selected_sbuffer) ||
                    !exact_gta5_selected_sbuffer_site || n != 4u) {
                    ok = false;
                    return true;
                }
                if (selected_sbuffer->selected_sbuffer_soffset ==
                        kGtaSelectedSbufferZeroChainSoffset ||
                    selected_sbuffer->selected_sbuffer_soffset ==
                        kGtaSelectedSbufferAllOobSoffset ||
                    selected_sbuffer->selected_sbuffer_soffset ==
                        kGtaSelectedSbufferNullRecord4Soffset) {
                    for (uint32_t k = 0; k < n; ++k) {
                        rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0u);
                        rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                    }
                    return true;
                }
                const auto soff = rs.sreg.find(in.src[1].value);
                if (soff == rs.sreg.end()) {
                    ok = false;
                    return true;
                }
                const uint32_t selected = b.ucmp(
                    Op_IEqual, soff->second,
                    b.uconst(selected_sbuffer->selected_sbuffer_soffset));
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + static_cast<int>(k)] = b.sel(
                        selected,
                        b.uconst(selected_sbuffer->selected_sbuffer_words[k]),
                        b.uconst(0u));
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                return true;
            }
            // The front half proved all four live SBASE words at this exact scalar-buffer load and
            // proved that its minimum unsigned offset begins beyond the effective scalar bound. The
            // result is therefore exact zero without declaring or touching a dummy storage binding.
            // Check this before SOFFSET tracking: even an unknown runtime offset cannot reduce it.
            const ShaderResource* zero_record_sbuffer =
                rt && in.opcode >= 0x8u ? rt->by_fetch_pc(in.pc) : nullptr;
            if (zero_record_sbuffer && is_zero_record_raw_buffer(*zero_record_sbuffer)) {
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                return true;
            }
            // The whole-CFG proof established that this register-offset or aligned-immediate
            // S_LOAD_DWORDX2 feeds only a complete V# (possibly after an exact scalar carry-pair
            // relocation), never ordinary scalar/address or mask state. Every descriptor observation
            // resolves through an exact-PC resource, so no runtime load belongs in the emitted module.
            // Keep this before generic SOFFSET handling: the offset selects front-half provenance,
            // not emitted scalar data.
            if (rt && in.opcode == kSmemOpcodeLoadDwordX2 &&
                rs.smem_x2_descriptor_fragment_loads.contains(in.pc)) {
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                return true;
            }
            // SOFFSET handling. Immediate-only loads encode SOFFSET = SGPR_NULL (125). A register
            // SOFFSET adds an SGPR-computed byte offset:
            //  * a DESCRIPTOR s_load (x4/x8 = V#/T#) with a computed offset is the bindless fetch's
            //    V#-table read (`s_load_dwordx4 s[8:11],s[24:25],vcc_hi`) — the const-fold
            //    (resolve_dynamic_fetch -> by_fetch_pc) resolves the fetch through that V#, so the
            //    SGPR result is unused per-invocation — no-op it (placeholder 0) so the VS still
            //    recompiles (the "mask=0xF draws never render" gap on PPSA01885). CONFIDENCE: HIGH.
            //  * an s_buffer_load (0x8..0xC) with a TRACKED scalar offset is a computed constant-
            //    buffer read (DOLL's bloom-combine PS: per-tap weights at `vcc_lo = 16*(tap/2)`
            //    inside its counted loop, #273) — model it as a DYNAMIC dword index into the
            //    resolved cbuf binding: idx = (soffset + imm) >> 2. An UNTRACKED offset register
            //    (a runtime user-SGPR we have no value for) still rejects — never fold as 0.
            const bool soff_null = (in.src[1].kind == OperandKind::Special && in.src[1].value == 125);
            uint32_t soff_bits = 0; bool soff_dyn = false;
            if (!soff_null) {
                if (rt && (in.opcode == 0x2 || in.opcode == 0x3)) {
                    for (uint32_t k = 0; k < n; k++) rs.sreg[in.dst.value + (int)k] = b.uconst(0);
                    return true;
                }
                bool tracked = false;
                if (in.opcode >= 0x8 && in.opcode <= 0xC) {
                    if (in.src[1].kind == OperandKind::SGPR ||
                        (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 && in.src[1].value <= 123)) {
                        auto it = rs.sreg.find(in.src[1].value);
                        if (it != rs.sreg.end()) { soff_bits = it->second; tracked = true; }
                    } else if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0) {
                        soff_bits = b.uconst((uint32_t)in.src[1].value); tracked = true;
                    }
                }
                if (!tracked) {
                    if (getenv("PROSPER_DBG"))
                        fprintf(stderr,
                                "[smem-reject] pc=%u reason=untracked-soffset op=0x%x "
                                "src1-kind=%d src1=%d\n",
                                in.pc, in.opcode, static_cast<int>(in.src[1].kind),
                                in.src[1].value);
                    ok = false; return true;
                }
                soff_dyn = true;
            } else if ((int32_t)in.literal < 0) { ok = false; return true; }   // negative imm-only would wrap
            uint32_t base_idx = soff_dyn ? 0 : in.literal >> 2;    // immediate byte offset -> dword index
            // Descriptor provenance: pick which bound constant buffer via the resource table, routing this
            // load to that buffer's OWN binding (N-buffer model) — so Unity's several constant buffers
            // (per-draw transform, per-frame, …) don't collapse onto one. For s_buffer_load, SBASE
            // (src[0]) is the V#: resolve it by an earlier s_load's SRT tag (indirect) or directly by its
            // user-data SGPR index (the V# was placed in SGPRs by the driver). Default binding 2.
            uint32_t binding = 2; bool cbuf_resolved = false;
            const ShaderResource* cbuf_resource = nullptr;
            if (rt) { const ShaderResource* res = nullptr;
                // Descriptor-table folding emits pc-only entries when a V# has no stable SRT key
                // (or that key collides). Exact per-use provenance must win, as it does for MUBUF/MIMG.
                res = rt->by_fetch_pc(in.pc);
                if (res && res->cls != ResourceClass::ConstantBuffer) res = nullptr;
                uint32_t srt_tag = 0;
                if (!res && sreg_srt_range_tag(rs, in.src[0].value, 4, srt_tag))
                    res = rt->by_srt_offset(srt_tag);
                // A scalar buffer load reads a CONSTANT buffer — resolve the SBASE SGPR to a constant
                // buffer specifically (the same SGPR may also hold a vertex-buffer V# elsewhere).
                if (!res && !sreg_range_written(rs, in.src[0].value, 4))
                    res = rt->by_sgpr_base_cls(in.src[0].value, ResourceClass::ConstantBuffer);
                if (res) {
                    binding = res->binding;
                    cbuf_resolved = true;
                    cbuf_resource = res;
                }
            }
            if (getenv("PROSPER_CBUFLOG"))
                fprintf(stderr, "[cbuf] pc=%u s_buffer_load x%u src0=s%d off=0x%x(dw%u) dyn=%d -> binding=%u %s\n",
                        in.pc, n, in.src[0].value, in.literal, base_idx, (int)soff_dyn, binding,
                        cbuf_resolved ? "resolved" : "DEFAULT-2");
            // Immediate s_load_dwordx4/x8 is the same descriptor-table fetch as the dynamic form
            // handled above. When the front-half table already decoded that V#/T#/S#, its raw words
            // are provenance only: emitting loads from fallback binding 2 creates a statically-used
            // descriptor the runtime table does not contain (#515). Preserve the SRT tag and use
            // placeholders, exactly like the dynamic descriptor-fetch path. A genuinely resolved
            // constant-buffer resource still executes the load below.
            if (rt && !cbuf_resolved && (in.opcode == 0x2 || in.opcode == 0x3)) {
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + (int)k] = b.uconst(0);
                    rs.sreg_srt[in.dst.value + (int)k] = in.literal;
                }
                return true;
            }
            // GTA V packs two adjacent T# descriptors into one immediate S_LOAD_DWORDX16. Admit
            // only the load PCs certified by the whole-stream proof above. The values are resource
            // provenance, not scalar data used by the emitted module, so placeholders are exact;
            // unlike x4/x8, DO NOT attach the load's one SRT offset to either half. Both resources
            // are key-less and every MIMG consumer resolves by its own fetch_pc.
            if (rt && !cbuf_resolved && in.opcode == kSmemOpcodeLoadDwordX16 &&
                rs.smem_x16_descriptor_loads.contains(in.pc)) {
                for (uint32_t k = 0; k < n; ++k) {
                    rs.sreg[in.dst.value + static_cast<int>(k)] = b.uconst(0);
                    rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
                }
                // A shared tag here aliases both halves. Keep the invariant executable so the
                // paired-resource regression fails under a same-site tag mutation.
                uint32_t accidental_tag = 0;
                if (sreg_srt_range_tag(rs, in.dst.value, 8, accidental_tag) ||
                    sreg_srt_range_tag(rs, in.dst.value + 8, 8, accidental_tag)) {
                    ok = false;
                }
                return true;
            }
            // A data load with a runtime table may retain the legacy binding-2 convention only when
            // that table actually binds a buffer there. VS/compute tables start at binding 2 and use
            // this path legitimately; PS tables start at 32, where keeping the fallback would emit an
            // interface the renderer cannot satisfy (#719).
            if (rt && !cbuf_resolved) {
                const auto fallback = std::find_if(
                    rt->resources.begin(), rt->resources.end(), [](const ShaderResource& resource) {
                        return resource.binding == 2 &&
                               (resource.cls == ResourceClass::ConstantBuffer ||
                                resource.cls == ResourceClass::VertexBuffer);
                    });
                const bool fallback_bound = fallback != rt->resources.end();
                if (!fallback_bound) {
                    if (getenv("PROSPER_DBG"))
                        fprintf(stderr,
                                "[smem-reject] pc=%u reason=unresolved-cbuf op=0x%x "
                                "src0=s%d dyn=%d\n",
                                in.pc, in.opcode, in.src[0].value, (int)soff_dyn);
                    ok = false; return true;
                }
                cbuf_resource = &*fallback;
            }
            // RDNA2 scalar-buffer loads use m_size=(STRIDE==0 ? 1 : NUM_RECORDS) dwords. STRIDE is
            // only a bounds-mode selector here; it does not scale scalar addresses. Preserve the
            // explicit scalar count: size/stride cannot reconstruct it after the binding is staged
            // to the independent dword-addressable span (and is outright shorter when STRIDE < 4).
            // For every component select index zero before OpAccessChain, then select architectural
            // zero after the safe load; OpSelect does not make an already-OOB Vulkan load harmless.
            const bool scalar_buffer_load = in.opcode >= 0x8u && in.opcode <= 0xCu;
            const uint32_t scalar_bound_dwords =
                scalar_buffer_load && cbuf_resource
                    ? cbuf_resource->scalar_buffer_dword_count : 0u;
            const bool scalar_bound_known = scalar_bound_dwords != 0u;
            if (scalar_bound_known &&
                shader_resource_buffer_binding_bytes(*cbuf_resource) <
                    static_cast<uint64_t>(scalar_bound_dwords) * sizeof(uint32_t)) {
                ok = false;
                return true;
            }
            auto bounded_cbuf_load = [&](uint32_t dword_index) {
                if (!scalar_bound_known)
                    return b.cbuf_load(dword_index, binding);
                if (scalar_bound_dwords == 0u)
                    return b.uconst(0u);
                const uint32_t in_bounds = b.ucmp(
                    Op_ULessThan, dword_index, b.uconst(scalar_bound_dwords));
                const uint32_t safe_index = b.sel(in_bounds, dword_index, b.uconst(0u));
                const uint32_t loaded = b.cbuf_load(safe_index, binding);
                return b.sel(in_bounds, loaded, b.uconst(0u));
            };
            // #2481: this exact instruction is the producer of a runtime-selected descriptor table.
            // Its SOFFSET is the byte offset of the chosen record, so the element index is
            // SOFFSET / record-stride. Publish it as the live selector for that array binding before
            // the consumer runs; the consumer is a later instruction on the same path, so the value
            // dominates it. `cbuf_element_ptr` still range-checks the index against the declared
            // arity, so an out-of-range runtime selector reads element zero rather than out of
            // bounds -- and the front half proved every declared element is a valid descriptor.
            if (rt && soff_dyn) {
                for (const ShaderResource& resource : rt->resources) {
                    if (resource.table_selector_mode !=
                            BufferTableSelectorMode::DynamicSbufferByteOffset ||
                        resource.table_load_pc != in.pc || resource.table_index_count == 0u ||
                        resource.table_entry_stride == 0u)
                        continue;
                    b.cbuf_table_selector_value[resource.binding] =
                        b.ibin(Op_UDiv, soff_bits, b.uconst(resource.table_entry_stride));
                    b.cbuf_table_selector_block[resource.binding] = b.cur_block;
                    if (getenv("PROSPER_DBG"))
                        fprintf(stderr,
                                "[selected-table] producer pc=%u binding=%u stride=%u arity=%u\n",
                                in.pc, resource.binding, resource.table_entry_stride,
                                resource.table_index_count);
                }
            }
            if (soff_dyn) {
                // Dynamic dword index: (soffset + signed imm) >> 2 (uint add == two's-complement add).
                uint32_t idx0 = b.ibin(Op_ShiftRightLogical,
                                       b.ibin(Op_IAdd, soff_bits, b.uconst(in.literal)), b.uconst(2));
                for (uint32_t k = 0; k < n; k++) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx0, b.uconst(k)) : idx0;
                    rs.sreg[in.dst.value + (int)k] = bounded_cbuf_load(kidx);
                    rs.sreg_srt.erase(in.dst.value + (int)k);   // data load: drop any stale descriptor tag
                }
                return true;
            }
            for (uint32_t k = 0; k < n; k++)
                rs.sreg[in.dst.value + (int)k] =
                    bounded_cbuf_load(b.uconst(base_idx + k));
            // A wide scalar load is a descriptor fetch — tag its dest SGPRs with the SRT offset so a
            // later buffer/image op using them resolves to the right resource (provenance). x4 = V#/S#
            // (buffers, samplers), x8 = T# (textures, 8 dwords).
            if (rt && (in.opcode == 0x2 || in.opcode == 0x3)) {
                for (uint32_t k = 0; k < n; k++) rs.sreg_srt[in.dst.value + (int)k] = in.literal;
            } else {
                // Scalar data loads overwrite the destination; they do not carry descriptor identity.
                for (uint32_t k = 0; k < n; k++) rs.sreg_srt.erase(in.dst.value + (int)k);
            }
            return true;
        }
        case Rdna2Format::FLAT: {
            const FlatAccessInfo access = flat_access_info(in.opcode);
            const IndirectPointerAccessProof* relocated_access =
                b.indirect_pointer_proof
                    ? rdna2_indirect_pointer_access(*b.indirect_pointer_proof, in)
                    : nullptr;
            if (relocated_access) {
                const bool static_footprint =
                    b.indirect_pointer_proof->bound_kind ==
                        IndirectPointerBoundKind::StaticFootprint &&
                    b.indirect_pointer_proof->guard_kind ==
                        IndirectPointerGuardKind::Full64NonZero;
                const bool descriptor_range =
                    b.indirect_pointer_proof->bound_kind ==
                        IndirectPointerBoundKind::DescriptorRange &&
                    b.indirect_pointer_proof->guard_kind ==
                        IndirectPointerGuardKind::None;
                const uint64_t directory_end =
                    static_cast<uint64_t>(b.indirect_pointer_segment_directory_byte_offset) +
                    static_cast<uint64_t>(b.indirect_pointer_segment_count) *
                        kIndirectBufferRelocationSegmentBytes;
                const uint64_t record_directory_end =
                    static_cast<uint64_t>(b.indirect_pointer_record_directory_byte_offset) +
                    static_cast<uint64_t>(b.indirect_pointer_record_count) *
                        kIndirectBufferRelocationRecordBytes;
                if (!access.valid || access.store || access.bits != 32u ||
                    access.components != 1u || in.flat_segment != 2u || in.flat_lds ||
                    in.flat_glc || in.flat_slc || in.flat_dlc ||
                    in.dst.kind != OperandKind::VGPR || in.dst.value < 0 ||
                    in.src[0].kind != OperandKind::VGPR ||
                    in.src[0].value < 0 ||
                    static_cast<uint32_t>(in.src[0].value) !=
                        relocated_access->address_vgpr ||
                    relocated_access->address_vgpr >= 255u ||
                    in.src[1].kind != OperandKind::Special || in.src[1].value != 125 ||
                    in.literal != relocated_access->immediate_byte_offset ||
                    relocated_access->component_bytes != sizeof(uint32_t) ||
                    relocated_access->components != 1u ||
                    b.indirect_pointer_proof->schema_version != 1u ||
                    (!static_footprint && !descriptor_range) ||
                    b.indirect_pointer_binding == UINT32_MAX ||
                    b.indirect_pointer_segment_directory_byte_offset % sizeof(uint32_t) != 0u ||
                    b.indirect_pointer_payload_byte_offset % sizeof(uint32_t) != 0u ||
                    b.indirect_pointer_carrier_bytes % sizeof(uint32_t) != 0u ||
                    b.indirect_pointer_carrier_bytes < sizeof(uint32_t) ||
                    directory_end > b.indirect_pointer_payload_byte_offset ||
                    b.indirect_pointer_payload_byte_offset >
                        b.indirect_pointer_carrier_bytes) {
                    ok = false;
                    return true;
                }
                if (descriptor_range &&
                    (!b.indirect_pointer_record_count ||
                     !b.indirect_pointer_source_stride ||
                     !b.indirect_pointer_source_record_var ||
                     !b.indirect_pointer_source_root_lo_var ||
                     !b.indirect_pointer_source_root_hi_var ||
                     b.indirect_pointer_record_directory_byte_offset % sizeof(uint32_t) != 0u ||
                     b.indirect_pointer_record_directory_byte_offset <
                         b.indirect_pointer_source_bytes ||
                     record_directory_end !=
                         b.indirect_pointer_segment_directory_byte_offset ||
                     b.indirect_pointer_proof->source_address_kind !=
                         IndirectBufferRelocationRecord::SourceAddressKind::
                             BufferDescriptorBase48)) {
                    ok = false;
                    return true;
                }
                const int address_vgpr =
                    static_cast<int>(relocated_access->address_vgpr);
                const uint32_t address_lo = vreg_old(b, rs, address_vgpr);
                const uint32_t address_hi = vreg_old(b, rs, address_vgpr + 1);
                const uint32_t value = descriptor_range
                    ? b.relocated_indirect_descriptor_load_dword(
                          address_lo, address_hi,
                          relocated_access->immediate_byte_offset)
                    : b.relocated_indirect_load_dword(
                          address_lo, address_hi,
                          relocated_access->immediate_byte_offset);
                const int reg = in.dst.value;
                const uint32_t old_value = vreg_old(b, rs, reg);
                rs.vreg[reg] = value;
                predicate_write(b, rs, reg, old_value);
                return true;
            }
            IndirectBufferShadowAccess packed_access{};
            if (b.indirect_buffer_dispatch_validated &&
                rdna2_gta5_packed_pointer_access(in, packed_access)) {
                if (!access.valid || access.store || access.bits != 32u || in.flat_lds ||
                    in.src[0].kind != OperandKind::VGPR || in.src[0].value != 0 ||
                    in.src[1].kind != OperandKind::Special || in.src[1].value != 125 ||
                    access.components != packed_access.components ||
                    b.indirect_buffer_binding == UINT32_MAX ||
                    b.indirect_buffer_slot_count == 0u ||
                    !b.indirect_buffer_contract_tag || !b.indirect_buffer_header_bytes ||
                    !b.indirect_buffer_slot_bytes) {
                    ok = false;
                    return true;
                }
                const uint32_t address_lo = vreg_old(b, rs, 0);
                const uint32_t address_hi = vreg_old(b, rs, 1);
                const uint32_t relative = b.ibin(
                    Op_ISub, address_lo,
                    b.uconst(b.indirect_buffer_source_bytes +
                             b.indirect_buffer_header_bytes));
                const uint32_t aligned = b.ucmp(
                    Op_IEqual,
                    b.ibin(Op_UMod, relative, b.uconst(b.indirect_buffer_slot_bytes)),
                    b.uconst(0));
                const uint32_t slot = b.ibin(
                    Op_UDiv, relative, b.uconst(b.indirect_buffer_slot_bytes));
                uint32_t valid = b.ucmp(
                    Op_IEqual, address_hi, b.uconst(b.indirect_buffer_contract_tag));
                valid = b.land(valid, aligned);
                valid = b.land(valid,
                               b.ucmp(Op_ULessThan, slot,
                                      b.uconst(b.indirect_buffer_slot_count)));
                for (uint32_t component = 0; component < packed_access.components; ++component) {
                    const uint32_t byte_address = b.ibin(
                        Op_IAdd, address_lo,
                        b.uconst(packed_access.byte_offset + component * sizeof(uint32_t)));
                    // OpSelect is not short-circuiting: select a safe address before OpLoad, then
                    // separately force architectural zero for a malformed runtime tag.
                    const uint32_t safe_byte = b.sel(valid, byte_address, b.uconst(0));
                    const uint32_t index = b.ibin(
                        Op_ShiftRightLogical, safe_byte, b.uconst(2));
                    const uint32_t loaded = b.cbuf_load(
                        index, b.indirect_buffer_binding);
                    const uint32_t value = b.sel(valid, loaded, b.uconst(0));
                    const int reg = in.dst.value + static_cast<int>(component);
                    const uint32_t old_value = vreg_old(b, rs, reg);
                    rs.vreg[reg] = value;
                    predicate_write(b, rs, reg, old_value);
                }
                return true;
            }
            // General (non-scratch) FLAT LOAD from a raw 64-bit guest address (#1171). If the executor
            // resolved this load's base pointer to user SGPRs s[flat_base_sgpr : +1] and bound the
            // containing guest allocation as an SSBO (a ConstantBuffer-class resource keyed by this
            // load's pc), lower it to an indexed read of that window at byte offset (address_lo -
            // base_lo). base_lo is the ORIGINAL push-constant value (the address the executor bound the
            // window at), so the module stays dispatch-independent and correct even if the shader later
            // reuses the base SGPR. Unresolved forms fall through to the scratch/reject path below.
            const ShaderResource* fw =
                (rt && access.valid && !access.store && !in.flat_lds &&
                 in.src[0].kind == OperandKind::VGPR &&
                 in.src[1].kind == OperandKind::Special && in.src[1].value == 125)
                    ? rt->by_fetch_pc(in.pc) : nullptr;
            if (fw && fw->flat_base_sgpr != 0xFFFFFFFFu) {
                const uint32_t addr_lo = val(in.src[0]);                     // low dword of the address
                const uint32_t base_lo = b.load_push_constant(fw->flat_base_sgpr);
                // Byte offset in the window: (address - base) mod 2^32, which equals the true offset for
                // any 0 <= offset < window <= 256 MiB (the low-dword subtraction wraps complementarily to
                // the address's own IAdd). A negative offset (address < base) becomes a huge unsigned index
                // -> out-of-window -> robustBufferAccess returns 0 (defined, but a loose-bounds divergence
                // from HW; decode kernels use non-negative offsets).
                const uint32_t byte0 = b.ibin(Op_ISub, addr_lo, base_lo);
                for (uint32_t c = 0; c < access.components; ++c) {
                    const uint32_t addr = access.bits == 32
                        ? b.ibin(Op_IAdd, byte0, b.uconst(c * 4)) : byte0;
                    const uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
                    uint32_t value;
                    if (access.bits == 32) {
                        value = b.cbuf_load(idx, fw->binding);
                    } else {
                        // Sub-dword (ubyte/ushort) extract, mirroring the MUBUF raw path: a 16-bit
                        // access may begin at byte 3 and straddle two dwords, so join the adjacent words.
                        const uint32_t dw0 = b.cbuf_load(idx, fw->binding);
                        const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
                        const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
                        uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
                        if (access.bits == 16) {
                            const uint32_t dw1 =
                                b.cbuf_load(b.ibin(Op_IAdd, idx, b.uconst(1)), fw->binding);
                            const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                                b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                            const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                            joined = b.ibin(Op_BitwiseOr, joined,
                                b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                        }
                        value = access.sign_extend
                            ? b.bfe_s(joined, b.uconst(0), b.uconst(access.bits))
                            : b.bfe_u(joined, b.uconst(0), b.uconst(access.bits));
                    }
                    const int reg = in.dst.value + static_cast<int>(c);
                    const uint32_t old_value = vreg_old(b, rs, reg);
                    rs.vreg[reg] = value;
                    predicate_write(b, rs, reg, old_value);
                }
                return true;
            }
            // Model the compiler's private spill area only. Each generated shader invocation owns
            // one Function-storage array, so the entry-provided hardware base has no host-visible
            // address to preserve. Other FLAT/GLOBAL forms remain deliberately unsupported.
            if (!access.valid || in.flat_segment != 1u || in.flat_lds || !b.guest_scratch ||
                in.src[0].kind != OperandKind::None || in.src[1].kind != OperandKind::SGPR ||
                in.src[1].value != b.guest_scratch_saddr ||
                rs.sreg_written.count(in.src[1].value)) {
                ok = false;
                return true;
            }
            const int32_t base = static_cast<int32_t>(in.literal);
            const int64_t storage_begin = b.guest_scratch_min_byte;
            const int64_t storage_end = storage_begin + static_cast<int64_t>(b.guest_scratch_dwords) * 4;
            for (uint32_t component = 0; component < access.components; ++component) {
                const int32_t byte_offset = base +
                    static_cast<int32_t>(component * (access.bits / 8u));
                const int64_t access_end = static_cast<int64_t>(byte_offset) + access.bits / 8u;
                if (byte_offset < storage_begin || access_end > storage_end) {
                    ok = false;
                    return true;
                }
                const int reg = in.dst.value + static_cast<int>(component);
                if (access.store) {
                    b.guest_scratch_store_bits(byte_offset, access.bits, vreg_old(b, rs, reg),
                                               rs.exec_narrowed, rs.exec);
                } else {
                    const uint32_t old_value = vreg_old(b, rs, reg);
                    rs.vreg[reg] = b.guest_scratch_load_bits(byte_offset, access.bits,
                                                            access.sign_extend);
                    predicate_write(b, rs, reg, old_value);
                }
            }
            return true;
        }
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: {
            // Untyped buffer LOAD — the per-lane fetch mechanism (vertex fetch et al.). Modeled as a
            // per-lane load from the bound constant buffer: byte addr = (offen ? VADDR : 0) + SOFFSET
            // + inst-offset; index = addr>>2; N dwords -> VDATA..+N-1. Descriptor (SRSRC), idxen*stride,
            // and the format-converting buffer_load_format_* variants are deferred. Compute-only (cbuf).
            // LDS bit set = transfer between LDS and memory instead of VGPRs (ISA Table 98). The
            // VDATA VGPRs stay untouched on hardware and the data lands in LDS — translating it as
            // a VGPR load would clobber a live register AND drop the LDS write. Reject until a
            // buffer->LDS model exists.
            if (in.fmt == Rdna2Format::MUBUF && in.mubuf_lds) { ok = false; return true; }
            // MTBUF opcodes 8..15 pack D16 results/inputs two per VGPR. Reusing the ordinary or raw
            // MUBUF cases below would silently apply the wrong register layout, so fail closed until
            // that packing is modeled.
            if (in.fmt == Rdna2Format::MTBUF && in.opcode >= 8u) { ok = false; return true; }
            // TFE appends a fault/status result after the data VGPRs. Dropping that write can corrupt
            // later register consumers, so reject until status-return semantics are implemented.
            if ((in.fmt == Rdna2Format::MUBUF && in.mubuf_tfe) ||
                (in.fmt == Rdna2Format::MTBUF && in.mtbuf_tfe)) {
                ok = false;
                return true;
            }
            uint32_t n = 0; bool is_format = false, is_store = false, is_atomic = false;
            bool is_atomic_x2 = false, is_atomic_fminmax = false, atomic_fmin = false;
            uint32_t atomic_op = 0;   // SPIR-V atomic RMW opcode for is_atomic (set by the switch)
            uint32_t atomic_x2_record_count = 0;
            bool raw_subword = false, raw_signed = false;
            uint32_t raw_bits = 32;
            switch (in.opcode) {
                case 0x8: n = 1; raw_subword = true; raw_bits = 8; break;   // buffer_load_ubyte
                case 0x9: n = 1; raw_subword = true; raw_signed = true; raw_bits = 8; break; // sbyte
                case 0xA: n = 1; raw_subword = true; raw_bits = 16; break;  // buffer_load_ushort
                case 0xB: n = 1; raw_subword = true; raw_signed = true; raw_bits = 16; break; // sshort
                case kMubufOpcodeLoadDword:   n = 1; break;
                case kMubufOpcodeLoadDwordX2: n = 2; break;
                case kMubufOpcodeLoadDwordX4: n = 4; break;
                case kMubufOpcodeLoadDwordX3: n = 3; break; // x3 sorts after x4 (llvm-mc gfx1010)
                case 0x0: n = 1; is_format = true; break;   // buffer_load_format_x  (vertex fetch)
                case 0x1: n = 2; is_format = true; break;   // buffer_load_format_xy
                case 0x2: n = 3; is_format = true; break;   // buffer_load_format_xyz
                case 0x3: n = 4; is_format = true; break;   // buffer_load_format_xyzw
                case kMubufOpcodeStoreDword: n = 1; is_store = true; break;
                case kMubufOpcodeStoreDwordX2: n = 2; is_store = true; break;
                case kMubufOpcodeStoreDwordX4: n = 4; is_store = true; break;
                case kMubufOpcodeStoreDwordX3: n = 3; is_store = true; break;
                case 0x4: n = 1; is_format = true; is_store = true; break;   // buffer_store_format_x
                case 0x5: n = 2; is_format = true; is_store = true; break;   // buffer_store_format_xy
                case 0x6: n = 3; is_format = true; is_store = true; break;   // buffer_store_format_xyz
                case 0x7: n = 4; is_format = true; is_store = true; break;   // buffer_store_format_xyzw
                // 32-bit atomic RMW family (RDNA2 MUBUF opcodes 0x30..0x3b). Each does
                // mem = mem OP VDATA and returns the PRE-op value in VDATA. They all share the generic
                // OpAtomic<Op>(ptr, Device, AcqRel, value) shape emitted by cbuf_atomic_rtn. VDATA is one
                // dword. CMPSWAP (0x31, two-operand), CSUB (0x34, conditional-subtract), INC/DEC
                // (0x3c/0x3d, wrap semantics), and most x2 64-bit variants stay deferred
                // (fail-visible) via the default case below. Opcodes 0x50/0x5a have separately
                // guarded GTA V lowerings: one true qword RMW, never two ordinary 32-bit atomics.
                case kMubufOpcodeAtomicSwap:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicExchange; break;
                case kMubufOpcodeAtomicAdd:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicIAdd; break;
                case kMubufOpcodeAtomicSub:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicISub; break;
                case kMubufOpcodeAtomicSmin:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicSMin; break;
                case kMubufOpcodeAtomicUmin:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicUMin; break;
                case kMubufOpcodeAtomicSmax:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicSMax; break;
                case kMubufOpcodeAtomicUmax:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicUMax; break;
                case kMubufOpcodeAtomicAnd:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicAnd; break;
                case kMubufOpcodeAtomicOr:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicOr; break;
                case kMubufOpcodeAtomicXor:
                    n = 1; is_atomic = true; atomic_op = Op_AtomicXor; break;
                case kMubufOpcodeAtomicFmin:
                    n = 1; is_atomic = true; is_atomic_fminmax = true;
                    atomic_fmin = true; break;
                case kMubufOpcodeAtomicFmax:
                    n = 1; is_atomic = true; is_atomic_fminmax = true;
                    atomic_fmin = false; break;
                case kMubufOpcodeAtomicSwapX2:
                    n = 2; is_atomic = true; is_atomic_x2 = true;
                    atomic_op = Op_AtomicExchange; break;
                case kMubufOpcodeAtomicOrX2:
                    n = 2; is_atomic = true; is_atomic_x2 = true;
                    atomic_op = Op_AtomicOr; break;
                default: ok = false; return true;           // remaining typed/atomic opcodes deferred
            }
            // Per-instruction buffer-op disposition (PROSPER_DBG). A buffer op whose V# decodes to
            // NUM_RECORDS=0 — or that matches one of the no-backing markers below — is folded away:
            // loads become a constant, stores are dropped, and no backing buffer is declared. That is
            // architecturally correct for an empty descriptor. It is also, in the emitted SPIR-V,
            // INDISTINGUISHABLE from an instruction that never reached the emitter at all: both leave
            // no access chain, no load, no trace of any kind. Those two have completely different
            // causes, and offline dissection of a program whose buffer traffic has vanished cannot
            // separate them without this line (#2709 built an argument for a whole new memory model
            // on exactly that ambiguity, and was wrong).
            //
            // Reported from a scope guard rather than at each decision site so EVERY exit path emits
            // exactly one line. A path nobody classified prints `unclassified` rather than printing
            // nothing — so a census can detect its own incompleteness instead of silently
            // under-counting, which a per-site call cannot do. The three pre-switch rejects above are
            // deliberately outside it: they set ok=false, and a rejected program is already reported
            // by the recompile-reject census -- as is the switch's own `default:` arm, which is
            // also above this point. That is FOUR uncovered rejects, not three, and the `default:`
            // one matters most to a census: an unmodelled buffer opcode is literally the "never
            // reached the emitter" case this line exists to identify. So the tempting identity
            // "instructions in the stream == lines emitted" holds only for a program with none of
            // the four; it happened to hold for 0x413dc6700 (41 == 41) and that is not a guarantee.
            // Hoisting the guard above the switch is not the fix: n/is_store/is_atomic are unset
            // there, so it would print `n=0 store=0 atomic=0`, which is less honest than no line.
            //
            // `rt=` is on the line because `unclassified` alone does not mean "instrument hole".
            // recompile_coverage() translates on a table-less compute shell, so every buffer op it
            // sees takes a no-table path and prints `unclassified` for a reason that has nothing to
            // do with coverage. `rt=0` marks those. A hole is `rt=1 ... unclassified`.
            struct BufOpDisposition {
                bool on;
                const SpirvCompute& b;
                const Rdna2Inst& in;
                const uint32_t& n;
                const bool& is_store;
                const bool& is_atomic;
                const ShaderResourceTable* rt;
                const char* how = "unclassified";
                ~BufOpDisposition() {
                    if (!on) return;
                    std::fprintf(stderr,
                                 "[buf-op] program=0x%llx pc=%u %s op=0x%x n=%u store=%d atomic=%d rt=%d %s\n",
                                 (unsigned long long)b.diagnostic.program_address, in.pc,
                                 in.fmt == Rdna2Format::MUBUF ? "MUBUF" : "MTBUF", in.opcode, n,
                                 (int)is_store, (int)is_atomic, (int)(rt != nullptr), how);
                }
            } buf_op{std::getenv("PROSPER_DBG") != nullptr, b, in, n, is_store, is_atomic, rt};
            uint32_t offset = in.literal & 0xFFFu;
            bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
            const bool indirect_pointer_descriptor_source_candidate =
                b.indirect_pointer_proof &&
                rdna2_indirect_pointer_source(*b.indirect_pointer_proof, in);
            // GLC/DLC ordinary loads must observe device-visible writes rather than a cached value;
            // GLC ordinary stores publish through the device cache. Atomics retain their existing
            // Device/AcquireRelease operands and GLC's separate return-pre-op-value contract below.
            const bool coherent_load = !is_store && !is_atomic &&
                                       (in.mubuf_glc || in.mubuf_dlc);
            const bool coherent_store = is_store && in.mubuf_glc;
            if (b.indirect_buffer_dispatch_validated &&
                rdna2_gta5_packed_pointer_atomic_site(in)) {
                // The exact producer at pc347 forces V# stride 256 and pc349..351 set
                // NUM_RECORDS=1, so pc355's qword at byte 24 is in bounds. Its scalar raw-pointer
                // resource was separately proven readable+writable through byte 31 and expanded to
                // that exact binding range. Emit one true qword OR; GLC=0 preserves v0:v1.
                if (!is_atomic_x2 || atomic_op != Op_AtomicOr || is_store || in.mubuf_glc ||
                    !b.storage_buffer_int64_atomics ||
                    b.indirect_buffer_atomic_binding == UINT32_MAX ||
                    b.indirect_buffer_atomic_byte_offset != offset ||
                    (offset & 7u) != 0u) {
                    buf_op.how = "reject-packed-pointer-atomic";
                    ok = false;
                    return true;
                }
                const auto lo = rs.vreg.find(in.dst.value);
                const auto hi = rs.vreg.find(in.dst.value + 1);
                if (lo == rs.vreg.end() || hi == rs.vreg.end()) {
                    buf_op.how = "reject-packed-pointer-atomic";
                    ok = false;
                    return true;
                }
                const uint32_t value = b.u64_from_lohi(lo->second, hi->second);
                const uint32_t access = rs.exec_narrowed ? rs.exec : b.btrue();
                (void)b.cbuf_atomic_x2_rtn(
                    Op_AtomicOr, b.uconst(offset / 8u), value,
                    b.indirect_buffer_atomic_binding, access, b.uconst64(0u));
                buf_op.how = "packed-pointer-atomic";
                return true;
            }
            // PC-relative EMBEDDED TABLE (#273): this load's V# was built from s_getpc_b64 and the
            // table bytes live inside the shader blob — detect_pcrel_tables already copied them out.
            // Fold to a compile-time constant lookup: dword index = (inst offset + offen VADDR) >> 2;
            // out-of-range indexes read 0 (the hardware's OOB contract for a bounded V#).
            //
            // The TYPED consumer of the same idiom (#2859) folds identically. Sonic Frontiers' three
            // Cyber Space scene kernels build the V# exactly as above and read it with
            // `tbuffer_load_format_x v, v, s[0:3], 0 offen` at BUF_FMT 22 (`32_FLOAT`), twice each --
            // so the untyped-only guard, not the idiom, is what refused them. detect_pcrel_tables
            // admits an MTBUF site only when its format stores 32 bits per component and its
            // component count matches the opcode, which makes the typed fetch a raw dword copy; that
            // proof is what licenses relaxing `is_format` here, and only for a pc it recorded.
            const bool mtbuf_pcrel_fold = in.fmt == Rdna2Format::MTBUF &&
                                          rs.mtbuf_pcrel_tables.count(in.pc) != 0;
            if ((!is_format || mtbuf_pcrel_fold) && !is_store && !is_atomic && !raw_subword) {
                const auto& pcrel_tables = mtbuf_pcrel_fold ? rs.mtbuf_pcrel_tables
                                                            : rs.mubuf_pcrel_tables;
                auto pt = pcrel_tables.find(in.pc);
                if (pt != pcrel_tables.end()) {
                    const std::vector<uint32_t>& tab = pt->second;
                    uint32_t addr = b.uconst(offset);
                    if (offen) { Operand ov{OperandKind::VGPR, in.src[0].value};
                                 addr = b.ibin(Op_IAdd, addr, val(ov)); }
                    uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
                    for (uint32_t k = 0; k < n; k++) {
                        int d = in.dst.value + (int)k;
                        uint32_t old = vreg_old(b, rs, d);
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        uint32_t acc = b.uconst(0);   // OOB -> 0
                        for (uint32_t t = 0; t < (uint32_t)tab.size(); t++)
                            acc = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)), b.uconst(tab[t]), acc);
                        rs.vreg[d] = acc;
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "pcrel-table";
                    return true;
                }
            }
            // A proven PC-relative table is self-contained shader data and needs no runtime resource
            // bindings. Only external buffer accesses require the stage's SMEM/MUBUF resource gate.
            // Keep this check after the fold so resource-free graphics shaders (Astro Bot's loading VS)
            // can consume their embedded lookup table without making unresolved V# accesses permissive.
            if (!allow_smem) { buf_op.how = "reject-no-smem"; ok = false; return true; }
            uint32_t binding = 2, stride = 0;   // overwritten by SRSRC resolution below whenever a resource
                                                // table is present (format AND raw ops); the binding-2 default
                                                // survives only on the table-less offline path (see below)
            // Format of the fetched components. Untyped buffer_load_dword* is raw 32-bit (comp_bytes=4);
            // buffer_load_format_* takes the format from the resolved V# descriptor.
            DataFormat fmt = DataFormat::Uint32;   // untyped default: raw dwords
            uint32_t fmt_ncomp = 0;    // the V#'s real component count (format loads only); 0 = don't default-fill
            bool dyn_vfetch = false;   // set when the V# came from by_fetch_pc — a const-folded per-vertex
                                       // attribute fetch, whose element address is exactly gl_VertexIndex*stride.
            bool instance_vfetch = false;
            bool folded_vfetch = false; // by-fetch V# base already includes OFFSET/SOFFSET
            const ShaderResource* resolved_buffer = nullptr;
            bool gta5_selected_sbuffer_consumer = false;
            bool indirect_pointer_descriptor_source = false;
            if (is_format) {
                // A format load reads a vertex/buffer attribute — it needs the V# descriptor for the
                // binding, stride, and data format. Resolve SRSRC (src[1]) via provenance: an s_load
                // tag (indirect) else the SGPR index (direct/user-data).
                const ShaderResource* res = nullptr;
                // A format load (vertex fetch) reads a VERTEX buffer — resolve the SRSRC SGPR to a vertex
                // buffer specifically (that SGPR may hold a constant-buffer V# at other points; the const-
                // fold-resolved vertex buffer is keyed by this SRSRC SGPR). Fall back to an s_load SRT tag.
                // Any write to the four-dword SRSRC range invalidates its entry-time direct V#.
                // Exact per-fetch and s_load provenance below may still identify the live descriptor,
                // but a missing/rejected dynamic result must never fall back to stale user data.
                //
                // Hoisted out of the `if (rt)` block below because the REJECT diagnostic needs it:
                // whether the shader built its own SRSRC is the single fact that separates "the
                // resource table has no entry for this fetch" from "it has one but direct-SGPR
                // provenance was deliberately suppressed", and those two point at different files
                // (#3137).
                const bool srsrc_rewritten = sreg_range_written(rs, in.src[1].value, 4);
                if (rt) {
                    // PER-FETCH first: a reloaded SRSRC holds a different V# per attribute, so match this
                    // exact fetch instruction's pc; fall back to untouched SGPR user data or an s_load
                    // SRT tag. A rewritten direct descriptor without either provenance stays unresolved.
                    // Only a VERTEX-buffer pc entry implies the vertex-index address model — a pc-keyed
                    // CONSTANT/structured buffer (a PS's per-lane table fetch, #273) keeps the faithful
                    // VADDR*stride+offset address below.
                    res = rt->by_fetch_pc(in.pc);
                    // A pc-keyed entry is not itself proof that OFFSET/SOFFSET was folded into the
                    // bound base. Shader mode deliberately binds DynFetch::unshifted_desc and must
                    // retain both VADDR terms (idxen+offen), OFFSET and SOFFSET exactly as encoded.
                    folded_vfetch = res && res->cls == ResourceClass::VertexBuffer &&
                                      res->fetch_index_mode != VertexFetchIndexMode::Shader;
                    // The NGG fetch-prologue shortcut applies only to an untouched ABI element index. It
                    // must NOT apply after the shader has selected or computed VADDR. DOLL lays out packed
                    // attributes as two/three descriptor records per vertex and computes
                    // 2*vertex+channel / 3*vertex+channel in v0/v4/v5/v6. DQ also uses a modeled
                    // v_cndmask merged-wave selector to choose instance_id for a per-instance transform
                    // lookup and vertex_id for positions. Replacing either shader value with
                    // gl_VertexIndex reads allocator metadata as transform indices and emits NaN/giant
                    // triangles.
                    // A pc-keyed VertexBuffer can also be a directly supplied structured V# whose SRSRC
                    // was never rewritten; retain the established non-v0 faithful-address behavior there.
                    if (res) {
                        const int vaddr = in.src[0].value;
                        switch (res->fetch_index_mode) {
                            case VertexFetchIndexMode::Vertex:
                                dyn_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Instance:
                                instance_vfetch = true;
                                break;
                            case VertexFetchIndexMode::Shader:
                                break;
                            case VertexFetchIndexMode::Automatic:
                                // Backward-compatible fallback for metadata resources, hand-built
                                // tests, and captures predating explicit dynamic-fold provenance.
                                dyn_vfetch = res->cls == ResourceClass::VertexBuffer &&
                                             (vaddr == 0 || srsrc_rewritten);
                                break;
                        }
                    }
                    // MTBUF must resolve through the exact dynamic-use entry. The fold validates the
                    // live V# FORMAT != INVALID before publishing that pc. Falling back to an older
                    // metadata resource here can resurrect an unbound V# that happens to share its
                    // SGPR/SRT identity. MUBUF retains its established metadata fallbacks.
                    if (in.fmt != Rdna2Format::MTBUF) {
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base_cls(in.src[1].value, ResourceClass::VertexBuffer);
                        uint32_t srt_tag = 0;
                        if (!res && sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag))
                            res = rt->by_srt_offset(srt_tag);
                        // DIRECT user-data V# of any class (#273 — DOLL's title post PSes format-fetch
                        // through a V# the metadata labels a CONSTANT buffer sharp at s[24:27]): the class
                        // label doesn't change the descriptor's fields. Only when the SGPR was never
                        // REWRITTEN in-shader (no rs.sreg entry in its four-dword range) — a reloaded
                        // register no longer holds the seed-time sharp, and trusting it would fetch through
                        // a stale descriptor.
                        if (!res && !srsrc_rewritten)
                            res = rt->by_sgpr_base(in.src[1].value);
                    }
                }
                if (!res) {
                    if (getenv("PROSPER_DBG")) {   // which provenance step failed for this format load
                        uint32_t srt_tag = 0;
                        const bool has_srt_tag = sreg_srt_range_tag(
                            rs, in.src[1].value, 4, srt_tag);
                        const ShaderResource* tagged_res = has_srt_tag && rt
                                                               ? rt->by_srt_offset(srt_tag)
                                                               : nullptr;
                        // `pc_res` and `rewritten` are the two fields the RAW path
                        // (`[mubuf-raw-unresolved]`) has always printed and this one did not, and
                        // their absence is what made #3137 diagnosable only by re-running the title
                        // under PROSPER_DYNTRACE_FAIL: with `pc_res=null rewritten=1` the line says
                        // outright that the const-fold published no descriptor for this exact fetch
                        // AND that the shader assembles its own SRSRC, so the direct-SGPR fallbacks
                        // were correctly suppressed rather than tried and missed. That is a
                        // front-half/user-data verdict, not a recompiler one. `rewritten=0` with
                        // `pc_res=null` is the opposite verdict: the table simply has no entry at
                        // this SGPR.
                        const ShaderResource* pc_res = rt ? rt->by_fetch_pc(in.pc) : nullptr;
                        fprintf(stderr, "[mubuf-unresolved] pc=%u srsrc=s%d srt_tag=%s0x%x key_res=%s "
                                        "pc_res=%s rewritten=%d (%zu res)\n",
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ",
                                has_srt_tag ? srt_tag : 0u,
                                !rt ? "no-table" : (tagged_res ? "yes" : "null"),
                                !rt ? "no-table" : (pc_res ? "yes" : "null"),
                                (int)srsrc_rewritten,
                                rt ? rt->resources.size() : 0u);
                    }
                    buf_op.how = "reject-unresolved";
                    ok = false; return true;
                }
                if (is_zero_record_raw_buffer(*res)) {
                    // The exact live V# is empty. The producer admitted only selectors whose OOB
                    // result is zero; re-check that contract so a malformed table cannot silently
                    // turn SQ_SEL_1 into zero. Apply the result only to active EXEC lanes, preserving
                    // the old destination in masked lanes, and never declare a backing buffer.
                    if (is_store) { buf_op.how = "reject-zero-record-format-store"; ok = false; return true; }
                    for (uint32_t k = 0; k < n; ++k) {
                        const uint32_t selector = res->swizzle[k];
                        if (selector != 0u && selector < 4u) { buf_op.how = "reject-zero-record-selector"; ok = false; return true; }
                        const int d = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "zero-record";
                    return true;
                }
                buf_op.how = "resolved";
                resolved_buffer = res;
                binding = res->binding;
                stride = res->stride;
                if (in.fmt == Rdna2Format::MTBUF) {
                    // Unlike MUBUF format ops, MTBUF owns the type in the instruction. gfx1030 uses
                    // the same combined 7-bit BUF_FMT table as Gen5 V# descriptors.
                    rdna2_buffer_format(in.mtbuf_format, &fmt, &fmt_ncomp);
                    if (fmt == DataFormat::Unknown || fmt_ncomp == 0) {
                        ok = false; return true;
                    }
                } else {
                    fmt = res->format;
                    fmt_ncomp = res->num_components;   // format default-fill below (#368)
                }
            } else if (rt) {
                // RAW (untyped) MUBUF with a resource table: resolve SRSRC (src[1]) exactly like the
                // format path — a raw buffer op targets whatever buffer its V# describes, NOT a fixed
                // binding (#91: the old hardcoded binding-2 silently read/wrote the wrong buffer for
                // any other target). Raw ops don't imply a resource class the way a format load implies
                // VertexBuffer, so the direct-SGPR lookup is class-unrestricted (by_sgpr_base).
                // Provenance order mirrors the format path: exact fetch pc, then s_load SRT tag
                // (indirect), then user-data SGPR (direct).
                const ShaderResource* res = rt->by_fetch_pc(in.pc);
                uint32_t srt_tag = 0;
                const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 4, srt_tag);
                if (!res && has_srt_tag)
                    res = rt->by_srt_offset(srt_tag);
                if (!res && !sreg_range_written(rs, in.src[1].value, 4))
                    res = rt->by_sgpr_base(in.src[1].value);
                if (!res) {
                    if (getenv("PROSPER_DBG")) {
                        const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                        fprintf(stderr,
                                "[mubuf-raw-unresolved] rt=%p pc=%u srsrc=s%d srt_tag=%s0x%x "
                                "key_res=%s pc_res=%s rewritten=%d (%zu res)\n",
                                (const void*)rt,
                                in.pc, in.src[1].value, has_srt_tag ? "" : "NONE ", srt_tag,
                                has_srt_tag && rt->by_srt_offset(srt_tag) ? "yes" : "null",
                                pp ? "yes" : "null",
                                sreg_range_written(rs, in.src[1].value, 4), rt->resources.size());
                    }
                    buf_op.how = "reject-raw-unresolved";
                    ok = false; return true;   // unresolvable V# -> reject; NEVER default to binding 2
                }
                if (indirect_pointer_descriptor_source_candidate) {
                    const bool zero_soffset =
                        (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                        (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    const bool exact_source =
                        b.indirect_pointer_proof->bound_kind ==
                            IndirectPointerBoundKind::DescriptorRange &&
                        b.indirect_pointer_proof->guard_kind ==
                            IndirectPointerGuardKind::None &&
                        b.indirect_pointer_proof->source_address_kind ==
                            IndirectBufferRelocationRecord::SourceAddressKind::
                                BufferDescriptorBase48 &&
                        in.fmt == Rdna2Format::MUBUF && !is_format && !is_store &&
                        !is_atomic && !raw_subword && n == 2u && idxen && !offen &&
                        offset == b.indirect_pointer_proof->pointer_byte_offset &&
                        zero_soffset && in.dst.kind == OperandKind::VGPR &&
                        in.dst.value >= 0 &&
                        static_cast<uint32_t>(in.dst.value) ==
                            b.indirect_pointer_proof->source_result_vgpr &&
                        in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 &&
                        static_cast<uint32_t>(in.src[0].value) ==
                            b.indirect_pointer_proof->source_record_index_vgpr &&
                        res->fetch_pc == b.indirect_pointer_proof->source_fetch_pc &&
                        is_indirect_pointer_relocation_resource(*res) &&
                        res->binding == b.indirect_pointer_binding &&
                        res->size == b.indirect_pointer_source_bytes &&
                        res->stride == b.indirect_pointer_source_stride &&
                        b.indirect_pointer_source_bytes % sizeof(uint32_t) == 0u &&
                        b.indirect_pointer_source_record_var &&
                        b.indirect_pointer_source_root_lo_var &&
                        b.indirect_pointer_source_root_hi_var;
                    if (!exact_source) {
                        buf_op.how = "reject-indirect-pointer-source";
                        ok = false;
                        return true;
                    }
                    indirect_pointer_descriptor_source = true;
                }
                if (is_gta5_cf9200_no_backing_marker_candidate(*res)) {
                    const GtaCf9200NoBackingAccess access =
                        rdna2_gta5_cf9200_no_backing_site(in);
                    if (!b.gta5_cf9200_no_backing_dispatch_validated ||
                        !is_proven_gta5_cf9200_no_backing(*res) ||
                        res->fetch_pc != in.pc || is_format || is_atomic ||
                        access == GtaCf9200NoBackingAccess::None ||
                        (access == GtaCf9200NoBackingAccess::LoadZero && is_store) ||
                        (access == GtaCf9200NoBackingAccess::DropStore && !is_store)) {
                        buf_op.how = "reject-cf9200";
                        ok = false;
                        return true;
                    }
                    if (access == GtaCf9200NoBackingAccess::LoadZero) {
                        const int destination = in.dst.value;
                        const uint32_t old = vreg_old(b, rs, destination);
                        rs.vreg[destination] = b.uconst(0u);
                        predicate_write(b, rs, destination, old);
                    }
                    buf_op.how = "cf9200-no-backing";
                    return true;
                }
                if (is_proven_null_nullable_raw_buffer(*res)) {
                    // The final compile boundary already re-established the exact production bytes,
                    // s7-sized x256 launch, marker set, and retained +0x20 zero-qword witness. Keep the
                    // instruction half fail-closed too: only one dword load may synthesize zero and
                    // only the three exact dword stores may disappear.
                    const Gta5NullableOutputAccess access =
                        rdna2_gta5_nullable_output_site(in);
                    if (res->fetch_pc != in.pc || is_format || is_atomic ||
                        access == Gta5NullableOutputAccess::None ||
                        (access == Gta5NullableOutputAccess::LoadDword && is_store) ||
                        (access == Gta5NullableOutputAccess::StoreDword && !is_store)) {
                        buf_op.how = "reject-proven-null-nullable";
                        ok = false;
                        return true;
                    }
                    if (access == Gta5NullableOutputAccess::LoadDword) {
                        const int d = in.dst.value;
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    buf_op.how = "proven-null-nullable";
                    return true;
                }
                if (is_optional_null_raw_load_buffer(*res)) {
                    // Unlike NUM_RECORDS=0, this marker records an application-level optional
                    // entry, not an architectural empty descriptor. Only the exact admitted RAW
                    // dword load may consume it; stores, atomics, wider loads, and packet variants
                    // stay fail-visible rather than inheriting drop-write behavior.
                    if (!rdna2_optional_null_raw_load_shape(in)) {
                        buf_op.how = "reject-optional-null-load";
                        ok = false;
                        return true;
                    }
                    const int d = in.dst.value;
                    const uint32_t old = vreg_old(b, rs, d);
                    rs.vreg[d] = b.uconst(0);
                    predicate_write(b, rs, d, old);
                    buf_op.how = "optional-null-load";
                    return true;
                }
                if (is_proven_null_guarded_raw_store(*res)) {
                    // The front half proved the exact pc74/76/78 consumer lies in GTA V's
                    // dispatch-specialized null-output region. Accept only the raw store family;
                    // a load, typed store, or atomic at the same pc must remain fail-visible rather
                    // than inheriting a no-op from stale/malformed resource metadata.
                    const bool supported_store = in.fmt == Rdna2Format::MUBUF && is_store &&
                        !is_format && !is_atomic &&
                        (in.opcode == kMubufOpcodeStoreDword ||
                         in.opcode == kMubufOpcodeStoreDwordX2 ||
                         in.opcode == kMubufOpcodeStoreDwordX4 ||
                         in.opcode == kMubufOpcodeStoreDwordX3) &&
                        rdna2_gta5_null_guarded_raw_store_site(in);
                    if (!supported_store || res->fetch_pc != in.pc) {
                        buf_op.how = "reject-null-guarded-store";
                        ok = false;
                        return true;
                    }
                    buf_op.how = "null-guarded-store";
                    return true;
                }
                if (is_zero_record_raw_buffer(*res)) {
                    // Safe at the TOP here, unlike the format path: none of this block's three arms
                    // (atomic, store-drop, load-fold) sets ok=false, so no later outcome can
                    // contradict the label. A fail-visible guard added below must move this down.
                    buf_op.how = "zero-record";
                    // The front half proved all four live V# words at this exact instruction and
                    // decoded NUM_RECORDS=0. RDNA2's OOB contract returns zero for every raw load
                    // component, drops raw stores, and performs no memory operation for an atomic.
                    // For atomics GLC=0 leaves VDATA untouched; GLC=1 returns the pre-op value, which
                    // is zero for an empty descriptor. Apply that return only to active EXEC lanes,
                    // exactly like an ordinary predicated VGPR write. Never declare or touch a dummy
                    // binding: one empty operation cannot then leak state into another.
                    if (is_atomic) {
                        if (in.mubuf_glc) {
                            const int d = in.dst.value;
                            const uint32_t old = vreg_old(b, rs, d);
                            rs.vreg[d] = b.uconst(0);
                            predicate_write(b, rs, d, old);
                        }
                        return true;
                    }
                    if (is_store) return true;
                    for (uint32_t k = 0; k < n; ++k) {
                        const int d = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, d);
                        rs.vreg[d] = b.uconst(0);
                        predicate_write(b, rs, d, old);
                    }
                    return true;
                }
                const bool exact_consumer =
                    (in.pc == 156u && in.opcode == kMubufOpcodeLoadDwordX4 && n == 4u &&
                     in.words[0] == 0xe0382000u && in.words[1] == 0x80020006u) ||
                    (in.pc == 158u && in.opcode == kMubufOpcodeLoadDwordX2 && n == 2u &&
                     in.words[0] == 0xe0342010u && in.words[1] == 0x80020406u);
                if (b.gta5_selected_sbuffer_dispatch_validated && exact_consumer) {
                    const bool exact_operands = in.fmt == Rdna2Format::MUBUF && !is_format &&
                        !is_store && !is_atomic && !raw_subword && in.len_dwords == 2u &&
                        in.dst.kind == OperandKind::VGPR &&
                        in.dst.value == (in.pc == 156u ? 0 : 4) &&
                        in.src[0].kind == OperandKind::VGPR && in.src[0].value == 6 &&
                        in.src[1].kind == OperandKind::SGPR && in.src[1].value == 8 &&
                        in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0;
                    if (!exact_operands) {
                        buf_op.how = "reject-selected-sbuffer-operands";
                        ok = false;
                        return true;
                    }
                    gta5_selected_sbuffer_consumer = true;
                }
                buf_op.how = "resolved";
                resolved_buffer = res;
                binding = res->binding;
                stride  = res->stride;
                // fmt stays raw Uint32: untyped ops move raw dwords regardless of the V#'s declared format.
            }
            if (raw_subword) {
                if (raw_bits == 8) fmt = raw_signed ? DataFormat::Sint8 : DataFormat::Uint8;
                else               fmt = raw_signed ? DataFormat::Sint16 : DataFormat::Uint16;
            }
            // The checkpoint array ABI models read-only raw buffer accesses. Typed MUBUF default-fill
            // and descriptor swizzle semantics need per-selected-entry treatment; stores and atomics
            // additionally need writeback authority. Reject all three before any helper can form an
            // access through the array variable.
            if (resolved_buffer && resolved_buffer->table_index_count != 0u &&
                (is_format || is_store || is_atomic)) {
                ok = false;
                return true;
            }
            if (is_atomic_x2) {
                const bool zero_soffset =
                    (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                    (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                const bool exact_shape = b.storage_buffer_int64_atomics && resolved_buffer &&
                    resolved_buffer->fetch_pc == in.pc &&
                    resolved_buffer->table_index_count == 0u &&
                    resolved_buffer->stride == 8u &&
                    resolved_buffer->atomic_x2_record_count != 0u &&
                    resolved_buffer->atomic_x2_record_count <= 0x02000000u &&
                    static_cast<uint64_t>(resolved_buffer->atomic_x2_record_count) * 8u ==
                        resolved_buffer->size &&
                    (resolved_buffer->gpu_addr & 7u) == 0u && idxen && !offen &&
                    offset == 0u && zero_soffset && !in.mubuf_dlc &&
                    !in.mubuf_lds && (in.words[0] & 0x00020000u) == 0u &&
                    (in.words[1] & 0x00e00000u) == 0u &&
                    in.src[0].kind == OperandKind::VGPR;
                if (!exact_shape) { ok = false; return true; }
                atomic_x2_record_count = resolved_buffer->atomic_x2_record_count;
            }
            // else (rt == nullptr): table-less offline compute shell (recompile_valu without a resource
            // table — the unit-test harness). Keep the legacy single-cbuf convention: binding 2, stride 0.
            // The live graphics path can never reach here table-less: recompile_vertex/recompile_fragment
            // set allow_smem = (rt != nullptr), so MUBUF already rejected above when rt is null there.
            const bool packed_10_11_11 = fmt == DataFormat::Float10_11_11;
            const bool packed_2_10_10_10 =
                fmt == DataFormat::Unorm2_10_10_10 || fmt == DataFormat::Snorm2_10_10_10 ||
                fmt == DataFormat::Uint2_10_10_10  || fmt == DataFormat::Sint2_10_10_10;
            const bool packed_word = packed_10_11_11 || packed_2_10_10_10;
            const uint32_t comp_bytes = data_format_bytes(fmt);
            if (!packed_word && comp_bytes == 0) { ok = false; return true; } // unknown / unsupported
            // Per-component decode. 4-byte formats (Float32/Uint32/Sint32) are a raw dword load — no
            // conversion in our bit model. Sub-dword formats are unpacked: UNORM/SNORM normalize an
            // integer field, Float16 unpacks a packed half. num_components components pack tightly.
            const bool packed = packed_word || comp_bytes < 4;
            bool is_snorm = (fmt == DataFormat::Snorm8 || fmt == DataFormat::Snorm16 ||
                             fmt == DataFormat::Snorm2_10_10_10);
            bool is_half  = (fmt == DataFormat::Float16);
            // Integer sub-dword formats deliver the raw (un-normalized) INTEGER in the VGPR — the
            // hardware's UINT/SINT format-load contract. DOLL's skinned scene VS fetches its bone
            // indices as Uint8 x4 (stride 8, paired with Unorm8 weights); rejecting them dropped
            // every scene-geometry draw (#273). Zero-/sign-extend the field; no normalization.
            bool is_uint = (fmt == DataFormat::Uint8 || fmt == DataFormat::Uint16 ||
                            fmt == DataFormat::Uint2_10_10_10);
            bool is_sint = (fmt == DataFormat::Sint8 || fmt == DataFormat::Sint16 ||
                            fmt == DataFormat::Sint2_10_10_10);
            // A buffer_load_FORMAT whose V# type is a sub-dword integer (Uint8/Sint8/Uint16/Sint16)
            // reads tightly-packed integer components at a RUNTIME byte address — exactly like the raw
            // buffer_load_ubyte/ushort path, NOT the descriptor-defined statically-aligned packing the
            // norm/half formats use. DOLL's post-process LUT compute kernels index a stride-1 Uint8
            // table this way (`buffer_load_format_x v, vIDX, V#`, idxen), which the aligned packed path
            // below wrongly rejected as unaligned. Handle it with a runtime byte/halfword extract.
            const bool int_subword = is_format && !raw_subword && (is_uint || is_sint) &&
                                     (comp_bytes == 1 || comp_bytes == 2);
            const bool zero_soffset =
                (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
            // Vulkan exposes storage buffers as u32 arrays in Prosper's portable ABI. Admit a two-byte
            // guest range only for Plucky Squire's exact one-record UINT16/FLOAT16 scalar format fetch:
            // the original element index itself is checked against zero, the sole load is constant
            // dword 0, and the host supplies a zero upper half. No raw/store/atomic/offset variant can
            // inherit this contract.
            const StorageBufferTailSemantic one_record_tail_semantic = !resolved_buffer
                ? StorageBufferTailSemantic::None
                : resolved_buffer->format == DataFormat::Uint16
                    ? StorageBufferTailSemantic::Uint16
                    : resolved_buffer->format == DataFormat::Float16
                        ? StorageBufferTailSemantic::Float16
                        : StorageBufferTailSemantic::None;
            const bool one_record_16bit_tail =
                in.fmt == Rdna2Format::MUBUF && in.opcode == 0u && n == 1u &&
                one_record_tail_semantic != StorageBufferTailSemantic::None &&
                resolved_buffer->num_components == 1u && resolved_buffer->stride == 2u &&
                resolved_buffer->size == 2u && idxen && !offen && offset == 0u && zero_soffset &&
                !folded_vfetch && in.src[0].kind == OperandKind::VGPR;
            float norm = 0.0f;
            switch (fmt) {
                case DataFormat::Unorm8:  norm = 255.0f;   break;
                case DataFormat::Snorm8:  norm = 127.0f;   break;
                case DataFormat::Unorm16: norm = 65535.0f; break;
                case DataFormat::Snorm16: norm = 32767.0f; break;
                default: break;
            }
            if (packed && !packed_word && !is_half && !is_uint && !is_sint && norm == 0.0f) {
                if (getenv("PROSPER_DBG"))
                    fprintf(stderr, "[mubuf-badfmt] pc=%u fmt=%u comp_bytes=%u stride=%u n=%u\n",
                            in.pc, (unsigned)fmt, comp_bytes, stride, n);
                ok = false; return true;
            }
            // Most packed (sub-dword) components below use static fields relative to a DWORD-ALIGNED
            // element base. Float16 LOADS have a general runtime-address path: each component is read
            // from addr+k*2, joining adjacent dwords if the 16-bit field starts at byte 3. Other packed
            // formats still require a proven aligned base; reject them instead of silently dropping the
            // low address bits. Aligned iff: inst offset %4==0; stride %4==0 when idxen; no offen; and
            // SOFFSET is NULL/0. Stores retain their existing stricter paths.
            bool dyn_half = false;   // Float16 components at an arbitrary runtime byte address
            bool dyn_int = false;    // integer sub-dword FORMAT component at a runtime (unaligned) byte addr
            bool dyn_norm = false;   // normalized sub-dword FORMAT component at an arbitrary byte addr
            bool dyn_int_store = false;  // integer sub-dword FORMAT store via race-free atomic clear+set
            if (packed && !raw_subword) {
                bool base_aligned;
                if (folded_vfetch) {
                    // The dyn_vfetch address path (below) is exactly gl_VertexIndex*stride and DROPS the
                    // shader's inst-offset / offen / SOFFSET — the resolved V# base already folds in this
                    // attribute's in-record byte offset. So packed-component alignment depends ONLY on the
                    // per-element stride being dword-aligned (vertex records are, and the V# base with
                    // them). Requiring soff_zero here (as the general path does) wrongly rejected Unorm8×4
                    // packed-color attributes whose fetch carries a register SOFFSET that dyn_vfetch drops
                    // anyway — the "mask=0xF draws never render" gap on PPSA01885. CONFIDENCE: HIGH.
                    base_aligned = !idxen || (stride & 3u) == 0;
                } else {
                    bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                     (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    base_aligned = ((offset & 3u) == 0) && !offen &&
                                   (!idxen || (stride & 3u) == 0) && soff_zero;
                }
                if (!base_aligned) {
                    // Float16 format loads use the complete runtime byte address for every requested
                    // component. This covers both the stride-2 single-half path (#273) and Astro's
                    // Float16x4 vertex record with a register SOFFSET. Loads only: packed half stores
                    // remain fail-visible until their race-free sub-dword write contract is modeled.
                    bool soff_zero = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                     (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                    dyn_half = !packed_word && !is_store && is_half;
                    // An integer sub-dword FORMAT load extracts each component at its runtime byte
                    // address (join the straddled dwords, then bfe), so it needs no static alignment.
                    // Only LOADS: the packed store path still rejects sub-dword ints (they don't pack).
                    if (!dyn_half) dyn_int = int_subword && !is_store;
                    // UNORM/SNORM 8/16 loads use the same runtime extraction, followed by the format's
                    // normalization. Astro's world-map VS fetches SNORM16x3 with a shader-computed
                    // SOFFSET; treating it as statically packed rejected the complete map draw.
                    if (!dyn_half && !dyn_int)
                        dyn_norm = !packed_word && !is_store && norm != 0.0f &&
                                   (comp_bytes == 1 || comp_bytes == 2);
                    // An integer sub-dword FORMAT STORE writes ONE lane's disjoint bit field of the
                    // containing dword. A plain masked read-modify-write would race (adjacent lanes' fields
                    // share a dword), but atomicAnd(clear field)+atomicOr(set field) COMMUTE across lanes
                    // writing DISJOINT fields, so the store is race-free. Requires the field to lie within
                    // a single dword: address comp_bytes-aligned, no runtime per-lane byte offset. (A
                    // straddling field would span two dwords -> deferred fail-visibly.)
                    if (!dyn_half && !dyn_int && is_store && int_subword && !offen && !dyn_vfetch &&
                        (offset % comp_bytes) == 0 && (!idxen || (stride % comp_bytes) == 0) && soff_zero)
                        dyn_int_store = true;
                    if (!dyn_half && !dyn_int && !dyn_norm && !dyn_int_store) {
                        if (getenv("PROSPER_DBG"))
                            fprintf(stderr, "[mubuf-unaligned] pc=%u fmt=%u off=%u offen=%d idxen=%d stride=%u\n",
                                    in.pc, (unsigned)fmt, offset, (int)offen, (int)idxen, stride);
                        ok = false; return true;
                    }
                }
            }
            // Byte address of the element (#148): idxen -> a VADDR VGPR is an element index (×stride);
            // offen -> a per-lane byte offset; both terms ADD; when idxen AND offen are set VADDR is TWO
            // consecutive VGPRs ([0]=index, [1]=byte offset). Plus the inst offset and SOFFSET.
            uint32_t addr;
            // Const-folded per-vertex attribute fetch (#206): the element address is exactly
            // gl_VertexIndex*stride. (1) The NGG fetch-shader prologue that computes the element index
            // isn't fully modeled and folds to a constant, so every vertex would read record 0 -> a
            // degenerate single point that rasterizes nothing (the whole scene stayed the blue clear).
            // (2) The resolved V# base ALREADY includes this attribute's byte offset within the
            // interleaved record, so the shader's inst-offset + SOFFSET must NOT be added again
            // (double-counting pushes the read OOB -> robustBufferAccess 0, the same collapse). So use
            // gl_VertexIndex*stride and drop the shader's VADDR/offset/SOFFSET; everything else keeps the
            // faithful address (incl. #148's idxen+offen both-terms fix). CONFIDENCE: HIGH — makes
            // PPSA01885 (Unity/IL2CPP) render real geometry instead of a degenerate collapse.
            if (folded_vfetch && idxen && stride) {
                const uint32_t element = dyn_vfetch ? b.load_vertex_index()
                                       : instance_vfetch ? b.load_instance_index()
                                                         : val(in.src[0]);
                addr = b.ibin(Op_IMul, element, b.uconst(stride));
            } else {
                addr = b.uconst(offset);
                if (idxen && stride) addr = b.ibin(Op_IAdd, addr, b.ibin(Op_IMul, val(in.src[0]), b.uconst(stride)));
                if (offen) {
                    Operand off_vgpr{ OperandKind::VGPR, idxen ? in.src[0].value + 1 : in.src[0].value };
                    addr = b.ibin(Op_IAdd, addr, val(off_vgpr));
                }
                addr = b.ibin(Op_IAdd, addr, val(in.src[2]));          // SOFFSET
            }
            uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            // Keep the instruction's cache policy at one ordinary-load propagation site. Every raw,
            // typed, packed, and dynamically addressed component below consumes dwords through this
            // wrapper, so adding another format shape cannot accidentally drop GLC/DLC semantics.
            auto load_dword = [&](uint32_t dword_idx) {
                if (indirect_pointer_descriptor_source) {
                    // The relocation carrier appends directories and packed segments after the
                    // original source table. The guest V# still ends at `source_bytes`; an OOB
                    // source-record fetch must return zero rather than accidentally reading the
                    // carrier metadata. Clamp the physical load first because OpSelect does not
                    // short-circuit it, then select the architectural OOB zero.
                    const uint32_t in_source = b.ucmp(
                        Op_ULessThan, dword_idx,
                        b.uconst(b.indirect_pointer_source_bytes / sizeof(uint32_t)));
                    const uint32_t safe_idx = b.sel(
                        in_source, dword_idx, b.uconst(0));
                    const uint32_t value = b.cbuf_load(
                        safe_idx, binding, coherent_load);
                    return b.sel(in_source, value, b.uconst(0));
                }
                if (gta5_selected_sbuffer_consumer) {
                    const auto soff = rs.sreg.find(106);
                    if (soff == rs.sreg.end()) {
                        ok = false;
                        return b.uconst(0u);
                    }
                    const uint32_t selected = b.ucmp(
                        Op_IEqual, soff->second,
                        b.uconst(b.gta5_selected_sbuffer_soffset));
                    // A SPIR-V select does not short-circuit its operands. Redirect rejected
                    // partitions to the proven non-empty binding's dword zero before loading, then
                    // select the architectural OOB zero, so an arbitrary guest v6 cannot create an
                    // out-of-range host access on a path whose V# was actually the zero descriptor.
                    const uint32_t safe_idx = b.sel(selected, dword_idx, b.uconst(0u));
                    const uint32_t value = b.cbuf_load(safe_idx, binding, coherent_load);
                    return b.sel(selected, value, b.uconst(0u));
                }
                return b.cbuf_load(dword_idx, binding, coherent_load);
            };
            if (is_atomic_x2) {
                const int d = in.dst.value;
                const uint32_t old_lo = vreg_old(b, rs, d);
                const uint32_t old_hi = vreg_old(b, rs, d + 1);
                const auto lo_it = rs.vreg.find(d), hi_it = rs.vreg.find(d + 1);
                const uint32_t value_lo = lo_it == rs.vreg.end() ? b.uconst(0) : lo_it->second;
                const uint32_t value_hi = hi_it == rs.vreg.end() ? b.uconst(0) : hi_it->second;
                const uint32_t value = b.u64_from_lohi(value_lo, value_hi);
                // Range-check the original qword record index against this dispatch's concrete V#.
                // Each valid index addresses its natural eight-byte record; larger indices are OOB.
                const uint32_t record_index = val(in.src[0]);
                const uint32_t in_bounds =
                    b.ucmp(Op_ULessThan, record_index, b.uconst(atomic_x2_record_count));
                const uint32_t access = rs.exec_narrowed
                    ? b.land(rs.exec, in_bounds) : in_bounds;
                uint32_t fallback = b.uconst64(0);
                if (rs.exec_narrowed)
                    fallback = b.sel64(rs.exec, fallback, b.u64_from_lohi(old_lo, old_hi));
                const uint32_t pre = b.cbuf_atomic_x2_rtn(
                    atomic_op, record_index, value, binding, access, fallback);
                // GLC=0 preserves BOTH data VGPRs. GLC=1 returns the pre-op qword; active OOB lanes
                // receive zero and inactive EXEC lanes receive their original pair via fallback.
                if (in.mubuf_glc) {
                    rs.vreg[d] = b.u64_lo(pre);
                    rs.vreg[d + 1] = b.u64_hi(pre);
                }
                return true;
            }
            if (is_atomic) {
                if (is_atomic_fminmax && b.compute_pgm_rsrc1 == UINT32_MAX) {
                    ok = false;
                    return true;
                }
                const int d = in.dst.value;
                const uint32_t old = vreg_old(b, rs, d);
                const auto it = rs.vreg.find(d);
                const uint32_t value = it == rs.vreg.end() ? b.uconst(0) : it->second;
                const uint32_t pre = is_atomic_fminmax
                    ? b.cbuf_atomic_fminmax_rtn(idx, value, binding, atomic_fmin,
                                                rs.exec_narrowed, rs.exec, old)
                    : b.cbuf_atomic_rtn(atomic_op, idx, value, binding,
                                        rs.exec_narrowed, rs.exec, old);
                // ISA 8.1 / Table 98: for atomics GLC means "return pre-op value to VGPR". With
                // GLC=0 hardware leaves VDATA untouched (it still holds the DATA operand) — the
                // unconditional write clobbered it (the exercised Astro packet 0xe0e00004 is GLC=0).
                if (in.mubuf_glc) rs.vreg[d] = pre;
                return true;
            }
            if (is_store) {
                auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                if (coherent_store) b.mark_cbuf_coherent(binding);
                // As with loads, keep Volatile propagation shared by raw and packed ordinary stores.
                auto store_dword = [&](uint32_t dword_idx, uint32_t value) {
                    b.cbuf_store(dword_idx, value, binding, rs.exec_narrowed, rs.exec,
                                 coherent_store);
                };
                if (dyn_int_store) {
                    // Integer sub-dword store: clear then set THIS lane's disjoint field of the containing
                    // dword with two atomics. Disjoint fields commute (And clears only this field's bits,
                    // Or sets only this field's bits), so no read-modify-write lock is needed; EXEC
                    // predication keeps inactive lanes from writing (like cbuf_store). CONFIDENCE: HIGH —
                    // this diverges from a hardware byte-enable store ONLY in already-UB situations: two
                    // lanes storing to the SAME element OR-merge instead of one-winner, and a racing reader
                    // could observe the transient post-And zero. Both require a data race a well-formed
                    // shader never has. A straddling field (excluded by the alignment guard above) is the
                    // one shape this can't express and stays deferred.
                    const uint32_t field_mask = comp_bytes == 2 ? 0xffffu : 0xffu;
                    for (uint32_t k = 0; k < n; k++) {
                        const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * comp_bytes)) : addr;
                        const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                        const uint32_t bitpos = b.ibin(Op_ShiftLeftLogical,
                                                       b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                        const uint32_t mask = b.ibin(Op_ShiftLeftLogical, b.uconst(field_mask), bitpos);
                        const uint32_t v = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, vread(in.dst.value + (int)k),
                                                         b.uconst(field_mask)), bitpos);
                        b.cbuf_atomic_rtn(Op_AtomicAnd, cidx, b.iun(Op_Not, mask), binding,
                                          rs.exec_narrowed, rs.exec, b.uconst(0));
                        b.cbuf_atomic_rtn(Op_AtomicOr, cidx, v, binding,
                                          rs.exec_narrowed, rs.exec, b.uconst(0));
                    }
                    return true;
                }
                // Store the VDATA VGPRs (in.dst..+n-1). Integer sub-dword formats reach the atomic path
                // above when in-dword-provable; anything else that can't pack (packed_word, or an
                // integer field that could straddle) rejects rather than mis-store.
                if (packed_word || (packed && (is_uint || is_sint))) { ok = false; return true; }
                // MTBUF's instruction format owns the physical component count. A wider opcode still
                // reads only those components (for example XY00), so Z/W must not spill into adjacent
                // memory. NOTE (#2869): "selection" here is the COMPONENT COUNT, not the descriptor's
                // DST_SEL channel routing -- those are separate V# fields, and MTBUF overriding the
                // format field says nothing about the selector one. `shader_resources.cpp:210` calls
                // DST_SEL "a FORMAT-fetch control" and binds it on possibly-typed consumers; no format
                // lowering here consults it at all. Do not read this line as settling that.
                const uint32_t store_n = in.fmt == Rdna2Format::MTBUF && fmt_ncomp < n
                                           ? fmt_ncomp : n;
                if (!packed) {
                    // Raw/Float32/Uint32: one dword per component.
                    for (uint32_t k = 0; k < store_n; k++) {
                        uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                        store_dword(kidx, vread(in.dst.value + (int)k));
                    }
                } else {
                    // Packed UNORM/SNORM/Float16: pack the components tightly into ceil(n*bytes/4) dwords
                    // (inverse of the packed load). Each dword ORs together the fields that land in it.
                    const uint32_t dwords = (store_n * comp_bytes + 3) / 4;
                    for (uint32_t d = 0; d < dwords; d++) {
                        uint32_t acc = b.uconst(0);
                        for (uint32_t k = 0; k < store_n; k++) {
                            uint32_t byte_off = k * comp_bytes;
                            if (byte_off / 4 != d) continue;
                            uint32_t field = is_half ? b.pack_half_lo(vread(in.dst.value + (int)k))
                                                     : b.pack_norm(vread(in.dst.value + (int)k), comp_bytes * 8, is_snorm, norm);
                            uint32_t sh = (byte_off % 4) * 8;
                            if (sh) field = b.ibin(Op_ShiftLeftLogical, field, b.uconst(sh));
                            acc = b.ibin(Op_BitwiseOr, acc, field);
                        }
                        uint32_t did = d ? b.ibin(Op_IAdd, idx, b.uconst(d)) : idx;
                        store_dword(did, acc);
                    }
                }
                return true;
            }
            // Integer-typed format? Its absent-component default for W/A is integer 1, not float 1.0.
            const bool fmt_is_int = (fmt == DataFormat::Uint8  || fmt == DataFormat::Uint16 || fmt == DataFormat::Uint32 ||
                                     fmt == DataFormat::Sint8  || fmt == DataFormat::Sint16 || fmt == DataFormat::Sint32 ||
                                     fmt == DataFormat::Uint2_10_10_10 || fmt == DataFormat::Sint2_10_10_10);
            for (uint32_t k = 0; k < n; k++) {
                int d = in.dst.value + (int)k;
                uint32_t old = vreg_old(b, rs, d);
                uint32_t value;
                // Format default-fill (#368): a requested component beyond the format's component
                // count is not read from adjacent memory. MUBUF takes the ABSENT-component default
                // from the V# contract (0 for G/B/Z, 1 for A/W). MTBUF's instruction format names the
                // present components directly (X000/XY00/XYZ0/XYZW), so every absent one is zero.
                // NOTE (#2869): this is about which components EXIST, not about how the descriptor
                // routes the ones that do -- DST_SEL is a separate field and no format lowering in
                // this file reads it. Not a statement that a typed fetch ignores DST_SEL.
                if (is_format && fmt_ncomp && k >= fmt_ncomp) {
                    uint32_t one = fmt_is_int ? 1u : 0x3f800000u;   // integer 1 vs float 1.0 (raw bits)
                    value = b.uconst(in.fmt != Rdna2Format::MTBUF && k == 3 ? one : 0u);
                } else if (one_record_16bit_tail) {
                    const uint32_t packed_value = b.cbuf_load_zero_padded_tail(
                        binding, one_record_tail_semantic, coherent_load);
                    const uint32_t in_bounds = b.ucmp(
                        Op_IEqual, val(in.src[0]), b.uconst(0));
                    const uint32_t typed_value =
                        one_record_tail_semantic == StorageBufferTailSemantic::Float16
                            ? b.unpack_half(packed_value, 0)
                            : b.bfe_u(packed_value, b.uconst(0), b.uconst(16));
                    value = b.sel(in_bounds, typed_value, b.uconst(0));
                } else if (raw_subword) {
                    // Raw byte/short loads use their full byte address, unlike typed packed-format
                    // loads whose component packing is descriptor-defined. A 16-bit access may begin
                    // at byte 3 and straddle two dwords, so join the adjacent words before extracting.
                    const uint32_t dw0 = load_dword(idx);
                    const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
                    uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
                    if (raw_bits == 16) {
                        const uint32_t dw1 = load_dword(
                            b.ibin(Op_IAdd, idx, b.uconst(1)));
                        const uint32_t inv_shift = b.ibin(
                            Op_BitwiseAnd,
                            b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                        const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                        joined = b.ibin(Op_BitwiseOr, joined,
                                        b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)),
                                              b.uconst(0), upper));
                    }
                    value = raw_signed
                        ? b.bfe_s(joined, b.uconst(0), b.uconst(raw_bits))
                        : b.bfe_u(joined, b.uconst(0), b.uconst(raw_bits));
                } else if (!packed) {
                    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
                    value = load_dword(kidx);  // raw 32-bit component
                } else if (packed_word) {
                    // All requested components share one packed dword. GFX10 names layouts from high
                    // field to low field, so 2_10_10_10 is logical R/G/B in bits 0/10/20 and A in 30;
                    // 10_11_11 is R/G/B in bits 0/11/22 with widths 11/11/10.
                    uint32_t dw = load_dword(idx);
                    uint32_t boff = packed_10_11_11 ? (k == 0 ? 0u : k == 1 ? 11u : 22u)
                                                    : (k == 0 ? 0u : k == 1 ? 10u : k == 2 ? 20u : 30u);
                    uint32_t bits = packed_10_11_11 ? (k < 2 ? 11u : 10u) : (k < 3 ? 10u : 2u);
                    if (packed_10_11_11) {
                        value = b.unpack_ufloat(dw, boff, bits);
                    } else if (is_uint) {
                        value = b.bfe_u(dw, b.uconst(boff), b.uconst(bits));
                    } else if (is_sint) {
                        value = b.bfe_s(dw, b.uconst(boff), b.uconst(bits));
                    } else {
                        float field_norm = is_snorm ? (bits == 2 ? 1.0f : 511.0f)
                                                    : (bits == 2 ? 3.0f : 1023.0f);
                        value = b.unpack_norm(dw, boff, bits, is_snorm, field_norm);
                    }
                } else if (dyn_half) {
                    // Float16 component k at byte address addr+k*2. Join the following dword when
                    // the field begins at byte 3; masking the inverse shift avoids SPIR-V's undefined
                    // shift-by-32 case, and the select discards that word when shift==0.
                    const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * 2)) : addr;
                    const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                    uint32_t joined = b.ibin(
                        Op_ShiftRightLogical, load_dword(cidx), shift);
                    const uint32_t dw1 = load_dword(
                        b.ibin(Op_IAdd, cidx, b.uconst(1)));
                    const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                        b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                    joined = b.ibin(Op_BitwiseOr, joined,
                                    b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                    value = b.unpack_half(joined, 0);
                } else if (dyn_int || dyn_norm) {
                    // Integer/normalized sub-dword FORMAT component k at a runtime byte address
                    // (addr + k*comp_bytes):
                    // shift the loaded dword right by (byteaddr&3)*8, join the next dword when a 16-bit
                    // field straddles the boundary, then extend or normalize the low field. Mirrors the
                    // raw_subword path but per packed component (stride-1 Uint8, unaligned u16/SNORM16).
                    const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * comp_bytes)) : addr;
                    const uint32_t cidx  = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
                    const uint32_t shift = b.ibin(Op_ShiftLeftLogical,
                                                  b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
                    uint32_t joined = b.ibin(
                        Op_ShiftRightLogical, load_dword(cidx), shift);
                    if (comp_bytes == 2) {
                        const uint32_t dw1 = load_dword(
                            b.ibin(Op_IAdd, cidx, b.uconst(1)));
                        const uint32_t inv_shift = b.ibin(Op_BitwiseAnd,
                            b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
                        const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
                        joined = b.ibin(Op_BitwiseOr, joined,
                                        b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
                    }
                    value = dyn_norm
                        ? b.unpack_norm(joined, 0, comp_bytes * 8, is_snorm, norm)
                        : is_sint ? b.bfe_s(joined, b.uconst(0), b.uconst(comp_bytes * 8))
                                  : b.bfe_u(joined, b.uconst(0), b.uconst(comp_bytes * 8));
                } else {
                    // Component k lives at byte k*comp_bytes within the element: pick its dword + field.
                    uint32_t byte_off = k * comp_bytes;
                    uint32_t drel = byte_off / 4, boff = (byte_off % 4) * 8;
                    uint32_t did = drel ? b.ibin(Op_IAdd, idx, b.uconst(drel)) : idx;
                    uint32_t dw  = load_dword(did);
                    value = is_half ? b.unpack_half(dw, boff ? 1u : 0u)
                          : is_uint ? b.bfe_u(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
                          : is_sint ? b.bfe_s(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
                                    : b.unpack_norm(dw, boff, comp_bytes * 8, is_snorm, norm);
                }
                rs.vreg[d] = value;
                predicate_write(b, rs, d, old);
            }
            if (indirect_pointer_descriptor_source) {
                const auto root_lo = rs.vreg.find(in.dst.value);
                const auto word1 = rs.vreg.find(in.dst.value + 1);
                if (root_lo == rs.vreg.end() || word1 == rs.vreg.end()) {
                    ok = false;
                    return true;
                }
                b.capture_indirect_pointer_descriptor_source(
                    val(in.src[0]), root_lo->second, word1->second,
                    rs.exec_narrowed, rs.exec);
            }
            return true;
        }
        case Rdna2Format::MIMG: {
            // Image op. Needs the resource table for the binding, so gated on allow_smem + rt. Two paths,
            // selected by the resolved resource's class: a STORAGE image (image_load/image_store, no
            // sampler — compute copy/blit) or a sampled TEXTURE (image_sample* / image_load via a combined
            // image+sampler). Other opcodes / NSA / gradient / compare variants are rejected (deferred).
            if (!allow_smem || !rt) { ok = false; return true; }
            const uint32_t SQ_DIM_2D = 1u;
            const uint32_t SQ_DIM_3D = 2u;
            const uint32_t SQ_DIM_2D_ARRAY = 5u;   // (x, y, layer) -- #2265
            auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };

            // IMAGE_BVH_INTERSECT_RAY (GFX10 opcode 0xe6) has an image encoding but consumes a
            // four-dword BVH descriptor and eleven NSA VGPR operands. Vulkan ray-query support is
            // not available on every backend/device Prosper supports, so lower the exact RTIP 1.1
            // contract used by Astro Bot to ordinary read-only SSBO loads and scalar ALU. The
            // front-half admits only TYPE=8, triangle-return-mode=1 descriptors; keeping
            // the instruction gate equally narrow makes every unverified variant fail visibly.
            if (in.opcode == 0xe6u) {
                const ShaderResource* bvh = rt->by_fetch_pc(in.pc);
                if (in.words[0] != 0xf1989f07u || in.len_dwords != 5u ||
                    in.mimg_dmask != 0xfu || !in.mimg_unorm || in.mimg_dim != 0u ||
                    in.mimg_glc || in.src[2].value != 0 || (in.words[4] & 0xffff0000u) != 0u ||
                    !bvh || bvh->cls != ResourceClass::ConstantBuffer ||
                    bvh->format != DataFormat::Uint32 || bvh->num_components != 1u ||
                    bvh->size < 64u || (bvh->size & 3u) != 0u) {
                    ok = false;
                    return true;
                }

                // The front-half only creates this marker when one mapped zero qword load taints all
                // four descriptor words and the ray instruction is dominated by an EXEC-narrowing
                // guard with no external region entry. This is an explicit empty acceleration
                // structure for this dispatch, not a fallback for an unresolved descriptor.
                if (is_proven_null_bvh(*bvh)) {
                    for (uint32_t k = 0; k < 4; ++k) {
                        const int vd = in.dst.value + static_cast<int>(k);
                        const uint32_t old = vreg_old(b, rs, vd);
                        rs.vreg[vd] = b.uconst(0xffffffffu);
                        predicate_write(b, rs, vd, old);
                    }
                    return true;
                }

                auto addr_vgpr = [&](uint32_t k) -> int {
                    if (k == 0u) return in.src[0].value;
                    const uint32_t j = k - 1u;
                    return static_cast<int>((in.words[2u + j / 4u] >> (8u * (j % 4u))) & 0xffu);
                };
                uint32_t a[11]{};
                for (uint32_t k = 0; k < 11; ++k) a[k] = vread(addr_vgpr(k));

                const uint32_t node = a[0];
                const uint32_t node_type = b.ibin(Op_BitwiseAnd, node, b.uconst(7u));
                const uint32_t node_offset = b.ibin(Op_BitwiseAnd, node, b.uconst(~7u));
                const uint32_t word_base = b.ibin(Op_ShiftLeftLogical, node_offset, b.uconst(1u));
                const uint32_t dword_count = bvh->size / 4u;
                const uint32_t valid64 = b.ucmp(
                    Op_ULessThanEqual, node_offset, b.uconst((bvh->size - 64u) / 8u));
                const uint32_t valid128 = bvh->size < 128u ? b.bfalse() : b.ucmp(
                    Op_ULessThanEqual, node_offset, b.uconst((bvh->size - 128u) / 8u));

                // Every speculative load is independently clamped to dword zero. Results from a
                // malformed/out-of-range pointer are discarded below, but the clamp also prevents
                // an invalid guest pointer from becoming an out-of-bounds Vulkan SSBO access.
                auto load_node = [&](uint32_t rel) {
                    const uint32_t idx = rel
                        ? b.ibin(Op_IAdd, word_base, b.uconst(rel)) : word_base;
                    const uint32_t safe_idx = b.sel(
                        b.ucmp(Op_ULessThan, idx, b.uconst(dword_count)), idx, b.uconst(0u));
                    return b.cbuf_load(safe_idx, bvh->binding);
                };
                uint32_t w[28]{};
                for (uint32_t k = 0; k < 28; ++k) w[k] = load_node(k);

                const uint32_t zero = b.uconst(0u);
                const uint32_t one = b.uconst(fbits(1.0f));
                const uint32_t invalid = b.uconst(0xffffffffu);
                const uint32_t is_tri0 = b.ucmp(Op_IEqual, node_type, b.uconst(0u));
                const uint32_t is_tri1 = b.ucmp(Op_IEqual, node_type, b.uconst(1u));
                const uint32_t is_box16 = b.ucmp(Op_IEqual, node_type, b.uconst(4u));
                const uint32_t is_box32 = b.ucmp(Op_IEqual, node_type, b.uconst(5u));
                const uint32_t tri_valid = b.land(b.lor(is_tri0, is_tri1), valid64);
                const uint32_t box_valid = b.lor(b.land(is_box16, valid64),
                                                 b.land(is_box32, valid128));

                auto fadd = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FAdd, x, y); };
                auto fsub = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FSub, x, y); };
                auto fmul = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FMul, x, y); };
                auto dot3 = [&](const uint32_t x[3], const uint32_t y[3]) {
                    return fadd(fadd(fmul(x[0], y[0]), fmul(x[1], y[1])), fmul(x[2], y[2]));
                };
                auto cross3 = [&](const uint32_t x[3], const uint32_t y[3], uint32_t out[3]) {
                    out[0] = fsub(fmul(x[1], y[2]), fmul(x[2], y[1]));
                    out[1] = fsub(fmul(x[2], y[0]), fmul(x[0], y[2]));
                    out[2] = fsub(fmul(x[0], y[1]), fmul(x[1], y[0]));
                };

                // Triangle node types 0 and 1 share a 64-byte quad. Type 0 selects V0,V1,V2;
                // type 1 selects V1,V3,V2. The four hardware return values are the unnormalised
                // t numerator, determinant, I numerator and J numerator (mode 1).
                uint32_t tv0[3] = {
                    b.sel(is_tri1, w[3], w[0]),
                    b.sel(is_tri1, w[4], w[1]),
                    b.sel(is_tri1, w[5], w[2]),
                };
                uint32_t tv1[3] = {
                    b.sel(is_tri1, w[9], w[3]),
                    b.sel(is_tri1, w[10], w[4]),
                    b.sel(is_tri1, w[11], w[5]),
                };
                uint32_t tv2[3] = {w[6], w[7], w[8]};
                uint32_t edge1[3], edge2[3], from_v0[3];
                for (uint32_t k = 0; k < 3; ++k) {
                    edge1[k] = fsub(tv1[k], tv0[k]);
                    edge2[k] = fsub(tv2[k], tv0[k]);
                    from_v0[k] = fsub(a[2u + k], tv0[k]);
                }
                uint32_t ray_dir[3] = {a[5], a[6], a[7]};
                uint32_t s1[3], s2[3];
                cross3(ray_dir, edge2, s1);
                cross3(from_v0, edge1, s2);
                const uint32_t t_num = dot3(edge2, s2);
                const uint32_t t_denom = dot3(s1, edge1);
                const uint32_t i_num = dot3(from_v0, s1);
                const uint32_t j_num = dot3(ray_dir, s2);
                const uint32_t t = b.fbin(Op_FDiv, t_num, t_denom);
                const uint32_t bary_i = b.fbin(Op_FDiv, i_num, t_denom);
                const uint32_t bary_j = b.fbin(Op_FDiv, j_num, t_denom);
                uint32_t tri_miss = b.lor(
                    b.fcmp(Op_FOrdLessThan, bary_i, zero),
                    b.fcmp(Op_FOrdGreaterThan, bary_i, one));
                tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, bary_j, zero));
                tri_miss = b.lor(tri_miss,
                    b.fcmp(Op_FOrdGreaterThan, fadd(bary_i, bary_j), one));
                tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, t, zero));
                uint32_t tri_out[4] = {
                    b.sel(tri_miss, b.uconst(0x7f800000u), t_num),
                    b.sel(tri_miss, one, t_denom),
                    i_num,
                    j_num,
                };

                // Undo the pair-compression vertex rotation encoded in the final triangle-node
                // dword. The two 2-bit fields select from {1-I-J, I, J}; valid RTIP 1.1 nodes never
                // use selector 3, whose conservative fallback here is 1-I-J.
                const uint32_t bary0_num = fsub(fsub(tri_out[1], i_num), j_num);
                const uint32_t tri_shift = b.ibin(
                    Op_ShiftLeftLogical,
                    b.ibin(Op_BitwiseAnd, node_type, b.uconst(3u)), b.uconst(3u));
                const uint32_t i_selector = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, w[15], tri_shift), b.uconst(3u));
                const uint32_t j_selector = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, w[15],
                           b.ibin(Op_IAdd, tri_shift, b.uconst(2u))),
                    b.uconst(3u));
                auto swizzled_bary = [&](uint32_t selector) {
                    return b.sel(b.ucmp(Op_IEqual, selector, b.uconst(1u)), i_num,
                        b.sel(b.ucmp(Op_IEqual, selector, b.uconst(2u)), j_num,
                              bary0_num));
                };
                tri_out[2] = swizzled_bary(i_selector);
                tri_out[3] = swizzled_bary(j_selector);

                // Convert each FP16 box payload to the same six-float representation as an FP32
                // box, then apply the slab test to all four children. BOX_SORT_EN orders valid hits
                // by increasing near intersection time; misses receive +INF and therefore remain at
                // the end. Strict compare-swaps retain physical child order for equal-time hits.
                uint32_t box_out[4]{};
                uint32_t box_key[4]{};
                uint32_t ray_inv[3] = {a[8], a[9], a[10]};
                for (uint32_t child = 0; child < 4; ++child) {
                    const uint32_t hbase = 4u + child * 3u;
                    uint32_t fp16_bounds[6] = {
                        b.unpack_half(w[hbase], 0u), b.unpack_half(w[hbase], 1u),
                        b.unpack_half(w[hbase + 1u], 0u), b.unpack_half(w[hbase + 1u], 1u),
                        b.unpack_half(w[hbase + 2u], 0u), b.unpack_half(w[hbase + 2u], 1u),
                    };
                    const uint32_t fbase = 4u + child * 6u;
                    uint32_t bmin[3], bmax[3];
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        bmin[axis] = b.sel(is_box16, fp16_bounds[axis], w[fbase + axis]);
                        bmax[axis] = b.sel(is_box16, fp16_bounds[3u + axis], w[fbase + 3u + axis]);
                    }
                    uint32_t near_axis[3], far_axis[3];
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        const uint32_t t0 = fmul(fsub(bmin[axis], a[2u + axis]), ray_inv[axis]);
                        const uint32_t t1 = fmul(fsub(bmax[axis], a[2u + axis]), ray_inv[axis]);
                        const uint32_t positive = b.fcmp(Op_FOrdGreaterThanEqual, ray_inv[axis], zero);
                        near_axis[axis] = b.sel(positive, t0, t1);
                        far_axis[axis] = b.sel(positive, t1, t0);
                    }
                    const uint32_t near_unclamped = b.fext2(
                        Glsl_FMax, b.fext2(Glsl_FMax, near_axis[0], near_axis[1]), near_axis[2]);
                    const uint32_t far_unclamped = b.fext2(
                        Glsl_FMin, b.fext2(Glsl_FMin, far_axis[0], far_axis[1]), far_axis[2]);
                    const uint32_t near_t = b.fext2(Glsl_FMax, near_unclamped, zero);
                    const uint32_t far_t = b.fext2(Glsl_FMin, far_unclamped, a[1]);
                    const uint32_t nan_interval = b.lor(
                        b.fcmp(Op_FUnordNotEqual, near_unclamped, near_unclamped),
                        b.fcmp(Op_FUnordNotEqual, far_unclamped, far_unclamped));
                    const float box_multiplier = 1.0f +
                        static_cast<float>(bvh->bvh_box_grow) * (1.0f / 16777216.0f);
                    const uint32_t hit = b.land(
                        b.logical_not(nan_interval),
                        b.fcmp(Op_FOrdLessThanEqual, near_t,
                               fmul(far_t, b.uconst(std::bit_cast<uint32_t>(box_multiplier)))));
                    const uint32_t valid_hit = b.land(box_valid, hit);
                    box_out[child] = b.sel(valid_hit, w[child], invalid);
                    box_key[child] = b.sel(valid_hit, near_t, b.uconst(0x7f800000u));
                }
                if (bvh->bvh_sort_enabled) {
                    auto compare_swap = [&](uint32_t left, uint32_t right) {
                        const uint32_t left_key = box_key[left];
                        const uint32_t right_key = box_key[right];
                        const uint32_t left_out = box_out[left];
                        const uint32_t right_out = box_out[right];
                        const uint32_t swap = b.fcmp(
                            Op_FOrdGreaterThan, left_key, right_key);
                        box_key[left] = b.sel(swap, right_key, left_key);
                        box_key[right] = b.sel(swap, left_key, right_key);
                        box_out[left] = b.sel(swap, right_out, left_out);
                        box_out[right] = b.sel(swap, left_out, right_out);
                    };
                    compare_swap(0, 1);
                    compare_swap(2, 3);
                    compare_swap(0, 2);
                    compare_swap(1, 3);
                    compare_swap(1, 2);
                }

                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t result = b.sel(tri_valid, tri_out[k], box_out[k]);
                    const int vd = in.dst.value + static_cast<int>(k);
                    const uint32_t old = vreg_old(b, rs, vd);
                    rs.vreg[vd] = result;
                    predicate_write(b, rs, vd, old);
                }
                return true;
            }

            // Which resource class THIS operation can accept, decided before any lookup runs.
            // image_store plus EVERY integer image atomic. This was the THIRD copy of the 0x08/0x0f/0x11
            // triple -- the others are the storage-image classifier in gpu_executor.cpp and the emitter's
            // own is_atomic test -- and all three must agree for an image atomic to compile. They did
            // agree, on the wrong set: exactly the ops the emitter happened to support.
            //
            // The failure mode when they DISAGREE is the instructive part. Widening the classifier alone
            // made this gate reject harder: the resource correctly became a StorageImage, and the clause
            // below then read "0x17 is not a storage op, so it must want a Texture; it is not a Texture"
            // and discarded it. A half-applied fix was worse than none (#2275).
            //
            // Opcodes verified by assembling each with llvm-mc, positive control image_atomic_add ->
            // word0 0xf0440128, byte-identical to the dword decoded from the guest here.
            const bool storage_only_op = in.opcode == 0x08 || in.opcode == 0x09 ||
                                         in.opcode == 0x0f ||
                                         (in.opcode >= 0x11 && in.opcode <= 0x1a && in.opcode != 0x13);
            const ImageResourceRequirement image_requirement =
                storage_only_op            ? ImageResourceRequirement::StorageOnly
              : (in.opcode == 0x00 ||
                 in.opcode == 0x0e)        ? ImageResourceRequirement::Either
                                           : ImageResourceRequirement::SampledOnly;

            // Resolve the T#/U# via SRSRC (src[1]) provenance: s_load tag (indirect) else user-data SGPR.
            //
            // Every lookup is CLASS-FILTERED. It used to take the first hit at each key and then null it
            // out on class further down, which is not the same thing and cost this file two distinct
            // failures: (a) an image resource sharing a key with an earlier buffer-class resource could
            // never be found -- one live vertex stage carries a ConstantBuffer and two VertexBuffers all
            // at sgpr_base 8, so the collision is real, not hypothetical -- and (b) a wrong-class hit on
            // the SRT route left `res` non-null, which SUPPRESSED the `!res` guard on the SGPR fallback,
            // so a route that could have resolved was never tried. Filtering inside the lookup makes the
            // search continue past a wrong-class entry instead of stopping on it.
            //
            // The ACCEPTED SET is unchanged -- `image_resource_class_satisfies` is exactly the old
            // post-check, verified by enumerating all 256x5 (opcode, class) pairs: 258 accepted, zero
            // disagreements, with a deliberately mis-mapped 0x0e control showing the comparison can
            // detect one. So this can never bind a class the op cannot use.
            //
            // What it CAN change, and the first version of this comment wrongly denied: ROUTE
            // PRECEDENCE. Where the first entry at a `fetch_pc` is buffer-class and a later one at
            // the same pc is image-class, the old code took the buffer hit, failed the class test and
            // fell through to the SRT/SGPR routes; this one lets the pc route win. That is the
            // intended contract -- exact per-use provenance outranks a table key, because a sample
            // and a store may share one key while needing different Vulkan classes -- but it means
            // the outcome can be a DIFFERENT, still class-valid resource, not only a rejection turned
            // into a resolution.
            //
            // NO REACHABLE INSTANCE IS KNOWN, and the first version of this comment claimed one that
            // cannot exist: "reachability needs the BVH piggyback to land on a texture-publishing pc".
            // It cannot -- that piggyback keys a ConstantBuffer by an IMAGE_BVH_INTERSECT_RAY pc, and
            // opcode 0xe6 returns from its own block above, before this lookup ever runs. The sentence
            // arrived verbatim from a review comment, carrying a file:line, and was promoted into
            // shipped code without anyone opening the cited path -- the #2049 -> #2052 shape CLAUDE.md
            // records, repeated. State the precedence change, which is real and provable from the
            // diff; do not attach a reachability story to it that has not been traced. #3126/#1634.
            const ShaderResource* res = rt->image_by_fetch_pc(in.pc, image_requirement);
            if (!res) {
                uint32_t srt_tag = 0;
                if (sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag))
                    res = rt->image_by_srt_offset(srt_tag, image_requirement);
                if (!res && !sreg_range_written(rs, in.src[1].value, 8))
                    res = rt->image_by_sgpr_base(in.src[1].value, image_requirement);
                // The SRSRC range reads as written when the shader STAGED a direct descriptor into
                // it with s_mov_b32, so the lookup above is skipped even though the bits are still
                // the driver's. Resolve through the copy to where those words actually live (#1773).
                if (int ud_origin = 0; !res && sreg_range_ud_alias(rs, in.src[1].value, 8, ud_origin))
                    res = rt->image_by_sgpr_base(ud_origin, image_requirement);
            }
            if (!res) {
                // Resolution-failure diagnostic: which provenance step failed for this image op.
                //
                // Ungated, and deduped per (pc, srsrc). It fires only when an image op has already
                // failed to resolve, so its volume is bounded by the defect it reports. Behind
                // PROSPER_DBG it was unreachable in practice on any routed boot.
                //
                // #3143: this report used to sit downstream of the PROSPER_MIMG_SOFT early return
                // below, so arming that switch to see whether softening changes a frame ALSO deleted
                // the one line that names which descriptor failed -- the switch was least informative
                // exactly when it was armed for its intended purpose ("softening changed nothing"
                // became indistinguishable from "there was nothing here to soften"). It now runs
                // before either the soft or the reject path decides what to DO about the failure, so
                // both keep the same descriptor identification; the soft path still gets its own
                // confirmation line below ([mimg-soft]) once the constant is written.
                static std::mutex mimg_mutex;
                // Keyed by PROGRAM as well as (pc, srsrc): every shader has a pc 16, so a
                // pc-only key lets the first program to reach one silence all later programs and
                // attribute its line to a shader the reader is not looking at.
                static std::set<std::tuple<uint64_t, uint32_t, int>> mimg_reported;
                bool first_report = false;
                {
                    std::lock_guard<std::mutex> lock(mimg_mutex);
                    first_report = mimg_reported.emplace(b.diagnostic.program_address, in.pc,
                                                         in.src[1].value).second;
                }
                if (first_report) {
                    uint32_t srt_tag = 0;
                    const bool has_srt_tag = sreg_srt_range_tag(rs, in.src[1].value, 8, srt_tag);
                    const ShaderResource* pk = has_srt_tag ? rt->by_srt_offset(srt_tag) : nullptr;
                    const ShaderResource* pp = rt->by_fetch_pc(in.pc);
                    // Report the copy-alias step too. Without it a staged descriptor (#1773) and a
                    // genuinely absent one print the identical line, which is the output-vs-input
                    // confusion that cost #1590 several sessions -- name which provenance route failed.
                    int ud_origin = 0;
                    const bool has_ud_alias = sreg_range_ud_alias(rs, in.src[1].value, 8, ud_origin);
                    const ShaderResource* pa = has_ud_alias ? rt->by_sgpr_base(ud_origin) : nullptr;
                    // sgpr_res: the DIRECT user-data route's own outcome, which this line did not
                    // report and which is the ONLY route that can fire for the commonest failure
                    // shape -- `written=0`, i.e. the shader never touched SRSRC, so by construction
                    // there is no s_load tag and no copy alias for the other three fields to carry.
                    // For such a site `srt_tag=NONE key_res=null ud_alias=NONE alias_res=null` is a
                    // restatement of `written=0`, not evidence: those routes did not run. Five sites
                    // on Stray (PPSA02101, #3126) printed exactly that quadruple while three of them
                    // had a resource sitting at the requested SGPR the whole time -- of the wrong
                    // class, discarded in silence. Print the class-BLIND hit so "nothing is here"
                    // and "something is here and it is a cbuf" stop looking identical. #3126/#1634.
                    //
                    // ...and say whether that route was CONSULTED, which is the same trap inverted
                    // (trap 254). The SGPR route is guarded by `!sreg_range_written` above, so on a
                    // `written=1` site it never runs -- and printing a bare `sgpr_res=tex` there
                    // tells the reader a Texture was found at the requested SGPR and rejected, when
                    // in fact nothing consulted it and the descriptor is not the entry user data at
                    // all. That is live on this title: `0x30131d0000` pc=77 is `written=1`. Print the
                    // status and the contents, never the contents alone.
                    const bool srsrc_written = sreg_range_written(rs, in.src[1].value, 8);
                    const ShaderResource* ps_direct = rt->by_sgpr_base(in.src[1].value);
                    auto cls_name = [](const ShaderResource* r) {
                        if (!r) return "null";
                        switch (r->cls) {
                            case ResourceClass::Texture:        return "tex";
                            case ResourceClass::StorageImage:   return "simg";
                            case ResourceClass::ConstantBuffer: return "cbuf";
                            case ResourceClass::VertexBuffer:   return "vbuf";
                            case ResourceClass::Sampler:        return "samp";
                        }
                        return "other-cls";
                    };
                    const char* need =
                        image_requirement == ImageResourceRequirement::StorageOnly ? "storage"
                      : image_requirement == ImageResourceRequirement::SampledOnly ? "sampled"
                                                                                   : "either";
                    // `not-consulted(<cls>)` keeps BOTH facts on the line: that the route did not
                    // run, and what happens to sit at that SGPR anyway. Dropping the second half
                    // would trade one misreading for a different one.
                    const std::string sgpr_res =
                        srsrc_written ? std::string("not-consulted(") + cls_name(ps_direct) + ")"
                                      : std::string(cls_name(ps_direct));
                    fprintf(stderr, "[mimg-unresolved] program=0x%llx pc=%u op=0x%02x storage=%d need=%s srsrc=s%d srt_tag=%s0x%x key_res=%s pc_res=%s ud_alias=%s%d alias_res=%s sgpr_res=%s written=%d (%zu res)\n",
                            (unsigned long long)b.diagnostic.program_address,
                            in.pc, in.opcode, (int)storage_only_op, need,
                            in.src[1].value, has_srt_tag ? "" : "NONE ",
                            has_srt_tag ? srt_tag : 0u,
                            cls_name(pk), cls_name(pp),
                            has_ud_alias ? "s" : "NONE ", has_ud_alias ? ud_origin : 0,
                            cls_name(pa), sgpr_res.c_str(),
                            srsrc_written ? 1 : 0,
                            rt->resources.size());
                    // #3126: say what WAS available, not only what failed. "key_res=null pc_res=null"
                    // tells you the lookups missed; it does not tell you whether the table holds the
                    // descriptor under a different key, holds it classified as something else, or does
                    // not hold it at all -- and those are three different bugs. Print every resource's
                    // class and provenance so the miss can be diagnosed from one line instead of a
                    // rebuild. Bounded, and printed once per site regardless of whether the site is
                    // then softened or rejected (#3143).
                    //
                    // Print EVERY key a resource carries, not one of them. The old form printed
                    // `srt=0x%x` whenever sgpr_base was unset, so a resource keyed only by fetch_pc
                    // -- which is how the const-fold publishes a seed/table-recovered T# -- rendered
                    // as `srt=0xffffffff` and read as "this entry has no lookup key at all". Four of
                    // the nine resources in Stray's failing pixel stage printed that way while being
                    // perfectly well keyed, just not by a key this reader could see (#3126).
                    std::string avail;
                    size_t shown = 0;
                    for (const ShaderResource& r : rt->resources) {
                        if (shown++ >= 16) { avail += " ..."; break; }
                        char one[128];
                        char keys[80];
                        int used = 0;
                        if (r.sgpr_base != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " s%u", r.sgpr_base);
                        if (r.srt_offset != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " srt=0x%x", r.srt_offset);
                        if (r.fetch_pc != 0xFFFFFFFFu)
                            used += snprintf(keys + used, sizeof keys - (size_t)used, " pc=%u", r.fetch_pc);
                        if (!used) snprintf(keys, sizeof keys, " unkeyed");
                        snprintf(one, sizeof one, " [b%u %s%s]", r.binding, cls_name(&r), keys);
                        avail += one;
                    }
                    fprintf(stderr, "[mimg-unresolved]   available:%s\n",
                            avail.empty() ? " (none)" : avail.c_str());
                }

                // PROSPER_MIMG_SOFT=1 -- diagnostic only, and deliberately WRONG output.
                //
                // Placed BEFORE the reject gate below on purpose. That gate fires once per
                // (program, pc, srsrc); downstream of it, only the FIRST compile of each site would be
                // softened and every later one -- reachable through ordinary shader-cache eviction --
                // would take the reject path unchanged. The recorded result would then be
                // indistinguishable from "nothing was softened", which is exactly the reading this
                // diagnostic exists to produce. When an image
                // op cannot resolve its descriptor, write a constant instead of failing the stage, so a
                // frame can be seen that would otherwise be discarded entirely. It answers one question
                // nothing else here can: how much of the picture is riding on THIS resolution failure.
                // On Stray it showed the answer was "none" -- the composite compiled and the frame did
                // not change -- which is what redirected #3126 away from the recompiler.
                //
                // Never acceptable in a run that produces progression evidence: every unresolved sample
                // reads a flat value.
                if (!res && getenv("PROSPER_MIMG_SOFT")) {
                    const uint32_t soft = b.uconst(fbits(0.5f));
                    uint32_t comps = 0;
                    for (uint32_t m = 0; m < 4; ++m) if (in.mimg_dmask & (1u << m)) ++comps;
                    if (!comps) comps = 1;
                    for (uint32_t k = 0; k < comps; ++k) {
                        const int dst = in.dst.value + static_cast<int>(k);
                        if (dst < 0) break;
                        const uint32_t old = vreg_old(b, rs, dst);
                        rs.vreg[dst] = soft;
                        predicate_write(b, rs, dst, old);
                    }
                    static std::mutex smx; static std::set<uint64_t> sseen;
                    std::lock_guard<std::mutex> lk(smx);
                    if (sseen.insert(((uint64_t)b.diagnostic.program_address << 16) | in.pc).second)
                        fprintf(stderr, "[mimg-soft] program=0x%llx pc=%u -> constant %u comps\n",
                                (unsigned long long)b.diagnostic.program_address, in.pc, comps);
                    return true;
                }
                if (!first_report) { ok = false; return true; }
            }
            if (!res) { ok = false; return true; }
            const bool uint_texture = res->format == DataFormat::Uint8 ||
                                      res->format == DataFormat::Uint16 ||
                                      res->format == DataFormat::Uint32;
            // D16 packs two floating-point result components into each VDATA dword. The image
            // helpers expose normalized/float resources as ordinary f32 bits, so the back-half
            // must explicitly convert and pack those values rather than writing one f32 per VGPR.
            // Likewise an IMAGE_STORE D16 consumes consecutive packed halves, not four full-width
            // VGPRs. GTA V's FSR1 RCAS pass depends on both halves of this contract: its dmask-f
            // R11G11B10F load writes v[12:13], and its dmask-f RGBA8 stores read v[14:15].
            // Integer D16 conversion has a distinct bit/integer contract and remains fail-visible.
            const bool d16_float_resource =
                res->format == DataFormat::Float32 || res->format == DataFormat::Float16 ||
                res->format == DataFormat::Unorm16 || res->format == DataFormat::Snorm16 ||
                res->format == DataFormat::Unorm8 || res->format == DataFormat::Snorm8 ||
                res->format == DataFormat::Float10_11_11 ||
                res->format == DataFormat::Unorm2_10_10_10 ||
                res->format == DataFormat::Snorm2_10_10_10;
            auto write_mimg_results = [&](const uint32_t out[4]) -> bool {
                const int vd = in.dst.value;
                if (!in.mimg_d16) {
                    int written = 0;
                    for (uint32_t component = 0; component < 4; ++component) {
                        if (!(in.mimg_dmask & (1u << component))) continue;
                        const uint32_t old = vreg_old(b, rs, vd + written);
                        rs.vreg[vd + written] = out[component];
                        predicate_write(b, rs, vd + written, old);
                        ++written;
                    }
                    return true;
                }
                if (!d16_float_resource) return false;

                uint32_t packed = b.uconst(0);
                int component_index = 0;
                int word_index = 0;
                auto write_packed = [&](uint32_t value) {
                    const uint32_t old = vreg_old(b, rs, vd + word_index);
                    rs.vreg[vd + word_index] = value;
                    predicate_write(b, rs, vd + word_index, old);
                    ++word_index;
                };
                for (uint32_t component = 0; component < 4; ++component) {
                    if (!(in.mimg_dmask & (1u << component))) continue;
                    const uint32_t half = b.pack_half_lo(out[component]);
                    if ((component_index & 1) == 0) {
                        packed = half;
                    } else {
                        packed = b.ibin(
                            Op_BitwiseOr, packed,
                            b.ibin(Op_ShiftLeftLogical, half, b.uconst(16)));
                        write_packed(packed);
                    }
                    ++component_index;
                }
                if (component_index & 1) write_packed(packed); // odd dmask: high half is zero.
                return true;
            };

            // --- Storage-image path: image_load (0x00), image_store[_mip] (0x08/0x09), and R32_UINT
            if (res->cls == ResourceClass::StorageImage) {
                const bool is_ld = in.opcode == 0x00;
                const bool is_zero_mip_st = in.opcode == 0x09;
                const bool is_st = in.opcode == 0x08 || is_zero_mip_st;
                // The backend exposes one materialized mip, but that alone is not permission to
                // discard a guest operand. IMAGE_STORE_MIP is admitted only for the exact captured
                // 2D packet whose mip VGPR was proven zero at this use and whose descriptor declares
                // one uncompressed base level. The live dmask-f packet is exact to the captured
                // fmt60 Uint8x4 resource; other formats and dynamic/nonzero/general or DCC-backed
                // forms remain fail-visible.
                if (is_zero_mip_st &&
                    (!rdna2_mimg_zero_mip_shape(in) || !res->proven_zero_mip ||
                     res->img_dim != SQ_DIM_2D || res->sample_count != 1u ||
                     res->declared_mip_levels != 1u || res->in_mip_tail ||
                     res->compression_enabled ||
                     (in.mimg_dmask == 0xfu &&
                      (res->format != DataFormat::Uint8 || res->num_components != 4u)))) {
                    ok = false;
                    return true;
                }
                // Opcodes verified with llvm-mc; add/positive-control matches the guest's own word0 (#2275).
                uint16_t spirv_atomic = 0;
                switch (in.opcode) {
                    case 0x0f: spirv_atomic = Op_AtomicExchange; break;
                    case 0x11: spirv_atomic = Op_AtomicIAdd;     break;
                    case 0x12: spirv_atomic = Op_AtomicISub;     break;
                    case 0x14: spirv_atomic = Op_AtomicSMin;     break;
                    case 0x15: spirv_atomic = Op_AtomicUMin;     break;
                    case 0x16: spirv_atomic = Op_AtomicSMax;     break;
                    case 0x17: spirv_atomic = Op_AtomicUMax;     break;
                    case 0x18: spirv_atomic = Op_AtomicAnd;      break;
                    case 0x19: spirv_atomic = Op_AtomicOr;       break;
                    case 0x1a: spirv_atomic = Op_AtomicXor;      break;
                    default: break;
                }
                const bool is_atomic = spirv_atomic != 0;
                if (!is_ld && !is_st && !is_atomic) { ok = false; return true; }
                uint32_t dim, ncoord; bool arrayed = false, ms = false;
                switch (in.mimg_dim) {   // SQ_RSRC dim -> SPIR-V Dim + coord count (+ array layer / MSAA sample)
                    case 0: dim = Dim_1D; ncoord = 1; break;                       // 1D
                    case 1: dim = Dim_2D; ncoord = 2; break;                       // 2D
                    case 2: dim = Dim_3D; ncoord = 3; break;                       // 3D
                    case 4: dim = Dim_1D; ncoord = 2; arrayed = true; break;       // 1D_ARRAY (x, layer)
                    case 5: dim = Dim_2D; ncoord = 3; arrayed = true; break;       // 2D_ARRAY (x, y, layer)
                    case 6: dim = Dim_2D; ncoord = 2; ms = true; break;            // 2D_MSAA (x, y) + sample index
                    case 7: dim = Dim_2D; ncoord = 3; arrayed = true; ms = true; break;  // 2D_MSAA_ARRAY (x,y,layer)+sample
                    default: ok = false; return true;   // cube storage images deferred
                }
                if (ms && (is_st || is_atomic)) { ok = false; return true; }
                const uint32_t components = res->num_components ? res->num_components : 1;
                // The live Astro Bot visibility image is an ordinary 2D R32_UINT surface. Keep the
                // first atomic implementation exact and fail-visible for every other image shape;
                // atomics require a typed integer image in Vulkan/SPIR-V rather than Format=Unknown.
                // 2D and 2D_ARRAY. The array layer was added for #2265: Sonic Racing: CrossWorlds
                // issues IMAGE_ATOMIC_ADD at dim=2D_ARRAY on a default launch, and the storage
                // load/store siblings had already been generalised over the layer while the atomic
                // had not. `res->depth` is the layer COUNT for an arrayed view, so it is required to
                // be exactly 1 only in the non-arrayed case.
                const bool atomic_2d_array = in.mimg_dim == SQ_DIM_2D_ARRAY && arrayed && !ms;
                // #2265: the RESOURCE half is now shader_resource_supports_atomic_image_buffer,
                // shared with the descriptor validator and the backend materialization. Those three
                // had drifted -- #2272 widened this gate over the array layer and left the other
                // two on the single-layer clause, so this lowering emitted a StorageBuffer binding
                // that its own validator then rejected as WrongType and the dispatch was skipped
                // every frame. The INSTRUCTION half stays here, because only this site sees it.
                if (is_atomic &&
                    ((in.mimg_dim != SQ_DIM_2D && !atomic_2d_array) || ms || in.len_dwords < 2 ||
                     in.mimg_unorm || in.mimg_d16 || in.mimg_dmask != 1u ||
                     arrayed != (res->img_dim == SQ_DIM_2D_ARRAY) ||
                     !shader_resource_supports_atomic_image_buffer(*res))) {
                    ok = false;
                    return true;
                }
                const bool native_2d_storage =
                    shader_resource_uses_native_2d_storage_image(
                        *res, dim == Dim_2D, arrayed, ms);
                const bool native_uint_2d_storage =
                    shader_resource_uses_native_uint_2d_storage_image(
                        *res, dim == Dim_2D, arrayed, ms);
                const bool ordinary_3d = in.mimg_dim == SQ_DIM_3D && res->img_dim == 2 &&
                                         res->depth && !arrayed && !ms &&
                                         !res->depth_compare;
                const uint32_t native_support_bit = ordinary_3d
                    ? native_storage_3d_format_support_bit(res->format, components)
                    : native_storage_format_support_bit(res->format, components);
                // Vulkan permits B10G11R11 storage conversion, but RADV stores finite values with
                // a systematic downward bias while the guest conversion is round-to-nearest-even
                // (#1790). Keep the exact shader-side R32ui pack authoritative in ordinary/single-
                // layer-arrayed 2D and 3D, even when the device advertises native typed storage.
                // Other float formats retain the device-gated native path below.
                const bool packed_r11 = b.packed_r11_storage &&
                    (native_2d_storage || ordinary_3d) && !ms && !is_atomic &&
                    res->format == DataFormat::Float10_11_11 && components == 3;
                const bool native_float = !packed_r11 &&
                    (native_2d_storage || ordinary_3d) &&
                    native_float_storage_image_supported(
                        res->format, components, res->srgb,
                        (b.native_storage_format_support & native_support_bit) != 0);
                const bool native_uint = !packed_r11 && native_uint_2d_storage && !is_atomic &&
                    native_uint_storage_image_supported(
                        res->format, components, res->srgb,
                        (b.native_storage_format_support & native_support_bit) != 0);
                const bool compute_atomic_buffer = is_atomic && b.is_compute;
                if (compute_atomic_buffer) {
                    if (!b.declare_compute_atomic_image_buffer(res->binding)) {
                        ok = false;
                        return true;
                    }
                } else {
                    const bool native_r32ui = is_atomic;
                    const uint32_t native_uint_format = res->format == DataFormat::Uint32
                        ? ImgFmt_R32ui : res->format == DataFormat::Uint16
                            ? ImgFmt_R16ui : components == 4
                                ? ImgFmt_Rgba8ui : ImgFmt_R8ui;
                    b.declare_storage_image(res->binding, dim, arrayed, ms, native_float,
                                            (native_r32ui || packed_r11)
                                                ? ImgFmt_R32ui
                                                : native_uint ? native_uint_format : ImgFmt_Unknown,
                                            packed_r11);
                }
                // Coordinate VGPR per axis. Non-NSA (len==2): consecutive from VADDR (src[0]). NSA (len>2):
                // split across the extra address dwords — coord0 = VADDR, coord k>=1 = byte (k-1) of
                // words[2..3] (dword2 = addr1..4, dword3 = addr5..8). Layout verified via llvm-mc gfx1010.
                const bool nsa = in.len_dwords > 2;
                int va = in.src[0].value;
                auto coord_vgpr = [&](uint32_t k) -> int {
                    if (!nsa || k == 0) return va + (nsa ? 0 : (int)k);
                    uint32_t j = k - 1;
                    return (int)((in.words[2 + j / 4] >> (8 * (j % 4))) & 0xFFu);
                };
                uint32_t coords[3] = {0, 0, 0};
                for (uint32_t k = 0; k < ncoord; k++) coords[k] = vread(coord_vgpr(k));
                uint32_t sample = ms ? vread(coord_vgpr(ncoord)) : 0;   // MSAA sample index = coord after the spatial ones
                if (is_ld) {
                    uint32_t out[4]; b.image_read(res->binding, dim, arrayed, ncoord, coords, out, ms, sample);
                    if (!write_mimg_results(out)) { ok = false; return true; }
                } else if (is_st) {
                    // image_store: gather the VDATA VGPRs selected by dmask into an RGBA texel (channels
                    // absent from dmask store as 0). Under a narrowed EXEC (e.g. a grid-tail bounds check),
                    // the write is EXEC-predicated so inactive lanes don't write out-of-range texels.
                    uint32_t vals[4] = { b.uconst(0), b.uconst(0), b.uconst(0), b.uconst(0) };
                    int vd = in.dst.value, w = 0;
                    if (in.mimg_d16 && !d16_float_resource) { ok = false; return true; }
                    for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                        vals[c] = in.mimg_d16
                            ? b.unpack_half(vread(vd + w / 2), static_cast<uint32_t>(w & 1))
                            : vread(vd + w);
                        ++w;
                    }
                    b.image_write(res->binding, dim, arrayed, ncoord, coords, vals, rs.exec_narrowed, rs.exec);
                } else {
                    // IMAGE_ATOMIC_SWAP/ADD reads its operand from VDATA. GLC=1 overwrites that VGPR
                    // with the pre-operation texel; GLC=0 leaves VDATA unchanged. The helper's phi
                    // preserves the old register value for EXEC-inactive lanes.
                    const int vd = in.dst.value;
                    const uint32_t old = vreg_old(b, rs, vd);
                    const uint16_t atomic_op = spirv_atomic;
                    uint32_t result;
                    if (compute_atomic_buffer) {
                        // #2265: the backend stages an arrayed view as layers tightly packed in
                        // layer-major order (live_compute.cpp detiles each slice from its physical
                        // stride into w*h*4 of linear), so the flat index is
                        // (z*height + y)*width + x. The layer MUST be in the bound as well as the
                        // index: Vulkan leaves an out-of-bounds image atomic undefined, robust
                        // buffer access does not cover atomics, and the 2D path's own comment
                        // records that RADV can spend seconds in one before resetting the device.
                        //
                        // `width`, `height` and `depth` are baked into the module as OpConstants,
                        // which is only safe because all three are part of the shader cache key, so
                        // a module compiled for one extent can never be reused against a smaller
                        // one: `gpu_executor.cpp` sets `compiled.width`/`height` under
                        // `atomic_extent` (an R32_UINT single-component storage image -- exactly
                        // this case) and `compiled.depth`/`img_dim` under `storage_image`, and
                        // `ShaderResourceCompileKey` has a defaulted member-wise `operator==`. Check
                        // that still holds before baking a fourth quantity in here.
                        const uint32_t row = arrayed
                            ? b.ibin(Op_IAdd, coords[1],
                                     b.ibin(Op_IMul, coords[2], b.uconst(res->height)))
                            : coords[1];
                        const uint32_t index = b.ibin(
                            Op_IAdd, coords[0],
                            b.ibin(Op_IMul, row, b.uconst(res->width)));
                        uint32_t active = b.land(
                            b.ucmp(Op_ULessThan, coords[0], b.uconst(res->width)),
                            b.ucmp(Op_ULessThan, coords[1], b.uconst(res->height)));
                        if (arrayed)
                            active = b.land(active,
                                            b.ucmp(Op_ULessThan, coords[2],
                                                   b.uconst(res->depth ? res->depth : 1u)));
                        if (rs.exec_narrowed) active = b.land(rs.exec, active);
                        result = b.cbuf_atomic_rtn(
                            atomic_op, index, vread(vd), res->binding,
                            true, active, old);
                    } else {
                        result = b.image_atomic_u32(
                            atomic_op, res->binding, ncoord, coords, vread(vd),
                            rs.exec_narrowed, rs.exec, old);
                    }
                    if (in.mimg_glc) rs.vreg[vd] = result;
                }
                return true;
            }

            // image_get_resinfo (0x0e): sampled-image dimensions at the integer LOD in VADDR. The
            // DOLL UE4 volume initializer uses dim:3D, dmask:xyz to bounds-check its 8x8x8 dispatch
            // before loading and writing the volume. Array/cube queries remain deferred with their
            // corresponding sampled-image representations.
            // #325: arrayed-ness is a property of the RESOURCE, not of the instruction. The
            // uploader picks the Vulkan view type from the guest T# and cannot see which opcode will
            // sample it, so every declaration of this binding must agree with the T# or the
            // descriptor is a mismatch. Non-array instructions reaching an array texture get a
            // three-component coordinate with layer 0 -- the same base slice the old base-slice 2D
            // view gave them. The predicate is guest_texture_is_uploaded_array(): img_dim 5 AND
            // more than one layer AND block-compressed. `img_dim == 5` alone is NOT it -- a depth-1
            // or non-BC array is a plain 2D image on both sides, and keying on img_dim is exactly
            // the mistake this comment used to describe.
            const bool res_arrayed = res && prosper::gpu::guest_texture_is_uploaded_array(
                                                res->img_dim, res->depth, res->format);
            if (in.opcode == 0x0e) {
                uint32_t dim;
                if (in.mimg_dim == 0u) dim = Dim_1D;
                else if (in.mimg_dim == 1u) dim = Dim_2D;
                else if (in.mimg_dim == 2u) dim = Dim_3D;
                // #2790: 2D_ARRAY (dim 5) queries as SPIR-V Dim_2D with Arrayed set, which is the
                // representation #325 already uploads and declares for these resources -- and
                // `image_get_resinfo` already branches on `tex_is_arrayed(binding)` to ask for the
                // ivec3 form and report its third component as the layer COUNT, which is exactly
                // what GET_RESINFO's third result means for a 2D_ARRAY T#. The lowering was widened
                // for the array case and this dispatch is the site that still declined it -- the
                // same lag #2265 records for the atomic coverage predicate.
                //
                // `res_arrayed` above, not `mimg_dim == 5`, decides which query shape is emitted,
                // and the distinction is load-bearing. `guest_texture_is_uploaded_array()` is dim-5
                // AND depth>1 AND block-compressed; a dim-5 T# that fails it (Sonic Frontiers' own
                // is depth=1) is a plain 2D image on both sides, so the ivec2 path runs and out[2]
                // keeps its default of 1 -- the true layer count for a one-layer array.
                //
                // What this does NOT do is report a layer count for a MULTI-layer non-BC array,
                // because prosper does not materialize one; the query then answers for the resource
                // as actually uploaded, consistent with what every sample of it reads.
                // CONFIDENCE: MED on that case alone -- no title in the corpus is known to issue
                // one, so it is reasoned rather than measured.
                else if (in.mimg_dim == 5u) dim = Dim_2D;
                else { ok = false; return true; }
                if (res->cls != ResourceClass::Texture) { ok = false; return true; }
                if (!b.declare_texture(res->binding, dim, uint_texture, res_arrayed)) {
                    ok = false; return true;
                }
                uint32_t out[4]; b.image_get_resinfo(res->binding, dim, vread(in.src[0].value), out);
                int vd = in.dst.value, w = 0;
                for (uint32_t c = 0; c < 4; c++) if (in.mimg_dmask & (1u << c)) {
                    uint32_t old = vreg_old(b, rs, vd + w); rs.vreg[vd + w] = out[c];
                    predicate_write(b, rs, vd + w, old); w++;
                }
                return true;
            }

            // IMAGE_GET_LOD (0x60): RDNA2 returns {sampler-clamped LOD, raw LOD}; SPIR-V's
            // OpImageQueryLod returns the same pair. House of the Dead 2's Unity scene shaders use
            // the ordinary non-NSA 2D form in fragment programs. Keep every unverified dimension,
            // Table 100 control, address shape, and output component fail-visible rather than guessing.
            if (in.opcode == 0x60) {
                if (!b.is_fragment || res->cls != ResourceClass::Texture ||
                    in.mimg_dim != SQ_DIM_2D || res->img_dim != SQ_DIM_2D ||
                    in.len_dwords != 2u || mimg_get_lod_has_unmodeled_controls(in) ||
                    !(in.mimg_dmask & 0x3u) || (in.mimg_dmask & ~0x3u) ||
                    res->unnormalized || res->depth_compare ||
                    !b.declare_texture(res->binding, Dim_2D, uint_texture)) {
                    ok = false;
                    return true;
                }
                uint32_t out[2];
                b.image_get_lod_2d(res->binding, vread(in.src[0].value),
                                   vread(in.src[0].value + 1), out);
                int vd = in.dst.value, written = 0;
                for (uint32_t component = 0; component < 2; ++component) {
                    if (!(in.mimg_dmask & (1u << component))) continue;
                    const uint32_t old = vreg_old(b, rs, vd + written);
                    rs.vreg[vd + written] = out[component];
                    predicate_write(b, rs, vd + written, old);
                    ++written;
                }
                return true;
            }

            // --- Sampled-texture path: image_sample* (0x20/0x24/0x25/0x27) / image_gather4_lz (0x47) /
            // image_load (0x00). 2D (any LOD variant) or 3D (implicit-LOD or LOD-0 sample); NSA allowed
            // (coords gathered below). image_sample = 0x20 (implicit-LOD), image_sample_l = 0x24
            // (explicit LOD in last coord), image_sample_b = 0x25 (implicit-LOD + BIAS in FIRST vaddr),
            // image_sample_lz = 0x27 (LOD 0), image_gather4_lz = 0x47 (2x2 single-channel gather,
            // base level), image_load = 0x00 (integer texel fetch). Opcodes round-trip-verified
            // via llvm-mc gfx1010 (#273).
            // op 0xa0 = the high-bit sibling of IMAGE_SAMPLE (0x20): the decoder builds the 8-bit
            // MIMG opcode as ((word0&1)<<7)|bits[24:18], so 0xa0 = 0x80|0x20. GTA V's (PPSA04263,
            // RAGE) intro/composite pipeline (es=0x2042d6a200 / ps=0x2042d83c00) issues it as a plain
            // 2D texture sample; rejecting it dropped that pipeline's draws and blacked the whole
            // frame (#1140). Lowered as an ordinary implicit-LOD sample it renders the animated
            // Rockstar Games intro logo correctly (live PPSA04263 capture). CONFIDENCE: MED — the base
            // op and the "samples a 2D texture" behavior are live-title evidence; the exact high-bit
            // family (an RDNA2 gfx10.3 sample variant) is not yet llvm-mc round-trip-verified, so a
            // future title exercising its distinguishing modifier may need a dedicated lowering.
            const bool is_sample = (in.opcode == 0x20) || (in.opcode == 0xa0);
            const bool is_zero_mip_load = in.opcode == 0x01;
            const bool is_load = in.opcode == 0x00 || is_zero_mip_load;
            const bool is_sample_l = (in.opcode == 0x24), is_sample_lz = (in.opcode == 0x27);
            const bool is_sample_b = (in.opcode == 0x25), is_gather_lz = (in.opcode == 0x47);
            const bool is_sample_c_lz = (in.opcode == 0x2f);
            // image_gather4_lz_o = 0x57 (gather at base level with the _o packed-offset operand in the
            // FIRST vaddr — llvm-mc gfx1030 round-trip on live DOLL bytes: 0xf15c0808 "image_gather4_lz_o
            // v[4:7], [v0, v18, v19], ..." — coords follow the offset, matching image_sample_b's
            // modifier-first vaddr convention). DOLL's FXAA/upsample pass PS (#294).
            const bool is_gather_lz_o = (in.opcode == 0x57);
            // image_sample_lz_o = 0x37 (LOD-0 sample with the _o packed-offset FIRST vaddr — llvm-mc
            // gfx1010 round-trip on live DOLL FXAA bytes: 0xf0dc0808/0xf0dc080a "image_sample_lz_o";
            // the offset-adjust folds into the normalized coords, see image_sample_lz_offset_2d).
            const bool is_sample_lz_o = (in.opcode == 0x37);
            // image_sample_d = 0x22: sample with EXPLICIT gradients. For 2D the vaddr packs four
            // derivative dwords FIRST — [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy] — then the (u,v) coords, per ISA
            // 8.2.4's "{derivative}{body}" order. Blue Prince (PPSA25009) issues it as a plain 2D texture
            // sample in a fragment shader (ps=0x2011d60100); rejecting it dropped that whole pipeline's
            // draws. Preserve those explicit derivatives with OpImageSampleExplicitLod + Grad: projected,
            // wrapped, atlas, and uniform coordinates do not share the screen quad's implicit derivative.
            // CONFIDENCE: HIGH — operand order is ISA-defined and an execution regression distinguishes the
            // requested gradient-selected mip from the implicit-derivative result.
            const bool is_sample_d = (in.opcode == 0x22);
            // #325: 2D_ARRAY resources are now uploaded and declared as real arrays -- see
            // res_arrayed above. The historical base-slice fallback this comment used to describe is
            // gone; what remains of it is that a non-array INSTRUCTION reaching an array resource
            // reads layer 0, which is the same slice it used to get.
            const bool dim2d = (in.mimg_dim == 1u || in.mimg_dim == 5u), dim3d = (in.mimg_dim == 2u);
            const bool dim_msaa = in.mimg_dim == 6u;
            const bool dimcube = (in.mimg_dim == 3u);   // CUBE: stacked-face 2D lowering (#273, below)
            if (in.mimg_dim == 5u && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[recompile] 2D_ARRAY image op: resource %s an uploaded array (#325)\n",
                        res_arrayed ? "IS" : "is NOT");
            if ((!is_sample && !is_load && !is_sample_l && !is_sample_lz && !is_sample_b &&
                 !is_sample_c_lz && !is_gather_lz &&
                 !is_gather_lz_o && !is_sample_lz_o && !is_sample_d) ||
                (!dim2d && !dim3d && !dimcube && !dim_msaa)) { ok = false; return true; }
            if (res->cls != ResourceClass::Texture) { ok = false; return true; }
            // #3048: the guest's mip selector no longer has to be discarded. When the compute
            // backend materializes this resource's whole declared chain -- one derivation,
            // `shader_resource_compute_mip_chain_levels`, read by BOTH this lowering and
            // live_compute's image creation so the two cannot disagree about how many levels exist
            // -- the operand reaches OpImageFetch's Lod. This is the only honest answer for the
            // dynamic case: Sonic Frontiers' three scene-width stage kernels issue IMAGE_LOAD_MIP
            // against a 12-level 2048x2048 R32G32_FLOAT surface with the mip NOT provably zero, so
            // the specialization below can never admit them.
            //
            // Compute only, deliberately: the graphics texture path still uploads one level (its
            // generated-chain gate in the render backend is narrower than this one), so a fragment
            // stage reaching here would fetch a level that does not exist.
            uint32_t dynamic_mip_vgpr = 0;
            const uint32_t materialized_mip_levels =
                prosper::gpu::shader_resource_compute_mip_chain_levels(*res);
            const bool dynamic_mip_load =
                is_zero_mip_load && b.is_compute && materialized_mip_levels > 1u &&
                rdna2_mimg_dynamic_mip_shape(in, &dynamic_mip_vgpr) &&
                in.mimg_dim == SQ_DIM_2D &&
                prosper::gpu::shader_resource_uses_ordinary_2d_image(*res, true, false, false) &&
                // `in_mip_tail` used to be refused here as well. It no longer is (#3134): a
                // tail-packed selected level is a MODELLED placement, and the level count above is
                // the authority on it -- `shader_resource_mip_chain_plan` admits the tail only
                // after proving its own in-block coordinates equal the ones the descriptor decode
                // published, and reports one level otherwise. Repeating the test here would have
                // kept refusing the whole small-texture class (Stray's 32x32 six-level pyramid is
                // smaller than one 64 KiB macroblock, so EVERY one of its levels is in the tail)
                // for a placement the backend now uploads.
                !res->unnormalized && !res->depth_compare &&
                res->sample_count == 1u && !res->compression_enabled;
            // IMAGE_LOAD_MIP's final address is a real guest mip selector. Specialize it away only
            // after the per-use fold and the materialized-resource checks agree. The 2D_ARRAY form
            // retains its slice coordinate; only the separately proven mip operand is discarded.
            if (is_zero_mip_load && !dynamic_mip_load &&
                (!rdna2_mimg_zero_mip_shape(in) || !res->proven_zero_mip ||
                 res->img_dim != in.mimg_dim || res->sample_count != 1u ||
                 res->declared_mip_levels != 1u || res->in_mip_tail ||
                 res->compression_enabled || (in.mimg_dim == 5u && !b.is_compute))) {
                // Name the sub-condition. Seven terms collapse into one
                // `mode=unresolved-operand`, and an investigation cannot act on that: "the mip is
                // not proven zero" and "the resource declares compression" are different pieces of
                // work. Ungated and deduped per pc, so it costs one line per declining site --
                // PROSPER_DBG, the usual home for this, produces a ~1.5 GB log and desyncs the pad
                // script badly enough that the route never reaches the phase being diagnosed.
                //
                // Measured on GTA V's 0x2042f49a00, a compute pass that reads the main depth and
                // stencil and writes two 4K storage images the frame goes on to sample:
                //   pc=16 shape=1 proven_zero_mip=0 ... compressed=1
                // so exactly two terms hold it, and one of them describes guest bytes that this
                // resource does not read -- 0x2052ac0000 resolves through the sampled-depth bridge,
                // whose pixels come from a retained Vulkan image rather than from compressed memory.
                static std::mutex mip_mutex;
                // Same program-scoped key as [mimg-mip-why]; see the note there.
                static std::set<std::pair<uint64_t, uint32_t>> mip_reported;
                bool first_mip = false;
                {
                    std::lock_guard<std::mutex> lock(mip_mutex);
                    first_mip = mip_reported.emplace(b.diagnostic.program_address, in.pc).second;
                }
                if (first_mip)
                    std::fprintf(stderr,
                                 "[mimg-mip] program=0x%llx image_load_mip declined pc=%u shape=%d "
                                 "proven_zero_mip=%d img_dim=%u/%u samples=%u mips=%u mip_tail=%d "
                                 "compressed=%d array_in_gfx=%d addr=0x%llx %ux%ux%u "
                                 "dataformat=%d ncomp=%u "
                                 "tile=%u dmask=0x%x unorm=%u glc=%u layer_stride=%u "
                                 "dyn_shape=%d nsa=%u materialized_mips=%u compute=%d\n",
                                 (unsigned long long)b.diagnostic.program_address, in.pc,
                                 (int)rdna2_mimg_zero_mip_shape(in),
                                 (int)res->proven_zero_mip, res->img_dim, in.mimg_dim,
                                 res->sample_count, res->declared_mip_levels,
                                 (int)res->in_mip_tail, (int)res->compression_enabled,
                                 (int)(in.mimg_dim == 5u && !b.is_compute),
                                 (unsigned long long)res->gpu_addr, res->width, res->height,
                                 res->depth, (int)res->format, res->num_components,
                                 res->tile_mode, in.mimg_dmask,
                                 (unsigned)in.mimg_unorm, (unsigned)in.mimg_glc,
                                 res->layer_stride_bytes,
                                 // The DYNAMIC route's own inputs. Without these the line names
                                 // only the zero-mip specialization's terms, and an investigation
                                 // reading it cannot tell "the address encoding is not modelled"
                                 // (#3134's original state) from "the chain is not materialized on
                                 // this stage" -- which are different pieces of work.
                                 (int)rdna2_mimg_dynamic_mip_shape(in), in.mimg_nsa,
                                 materialized_mip_levels, (int)b.is_compute);
                ok = false;
                return true;
            }
            // Guest 2D_MSAA IMAGE_LOAD is represented exactly as a host single-sample 2D array: the
            // guest sample coordinate selects the array layer. Asterix's resolve PS mixes the ordinary
            // consecutive-vaddr packet with three one-extra-dword NSA packets; llvm-mc gfx1030 confirms
            // those NSA addresses are [x,y,sample] in bytes {VADDR, words[2].lo, words[2].byte1}.
            // Accept only those two exact address shapes. Nonzero unused NSA bytes could carry further
            // operands, and every other MSAA op/count/address shape stays fail-visible.
            const bool msaa_address_shape = in.len_dwords == 2u ||
                (in.len_dwords == 3u && (in.words[2] & 0xffff0000u) == 0u);
            const bool msaa_array_fetch = dim_msaa && is_load && msaa_address_shape &&
                !in.mimg_unorm && !in.has_modifier && res->img_dim == 6u &&
                res->sample_count == 4u && res->depth == 1u &&
                res->declared_mip_levels == 1u && !res->depth_compare;
            if (dim_msaa && !msaa_array_fetch) { ok = false; return true; }
            // S# FORCE_UNNORMALIZED makes only the spatial sample coordinates texel-space. Vulkan's
            // native unnormalized sampler cannot represent the guest contract in general (R-Type's
            // live S# combines it with wrap addressing and a nonzero max LOD), so retain the ordinary
            // sampler and convert texels to normalized coordinates in the shader. Array layers, cube
            // faces, explicit LOD/bias, DREF and packed offsets are not spatial coordinates. Explicit
            // gradients receive the same per-axis scale so mip selection remains identical.
            const bool normalize_sampler_coordinates = res->unnormalized &&
                !getenv("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE");
            auto normalized_spatial = [&](uint32_t coordinate, uint32_t extent) {
                if (!normalize_sampler_coordinates) return coordinate;
                if (!extent) { ok = false; return b.uconst(0); }
                return b.fbin(Op_FMul, coordinate,
                              b.uconst(fbits(1.0f / static_cast<float>(extent))));
            };
            // UNRM=1 supplies unnormalized (texel-space) coordinates (Table 100). Compilers set it
            // only on loads/stores/atomics — and our fetch paths are already texel-space — but a
            // SAMPLER op with UNRM set would treat texel coords as normalized and sample a wildly
            // wrong location. Reject the sampler forms until a live title exercises one.
            if (in.mimg_unorm && !is_load) { ok = false; return true; }
            // coords (normalized float for sample, integer texel for load). Non-NSA: consecutive from VADDR.
            // NSA (len>2): coord0 = VADDR, coord k>=1 = byte (k-1) of the extra address dwords words[2..3].
            const bool nsa = in.len_dwords > 2;
            // GTA V's FSR1 RCAS packet uses ordinary 2D IMAGE_LOAD with A16: unsigned x/y
            // texel coordinates occupy the low/high 16-bit halves of ONE VADDR VGPR (LLVM prints
            // `image_load ..., v0, ... a16 d16`, not `v[0:1]`). Sample-family A16 carries fp16
            // addresses, and NSA/array/MSAA have different operand shapes; retain only the exact
            // integer-load contract until those forms have independent evidence.
            if (in.mimg_a16 &&
                (in.opcode != 0x00 || in.mimg_dim != SQ_DIM_2D ||
                 res->img_dim != SQ_DIM_2D || nsa)) {
                ok = false;
                return true;
            }
            int va = in.src[0].value;
            auto cvg = [&](uint32_t k) -> int { if (!nsa || k == 0) return va + (nsa ? 0 : (int)k);
                uint32_t j = k - 1; return (int)((in.words[2 + j / 4] >> (8 * (j % 4))) & 0xFFu); };
            uint32_t out[4];
            if (dimcube) {
                // Cube-processed coordinates encode face selection and a normalized in-face pair;
                // no axis is a plain texel coordinate. FORCE_UNNORMALIZED on this combination has no
                // representable generic meaning, so keep it fail-visible instead of scaling a face.
                if (normalize_sampler_coordinates) { ok = false; return true; }
                // CUBE sample (#273 — DOLL's title post PSes sample their reflection probes /
                // skybox with `image_sample_l ..., dim:CUBE`). The compiled coords are the standard
                // AMD cube-processed form (Mesa ac_prepare_cube_coords): vaddr = { sc*rcp(|ma|)+1.5,
                // tc*rcp(|ma|)+1.5, face_id [, lod] } — face coords centered at 1.5 (span [1,2]),
                // face id an integral float from v_cubeid. Our texture backend uploads the cube as
                // SIX FACES STACKED VERTICALLY in one 2D image (w x 6h — see the live_renderer cube
                // path), so the sample lowers to a plain 2D sample at u = x-1, v = (face + clamp
                // (y-1)) / 6 at base LOD (mips aren't uploaded; a >0 LOD clamps to the one level).
                // The in-face clamp stops bilinear bleed across face seams. CONFIDENCE: MED — the
                // coordinate convention is Mesa-verified; face memory layout is validated visually.
                if (!is_sample && !is_sample_l && !is_sample_lz && !is_sample_b &&
                    !is_sample_c_lz) { ok = false; return true; }
                // Modifier-first order: _b carries bias in vaddr0; _c_lz carries DREF there.
                const uint32_t ci = (is_sample_b || is_sample_c_lz) ? 1u : 0u;
                uint32_t x = vread(cvg(ci)), y = vread(cvg(ci + 1)), fid = vread(cvg(ci + 2));
                const uint32_t one = b.uconst(fbits(1.0f)), zero = b.uconst(fbits(0.0f));
                uint32_t uf = b.fbin(Op_FSub, x, one);
                uint32_t vf = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, b.fbin(Op_FSub, y, one), one), zero);
                uint32_t layer = b.fext1(Glsl_RoundEven,
                                     b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, fid, b.uconst(fbits(5.0f))), zero));
                uint32_t v6 = b.fbin(Op_FMul, b.fbin(Op_FAdd, layer, vf), b.uconst(fbits(1.0f / 6.0f)));
                if (is_sample_c_lz) {
                    // Point-light shadow maps use the same cube-processed coordinates as ordinary
                    // samples, with DREF prepended: [dref, x, y, face]. The renderer uploads cube
                    // faces as one vertical 2D stack, so compare the transformed face coordinate
                    // manually through its ordinary non-compare sampler (#1167/#1169).
                    if (uint_texture || !res->depth_compare) { ok = false; return true; }
                    if (!b.declare_texture(res->binding, Dim_2D, false, res_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(res->binding, uf, v6, vread(cvg(0)),
                                                  res->depth_compare_func,
                                                  res->mag_filter != 0u, res->addr_uvw[0],
                                                  res->addr_uvw[1], res->border_color_type, out);
                } else {
                    if (!b.declare_texture(res->binding, Dim_2D, uint_texture, res_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_lod_2d(res->binding, uf, v6, b.uconst(0), out);
                }
            } else if (dim3d) {
                // 3D: implicit-LOD / LOD-0 sample, or an integer texel FETCH (image_load — DOLL's
                // color-grade 3D LUT, #273).
                if (!is_sample && !is_sample_lz && !is_load) { ok = false; return true; }
                if (!b.declare_texture(res->binding, Dim_3D, uint_texture)) {
                    ok = false; return true;
                }
                uint32_t cu = vread(cvg(0)), cv = vread(cvg(1)), cw = vread(cvg(2));
                if (!is_load) {
                    cu = normalized_spatial(cu, res->width);
                    cv = normalized_spatial(cv, res->height);
                    cw = normalized_spatial(cw, res->depth);
                }
                if (is_sample)      b.image_sample_3d(res->binding, cu, cv, cw, out);
                else if (is_load)   b.image_fetch_3d(res->binding, cu, cv, cw, out);
                else                b.image_sample_lod_3d(res->binding, cu, cv, cw,
                                                          b.uconst(0), out);   // _lz: base level
            } else if (is_sample_c_lz) {
                // Astro Bot shadow/visibility packet (opcode 0x2f, dim 2D_ARRAY, NSA). ISA 8.2.5
                // vaddr order is "{offset}{bias}{z-compare}{derivative}{body}" — the z-compare
                // reference PRECEDES the coordinates, so SAMPLE_C_LZ 2D_ARRAY reads
                // [dref, u, v, slice] (the same modifier-first rule the _b/_o paths already use).
                // The earlier [u,v,layer,dref] mapping rotated every operand (#883).
                // EVIDENCE: no rendered-frame validation exists for either order — Astro crashes
                // in engine init before it reaches shadow rendering (#825), and the prior
                // "[u,v,layer,dref]" comment was an un-round-tripped interpretation of the packet,
                // not a captured/disassembled fact. This order is grounded in the ISA plus an
                // llvm-mc gfx1030 round-trip of a canonical c_lz 2D_ARRAY NSA packet
                // (image_sample_c_lz v5, [v10,v11,v12,v13], ... = 0xf0bc012a 0x0040050a 0x000d0c0b:
                // four consecutive address VGPRs, dref at slot 0 per 8.2.5) and LLVM's own
                // llvm.amdgcn.image.sample.c.lz lowering (dref before coords). CONFIDENCE: MED —
                // the #825 lane must re-validate against a real rendered shadow once it lights up.
                // Integer views are not legal for this lowering, and a non-shadow S# means the
                // provenance drifted; reject rather than silently turning a comparison sample into
                // an ordinary color read.
                if (uint_texture || !res->depth_compare) { ok = false; return true; }
                if (in.mimg_dim == 5u) {
                    if (b.is_compute) {
                        // Compute has a backend-reflected 2D-array path. Preserve [dref,u,v,slice]
                        // and perform the comparison manually over an ordinary color sampler, so no
                        // compare-sampler/Dref Vulkan contract is required. Astro Bot's world-map
                        // visibility shader selects among sixteen depth layers here.
                        if (!b.declare_texture(res->binding, Dim_2D, false, true)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding,
                            normalized_spatial(vread(cvg(1)), res->width),
                            normalized_spatial(vread(cvg(2)), res->height),
                            vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type,
                            out, true, vread(cvg(3)));
                    } else {
                        // #325: the graphics fallback used to declare a base-slice 2D image here,
                        // "until its resource uploader can create matching array views". It now
                        // does -- but ONLY for resources the uploader actually uploads as arrays,
                        // which is what res_arrayed answers. A DIM=5 instruction is not sufficient:
                        // this title's shadow maps are declared img_dim 5 with **depth 1**, so they
                        // are uploaded as plain 2D, and declaring them Arrayed produced
                        // VUID-vkCmdDraw-viewType-07752 (caught by tools/vkval, not by ctest --
                        // validation errors do not fail a test).
                        if (!b.declare_texture(res->binding, Dim_2D, false, res_arrayed)) {
                            ok = false; return true;
                        }
                        b.image_sample_dref_manual_2d(
                            res->binding,
                            normalized_spatial(vread(cvg(1)), res->width),
                            normalized_spatial(vread(cvg(2)), res->height),
                            vread(cvg(0)),
                            res->depth_compare_func, res->mag_filter != 0u,
                            res->addr_uvw[0], res->addr_uvw[1], res->border_color_type,
                            out, res_arrayed, vread(cvg(3)));
                    }
                } else if (in.mimg_dim == 1u) {
                    // Plain 2D form (Blue Prince's lit-material PSes, #1271: 436 rejects/run, all
                    // op 0x2f dim 1 dmask 0x1 — the entire shadowed lighting pass dropped and the
                    // scene rendered unattenuated/blown-out). Same 8.2.5 vaddr order with no array
                    // slice: [dref, u, v]. Lowered as a manual compare against the color-sampled
                    // shadow map (see image_sample_dref_manual_2d for why not a compare sampler).
                    if (!b.declare_texture(res->binding, Dim_2D, false, res_arrayed)) {
                        ok = false; return true;
                    }
                    b.image_sample_dref_manual_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(1)), res->width),
                        normalized_spatial(vread(cvg(2)), res->height),
                        vread(cvg(0)), res->depth_compare_func,
                        res->mag_filter != 0u, res->addr_uvw[0],
                        res->addr_uvw[1], res->border_color_type, out);
                } else { ok = false; return true; }
            } else if (is_gather_lz || is_gather_lz_o) {
                // gather4 dmask selects ONE channel (must be a single bit); the result is always the
                // four texels of that channel, with gather order preserved. D16 changes only the
                // physical VDATA layout: the four fp16 results occupy two consecutive VGPRs. GTA V's
                // FSR1 EASU shaders consume those packed halves with VOP3P; writing four f32 VGPRs
                // clobbers the following temporaries and turns an otherwise intact scene into bands.
                uint32_t dm = in.mimg_dmask;
                if (dm != 1u && dm != 2u && dm != 4u && dm != 8u) { ok = false; return true; }
                uint32_t comp = dm == 1u ? 0u : dm == 2u ? 1u : dm == 4u ? 2u : 3u;
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture, res_arrayed)) {
                    ok = false; return true;
                }
                if (is_gather_lz_o)   // vaddr order for _o: [packed offset, u, v]
                    b.image_gather_offset_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(1)), res->width),
                        normalized_spatial(vread(cvg(2)), res->height),
                        comp, vread(cvg(0)), out);
                else
                    b.image_gather_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(0)), res->width),
                        normalized_spatial(vread(cvg(1)), res->height), comp, out);
                int vd = in.dst.value;
                if (in.mimg_d16) {
                    if (!d16_float_resource) { ok = false; return true; }
                    for (uint32_t word = 0; word < 2; ++word) {
                        const uint32_t old = vreg_old(b, rs, vd + static_cast<int>(word));
                        rs.vreg[vd + static_cast<int>(word)] =
                            b.pack_half2x16(out[word * 2], out[word * 2 + 1]);
                        predicate_write(b, rs, vd + static_cast<int>(word), old);
                    }
                } else {
                    for (uint32_t c = 0; c < 4; c++) {
                        uint32_t old = vreg_old(b, rs, vd + static_cast<int>(c));
                        rs.vreg[vd + static_cast<int>(c)] = out[c];
                        predicate_write(b, rs, vd + static_cast<int>(c), old);
                    }
                }
                return true;
            } else {
                const bool load_2d_array = b.is_compute && in.opcode == 0x00 &&
                    in.mimg_dim == 5u && res->img_dim == 5u;
                const bool mip_load_2d_array = b.is_compute && is_zero_mip_load &&
                    in.mimg_dim == 5u && res->img_dim == 5u;
                // #325: array SAMPLE was restricted to compute. Nothing about an array slice is
                // compute-specific -- the restriction recorded where it was first needed, not a
                // constraint -- and graphics paid for it: Tomb Raider I-III Remastered textures its
                // entire world from ONE 256-slice array, so every world surface sampled slice 0 and
                // rendered flat. Measured on a captured Croft Manor frame, zeroing slices 1..255
                // changed 0.0% of pixels while zeroing all 256 changed 62.9%.
                // `res_arrayed` and not `img_dim == 5` alone: the uploader arrays a resource only
                // when it is ALSO multi-layer and block-compressed, so keying on img_dim would
                // declare Arrayed for this title's depth-1 shadow maps and for Float32 arrays that
                // get a plain 2D view -- VUID-vkCmdDraw-viewType-07752, the same failure as the
                // hardcoded flag two branches up, at a different site. Compute keeps its own path:
                // it picks the view from the SPIR-V reflection, so it is not bound by that.
                const bool array_sample = in.mimg_dim == 5u &&
                    (b.is_compute || res_arrayed) &&
                    res->img_dim == 5u && (is_sample || is_sample_l || is_sample_lz);
                const bool host_array = load_2d_array || mip_load_2d_array ||
                    array_sample || msaa_array_fetch;
                if (!b.declare_texture(res->binding, Dim_2D, uint_texture, host_array || res_arrayed)) {
                    ok = false; return true;
                }
                if (msaa_array_fetch || load_2d_array || mip_load_2d_array) {
                    b.image_fetch_2d_array(
                        res->binding, vread(cvg(0)), vread(cvg(1)), vread(cvg(2)), out);
                } else if (array_sample) {
                    // All three retain the 2D-array slice in the SPIR-V coordinate. SAMPLE_LZ
                    // resolves level zero and SAMPLE_L carries its own LOD, so both take the
                    // explicit-LOD form. Plain SAMPLE must NOT: in a fragment stage its LOD comes
                    // from quad derivatives, and forcing level zero there would sample the base mip
                    // across every textured surface -- correct in compute, wrong on a wall.
                    // image_sample_2d_array picks implicit or explicit by stage exactly as the
                    // non-array image_sample_2d does, so compute behaviour is preserved bit for bit.
                    const uint32_t au = normalized_spatial(vread(cvg(0)), res->width);
                    const uint32_t av = normalized_spatial(vread(cvg(1)), res->height);
                    const uint32_t layer = vread(cvg(2));
                    if (is_sample)
                        b.image_sample_2d_array(res->binding, au, av, layer, out);
                    else
                        b.image_sample_lod_2d_array(
                            res->binding, au, av, layer,
                            is_sample_l ? vread(cvg(3)) : b.uconst(fbits(0.0f)), out);
                } else if (is_sample_b) {      // vaddr order for _b: [bias, u, v]
                    b.image_sample_bias_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(1)), res->width),
                        normalized_spatial(vread(cvg(2)), res->height),
                        vread(cvg(0)), out);
                } else if (is_sample_lz_o) {   // vaddr order for _o: [packed offset, u, v]
                    b.image_sample_lz_offset_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(1)), res->width),
                        normalized_spatial(vread(cvg(2)), res->height),
                        vread(cvg(0)), out);
                } else if (is_sample_d) {   // vaddr order for _d: [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy, u, v]
                    b.image_sample_grad_2d(
                        res->binding,
                        normalized_spatial(vread(cvg(4)), res->width),
                        normalized_spatial(vread(cvg(5)), res->height),
                        normalized_spatial(vread(cvg(0)), res->width),
                        normalized_spatial(vread(cvg(1)), res->height),
                        normalized_spatial(vread(cvg(2)), res->width),
                        normalized_spatial(vread(cvg(3)), res->height), out);
                } else {
                    uint32_t cu, cv;
                    if (in.mimg_a16) {
                        const uint32_t packed = vread(cvg(0));
                        cu = b.bfe_u(packed, b.uconst(0), b.uconst(16));
                        cv = b.bfe_u(packed, b.uconst(16), b.uconst(16));
                    } else {
                        cu = vread(cvg(0));
                        cv = vread(cvg(1));
                    }
                    if (!is_load) {
                        cu = normalized_spatial(cu, res->width);
                        cv = normalized_spatial(cv, res->height);
                    }
                    if (is_sample)         b.image_sample_2d(res->binding, cu, cv, out);
                    else if (is_sample_lz) b.image_sample_lod_2d(res->binding, cu, cv, b.uconst(0), out);      // LOD 0
                    // Explicit-LOD plain 2D uses [u,v,lod]. Graphics keeps the established
                    // base-slice view for DIM=5, whose address is [u,v,slice,lod].
                    else if (is_sample_l)  b.image_sample_lod_2d(res->binding, cu, cv,
                        vread(cvg(in.mimg_dim == 5u && res->img_dim == 5u ? 3u : 2u)), out);
                    else if (dynamic_mip_load) {
                        // Clamp the selector to the levels the host image actually has.
                        //
                        // The half that is certain: an out-of-range `Lod` operand is UNDEFINED in
                        // SPIR-V, so some clamp is mandatory whatever the hardware does.
                        //
                        // The half that is inference: RDNA 2 ISA doc 70648 gives the image
                        // descriptor BASE_LEVEL/LAST_LEVEL fields as the addressable level range,
                        // and prosper reads `declared_mip_levels` off exactly that pair -- but the
                        // guide's MIMG section does not spell out the saturating behaviour for an
                        // out-of-range IMAGE_LOAD_MIP operand, and no live capture here exercises
                        // one. Saturating to the last level is the reading consistent with the
                        // field's purpose; returning zeros would be the other candidate.
                        // CONFIDENCE: MED -- revisit if a title is measured issuing an
                        // out-of-range mip.
                        // The register the shape predicate itself identified, not a second
                        // derivation of it: for this packet family they are the same VGPR, and
                        // reading it from one place keeps them that way.
                        const uint32_t requested = vread(static_cast<int>(dynamic_mip_vgpr));
                        const uint32_t lod = b.uext2(
                            Glsl_UMin, requested, b.uconst(materialized_mip_levels - 1u));
                        b.image_fetch_2d_lod(res->binding, cu, cv, lod, out);
                    }
                    else                   b.image_fetch_2d (res->binding, cu, cv, out);
                }
            }
            // D16 changes the VDATA layout, not the logical image result: selected components are
            // still in dmask order, but two binary16 values share each consecutive VGPR.
            if (!write_mimg_results(out)) { ok = false; return true; }
            return true;
        }
        case Rdna2Format::DS: {
            if (in.opcode == kDsOpcodeBpermuteB32) {
                // A native subgroup exactly matching the guest wave gives each architectural
                // 32-lane half a valid shuffle domain. Portable compute needs a workgroup-scratch
                // gather and remains fail-visible until that separate synchronized route exists.
                if (!b.is_compute || in.ds_gds || !b.native_subgroup_size ||
                    b.native_subgroup_size != b.wave_size) {
                    ok = false;
                    return true;
                }
                const auto address = rs.vreg.find(in.src[0].value);
                const auto source = rs.vreg.find(in.src[1].value);
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.ds_bpermute_b32(
                    address == rs.vreg.end() ? b.uconst(0) : address->second,
                    source == rs.vreg.end() ? b.uconst(0) : source->second,
                    rs.exec, b.uconst(in.literal));
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            if (in.opcode == 0x35) {                    // ds_swizzle_b32: VDST = lane-gather(ADDR)
                if (!b.is_compute && !b.is_fragment) { ok = false; return true; }
                uint32_t source_lane = 0;
                if (!b.ds_swizzle_source_lane(in.literal, &source_lane)) {
                    ok = false; return true;
                }
                auto source = rs.vreg.find(in.src[0].value);
                const uint32_t source_value =
                    source == rs.vreg.end() ? b.uconst(0) : source->second;
                const uint32_t shuffled = b.subgroup_shuffle(source_value, source_lane);
                const uint32_t active_word = b.sel(rs.exec, b.uconst(1), b.uconst(0));
                const uint32_t fetched_active = b.subgroup_shuffle(active_word, source_lane);
                const uint32_t result = b.sel(
                    b.ucmp(Op_INotEqual, fetched_active, b.uconst(0)),
                    shuffled, b.uconst(0));
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = result;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            // GDS=1 targets the device-global data share, not workgroup LDS — running it against
            // our LDS model would silently give per-workgroup semantics. Reject — EXCEPT
            // ds_append/ds_consume (0x3e/0x3d): Astro Bot's live counter words carry GDS=1
            // (0xd8fa/0xd8f6 — llvm-mc gfx1030 round-trip: plain append is 0xd8f8), and the
            // existing wave_append model was landed and live-validated against exactly those
            // packets (#554/#580). Kept as a documented per-workgroup approximation of the
            // device-global counter (exact for the exercised dispatch shapes). CONFIDENCE: MED.
            if (in.ds_gds && in.opcode != 0x3d && in.opcode != 0x3e &&
                !(b.is_compute && in.opcode == 0x0d)) { ok = false; return true; }
            // ds_write_addtid_b32 (0xb0) / ds_read_addtid_b32 (0xb1). AMD RDNA2 ISA 10.4:
            //   LDS address = LDS_BASE + immediate + M0[15:0] + TID(0..63)*4.
            // TID is the lane within the hardware wave, not the workgroup-linear invocation id.
            // Compute therefore uses real Workgroup LDS and wraps only the lane id at wave_size;
            // M0 supplies a distinct base when a workgroup contains more than one wave. The final
            // address itself does not wrap (RDNA2 LDS allocations explicitly do not wrap).
            // Astro Bot's world-map material kernel uses several such lane arrays before reducing
            // them through explicit DS reads and barriers.
            if (b.is_compute && (in.opcode == 0xb0 || in.opcode == 0xb1)) {
                auto m0 = rs.sreg.find(124);
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                b.declare_lds();
                if (!b.lds_var) { ok = false; return true; }
                const uint32_t base = b.ibin(Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
                const uint32_t tid = b.ibin(
                    Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1u));
                const uint32_t byte_addr = b.ibin(
                    Op_IAdd,
                    b.ibin(Op_IAdd, base, b.uconst(in.literal)),
                    b.ibin(Op_ShiftLeftLogical, tid, b.uconst(2)));
                const uint32_t idx = b.ibin(Op_ShiftRightLogical, byte_addr, b.uconst(2));
                if (in.opcode == 0xb0) {
                    auto value = rs.vreg.find(in.src[1].value);
                    b.lds_store(idx, value == rs.vreg.end() ? b.uconst(0) : value->second,
                                rs.exec_narrowed, rs.exec,
                                b.atomicized_lds_store_pcs.contains(in.pc));
                } else {
                    const uint32_t old = vreg_old(b, rs, in.dst.value);
                    rs.vreg[in.dst.value] = b.lds_load(
                        idx, rs.exec_narrowed, rs.exec, old);
                }
                return true;
            }
            // In a GRAPHICS stage (#273), ADDTID is a compiler spill/reload through per-wave LDS:
            // a per-lane VGPR spill through LDS (addr = M0 + offset + tid*4) — DOLL's title post
            // PSes spill v15 before their accumulation loop and reload it after. Per-invocation the
            // slot is ONE value: track it in rs.lds_addtid keyed by (M0 SSA id, inst offset); the
            // matching read returns the spilled SSA value. A write with UNTRACKED M0 still no-ops
            // (nothing in this model can observe it) but poisons nothing; a read with untracked M0
            // or a never-written slot rejects loudly. VERIFIED(round-trip llvm-mc gfx1010:
            // 0xdac00000/0x00000f00 -> ds_write_addtid_b32 v15; llvm-mc gfx1030:
            // 0xdac40000/0x0f000000 -> ds_read_addtid_b32 v15).
            if (in.opcode == 0xb0 || in.opcode == 0xb1) {
                auto m0 = rs.sreg.find(124);
                if (in.opcode == 0xb0) {
                    if (m0 != rs.sreg.end()) {
                        auto vr = rs.vreg.find(in.src[1].value);
                        rs.lds_addtid[((uint64_t)m0->second << 32) | in.literal] =
                            vr != rs.vreg.end() ? vr->second : b.uconst(0);
                    }
                    return true;   // spill: unobservable beyond the tracked slot
                }
                if (m0 == rs.sreg.end()) { ok = false; return true; }
                auto slot = rs.lds_addtid.find(((uint64_t)m0->second << 32) | in.literal);
                if (slot == rs.lds_addtid.end()) { ok = false; return true; }
                uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = slot->second;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            if (in.opcode == 0x3d || in.opcode == 0x3e) {
                auto m0 = rs.sreg.find(124);
                if (!allow_wave || m0 == rs.sreg.end()) { ok = false; return true; }
                const uint32_t idx = ds_append_consume_index(
                    b, m0->second, in.literal, in.ds_gds);
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                if (b.is_fragment && in.ds_gds) {
                    rs.vreg[in.dst.value] = b.native_gds_append(
                        idx, rs.exec, in.opcode == 0x3d);
                } else if (b.is_compute) {
                    if (!in.ds_gds) b.declare_lds();
                    rs.vreg[in.dst.value] = b.wave_append(
                        idx, rs.exec, in.opcode == 0x3d, in.ds_gds);
                } else {
                    ok = false; return true;
                }
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            // Compute uses Workgroup LDS. Only a mechanically proven NGG projection may use the
            // private graphics backing store; every other graphics DS shape remains fail-closed.
            if ((!b.is_compute && !b.ngg_private_lds) ||
                (in.opcode != 0x00 && in.opcode != 0x05 && in.opcode != 0x06 &&
                                  in.opcode != 0x07 && in.opcode != 0x08 &&
                                  in.opcode != 0x09 && in.opcode != 0x0a &&
                                  in.opcode != 0x0b && in.opcode != 0x12 &&
                                  in.opcode != 0x13 && in.opcode != 0x20 &&
                                  in.opcode != 0x2d &&
                                  in.opcode != 0x0d && in.opcode != 0x0e &&
                                  in.opcode != 0x36 && in.opcode != 0x37 &&
                                  in.opcode != 0x3d && in.opcode != 0x3e && in.opcode != 0x4d && in.opcode != 0x4e &&
                                  in.opcode != 0x76 && in.opcode != 0x77 &&
                                  in.opcode != 0xde && in.opcode != 0xdf &&
                                  in.opcode != 0xfe && in.opcode != 0xff)) {
                ok = false; return true;
            }
            b.declare_lds();
            if (!b.lds_var) { ok = false; return true; }
            auto vread = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
            const bool atomicize_store = b.atomicized_lds_store_pcs.contains(in.pc);
            if (in.opcode == 0x0e) {                    // ds_write2_b32: two dwords at offset0/offset1
                // AMD RDNA2 ISA 12.13: MEM[ADDR + OFFSET0/1 * 4] = DATA0/1. The packed offsets
                // mirror ds_read2_b32 below; Astro Bot's world-map reduction uses both operations.
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu));
                b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec,
                            atomicize_store);
                // Equal offsets encode one memory access and use DATA0; DATA1 is ignored.
                if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu))
                    b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec,
                                atomicize_store);
                return true;
            }
            if (in.opcode == 0x4e) {                    // ds_write2_b64: two VGPR pairs at offset0/offset1
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
                b.lds_store(idx0, vread(in.src[1].value), rs.exec_narrowed, rs.exec,
                            atomicize_store);
                b.lds_store(b.ibin(Op_IAdd, idx0, b.uconst(1)), vread(in.src[1].value + 1),
                            rs.exec_narrowed, rs.exec, atomicize_store);
                // OFFSET0 == OFFSET1 selects ONE address: the hardware performs a single write and
                // uses DATA0 only (RDNA2 ISA 70648 §10.4.3). Writing DATA1 as well leaves the later
                // store winning, so a subsequent LDS read observes DATA1 where hardware preserves
                // DATA0 -- silent wrong data rather than a fault. The ds_write2_b32 case above
                // already guards this; the 64-bit variant did not (#1473).
                if ((in.literal & 0xFFu) != ((in.literal >> 8) & 0xFFu)) {
                    b.lds_store(idx1, vread(in.src[2].value), rs.exec_narrowed, rs.exec,
                                atomicize_store);
                    b.lds_store(b.ibin(Op_IAdd, idx1, b.uconst(1)), vread(in.src[2].value + 1),
                                rs.exec_narrowed, rs.exec, atomicize_store);
                }
                return true;
            }
            // NOTE: a SECOND ds_write2_b32 (0x0e) block used to sit here, unreachable because the
            // handler above claims 0x0e first and returns. It was a duplicate carrying the
            // PRE-FIX behaviour -- both stores, no equal-offset guard -- so the hazard was never
            // that it ran. It was that adding any condition to the reachable block (a wave-size
            // check, an encoding refinement) would have silently handed 0x0e to a copy with the
            // #1473 defect still in it, and the tests would have kept passing. Deleted (#1473).
            if (in.opcode == 0x77) {                    // ds_read2_b64: two pairs at scaled offsets
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t idx0 = b.ibin(Op_IAdd, base, b.uconst((in.literal & 0xFFu) * 2u));
                const uint32_t idx1 = b.ibin(Op_IAdd, base, b.uconst(((in.literal >> 8) & 0xFFu) * 2u));
                const uint32_t indices[4] = {
                    idx0, b.ibin(Op_IAdd, idx0, b.uconst(1)),
                    idx1, b.ibin(Op_IAdd, idx1, b.uconst(1)),
                };
                for (int k = 0; k < 4; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    rs.vreg[in.dst.value + k] = b.lds_load(
                        indices[k], rs.exec_narrowed, rs.exec, old);
                }
                return true;
            }
            if (in.opcode == 0x37) {                    // ds_read2_b32: two dwords at offset0/offset1
                // AMD RDNA2 ISA 12.13: RETURN_DATA[0/1] = MEM[ADDR + OFFSET0/1 * 4].
                // The two 8-bit offsets share the instruction's 16-bit offset field. Astro's
                // loading compositor uses both adjacent (0,1) and non-zero (16,17) pairs.
                const uint32_t base = b.ibin(Op_ShiftRightLogical, vread(in.src[0].value), b.uconst(2));
                const uint32_t indices[2] = {
                    b.ibin(Op_IAdd, base, b.uconst(in.literal & 0xFFu)),
                    b.ibin(Op_IAdd, base, b.uconst((in.literal >> 8) & 0xFFu)),
                };
                for (int k = 0; k < 2; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    rs.vreg[in.dst.value + k] = b.lds_load(
                        indices[k], rs.exec_narrowed, rs.exec, old);
                }
                return true;
            }
            if (b.ngg_private_lds && in.opcode == 0x36 &&
                in.pc == b.ngg_vertex_index_read_pc && b.ngg_vertex_index_value) {
                // The merged NGG prologue distributes hardware vertex ids between wave lanes through
                // LDS, then the ES consumes that value as its first MUBUF index. In the one-invocation
                // Vulkan shell there is no peer-lane LDS routing: the equivalent value is precisely
                // BuiltIn VertexIndex. This handoff is identified in recompile_vertex from the first
                // MUBUF's vaddr reaching back to this DS read; all other vertex LDS reads remain real.
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.ngg_vertex_index_value;
                predicate_write(b, rs, in.dst.value, old);
                return true;
            }
            uint32_t addr = b.ibin(Op_IAdd, vread(in.src[0].value), b.uconst(in.literal));
            if (in.ds_gds)
                addr = b.ibin(Op_BitwiseAnd, addr, b.uconst(0xFFFFu));
            uint32_t idx  = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
            if (in.ds_gds && in.opcode == 0x0d) {
                b.compute_gds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec);
                return true;
            }
            if (b.ngg_private_lds && (in.opcode == 0x00 || in.opcode == 0x07 || in.opcode == 0x08 ||
                                     in.opcode == kDsOpcodeMinF32 ||
                                     in.opcode == kDsOpcodeMaxF32 ||
                                     in.opcode == 0x20 || in.opcode == 0x2d)) {
                // Cross-lane/atomic NGG LDS effects do not have a proven single-lane reduction yet.
                ok = false; return true;
            }
            if (in.opcode == 0x00) {                     // ds_add_u32: LDS += DATA0, no VGPR return
                b.lds_atomic(Op_AtomicIAdd, idx, vread(in.src[1].value),
                             rs.exec_narrowed, rs.exec);
            } else if (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32) {
                // Exact ordinary GFX10 encoding: ADDR + DATA0 only. DATA1 and VDST are reserved for
                // these non-returning forms (llvm-mc gfx1030 emits both as zero); reject a packet
                // carrying either field instead of interpreting an unmodeled SRC2/return variant.
                if ((in.words[1] & 0xffff0000u) != 0u ||
                    b.compute_pgm_rsrc1 == UINT32_MAX) { ok = false; return true; }
                b.lds_atomic_fminmax(idx, vread(in.src[1].value),
                                     in.opcode == kDsOpcodeMinF32,
                                     rs.exec_narrowed, rs.exec);
            } else if (in.opcode >= 0x05 && in.opcode <= 0x0b) {
                // Non-returning 32-bit LDS atomics map directly to SPIR-V atomics. DS_OR_B32
                // (0x0a) occurs in Plucky's first-gameplay compute path; supporting the adjacent
                // signed/unsigned min/max and bitwise family keeps this architectural rather than
                // making the acceptance rule title-specific.
                const uint32_t atomic_op =
                    in.opcode == 0x05 ? Op_AtomicSMin :
                    in.opcode == 0x06 ? Op_AtomicSMax :
                    in.opcode == 0x07 ? Op_AtomicUMin :
                    in.opcode == 0x08 ? Op_AtomicUMax :
                    in.opcode == 0x09 ? Op_AtomicAnd  :
                    in.opcode == 0x0a ? Op_AtomicOr   : Op_AtomicXor;
                b.lds_atomic(atomic_op, idx, vread(in.src[1].value),
                             rs.exec_narrowed, rs.exec);
            } else if (in.opcode == 0x20) {             // ds_add_rtn_u32: VDST = old LDS; LDS += DATA0
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_atomic_rtn(
                    Op_AtomicIAdd, idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec, old);
                predicate_write(b, rs, in.dst.value, old);
            } else if (in.opcode == 0x2d) {             // ds_wrxchg_rtn_b32: swap DATA0, return old LDS
                const uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_atomic_rtn(
                    Op_AtomicExchange, idx, vread(in.src[1].value),
                    rs.exec_narrowed, rs.exec, old);
                predicate_write(b, rs, in.dst.value, old);
            } else if (in.opcode == 0x0d) {             // ds_write_b32: LDS[idx] = DATA0
                b.lds_store(idx, vread(in.src[1].value), rs.exec_narrowed, rs.exec,
                            atomicize_store);
            } else if (in.opcode == 0x4d || in.opcode == 0xde || in.opcode == 0xdf) {
                // ds_write_b64 / ds_write_b96 / ds_write_b128. The wide forms consume consecutive
                // DATA0 VGPRs and write consecutive LDS dwords from the ordinary byte address.
                const int dwords = in.opcode == 0x4d ? 2 : in.opcode == 0xde ? 3 : 4;
                for (int k = 0; k < dwords; ++k) {
                    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
                    b.lds_store(at, vread(in.src[1].value + k), rs.exec_narrowed, rs.exec,
                                atomicize_store);
                }
            } else if (in.opcode == 0x76 || in.opcode == 0xfe || in.opcode == 0xff) {
                // ds_read_b64 / ds_read_b96 / ds_read_b128. RDNA2 ISA 12.13 opcodes 118/254/255;
                // all return consecutive dwords beginning at the ordinary LDS byte address.
                const int dwords = in.opcode == 0x76 ? 2 : in.opcode == 0xfe ? 3 : 4;
                for (int k = 0; k < dwords; ++k) {
                    const uint32_t old = vreg_old(b, rs, in.dst.value + k);
                    const uint32_t at = k ? b.ibin(Op_IAdd, idx, b.uconst((uint32_t)k)) : idx;
                    rs.vreg[in.dst.value + k] = b.lds_load(
                        at, rs.exec_narrowed, rs.exec, old);
                }
            } else {                                    // ds_read_b32: VDST = LDS[idx]
                uint32_t old = vreg_old(b, rs, in.dst.value);
                rs.vreg[in.dst.value] = b.lds_load(
                    idx, rs.exec_narrowed, rs.exec, old);
            }
            return true;
        }
        case Rdna2Format::VINTRP: {
            // Pixel-shader attribute interpolation. The rasterizer performs the interpolation; v_interp_p1
            // (opcode 0) initializes and is a no-op here, while v_interp_p2 (1) and v_interp_mov (2)
            // deliver the interpolated attribute component from the matching Input varying. Fragment-only.
            if (!b.is_fragment) { ok = false; return true; }
            if (in.opcode == 0) return true;   // p1: no-op (p2/mov produce the value)
            uint32_t old = vreg_old(b, rs, in.dst.value);
            if (in.opcode == 2) {
                const uint32_t parameter = b.interp_parameter(
                    in.vintrp_attr, in.vintrp_chan, in.src[0].value);
                if (!parameter) { ok = false; return true; }
                rs.vreg[in.dst.value] = parameter;
            } else {
                rs.vreg[in.dst.value] = b.interp_read(in.vintrp_attr, in.vintrp_chan);
            }
            predicate_write(b, rs, in.dst.value, old);
            return true;
        }
        default: return false;   // not a VALU format
    }
}

namespace {


} // namespace

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.

} // namespace

} // namespace prosper::gpu
