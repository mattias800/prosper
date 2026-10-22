#pragma once
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_uniform_branch.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_divergent_detector.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_loop_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_counted_loop.hpp" // IWYU pragma: export

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_cfg_support.hpp — helpers shared by rdna2_emit_cfg.cpp and rdna2_to_spirv.cpp.
// INTERNAL to src/gpu/recompiler/.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
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
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp" // IWYU pragma: export

namespace prosper::gpu {

uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);
// The VALU sources that are lane masks: SRC2 of v_cndmask_b32_e64 and the VOP3B carry-in forms.
bool valu_lane_mask_source(const Rdna2Inst& in, uint32_t source);
// scalar_alu_source_words for a program of a known wave width: a Wave32 lane mask is ONE SGPR,
// where the width-free form charges every lane mask the Wave64 pair (#5202). Any other wave_size
// value (64, or unknown) keeps the pair.
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
// The words the source MARKS charge (scalar_source_marks and the dispatcher's
// spilled_vector_source_marks): the wave-aware width, except that a Wave32 VCC_LO lane mask still
// charges VCC_HI, the only witness the marks have of an unprovided entry VCC.
uint32_t scalar_source_mark_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);

// What a surviving read of R is allowed to be.
//
// AnyRead     -- R must not be read at all before it is redefined. The original and default proof.
// MaskDomainOnly -- R may be read as 32-bit scalar DATA, but not by a lane-mask consumer. Only
//                meaningful for a VCC half, and only sound because prosper models a VCC half written
//                by a B32 scalar op as scalar data with its own storage: a 32-bit read of that half
//                reproduces the written dword exactly, while a MASK read would need the per-lane
//                predicate the scalar model deliberately does not reconstruct.
//
// The distinction is 32-bit-ness, not explicitness, and that is the whole subtlety: a VOP3
// `v_cndmask_b32 v0, v1, v2, vcc` names VCC explicitly and still reads the mask domain. Every
// mask consumer reads the PAIR, so requiring `scalar_alu_source_words(...) == 1` excludes all of
// them -- VOP3 cndmask, VOP3B carry-in, `s_mov_b64 exec, vcc`, and B64 mask logic alike -- without
// enumerating them. The e32 implicit readers and the vccz/vccnz branches are rejected earlier and
// unconditionally, so they are excluded on both paths.
enum class ScalarMergeProof { AnyRead, MaskDomainOnly };

// Why the proof failed, for the reject message. A liveness proof that reports only "failed" makes
// every widening of it a guess: the first attempt at MaskDomainOnly did not accept the GTA V kernel
// it was written for, and without the blocking pc there was no way to tell whether the walk stopped
// at a mask read, a data read, or an unmodelled opcode -- three different follow-ups.
struct ScalarMergeBlocker {
    uint32_t pc = UINT32_MAX;
    const char* kind = "none";
};

struct ScalarMergeMaskWriteWidths {
    bool carry_pair = true;
    bool implicit_vcc_pair = true;
    bool explicit_vcc_pair = true;
};

inline bool
sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R,
                   ScalarMergeProof proof = ScalarMergeProof::AnyRead,
                   ScalarMergeBlocker* blocker = nullptr,
                   // Explicit compare HIGH-word kills require the caller's pair-width
                   // proof. A Wave32 compare always replaces LOW but can leave HIGH live.
                   bool vopc_sgpr_pair_kills = false,
                   // Match physical mask writes to the caller's actual emitter mode.
                   ScalarMergeMaskWriteWidths mask_widths = {}) {
    const auto block = [&](const Rdna2Inst& at, const char* kind) {
        if (blocker && blocker->pc == UINT32_MAX) {
            blocker->pc = at.pc;
            blocker->kind = kind;
        }
        return false;
    };
    // MaskDomainOnly is a VCC-half question by construction: an ordinary SGPR has no mask domain,
    // so the relaxation would silently become "reads are fine", which is not a proof of anything.
    if (proof == ScalarMergeProof::MaskDomainOnly && R != 106 && R != 107) return false;
    // M0 has readers that never name it: the relative moves, v_movrels_b32, the LDS and
    // append/consume forms, s_sendmsg. The operand scan below sees none of them, and a walk
    // that reached s_endpgm used to answer "dead" for it.
    if (R == 124) return false;
    const bool data_read_ok = proof == ScalarMergeProof::MaskDomainOnly;
    // Prove liveness over the actual scalar CFG, not just lexical order. UE4 commonly places another
    // forward EXECZ after an if/merge and only then overwrites the scratch SGPR/VCC value. A linear
    // scan used to reject that safe shape merely because it encountered the second branch. Explore
    // BOTH successors of every recognized conditional branch; a read on either path fails the proof,
    // while a redefinition/end terminates only that path. The visited set also bounds back-edges.
    std::unordered_map<uint32_t, size_t> pc_to_index;
    for (size_t i = 0; i < ins.size(); ++i) pc_to_index.emplace(ins[i].pc, i);
    const auto start = pc_to_index.find(target);
    if (start == pc_to_index.end()) return false;
    std::vector<size_t> pending{start->second};
    std::vector<uint8_t> visited(ins.size(), 0);
    while (!pending.empty()) {
        size_t index = pending.back();
        pending.pop_back();
        if (index >= ins.size() || visited[index]) continue;
        visited[index] = 1;
        const auto& in = ins[index];
        if (in.is_end) continue;   // this path ends without a read
        // VCC (R = 106/107) has IMPLICIT readers the operand scan can't see: the e32 VOP2 carry /
        // cndmask ops (0x01, 0x28-0x2A) and the vccz/vccnz branches. Flag those as reads up front so
        // the redef checks below are sound for VCC too (which the s_mov carve-out excludes, but the
        // SMEM carve-out needs — compilers love `s_buffer_load_dword vcc_lo, …` scratch loads).
        if (R == 106 || R == 107) {
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2A)))
                return block(in, "vop2-implicit-vcc");
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
                return block(in, "vccz-branch");
            // NOTE: v_div_fmas_f32/f64 (VOP3 0x16f/0x170) also read VCC implicitly and are NOT
            // listed above. Neither is emitted by the recompiler today (an unsupported opcode
            // fails the whole recompile), so no liveness question reaches them; add the reader
            // here before one is lowered.
        }
        switch (in.fmt) {
            // Most SOPK instructions remain fail-closed: s_addk/s_mulk/s_cmovk/s_cmpk read or
            // read-modify-write their encoded "dst" while decode exposes no source. MOVK_I32 is a
            // pure one-word definition and WAITCNT_VSCNT is register-transparent; GTA's low-only
            // VCC lifetime crosses both before the actual pair replacement.
            case Rdna2Format::SOPK:
                if (in.opcode == kSopkOpcodeWaitcntVscnt && in.dst.kind == OperandKind::SGPR &&
                    in.dst.value == 125)
                    break;
                if (in.opcode == kSopkOpcodeMovkI32 && in.dst.kind == OperandKind::SGPR) {
                    if (in.dst.value == R) continue;
                    break;
                }
                // Under MaskDomainOnly the whole read-modify-write SOPK family is admissible. Every
                // SOPK operates on ONE dword: s_addk_i32/s_mulk_i32 read and rewrite their encoded
                // destination, s_cmpk_* read it and write SCC, s_cmovk_i32 reads it and may rewrite
                // it. All three are 32-bit scalar-DATA touches of the half; none can consume it as a
                // 64-bit lane mask, which is the only thing this proof needs to exclude.
                //
                // Deliberately does NOT claim the half died, even for the forms that always write
                // it: continuing the walk is the conservative direction, and a later genuine mask
                // read still fails the proof. Being wrong about the kill would be unsound; being
                // silent about it only costs acceptances.
                if (data_read_ok && in.dst.kind == OperandKind::SGPR) break;
                return block(in, "unmodelled-sopk");
            case Rdna2Format::SOPP:
                // Hint/sync SOPPs read and write nothing — scan through them (s_waitcnt is ubiquitous
                // after the merge). A barrier is also scalar-register-transparent; it changes only
                // workgroup execution order. For branches, enqueue every possible successor. Unknown
                // scalar control flow remains fail-closed.
                if (sopp_is_noop(in) || in.opcode == 0x0a) break;
                if (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09)) {
                    const int64_t branch_pc =
                        static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
                    const auto branch = branch_pc >= 0
                                            ? pc_to_index.find(static_cast<uint32_t>(branch_pc))
                                            : pc_to_index.end();
                    if (branch == pc_to_index.end()) return false;
                    pending.push_back(branch->second);
                    if (in.opcode != 0x02) {
                        if (index + 1 >= ins.size()) return false;
                        pending.push_back(index + 1);
                    }
                    continue;
                }
                return block(in, "unmodelled-sopp");
            case Rdna2Format::SMEM: {
                // A load reads SBASE (src[0]: a pair, or the buffer forms' V# quad) and SOFFSET
                // (src[1]) and redefines N consecutive SDATA SGPRs; the effects table gives both
                // widths. Anything not a load -> bail. (This read only SBASE's first two words, so
                // the last two words of a buffer load's V# were not seen as reads.)
                const uint32_t n = rdna2_smem_load_dwords(in.opcode);
                if (!n) return block(in, "unmodelled-smem");
                if (R >= in.src[0].value &&
                    R < in.src[0].value + static_cast<int>(rdna2_smem_address_dwords(in.opcode)))
                    return block(in, "smem-sbase");   // SBASE read
                // SOFFSET is one dword and is consumed as an address, never as a lane mask.
                if ((in.src[1].kind == OperandKind::SGPR ||
                     in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value == R && !data_read_ok)
                    return block(in, "smem-soffset");   // SOFFSET read
                if (R >= in.dst.value && R < in.dst.value + (int)n)
                    continue;   // redefined: this path is dead
                break;
            }
            case Rdna2Format::SOP1:
            case Rdna2Format::SOP2:
            case Rdna2Format::SOPC:
            case Rdna2Format::VOP1:
            case Rdna2Format::VOP2:
            case Rdna2Format::VOP3:
            case Rdna2Format::VOPC:
            case Rdna2Format::
                DS:   // DS operands are VGPR/M0 only; it cannot read an ordinary SGPR/VCC
            case Rdna2Format::EXP: {   // EXP data sources are all VGPRs — it can never read an SGPR
                // Conditional moves, SOPK immediate operations, and SOP1 bitsets may read their
                // encoded destination before replacing it. Their decoded source list does not
                // contain that old value, so consult the shared implicit-read inventory first.
                const uint32_t implicit_read_width = scalar_implicit_destination_read_width(in);
                if (implicit_read_width && R >= in.dst.value &&
                    R < in.dst.value + static_cast<int>(implicit_read_width) &&
                    !(data_read_ok && implicit_read_width == 1))
                    return block(in, "implicit-dst-rmw");
                for (int k = 0; k < in.n_src; k++) {
                    if (in.src[k].kind != OperandKind::SGPR &&
                        in.src[k].kind != OperandKind::Special)
                        continue;
                    // Operand does not carry B32/B64 width. Use the shared opcode inventory so a
                    // physical low-half read (notably GTA's CSELECT_B32 -> V_ADD3 chain) cannot
                    // falsely keep the sibling live, while unlisted VOP3 sources remain pairs.
                    const uint32_t words = scalar_alu_source_words(in, k);
                    if (words == UINT32_MAX) continue;
                    if (R >= in.src[k].value && R < in.src[k].value + static_cast<int>(words)) {
                        // A one-dword source is a scalar-data read. Anything wider reads the pair,
                        // which is exactly the set of lane-mask consumers.
                        if (data_read_ok && words == 1) continue;
                        return block(in, words == 1 ? "source-dword" : "source-pair");
                    }
                }
                // Check all sources before the kill: carry-in can read the old destination.
                const bool carry_low_only =
                    !mask_widths.carry_pair &&
                    (vop3b_fresh_carry_output(in) ||
                     (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 && in.opcode <= 0x12a));
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || (!carry_low_only && in.sdst.value + 1 == R)))
                    continue;
                // A non-X VOPC e32 compare writes the whole VCC pair — kills both halves. V_CMPX
                // writes EXEC ONLY on RDNA2 (ISA 12.9: "EXEC[threadId] = ..."; no VCC/SDST dest),
                // so it must NOT count as a VCC kill — miscounting it let safe_execz linearize a
                // block whose vcc scratch-write hardware would have skipped.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.value == 106 || in.dst.value == 107)) {
                    const bool pair = in.dst.kind == OperandKind::SGPR
                                          ? mask_widths.explicit_vcc_pair
                                          : mask_widths.implicit_vcc_pair;
                    if (pair ? (R == 106 || R == 107) : R == in.dst.value) continue;
                }
                // CONFIDENCE: HIGH. RDNA2 ISA 3.9/6.8: a non-CMPX explicit compare
                // replaces its complete low mask word, writing zero for EXEC-off lanes.
                // SDWAB has no partial destination selection. Check sources first: a
                // compare may read its own old scalar destination. Keep HIGH conservative.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
                    R == in.dst.value)
                    continue;
                if (vopc_sgpr_pair_kills && in.fmt == Rdna2Format::VOPC &&
                    !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::SGPR &&
                    in.dst.value >= 0 && in.dst.value <= 104 &&
                    (R == in.dst.value || R == in.dst.value + 1))
                    continue;
                // A scalar redefinition kills every word covered by its opcode's write width. Keep
                // this after the explicit/implicit read checks above: B64 read-modify-write forms
                // must observe the old pair before replacing it. EXEC/M0 (124/126/127) still cannot
                // be proven dead (implicit reads everywhere), and cmpx writes EXEC rather than its
                // decoded SGPR destination.
                const uint32_t dst_width = scalar_write_width(in);
                if (dst_width && in.dst.kind == OperandKind::SGPR && R <= 107 &&
                    R >= in.dst.value && R < in.dst.value + static_cast<int>(dst_width) &&
                    !(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)))
                    continue;
                break;
            }
            case Rdna2Format::MIMG:
                // Image packets fully decode both scalar descriptor operands: the eight-dword T#/U#
                // at src[1] and the four-dword S# at src[2]. Some opcodes do not consume the sampler,
                // but accounting for both ranges is deliberately conservative and lets the proof scan
                // through every MIMG without needing a second opcode inventory.
                if (in.src[1].kind != OperandKind::SGPR || in.src[2].kind != OperandKind::SGPR)
                    return false;
                if (R >= in.src[1].value && R < in.src[1].value + 8) return false;
                if (R >= in.src[2].value && R < in.src[2].value + 4) return false;
                break;
            case Rdna2Format::MUBUF:
            case Rdna2Format::MTBUF:
                // Buffer packets fully decode their scalar inputs: SRSRC is the four-dword
                // descriptor at src[1], and SOFFSET is the single src[2] operand.  Treating every
                // buffer access as an unknowable SGPR read made an otherwise-dead VCC_HI scratch
                // word look live across Astro Bot's world-map setup pass.  Keep the proof exact by
                // checking the complete descriptor range and explicit offset before scanning on.
                if (in.src[1].kind != OperandKind::SGPR) return false;
                if (R >= in.src[1].value && R < in.src[1].value + 4) return false;
                if ((in.src[2].kind == OperandKind::SGPR ||
                     in.src[2].kind == OperandKind::Special) &&
                    in.src[2].value == R)
                    return false;
                break;
            case Rdna2Format::FLAT:
                // FLAT/GLOBAL/SCRATCH packets keep every data/address operand in VGPRs except
                // SADDR, the optional scalar address pair at src[1]. NULL/off (125) reads no
                // scalar register. An LDS transfer also observes M0 implicitly, so keep that
                // uncommon form fail-closed. Account for the complete SADDR pair conservatively;
                // this liveness proof does not imply that the memory operation is translatable.
                if (in.flat_lds) return false;
                if ((in.src[1].kind == OperandKind::SGPR ||
                     in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value != 125 && (in.src[1].value == R || in.src[1].value + 1 == R))
                    return false;
                break;
            case Rdna2Format::VINTRP:
                // CONFIDENCE: HIGH -- AMD 70648 10.4.2/Table 53: P1/P2/MOV read VGPRs
                // and implicit M0, never an ordinary scalar word. EXEC still predicates them;
                // M0 is excluded above. Reserved opcodes and inconsistent decoded forms prove
                // nothing. This only extends the ordinary scalar/VCC-word liveness question.
                if (R < 0 || R > 107 || in.opcode > 2 || in.len_dwords != 1 || in.n_src != 1 ||
                    in.dst.kind != OperandKind::VGPR || in.dst.value < 0 || in.dst.value > 255 ||
                    in.src[0].kind != OperandKind::VGPR || in.src[0].value < 0 ||
                    in.src[0].value > 255 || in.vintrp_attr > 32 || in.vintrp_chan > 3 ||
                    (in.opcode == 2 && in.src[0].value > 2) || in.has_modifier || in.has_sdwa ||
                    in.has_dpp || in.has_literal || in.sdst.kind != OperandKind::None ||
                    in.words[0] !=
                        (0xc8000000u | (in.opcode << 16u) |
                         (static_cast<uint32_t>(in.dst.value) << 18u) | (in.vintrp_attr << 10u) |
                         (in.vintrp_chan << 8u) | static_cast<uint32_t>(in.src[0].value)))
                    return block(in, "unmodelled-vintrp");
                break;
            default: return false;   // memory/branch/unknown: can't bound reads of R -> assume live
        }
        if (index + 1 < ins.size()) pending.push_back(index + 1);
    }
    return true;   // every reachable path hit a redefinition/end without a read
}

inline std::unordered_set<uint32_t>
proven_cselect_b64_low_only_pcs(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        // GTA V 0x413cf9a00 pc60 selects two scalar-data pairs into VCC, then reads only
        // VCC_LO as one scalar dword. Its second alternative currently has only the low word
        // available (the zero-record S_BUFFER load at pc16 writes s16, not s17). Admit that
        // incomplete-source form only for the exact ordinary-SGPR-to-VCC packet shape and only
        // when the shared CFG liveness walk proves VCC_HI dead from the following instruction.
        // An implicit VCC consumer counts as a high-half read in that walk, so no per-lane mask
        // can observe the unavailable selected word.
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0bu && in.dst.kind == OperandKind::SGPR &&
            in.dst.value == 106 && in.src[0].kind == OperandKind::SGPR &&
            in.src[1].kind == OperandKind::SGPR &&
            sgpr_dead_at_merge(ins, in.pc + in.len_dwords, 107))
            result.insert(in.pc);
    }
    return result;
}

// A one-dword scalar write to Wave64 VCC_LO normally has to preserve VCC_HI and rebuild the
// architectural per-lane predicate from both physical words. GTA V also recycles only the low word
// as scalar scratch, then replaces the complete pair before anything can observe the untouched high
// word. The scalar-data-vs-mask decision remains path-local in emit_alu; this predicate only names
// B32 packets for which a dead-high proof can make that scalar path discard the old predicate.
inline bool is_wave64_vcc_lo_scalar_b32_candidate(const Rdna2Inst& in) {
    const auto scalar_constant = [](const Operand& operand) {
        return operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
               operand.kind == OperandKind::Literal;
    };
    const bool scalar_cselect_hi = in.fmt == Rdna2Format::SOP2 &&
                                   in.opcode == kSop2OpcodeCselectB32 &&
                                   in.dst.kind == OperandKind::SGPR && in.dst.value == 107 &&
                                   scalar_constant(in.src[0]) && scalar_constant(in.src[1]);
    return is_wave64_vcc_lo_scalar_cselect(in) || scalar_cselect_hi ||
           (in.fmt == Rdna2Format::SOP2 && sop2_is_b32_logical(in.opcode) &&
            in.dst.kind == OperandKind::SGPR && (in.dst.value == 106 || in.dst.value == 107));
}

inline std::unordered_set<uint32_t>
proven_wave64_vcc_b32_low_only_pcs(const std::vector<Rdna2Inst>& ins, bool include_logical = true,
                                   bool include_high_cselect = false) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        const bool candidate =
            (include_high_cselect && is_wave64_vcc_lo_cselect_reading_vcc_hi(in)) ||
            (include_logical ? is_wave64_vcc_lo_scalar_b32_candidate(in)
                             : is_wave64_vcc_lo_scalar_cselect(in));
        // Scalar scratch may use either physical VCC word and combine the two as ordinary dwords.
        // The sibling may be read as scalar data, but it must be dead before any pair/mask-domain
        // observation. MaskDomainOnly is exactly that whole-CFG proof.
        const int sibling = in.dst.value == 107 ? 106 : 107;
        if (candidate && sgpr_dead_at_merge(ins, in.pc + in.len_dwords, sibling,
                                            ScalarMergeProof::MaskDomainOnly))
            result.insert(in.pc);
    }
    return result;
}
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
// Same proof, applied to a scalar branch's SCC instead of a VCC lane mask (#3713).
bool scc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);

// The header EXEC is a subset of its preheader only when every possible iteration preserves or
// narrows EXEC. This code-only certificate does not authenticate any carried register's data.
bool structured_loop_only_narrows_exec(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
bool narrowing_loop_exec_admission(const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                                   const RegState& entry);
// A wave-scalar word the loop [header, back edge] may write that is live at its exit: an SGPR,
// VCC or M0 word (a mask pair counts, as scalar data), or kLoopLiveScc. -1 when there is none.
// The narrowing certificate requires -1; rdna2_loop_wave_scalar_route.cpp routes the others.
constexpr int kLoopLiveScc = 253;
int loop_live_wave_scalar(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
// The launch and stream a fresh exact-dispatcher attempt of this program would compile: an
// authentic native Wave64 compute entry whose original words are `code`.
bool native_exact_retry_authentic(const SpirvCompute& b, const uint32_t* code, size_t dwords);
// A refused native store-loop header may select the existing exact dispatcher. This proves
// only eligibility to try a fresh builder, never DATA, scalar, resource or effect admission.
bool closed_compute_store_loop_data_retry(const SpirvCompute& b, const RegState& rs,
                                          const DivLoop& loop, const Rdna2Inst& header,
                                          uint32_t preheader_exec, const uint32_t* code,
                                          size_t dwords);
uint32_t make_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                      std::size_t& patch, bool admitted);
// The header EXEC phi is a subset of the preheader EXEC when every back edge carries a subset of
// it (induction over iterations). Two certificates grant that fact: the code-only compute one
// (narrowing_loop_exec_admission), and a checked one for fragment loops whose EXEC writes are
// all narrowing forms: assumed at the header, then verified at the back edge from the
// EXEC subset facts the body actually recorded. A failed check refuses the loop (#5039).
// Implemented in rdna2_loop_exec_subset.cpp.
class LoopExecSubset {
public:
    LoopExecSubset(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                   const RegState& entry);
    bool admitted() const { return certified_ || checked_; }
    void open(const RegState& preheader) {
        preheader_exec_ = preheader.exec;
        preheader_full_ = !preheader.exec_narrowed;
    }
    // At the back edge, with `rs` holding the body's final EXEC.
    bool holds(SpirvCompute& b, const RegState& rs) const;

private:
    bool certified_ = false;
    bool checked_ = false;
    uint32_t preheader_exec_ = 0;
    bool preheader_full_ = false;
};
// The EXEC phi at a one-armed structured if's merge. When the taken arm only narrowed the skipped
// edge's EXEC, the merge is a subset of it too (rdna2_loop_exec_subset.cpp).
uint32_t merge_exec_arm(SpirvCompute& b, RegState& rs, uint32_t skipped_exec,
                        uint32_t skipped_block, uint32_t taken_exec, uint32_t taken_block);
template <class PhiRecords>
void open_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                  PhiRecords& phis, LoopExecSubset& proof) {
    std::size_t patch;
    proof.open(rs);
    const uint32_t phi = make_narrowing_loop_exec_phi(b, rs, preheader, patch, proof.admitted());
    phis.push_back({0, 4, phi, patch});
}

// A scalar VCCZ branch tests whether ANY active lane set VCC, not this lane's bit. Stages without
// fragment's forced wave64 vote may lower it as structured control only when the VCC producer is
// provably uniform across the wave. Dead Cells' light loops use the narrow compiler shape
//   v_cvt_i32_f32 vBOUND, sBOUND; ...; v_cmp_* vcc, sCOUNTER, vBOUND; s_cbranch_vccz EXIT
// so every active lane makes the same comparison. Unity also derives the bound through a short
// lane-local VALU chain fed only by scalar sources. Trace that chain backwards: uniform inputs stay
// uniform through ordinary VOP1/VOP2 arithmetic, while lane permutations, VCC-dependent selects,
// carry chains, interpolants, and unresolved VGPR inputs remain rejected.
// True when `in` provably cannot overwrite VCC — used by vcc_exit_is_wave_uniform's compare-finding
// walk (#590) to look past instructions scheduled between the VOPC and its consuming branch (real
// UE4 compute kernels hoist ~15 unrelated ALU ops in between). Anything not provably safe stops the
// walk conservatively: VOPC itself (the implicit dst), the VOP2 carry-chain ops (v_*_co_ci_u32,
// 0x28-0x2a), VOP3 windows that may carry a scalar dest (VOPC-as-VOP3 sdst at 0x000-0x13f, VOP3B
// carry-out / v_readlane at 0x300+ — the decoder does not expose their sdst), and any scalar or
// SMEM write whose dest range could cover SGPR 106/107 (the shaders use s106/s107 as scratch).
inline bool cannot_write_vcc(const Rdna2Inst& in) {
    auto sgpr_dst_misses_vcc = [&](uint32_t n_dwords) {
        if (in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) return true;
        const int lo = in.dst.value, hi = in.dst.value + (int)n_dwords - 1;
        return hi < 106 || lo > 107;
    };
    // The two lane reads write an SGPR although the decoder files their destination as a VGPR
    // operand (scalar_write_width knows them by opcode). Judge them by register number, or a
    // `v_readfirstlane_b32 vcc_lo` / `v_readlane_b32 vcc_lo` would pass as VCC-preserving (#4425).
    const auto lane_read_misses_vcc = [&] { return in.dst.value != 106 && in.dst.value != 107; };
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            if (in.opcode == 0x02) return lane_read_misses_vcc();   // v_readfirstlane_b32
            return true;
        case Rdna2Format::VINTRP:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::DS:
        case Rdna2Format::FLAT:
        case Rdna2Format::EXP:   // VGPR/memory dsts only
        case Rdna2Format::SOPP:   // branches/hints/waitcnt: no dst
        case Rdna2Format::SOPC: return true;   // writes SCC only
        case Rdna2Format::VOP2: return in.opcode < 0x28 || in.opcode > 0x2a;
        case Rdna2Format::VOP3P: return true;   // packed VALU: VGPR dst, no carry
        case Rdna2Format::VOP3:
            // gfx10 v_writelane_b32 writes only its VGPR lane slot. v_readlane_b32 writes the
            // scalar destination encoded in the decoder's dst field, so it is VCC-preserving as
            // long as that destination does not overlap s106:s107. UE4 schedules long scalar-spill
            // sequences of these two ops between a uniform VOPC and its s_cbranch_vccz.
            if (in.opcode == 0x361) return true;
            if (in.opcode == 0x360) return lane_read_misses_vcc();   // v_readlane_b32
            // Decoder storage has an SDST-shaped field for every VOP3 packet, but only VOP3B
            // operations architecturally write it. Treat ordinary VOP3A VALU as VCC-transparent;
            // otherwise a MAC between a uniform compare and VCCZ creates a false Wave64 vote.
            if (vop3_writes_mask_sdst(in)) return in.sdst.value != 106 && in.sdst.value != 107;
            return in.dst.kind == OperandKind::VGPR;
        case Rdna2Format::SOP1:
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPK: return sgpr_dst_misses_vcc(2);   // conservative 64-bit pair
        case Rdna2Format::SMEM: {
            // What SDATA receives, from the effects table (this used to read the width from
            // opcode & 7, which charged s_memtime sixteen SGPRs and every opcode 5-7 mod 8 none).
            if (!rdna2_instruction_effects(in).known()) return false;
            const uint32_t n = rdna2_smem_destination_dwords(in);
            return !n || sgpr_dst_misses_vcc(n);
        }
        default: return false;   // VOPC / unknown: stops the walk
    }
}

inline bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    const Rdna2Inst* compare = nullptr;
    for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
        if (it->pc >= branch_pc) continue;
        if (sopp_is_noop(*it)) continue;
        if (it->fmt == Rdna2Format::VOPC) {
            compare = &*it;
            break;
        }
        if (cannot_write_vcc(*it)) continue;   // (#590) look past provably-VCC-preserving instrs —
        break;   // VCC's value is frozen at the compare; only a real
        // rewrite between compare and branch invalidates it
    }
    if (!compare || compare->fmt != Rdna2Format::VOPC || vopc_is_cmpx(compare->opcode))
        return false;

    std::function<bool(const Operand&, uint32_t)> uniform_operand_at;
    uniform_operand_at = [&](const Operand& operand, uint32_t use_pc) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) {
            // "Lives in a scalar register" is NOT the same as "is wave-uniform" (#3607).
            // `v_readfirstlane_b32` writes an SGPR while producing THIS lane's value under prosper's
            // per-lane model -- its lowering is annotated SPECULATIVE(confidence: med) in
            // rdna2_emit_alu.cpp for exactly that reason -- so accepting the register unconditionally
            // lets a lane-varying value satisfy a wave-uniformity proof.
            //
            // `Special` is in this branch and NOT with the constants below, which is the half a
            // first version of this guard got wrong. `decode_src_field` maps 0..105 to `SGPR` but
            // sends VCC_LO/HI (106/107), EXEC, M0 and ttmp to `Special` -- so screening `SGPR` alone
            // left `v_readfirstlane_b32 s106, v1` writing VCC_LO and then satisfying the proof
            // through the very branch that used to assert "scalar-register sources (including VCC
            // halves) broadcast one wave value". This file's own comments note the shaders use
            // s106/s107 as scratch, so that is a reachable shape, and #3596's emitter-side sibling
            // already screens `Special` for exactly this reason. Matching on `value` works uniformly
            // across both kinds because the destination carries the raw register number.
            //
            // Refuse when any earlier instruction read this register out of a lane that way. Two
            // notes on the shape, both load-bearing:
            //
            //  - The match is on `dst.value`, NOT on `dst.kind`. The decoder builds every VOP1 dst as
            //    `vgpr(w >> 17)` (rdna2_decode.cpp:269), so this instruction's dst reports kind VGPR
            //    while its VALUE is the SGPR number -- which is what `rdna2_vgpr_write_count` returning
            //    0 here and the emitter's `rs.sreg[in.dst.value]` both rely on.
            //  - It does NOT stop at a later redefinition, so an SGPR that readfirstlane wrote and a
            //    subsequent `s_mov_b32` made uniform again keeps being refused. That is deliberate:
            //    the only way to stop at a redefinition is a "this instruction writes SGPR n"
            //    predicate, which does not exist here and would have to enumerate every SGPR-writing
            //    format by hand. A format missed by such a predicate would miss a *readfirstlane*
            //    write and return true -- the unsafe direction. Over-refusal costs an optimisation and
            //    cannot produce a wrong answer, because every caller treats false as "not proven
            //    uniform" and falls back to the general path.
            //
            // Deliberately narrow otherwise: anything else keeps the previous answer. The wider
            // question -- whether ANY scalar reached through arbitrary ALU is uniform -- is #3606.
            for (const auto& prior : ins)
                if (prior.pc < use_pc && prior.fmt == Rdna2Format::VOP1 && prior.opcode == 0x02u &&
                    prior.dst.value == operand.value)
                    return false;
            return true;
        }
        if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true;   // a constant is uniform by construction and nothing can write it
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= use_pc) continue;
            if (operand.value == rdna2_tfe_status_vgpr(*it)) return false;
            const uint32_t writes = rdna2_vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes)
                continue;
            if (it->dst.value != operand.value || it->has_dpp) return false;
            // The write makes every currently active lane uniform. Keep the original conservative
            // EXEC rule: any later mask change could expose a lane that retained an older value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    rdna2_instruction_may_change_exec(between))
                    return false;

            bool pure_lane_alu = false;
            if (it->fmt == Rdna2Format::VOP1) {
                pure_lane_alu = it->n_src == 1;
            } else if (it->fmt == Rdna2Format::VOP2) {
                // v_cndmask consumes VCC; the carry family consumes/produces VCC. Neither preserves
                // scalar-source uniformity as a simple lane-local expression.
                pure_lane_alu = it->n_src == 2 && it->opcode != 0x01 &&
                                (it->opcode < 0x28 || it->opcode > 0x2a);
            }
            if (!pure_lane_alu) return false;
            for (uint32_t i = 0; i < it->n_src; ++i)
                if (!uniform_operand_at(it->src[i], it->pc)) return false;
            return true;
        }
        return false;
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand_at(compare->src[i], compare->pc)) return false;
    return compare->n_src != 0;
}
// Whether `in` is an s_movrels_b32 the lowering can emit (rdna2_movrels.cpp). What it writes is
// then a scalar word: the select over what each candidate holds. A candidate in another domain (a
// saved mask half, an entry-M0 token) refuses the whole instruction. A candidate with no tracked
// value does NOT refuse; it contributes the zero placeholder every scalar read uses, so the
// result is exactly as good as the candidate M0 selects, and wrong when that one was untracked.
//
// The dispatcher's scalar-word transfer asks this INSTEAD of charging the candidates, and the
// reason is what failing that transfer means. It is not a refusal: the destination is dropped
// from the words a case boundary reloads, and reads the zero placeholder afterwards. So dropping
// can only turn a right result into zero, never a wrong one into a refusal. Charged the base
// alone, a table whose first register was never written lost its result at the next join.
// Charged the whole range s[base..105], every result did, since almost no program defines all of
// those words. scalar_alu_source_words describes the read range for refusing proofs; this is
// for the one that drops.
inline bool s_movrels_b32_result_is_scalar_data(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovrelsB32 &&
           in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
           in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 && in.src[0].value <= 105;
}
// Scalar instructions whose encoded destination is also an implicit source. These reads must be
// observed before writer transfer functions invalidate the old lifetime. Every SOPK form whose SDST
// field is READ belongs here, not merely the arithmetic band: none of them decodes a src operand at
// all, so a caller scanning `n_src` sees nothing for any of them. SOPK has no decoded src
// operands at all; conditional moves preserve the old destination on one SCC outcome, comparisons
// consume SDST without writing it, and ADDK/MULK are ordinary read-modify-write operations. SOP1
// conditional moves and bitset operations have an explicit source too, but it is the replacement
// value/bit index rather than the old destination.
inline uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::SGPR) return 0;
    if (in.fmt == Rdna2Format::SOPK) {
        if (in.opcode == 0x02 ||   // s_cmovk_i32
            (in.opcode >= 0x03 && in.opcode <= 0x10) ||   // s_cmpk_*, s_addk, s_mulk
            // s_setreg_b32: the encoded SDST field is the SOURCE SGPR, which rdna2_emit_alu.cpp's
            // own handler says in as many words. It reads the register and writes none.
            in.opcode == 0x13 ||
            // s_subvector_loop_begin/end read SDST as the loop counter.
            in.opcode == 0x1b || in.opcode == 0x1c)
            return 1;
        return 0;
    }
    if (in.fmt != Rdna2Format::SOP1) return 0;
    switch (in.opcode) {
        case kSop1OpcodeCmovB32: return 1;
        case kSop1OpcodeCmovB64: return 2;
        case kSop1OpcodeBitset0B32: return 1;
        case 0x1c: return 2;   // s_bitset0_b64
        case kSop1OpcodeBitset1B32: return 1;
        case 0x1e: return 2;   // s_bitset1_b64
        default: return 0;
    }
}

// The input marks of one instruction (#4706; RegState::sreg_merge_placeholder). Defined here, not
// in rdna2_to_spirv_internal.hpp, because it reads the two width inventories above; every caller
// of snapshot_saved_b64_masks() includes this header.
inline ScalarSourceMarks scalar_source_marks(const RegState& rs, const Rdna2Inst& in) {
    ScalarSourceMarks marks = vector_source_marks(rs, in);
    const auto word = [&](int r) {
        const bool fabricated = sreg_word_may_be_fabricated(rs, r);
        marks.placeholder = marks.placeholder || fabricated;
        // A scalar word is unconfined: it may be absent in every lane, so no EXEC confinement survives it.
        // Nor is it a never-loaded VGPR: scalar data stays hard under #5174 and #5185. A loop
        // assumption's word is not hard: like a soft vector word it adds only its roots, which an
        // arbitrary result keeps as obligations (#5193 companion rule).
        const auto blanket = rs.sreg_loop_blanket.find(r);
        if (fabricated) marks.outside_exec = 0;
        if (fabricated && blanket == rs.sreg_loop_blanket.end()) marks.arbitrary = false;
        if (fabricated && blanket == rs.sreg_loop_blanket.end())
            marks.hard = true;
        else if (fabricated)
            marks.blanket_roots.insert(blanket->second.begin(), blanket->second.end());
        marks.memory = marks.memory || rs.sreg_memory_pattern.contains(r);
    };
    if (in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG ||
        in.fmt == Rdna2Format::FLAT) {   // load addresses are not the loaded data
        marks.memory = true;
        return marks;
    }
    const bool lane_read = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360;
    if (lane_read) marks = vector_lane_read_marks(rs, in);
    // Numeric SCC copies must retain the current condition's missing-input mark, including
    // WRITELANE's data role before its early return. Loaded data remains separate from addresses.
    if (scalar_reads_scc(in) && (!rs.scc || rs.scc_merge_placeholder)) {
        ScalarSourceMarks condition;
        condition.placeholder = true;
        condition.hard = !rs.scc || rs.scc_loop_blanket.empty();
        condition.blanket_roots = rs.scc_loop_blanket;
        union_source_marks(marks, condition);
    }
    if (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x360 || in.opcode == 0x361)) {
        // A dynamic selector is a numeric input too. Its storage value does not prove which
        // peer a READLANE consumes or which peer a WRITELANE replaces.
        if (in.src[1].kind == OperandKind::SGPR ||
            (in.src[1].kind == OperandKind::Special && in.src[1].value <= 124))
            word(in.src[1].value);
        if (!lane_read && (in.src[0].kind == OperandKind::SGPR ||
                           (in.src[0].kind == OperandKind::Special && in.src[0].value <= 124))) {
            word(in.src[0].value);   // v_writelane's data word
        }
        return marks;
    }
    for (uint32_t k = 0; k < in.n_src && k < 3; ++k) {
        const Operand& o = in.src[k];
        // Every register whose marks propagation can set: SGPRs, and the special data words VCC,
        // ttmp0-15 and M0 (106-124).
        const bool scalar = o.kind == OperandKind::SGPR ||
                            (o.kind == OperandKind::Special && o.value >= 106 && o.value <= 124);
        if (!scalar) continue;
        // CNDMASK's third operand consumes the current Bool domain, not the saved mask's
        // physical scalar halves. A fresh predicate can be genuine while those halves cannot
        // be reconstructed. Retain actual merge poison and DATA roles; map membership alone
        // must not authenticate an absent, relayed or numerically overwritten mask.
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101 && k == 2 &&
            o.kind == OperandKind::SGPR && o.value >= 0 && o.value <= 105) {
            const auto mask = rs.sreg_bool.find(o.value);
            const uint32_t mask_words = rs.sreg_bool_b32.contains(o.value) ? 1u : 2u;
            bool genuine = mask != rs.sreg_bool.end() && mask->second;
            for (uint32_t w = 0; w < mask_words; ++w) {
                const int reg = o.value + static_cast<int>(w);
                genuine = genuine && !rs.sreg_merge_placeholder.contains(reg);
                marks.memory = marks.memory || rs.sreg_memory_pattern.contains(reg);
            }
            if (genuine) continue;
        }
        // A Wave32 lane mask is one SGPR (#5202), VCC_LO excepted (scalar_source_mark_words).
        uint32_t width = scalar_source_mark_words(in, k, rs.vector_wave_size);
        if (width == 0 || width == UINT32_MAX) width = 2;   // unknown: both words of a pair
        if (width > 2) width = static_cast<uint32_t>(std::max(0, 125 - o.value));   // a range
        for (uint32_t w = 0; w < width; ++w) word(o.value + static_cast<int>(w));
    }
    // A read-modify-write keeps the destination's old bits (s_bitset*, s_cmov*, s_addk, ...).
    const uint32_t implicit = scalar_implicit_destination_read_width(in);
    for (uint32_t w = 0; w < implicit; ++w) word(in.dst.value + static_cast<int>(w));
    return marks;
}

inline SavedB64MaskSnapshot snapshot_saved_b64_masks(const RegState& rs, const Rdna2Inst& in) {
    SavedB64MaskSnapshot snapshot;
    snapshot.source_marks = scalar_source_marks(rs, in);
    // The widest write form is deliberate: this set only FILTERS what record_scalar_write may
    // expire, and that function applies its own exact `effective_width`, so an extra candidate root
    // here can never widen the erase set.
    for_each_scalar_write(
        in,
        [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int root = base + static_cast<int>(word);
                for (int candidate : {root - 1, root}) {
                    if (candidate < 0 || candidate > 105 || rs.sreg_bool_b32.contains(candidate))
                        continue;
                    const auto mask = rs.sreg_bool.find(candidate);
                    if (mask != rs.sreg_bool.end())
                        snapshot.entries.emplace_back(candidate, mask->second);
                }
            }
        },
        /*wave32_one_word_masks*/ false);
    return snapshot;
}

// Explicit emission-domain snapshot: host load_state reconstruction is not an architectural write.
inline SavedB64MaskSnapshot snapshot_saved_b64_masks(SpirvCompute& b, const RegState& rs,
                                                     const Rdna2Inst& in) {
    auto snapshot = snapshot_saved_b64_masks(rs, in);
    note_memory_address(b, rs, in, snapshot.source_marks);   // #5193, before the roots are cleared
    if (in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF) {
        // Address assumptions are read before an aliasing write. Keep them on the loaded value:
        // a weak back edge makes it arbitrary, so a later strong address consumer must charge
        // these same roots rather than treating the loaded value as unconditional DATA (#5204).
        auto& roots = snapshot.source_marks.blanket_roots;
        b.loop_blanket_roots_read.insert(roots.begin(), roots.end());
        // Scalar roots cannot hold weakly; retire their address-only dependencies as before.
        std::erase_if(roots, [](uint64_t root) {
            return static_cast<int>(root & 0xffffu) < kVectorBlanketBase;
        });
        if (!roots.empty()) snapshot.source_marks.placeholder = true;
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43)
        snapshot.source_marks = vector_relative_source_marks(b, rs, in);
    snapshot.portable_numeric_replacement = ordinary_portable_wave64_masks(b);
    return snapshot;
}

// ---------------------------------------------------------------------------------------------
// Workgroup-uniform wave branch (#1554).
//
// A compute `s_cbranch_vccz`/`vccnz` decides per WAVE, so two waves of one workgroup may take
// different arms. A guest `S_BARRIER` inside such an arm violates Vulkan's uniform-participation
// requirement for OpControlBarrier, so detect_forward_ifs rejects that combination.
//
// Real kernels nevertheless build some of these masks entirely out of launch state:
//
//     s_load_dwordx4        s[8:11], s[12:13], 0x20    ; s12:s13 is entry data
//     s_buffer_load_dwordx2 s[16:17], s[8:11], 0xc
//     s_add_i32             vcc_lo, s17, -1            ; VCC_LO used as scalar scratch
//     s_cmp_gt_u32          s16, 0
//     s_cselect_b64         s[10:11], exec, 0
//     s_cmp_eq_u32          s14, vcc_lo
//     s_cselect_b64         vcc, exec, 0
//     s_and_b64             vcc, vcc, s[10:11]
//     s_cbranch_vccz        END
//
// With EXEC full, VCC is nonzero exactly when the scalar predicate holds, and that predicate reads
// only compute launch SGPRs. recompile_compute seeds those from push-constant user data and from
// WorkGroupId system SGPRs (see the entry seeding in recompile_compute), both of which are identical
// for every wave of a workgroup — so the branch is workgroup-uniform and its arm may synchronize.
//
// Two obligations are proved independently and mechanically:
//
//   (1) EXEC is full at the branch on EVERY path. Without it, VCCZ is also true for a wave whose
//       EXEC happens to be empty, which is a genuinely per-wave property. A forward must-dataflow
//       proves this, so a loop that restores `s_mov_b64 exec, -1` before its back-edge still
//       qualifies while an arm that narrows EXEC does not.
//   (2) VCC is `SCC ? EXEC : 0`, optionally combined by s_and_b64/s_or_b64; every contributing SCC
//       comes from a scalar compare, and every scalar input traces to launch data.
//
// Everything not proved stays rejected: EXEC or SCC read as data, V_READFIRSTLANE/V_READLANE,
// VOPC-written masks, SGPRs above the launch range, and any definition reaching the branch from
// outside its own straight-line region unless it provably still holds its entry value.
//
// CONFIDENCE: HIGH — the accepted provenance set is closed and every rejection is fail-visible.

// Forward "must" dataflow over the decoded stream. `fact[i]` is true only when the tracked property
// holds at entry to ins[i] along EVERY path that reaches it. `gen` re-establishes the property,
// `kill` destroys it, and anything else passes it through. Optimistic initialization followed by
// iteration to a fixpoint is the standard greatest-fixpoint formulation, which is what makes the
// result sound across back-edges. Returns an empty vector when the stream contains an edge this walk
// cannot resolve, so every caller treats "unanalyzable" as "not proved".
template <typename Gen, typename Kill>
std::vector<uint8_t> must_fact_at(const std::vector<Rdna2Inst>& ins, Gen&& gen, Kill&& kill,
                                  bool entry_holds) {
    if (ins.empty()) return {};
    std::unordered_map<uint32_t, size_t> index_of;
    for (size_t i = 0; i < ins.size(); ++i) index_of[ins[i].pc] = i;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };

    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) continue;
        if (is_branch(in)) {
            const uint32_t target = branch_target(in);
            auto found = index_of.find(target);
            if (found != index_of.end())
                succ[i].push_back(found->second);
            else if (target <= ins.back().pc)
                return {};   // an edge this walk cannot resolve
            // A target past the decoded stream is an early-out that terminates the wave; it
            // contributes no successor inside this function.
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02)
            continue;   // s_branch: no fallthrough
        if (i + 1 < ins.size()) succ[i].push_back(i + 1);
    }

    std::vector<uint8_t> fact(ins.size(), 1);
    if (!entry_holds) fact[0] = 0;
    // Each round can only turn a 1 into a 0, so the fixpoint is reached in at most size()+1 rounds.
    for (size_t round = 0; round <= ins.size() + 1; ++round) {
        std::vector<uint8_t> next(ins.size(), 1);
        if (!entry_holds) next[0] = 0;
        for (size_t i = 0; i < ins.size(); ++i) {
            uint8_t out = fact[i];
            if (gen(ins[i]))
                out = 1;
            else if (kill(ins[i]))
                out = 0;
            if (out) continue;
            for (size_t s : succ[i]) next[s] = 0;
        }
        if (next == fact) return fact;
        fact.swap(next);
    }
    return fact;
}

// `s_mov_b64 exec, -1` is the only accepted full-mask definition. A 32-bit literal written to a
// 64-bit scalar destination is zero-extended, so it activates only the low half and is NOT a full
// wave64 mask; the inline -1 constant is sign-extended and is.
inline bool exec_write_sets_full_mask(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.n_src != 1) return false;
    if ((in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) ||
        in.dst.value != 126)
        return false;
    return in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1;
}

inline bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

inline uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

// Arbitrary CFG fallback. Some UE4 volume-lighting kernels and Astro Bot material shaders contain
// nested EXEC loops plus scalar/VCC branches. They are valid reducible machine CFGs, but deliberately
// exceed the narrow pattern structurizer below. Lower them as a structured dispatcher loop: one
// switch case per basic block, with the emulated register file persisted in Function variables
// between iterations. Graphics stages can branch directly on this invocation's SCC/VCC/EXEC bit;
// compute needs the guest-wave reduction described below.
// VCCZ/EXECZ reduce over the dispatch's 32/64-lane hardware wave, not the host Vulkan subgroup.
// Native subgroup widths are implementation-defined (llvmpipe is commonly 8), so every invocation
// remains in the dispatcher until the whole workgroup is done and exchanges its vote at the common
// switch merge. Dedicated Workgroup scratch keeps multiple hardware waves independent. The fallback
// is gated by emit_body to complex compute CFGs without guest barriers. Ordinary LDS reads, writes,
// and atomics remain valid while waves visit different cases. V_MBCNT is split into a dedicated
// common phase so every workgroup invocation reaches its synthesized barriers in uniform control flow.
// PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Bound the CFG DISPATCHER's loop at N iterations, on
// whichever back-edge it emits (direct or portable). PROSPER_CFG_TRIP_BOUND_PROGRAM selects one
// program; PROSPER_CFG_TRIP_BOUND_PHASE selects one phase and is REQUIRED — see
// emitted_loop_trip_bound for why a record spanning two phases has no true reading. A program that
// accesses GDS itself is refused, because the witness lives in guest-addressable storage.
//
// It does NOT cover the two structured loop emitters. An earlier revision of this comment claimed it
// did; it never called this helper from either, so the claim was false and would have made a null
// result from a structured-loop program read as evidence.
//
// A loop that never terminates hangs the GPU into a driver reset, which costs the whole process its
// compute backend and every later indirect draw — and from outside it is indistinguishable from a
// slow shader or from a defect anywhere else in the submit. Bounding the dispatcher's loop turns
// "is this a non-terminating loop at all?" into one run for the programs it covers.
//
// (This paragraph used to end "and covering all three emitters means the answer is not confined to
// one lowering path" — the very claim the paragraph above corrects. The correction landed two lines
// up and the boast survived underneath it, which is how a retracted statement keeps being read as
// current. `tests/shared/diagnostics/test_cfg_trip_bound.cpp` now pins both halves: a structured loop is unchanged when
// armed, a dispatcher loop is not.)
//
// Unset, nothing is emitted and modules are byte-identical. It is NOT a fix: truncating a guest
// program's control flow produces wrong results by construction, which is why it is opt-in and says
// so when it arms.
// See kComputeTripWitnessDword. Deliberately ignores the phase selector: if ANY phase of this program
// will be bounded, the witness destination has to be bound for the whole dispatch.
// The complete selector state, read fresh rather than cached in function-local statics.
//
// Two reasons it is a struct and not three scattered getenv sites. First, the shader cache is keyed
// on the program's CODE BYTES and never on its address, so a target and a non-target with the same
// body collide on one entry: whichever compiled first would serve the other, handing the bounded
// module to an excluded program or the unbounded module to the target. Either direction silently
// destroys the isolation that makes a bounded run evidence about ONE program. The cache key therefore
// mixes this whole struct in (see gpu_executor.cpp), and a struct is what makes that complete by
// construction — a future selector added here cannot be forgotten there.
//
// Second, function-local statics parse the environment exactly once per process, which makes the
// selectors untestable in-process: no test could arm a bound, assert, then disarm and assert again.
// So this parses on demand, and `TripBoundOperation` (rdna2_to_spirv.hpp) is what keeps one
// operation's reads coherent without freezing them for the process.
//
// Work counter for the operation scope. See trip_bound_parses().
inline std::atomic<uint64_t>& trip_bound_parse_counter() {
    static std::atomic<uint64_t> parses{0};
    return parses;
}

// The innermost TripBoundOperation's owned settings, or null when none is open on this thread.
// Per thread because compilation runs on whatever worker realized the draw.
inline const ComputeTripBoundSettings*& trip_bound_pin() {
    static thread_local const ComputeTripBoundSettings* pinned = nullptr;
    return pinned;
}

// THE COMMENT HERE USED TO SAY the re-reads happen "during RECOMPILATION only (never per draw or
// per dispatch — cf. #2214)", with CONFIDENCE: HIGH. That was wrong, and wrong in the way this
// project's charter warns about most: a true-sounding claim wearing a citation to the very issue
// about removing per-draw getenv, in a place nobody re-derives. Three of the callers are not in the
// recompiler at all — gpu_executor.cpp builds the shader-cache KEY from these settings, so they are
// read on every cache HIT as well as every miss, i.e. once or twice per draw. Measured on a 240 s
// routed Grand Theft Auto V window with the switch unset (#3714): 3,989,848 reads of
// PROSPER_CFG_TRIP_BOUND at `ec02a77f6`, ~16,600/s. That is nominally the largest remaining getenv
// name, but it leads PROSPER_GFXLOG by 0.39% while GFXLOG itself varies by 35,143 between two runs
// of the same route -- so read the two as jointly largest, not ranked. An unset name is the
// expensive case on glibc -- a full scan of environ -- so those are all misses.
//
// The body. `parse_trip_bound_settings()` and `compute_trip_bound_settings()` are declared in the
// public header and defined ONCE in rdna2_to_spirv.cpp on top of this. The reason is
// [basic.def.odr]/11: an `inline` function must be defined in every translation unit that uses it,
// and this header is internal to src/gpu/recompiler/, so a TU outside it -- the new
// `test_trip_bound_operation` -- cannot legally see a definition at all.
//
// Do NOT restate that as "no out-of-line copy survives". It does: on `ec02a77f6` gpu_executor.cpp
// calls `compute_trip_bound_settings()` through the public declaration WITHOUT including this
// header, and links. What was actually observed is narrower: linking test_trip_bound_operation
// against libprosper_core.a failed with `undefined reference to
// 'prosper::gpu::compute_trip_bound_settings()'` -- the PRE-EXISTING function, not any of the
// symbols this change adds, which is the only reason the observation bears on the inline question
// at all. The mechanism for it was not established and is deliberately not guessed at here.
inline ComputeTripBoundSettings parse_trip_bound_settings_impl() {
    trip_bound_parse_counter().fetch_add(1, std::memory_order_relaxed);
    ComputeTripBoundSettings settings;
    const char* spec = getenv("PROSPER_CFG_TRIP_BOUND");
    if (!spec || !*spec) return settings;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(spec, &end, 0);
    if (!end || *end || !parsed || parsed > 0xffffffffull) return settings;
    settings.bound = static_cast<uint32_t>(parsed);

    // PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR — restrict the bound to ONE program, leaving every other
    // recompiled module byte-identical. Without it, a bound low enough to be interesting truncates
    // MANY shaders, and any later change of behaviour has a second explanation: an earlier truncated
    // shader fed different data downstream. Targeting one program removes that alternative, which is
    // what turns "the run got further" into evidence about the program under test.
    if (const char* only = getenv("PROSPER_CFG_TRIP_BOUND_PROGRAM"); only && *only) {
        char* only_end = nullptr;
        const unsigned long long wanted = strtoull(only, &only_end, 0);
        if (only_end && !*only_end) settings.only_program = wanted;
    }
    // PROSPER_CFG_TRIP_BOUND_PHASE=K — bound only the K-th dispatcher of the selected program.
    //
    // A barrier-phased compute program emits one dispatcher per barrier-free phase, each covering a
    // different guest pc range. Bounding them together answers "does some loop in this program run
    // away" but not which, and they are not equivalent: an acyclic phase that needs a bound is a CFG
    // state-transition defect, while the phase containing the guest's own loop might merely be slow.
    if (const char* phase = getenv("PROSPER_CFG_TRIP_BOUND_PHASE"); phase && *phase) {
        char* phase_end = nullptr;
        const unsigned long long parsed_phase = strtoull(phase, &phase_end, 0);
        if (phase_end && !*phase_end && parsed_phase <= 0xfffffffeull)
            settings.only_phase = static_cast<uint32_t>(parsed_phase);
    }
    // PROSPER_CFG_TRIP_BOUND_ORDINAL=K — count only traversals about to dispatch ordinal K.
    // See ComputeTripBoundSettings::only_ordinal for why a per-loop cap is the thing that localizes
    // a runaway inside a program whose dispatcher serves several guest loops (#3193).
    if (const char* ordinal = getenv("PROSPER_CFG_TRIP_BOUND_ORDINAL"); ordinal && *ordinal) {
        char* ordinal_end = nullptr;
        const unsigned long long parsed_ordinal = strtoull(ordinal, &ordinal_end, 0);
        if (ordinal_end && !*ordinal_end && parsed_ordinal <= 0xfffffffeull)
            settings.only_ordinal = static_cast<uint32_t>(parsed_ordinal);
    }
    return settings;
}

struct BarrierPhasedCompute {
    size_t guard_index = 0;
    size_t end_index = 0;
    std::vector<size_t> barriers;
    bool guarded = false;
    bool found = false;
};

// Prove the SCC consumed by a terminal compute guard is identical for every wave in the workgroup,
// while allowing unrelated lane-local setup to precede it. The legacy barrier-phase proof required
// every instruction before the guard to be scalar; generated kernels commonly interleave vector
// address setup with a scalar descriptor load that independently produces the guard condition.
//
// This is deliberately a narrow backwards slice. The last SCC writer must be an SOPC, and every
// scalar word feeding that compare must trace through launch SGPRs, scalar ALU, or scalar memory.
// V_READFIRSTLANE/V_READLANE, vector mask writes, EXEC, and unresolved special registers terminate
// the proof. Existing all-scalar prefixes retain their established admission path below.
inline bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                                    size_t guard_index) {
    // This is a textual backwards slice, not a reaching-definition analysis. Keep the prefix
    // branch-free so every scalar definition it finds dominates the terminal guard. In particular,
    // a VCC/EXEC branch may vary between guest waves even when the literal or SMEM value written in
    // one arm is itself uniform.
    for (size_t i = 0; i < guard_index; ++i) {
        const Rdna2Inst& prefix = ins[i];
        if ((prefix.fmt == Rdna2Format::SOPP && !sopp_is_noop(prefix)) ||
            (prefix.fmt == Rdna2Format::SOP1 && prefix.opcode >= 0x20u && prefix.opcode <= 0x22u))
            return false;
    }

    std::function<bool(int, uint32_t, uint32_t)> uniform_reg_at;
    auto operand_uniform_at = [&](const Operand& operand, uint32_t words, uint32_t use_pc,
                                  auto&& self) -> bool {
        switch (operand.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal: return true;
            case OperandKind::Special:
                if (operand.value == 125) return true;   // SGPR_NULL
                if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                    return false;   // EXEC and SCC are not scalar launch data.
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!self(operand.value + static_cast<int>(word), use_pc, 0)) return false;
                return true;
            default: return false;
        }
    };

    uniform_reg_at = [&](int reg, uint32_t use_pc, uint32_t depth) -> bool {
        if (depth > 64) return false;
        for (size_t i = ins.size(); i-- > 0;) {
            const Rdna2Inst& writer = ins[i];
            if (writer.pc >= use_pc) continue;
            bool writes = false;
            for_each_scalar_write(writer, [&](int base, uint32_t width) {
                writes |= reg >= base && reg < base + static_cast<int>(width);
            });
            // Implicit VOPC writes target architectural VCC even when the decoder exposes it as a
            // Special operand rather than an ordinary SGPR destination.
            if ((reg == 106 || reg == 107) && writer.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(writer.opcode))
                writes = true;
            if (!writes) continue;

            auto uniform_operand = [&](const Operand& operand, uint32_t words) {
                switch (operand.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal: return true;
                    case OperandKind::Special:
                        if (operand.value == 125) return true;
                        if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                            return false;
                        [[fallthrough]];
                    case OperandKind::SGPR:
                        for (uint32_t word = 0; word < words; ++word)
                            if (!uniform_reg_at(operand.value + static_cast<int>(word), writer.pc,
                                                depth + 1))
                                return false;
                        return true;
                    default: return false;
                }
            };

            switch (writer.fmt) {
                case Rdna2Format::SOP1: {
                    // GETPC is identical for every wave. Mask/saveexec operations consume EXEC and
                    // are rejected by rdna2_instruction_may_change_exec; ordinary scalar operations trace
                    // their source at the destination's architectural width.
                    if (writer.opcode == 0x1f) return true;
                    if (rdna2_instruction_may_change_exec(writer) || writer.n_src != 1)
                        return false;
                    return uniform_operand(writer.src[0],
                                           scalar_write_width(writer) == 2 ? 2u : 1u);
                }
                case Rdna2Format::SOP2: {
                    // Carry/cselect forms consume SCC in addition to their decoded operands. They are
                    // outside this narrow slice until SCC itself is represented as a recursive value.
                    if (writer.opcode == 0x04 || writer.opcode == 0x05 || writer.opcode == 0x0a ||
                        writer.opcode == 0x0b)
                        return false;
                    const uint32_t words = scalar_write_width(writer) == 2 ? 2u : 1u;
                    return writer.n_src == 2 && uniform_operand(writer.src[0], words) &&
                           uniform_operand(writer.src[1], words);
                }
                case Rdna2Format::SMEM: {
                    // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads use a
                    // base pair. The scalar offset is one dword in both forms. Only a load's
                    // result is a function of its address operands.
                    if (!rdna2_smem_load_dwords(writer.opcode) || writer.n_src < 1) return false;
                    const uint32_t base_words = rdna2_smem_address_dwords(writer.opcode);
                    if (!uniform_operand(writer.src[0], base_words)) return false;
                    return writer.n_src < 2 || uniform_operand(writer.src[1], 1);
                }
                case Rdna2Format::SOPK:
                    // S_MOVK is a literal definition. Other SOPK data writers are read/modify/write
                    // or hardware-register reads and stay outside this proof.
                    return writer.opcode == 0x00;
                default: return false;
            }
        }
        // Ordinary compute launch SGPRs (user data and enabled system SGPRs) are uniform within one
        // workgroup. Special registers are numbered above this range and were handled explicitly.
        return reg >= 0 && reg <= 105;
    };

    for (size_t i = guard_index; i-- > 0;) {
        const Rdna2Inst& candidate = ins[i];
        if (sopp_is_noop(candidate)) continue;
        if (candidate.fmt == Rdna2Format::SOPC) {
            const uint32_t words = candidate.opcode == 0x12 || candidate.opcode == 0x13 ? 2u : 1u;
            for (uint32_t source = 0; source < candidate.n_src; ++source)
                if (!operand_uniform_at(candidate.src[source], words, candidate.pc, uniform_reg_at))
                    return false;
            return candidate.n_src != 0;
        }
        // Vector/memory operations cannot write SCC. Any scalar ALU or SOPK instruction might, so
        // stop rather than accidentally consuming an older compare through an unmodeled SCC writer.
        if (candidate.fmt == Rdna2Format::SOP1 || candidate.fmt == Rdna2Format::SOP2 ||
            candidate.fmt == Rdna2Format::SOPK)
            return false;
    }
    return false;
}

// Recognize barrier-separated compute phases. A generated kernel may wrap the phases in one proven
// workgroup-uniform terminal SCC guard, but the guard is not essential: an unconditional top-level
// barrier is itself a valid phase boundary. In either form, every explicit scalar edge must stay in
// one phase and indirect PC changes fail closed. Keeping this proof shared is important: emit_body
// uses it to split the shader, while the native subgroup policy uses the same result to make nested
// wave votes legal inside an individual phase.
inline BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
    BarrierPhasedCompute result;
    result.end_index = ins.size();
    uint32_t end_pc = UINT32_MAX;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].is_end) continue;
        result.end_index = i;
        end_pc = ins[i].pc;
        break;
    }

    result.guard_index = ins.size();
    size_t first_barrier = ins.size();
    size_t branch_count = 0;
    for (size_t i = 0; i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a && first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03)
            continue;
        ++branch_count;
        if (result.guard_index == ins.size() && (in.opcode == 0x04 || in.opcode == 0x05) &&
            branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool guarded = result.guard_index < first_barrier && first_barrier < result.end_index;
    bool scalar_prefix = guarded;
    for (size_t i = 0; scalar_prefix && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC ||
                            in.fmt == Rdna2Format::SMEM ||
                            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            scalar_prefix = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
                   (operand.value == 126 || operand.value == 127);
        };
        if (is_exec(in.dst)) scalar_prefix = false;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (is_exec(in.src[source])) scalar_prefix = false;
    }
    if (guarded && !scalar_prefix)
        guarded = terminal_guard_scc_is_workgroup_uniform(ins, result.guard_index);

    // If a candidate terminal guard was not proved uniform, include it in the ordinary edge proof.
    // Since it jumps across every barrier to S_ENDPGM, that proof rejects the conditional barrier.
    // A proved guard is deliberately excluded: every invocation in the workgroup takes it together.
    result.guarded = guarded;
    const size_t body_begin = guarded ? result.guard_index + 1 : 0;
    // Safety comes from the cross-edge, trap and EXEC-save checks below. A straight-line kernel
    // whose only control-flow instruction is a barrier satisfies that proof trivially; requiring
    // an unrelated branch-count threshold dropped GTA V's partial-workgroup barrier passes.
    bool valid = result.end_index < ins.size();

    for (size_t i = body_begin; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a) result.barriers.push_back(i);
    valid = valid && !result.barriers.empty();

    for (size_t i = body_begin; valid && i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // A trap deactivates its guest wave in the dispatcher. Emitting a host workgroup barrier
        // afterward would then require an invocation that has left the phase to participate.
        // Final-phase traps are fine because no outer barrier follows them.
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12 && i < result.barriers.back())
            valid = false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) valid = false;
    }
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = body_begin; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid;
    return result;
}

}   // namespace prosper::gpu
