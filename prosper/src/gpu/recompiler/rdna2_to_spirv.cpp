// rdna2_to_spirv.cpp — see rdna2_to_spirv.hpp. Internal SpirvCompute builder + the VALU translator.
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
#include "rdna2_to_spirv_internal.hpp"
#include "rdna2_alu_support.hpp"

namespace prosper::gpu {

namespace {


} // namespace

void record_recompile_reject_reason_for_test(const RecompileDiagnosticContext& diagnostic,
                                             const char* tag, const char* role,
                                             const char* payload) {
    log_recompile_diagnostic(diagnostic, tag, role, "%s", payload);
}

std::string last_terminal_reject_reason(uint64_t program_address) {
    std::lock_guard lock(terminal_reject_mutex());
    const auto& reasons = terminal_reject_reasons();
    const auto found = reasons.find(program_address);
    return found == reasons.end() ? std::string() : found->second;
}

void log_compute_recompile_skip_diagnostic(const RecompileDiagnosticContext& diagnostic) {
    log_recompile_diagnostic(diagnostic, "compute-recompile-reject", "consequent",
                             "reason=empty-result dispatch-skipped");
}

// #2319: render exactly the dwords the instruction HAS, not a fixed two.
//
// `words[]` is `uint32_t words[5] = {0,...}`, so a ONE-dword instruction printed as
// `words=bf860051,00000000` is indistinguishable from a two-dword instruction whose second dword
// is genuinely zero. That defeats the purpose of printing raw words (#2312), which is that a
// reader can name the instruction with one command:
//
//     echo '0x0e,0x16,0xea,0xbe' | llvm-mc -arch=amdgcn -mcpu=gfx1010 -disassemble
//
// A reader who takes `words=x,00000000` at face value feeds EIGHT bytes, and llvm-mc decodes a
// second, entirely fictitious instruction from the zero dword -- output that looks exactly as
// authoritative as the correct answer. On the CrossWorlds census pc=101 needed four bytes and
// pc=2038 needed eight, and nothing in either line said which.
//
// Clamped to the array bound as well as to len_dwords: len_dwords comes from the decoder, and a
// decode that failed badly enough to be rejected here is not a source to trust for indexing.
inline std::string reject_words_text(const Rdna2Inst& in) {
    const uint32_t count = std::min<uint32_t>(in.len_dwords ? in.len_dwords : 1u,
                                              (uint32_t)(sizeof(in.words) / sizeof(in.words[0])));
    std::string out;
    char buf[16];
    for (uint32_t i = 0; i < count; ++i) {
        std::snprintf(buf, sizeof buf, "%08x", in.words[i]);
        if (i) out += ',';
        out += buf;
    }
    return out;
}


FragmentInterpolationLayout::FragmentInterpolationLayout() {
    for (auto& locations : parameter_locations) locations.fill(kUnusedLocation);
    system_locations.fill(kUnusedLocation);
}

namespace {


// The f16 bit pattern an inline float constant supplies in a 16-bit operand position (ISA Table 10
// lists per-width encodings: "0.5 ... half: 0x3800" etc.). Only 1/(2*pi) (code 248, 0x3118) differs
// from rounding the f32 value — the f32 table entry 0.15915494 would round to a different last bit
// than the documented operand, so 16-bit consumers must use these bits, not the f32 constant.

}  // namespace

// Extend the narrow compiler shape
//
//   s_cbranch_scc* ELSE
//   THEN...
//   s_endpgm
// ELSE:
//   ELSE...
//   s_endpgm
//
// into the existing structured-if/else input. The first s_endpgm is represented as a synthetic
// s_branch to the second end, which is semantically exact: it skips the other terminating arm, while
// the shared SPIR-V shell may still converge solely to publish outputs/return. Keep this deliberately
// conservative: one scalar conditional, adjacent straight-line else arm, and a real terminating end.
// Immediate-end/nop-end tails remain the established early-out shape and are not rewritten.
static bool extend_terminating_if_else(const uint32_t* code, size_t dwords,
                                       std::vector<Rdna2Inst>& instructions,
                                       size_t* required_dwords = nullptr,
                                       uint32_t* synthetic_branch_pc = nullptr) {
    if (synthetic_branch_pc) *synthetic_branch_pc = UINT32_MAX;
    if (!code || instructions.empty()) return false;
    auto first_end = std::find_if(instructions.begin(), instructions.end(),
                                  [](const Rdna2Inst& in) { return in.is_end; });
    if (first_end == instructions.end()) return false;

    const Rdna2Inst* branch = nullptr;
    for (auto it = instructions.begin(); it != first_end; ++it) {
        if (it->fmt != Rdna2Format::SOPP) continue;
        if (it->opcode == 0x04 || it->opcode == 0x05) {
            if (branch) return false;
            branch = &*it;
        } else if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) {
            return false;
        }
    }
    if (!branch || branch->simm16 <= 0) return false;
    const int64_t target64 = static_cast<int64_t>(branch->pc) + branch->len_dwords + branch->simm16;
    if (target64 < 0 || static_cast<uint64_t>(target64) >= dwords) return false;
    const uint32_t target = static_cast<uint32_t>(target64);
    if (target != first_end->pc + first_end->len_dwords) return false;

    std::vector<Rdna2Inst> tail;
    const size_t tail_dwords = rdna2_walk(code + target, dwords - target, tail);
    if (tail.empty() || !tail.back().is_end) return false;
    bool has_real_else = false;
    for (auto it = tail.begin(); it != tail.end() - 1; ++it) {
        if (it->fmt == Rdna2Format::SOPP) {
            if (it->opcode == 0x00) continue; // padding is harmless but not a real else arm
            if (it->opcode >= 0x02 && it->opcode <= 0x09 && it->opcode != 0x03) return false;
        }
        has_real_else = true;
    }
    if (!has_real_else) return false;

    for (auto& in : tail) in.pc += target;
    const uint32_t merge_pc = tail.back().pc;
    const uint32_t skip_dwords = merge_pc - (first_end->pc + first_end->len_dwords);
    if (!skip_dwords || skip_dwords > static_cast<uint32_t>(INT16_MAX)) return false;
    first_end->fmt = Rdna2Format::SOPP;
    first_end->opcode = 0x02; // s_branch merge_pc: terminates the lexical then arm
    first_end->simm16 = static_cast<int32_t>(skip_dwords);
    first_end->words[0] = 0xbf820000u | (skip_dwords & 0xffffu);
    first_end->is_end = false;
    if (synthetic_branch_pc) *synthetic_branch_pc = first_end->pc;
    instructions.insert(instructions.end(), tail.begin(), tail.end());
    if (required_dwords) *required_dwords = target + tail_dwords;
    return true;
}

FragmentInterpolationLayout fragment_interpolation_layout(
        const uint32_t* code, size_t dwords,
        const PixelSystemInputMapping* system_inputs,
        const PixelInputMapping* pixel_inputs) {
    FragmentInterpolationLayout layout;
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    std::array<uint8_t, 32> selectors{};
    uint32_t highest_attribute = 0;
    bool has_attribute = false;
    for (const auto& instruction : instructions) {
        if (instruction.is_end) break;
        if (instruction.fmt != Rdna2Format::VINTRP || instruction.vintrp_attr >= 32) continue;
        const uint32_t attr = instruction.vintrp_attr;
        layout.attribute_mask |= 1u << attr;
        highest_attribute = std::max(highest_attribute, attr);
        has_attribute = true;
        if (instruction.opcode == 0 || instruction.opcode == 1)
            layout.smooth_mask |= 1u << attr;
        else if (instruction.opcode == 2 && instruction.src[0].value < 3)
            selectors[attr] |= static_cast<uint8_t>(1u << instruction.src[0].value);
    }
    if (pixel_inputs)
        layout.passthrough_mask =
            pixel_inputs->effective_passthrough_mask() & layout.attribute_mask;

    // P10/P20 have no ordinary Vulkan varying equivalent. P0 can retain the cheap Flat-input path
    // when it is the attribute's only interpolation mode; mixed P0+smooth needs the geometry copy too.
    for (uint32_t attr = 0; attr < 32; ++attr) {
        if (selectors[attr] & 0x3u) layout.requires_geometry = true;
        if ((selectors[attr] & 0x4u) && (layout.smooth_mask & (1u << attr)))
            layout.requires_geometry = true;
    }
    if (!layout.requires_geometry) return layout;

    uint32_t location = has_attribute ? highest_attribute + 1 : 0;
    for (uint32_t attr = 0; attr < 32; ++attr) {
        for (uint32_t selector = 0; selector < 3; ++selector) {
            if (!(selectors[attr] & (1u << selector))) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.parameter_locations[attr][selector] = location++;
        }
    }
    if (system_inputs) {
        for (uint32_t field = 0; field < 7; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit) || !(system_inputs->ena & bit)) continue;
            if (location >= 32) { layout.valid = false; return layout; }
            layout.system_locations[field] = location++;
        }
    }
    return layout;
}

std::vector<uint32_t> recompile_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_position) {
    SpirvCompute builder;
    return builder.build_interpolation_geometry(layout, capture_position);
}

bool sopp_is_noop(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPP) return false;
    switch (in.opcode) {
        case 0x00:   // s_nop
        case 0x0c:   // s_waitcnt
        case 0x20:   // s_inst_prefetch
        case 0x21:   // s_clause
        case 0x22:   // s_wait_idle
            // NOTE: s_waitcnt_vscnt is NOT SOPP on gfx10 — it is SOPK opcode 0x17
            // (round-trip: llvm-mc gfx1010 encodes it 0xBBFD0000). Handled in the SOPK case.
            return true;
        default:
            return false;
    }
}

namespace {
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);
}

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

inline bool sgpr_dead_at_merge(const std::vector<Rdna2Inst>& ins, uint32_t target, int R,
                               ScalarMergeProof proof = ScalarMergeProof::AnyRead,
                               ScalarMergeBlocker* blocker = nullptr) {
    const auto block = [&](const Rdna2Inst& at, const char* kind) {
        if (blocker && blocker->pc == UINT32_MAX) { blocker->pc = at.pc; blocker->kind = kind; }
        return false;
    };
    // MaskDomainOnly is a VCC-half question by construction: an ordinary SGPR has no mask domain,
    // so the relaxation would silently become "reads are fine", which is not a proof of anything.
    if (proof == ScalarMergeProof::MaskDomainOnly && R != 106 && R != 107)
        return false;
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
        if (in.is_end) continue;                           // this path ends without a read
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
        }
        switch (in.fmt) {
            // Most SOPK instructions remain fail-closed: s_addk/s_mulk/s_cmovk/s_cmpk read or
            // read-modify-write their encoded "dst" while decode exposes no source. MOVK_I32 is a
            // pure one-word definition and WAITCNT_VSCNT is register-transparent; GTA's low-only
            // VCC lifetime crosses both before the actual pair replacement.
            case Rdna2Format::SOPK:
                if (in.opcode == kSopkOpcodeWaitcntVscnt &&
                    in.dst.kind == OperandKind::SGPR && in.dst.value == 125)
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
                if (in.opcode == 0x02 ||
                    (in.opcode >= 0x04 && in.opcode <= 0x09)) {
                    const int64_t branch_pc = static_cast<int64_t>(in.pc) +
                        in.len_dwords + in.simm16;
                    const auto branch = branch_pc >= 0
                        ? pc_to_index.find(static_cast<uint32_t>(branch_pc)) : pc_to_index.end();
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
                // s_load/s_buffer_load: reads the SBASE pair (src[0], 2 regs) + SOFFSET (src[1]);
                // redefines N consecutive dst SGPRs. Anything not a plain load -> bail.
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;
                    default: return block(in, "unmodelled-smem");
                }
                if (in.src[0].value == R || in.src[0].value + 1 == R)
                    return block(in, "smem-sbase");                                          // SBASE read
                // SOFFSET is one dword and is consumed as an address, never as a lane mask.
                if ((in.src[1].kind == OperandKind::SGPR || in.src[1].kind == OperandKind::Special) &&
                    in.src[1].value == R && !data_read_ok)
                    return block(in, "smem-soffset");                                        // SOFFSET read
                if (R >= in.dst.value && R < in.dst.value + (int)n) continue;  // redefined: this path is dead
                break;
            }
            case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPC:
            case Rdna2Format::VOP1: case Rdna2Format::VOP2: case Rdna2Format::VOP3: case Rdna2Format::VOPC:
            case Rdna2Format::DS:    // DS operands are VGPR/M0 only; it cannot read an ordinary SGPR/VCC
            case Rdna2Format::EXP: { // EXP data sources are all VGPRs — it can never read an SGPR
                // Conditional moves, SOPK immediate operations, and SOP1 bitsets may read their
                // encoded destination before replacing it. Their decoded source list does not
                // contain that old value, so consult the shared implicit-read inventory first.
                const uint32_t implicit_read_width =
                    scalar_implicit_destination_read_width(in);
                if (implicit_read_width && R >= in.dst.value &&
                    R < in.dst.value + static_cast<int>(implicit_read_width) &&
                    !(data_read_ok && implicit_read_width == 1))
                    return block(in, "implicit-dst-rmw");
                for (int k = 0; k < in.n_src; k++) {
                    if (in.src[k].kind != OperandKind::SGPR &&
                        in.src[k].kind != OperandKind::Special) continue;
                    // Operand does not carry B32/B64 width. Use the shared opcode inventory so a
                    // physical low-half read (notably GTA's CSELECT_B32 -> V_ADD3 chain) cannot
                    // falsely keep the sibling live, while unlisted VOP3 sources remain pairs.
                    const uint32_t words = scalar_alu_source_words(in, k);
                    if (words == UINT32_MAX) continue;
                    if (R >= in.src[k].value &&
                        R < in.src[k].value + static_cast<int>(words)) {
                        // A one-dword source is a scalar-data read. Anything wider reads the pair,
                        // which is exactly the set of lane-mask consumers.
                        if (data_read_ok && words == 1) continue;
                        return block(in, words == 1 ? "source-dword" : "source-pair");
                    }
                }
                // A VOP3B carry-out (sdst) is a 64-bit mask write — reads none of R beyond its sources.
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || in.sdst.value + 1 == R)) continue;   // redefined (pair)
                // A non-X VOPC e32 compare writes the whole VCC pair — kills both halves. V_CMPX
                // writes EXEC ONLY on RDNA2 (ISA 12.9: "EXEC[threadId] = ..."; no VCC/SDST dest),
                // so it must NOT count as a VCC kill — miscounting it let safe_execz linearize a
                // block whose vcc scratch-write hardware would have skipped.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.value == 106 || in.dst.value == 107) && (R == 106 || R == 107))
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
                if (in.src[1].kind != OperandKind::SGPR ||
                    in.src[2].kind != OperandKind::SGPR) return false;
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
                     in.src[2].kind == OperandKind::Special) && in.src[2].value == R)
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
                    in.src[1].value != 125 &&
                    (in.src[1].value == R || in.src[1].value + 1 == R))
                    return false;
                break;
            default:
                return false;   // memory/branch/interp/unknown: can't bound reads of R -> assume live
        }
        if (index + 1 < ins.size()) pending.push_back(index + 1);
    }
    return true;   // every reachable path hit a redefinition/end without a read
}

std::unordered_set<uint32_t> proven_cselect_b64_low_only_pcs(
        const std::vector<Rdna2Inst>& ins) {
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
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0bu &&
            in.dst.kind == OperandKind::SGPR && in.dst.value == 106 &&
            in.src[0].kind == OperandKind::SGPR &&
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
bool is_wave64_vcc_lo_scalar_b32_candidate(const Rdna2Inst& in) {
    return is_gtav_wave64_vcc_lo_scalar_cselect(in) ||
        (in.fmt == Rdna2Format::SOP2 && sop2_is_b32_logical(in.opcode) &&
         in.dst.kind == OperandKind::SGPR && in.dst.value == 106);
}

std::unordered_set<uint32_t> proven_wave64_vcc_b32_low_only_pcs(
        const std::vector<Rdna2Inst>& ins, bool include_logical = true) {
    std::unordered_set<uint32_t> result;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        const bool candidate = include_logical
            ? is_wave64_vcc_lo_scalar_b32_candidate(in)
            : is_gtav_wave64_vcc_lo_scalar_cselect(in);
        if (candidate &&
            sgpr_dead_at_merge(ins, in.pc + in.len_dwords, 107))
            result.insert(in.pc);
    }
    return result;
}

// #2418: does this shader read SCC anywhere? Used to decide whether the fragment stage should pay for
// an exact wave vote when a mask op writes SCC. Deliberately CONSERVATIVE and whole-shader: it answers
// "could SCC ever be consumed", not "is this particular write live". A false positive costs one shader a
// subgroup-size requirement it might not have needed; a false NEGATIVE would silently drop the vote a
// consumer depends on, so every reader is listed even where the recompiler handles it elsewhere.
//
// SCC readers on RDNA2 (doc 70648): SOP2 s_cselect_b32/b64 (0x0a/0x0b) and the carry-in forms
// s_addc_u32/s_subb_u32 (0x04/0x05); SOPP s_cbranch_scc0/scc1 (0x04/0x05); SOP1 s_cmov_b32/b64
// (0x02/0x03); SOPK s_cmovk_i32 (0x02). SOPC and s_cmp_*/s_cmpk_* WRITE SCC and are not readers.
//
// SOPK was omitted in the first version of this scan and that was a genuine false negative -- the
// direction this function's own contract calls the dangerous one, since a missed reader silently
// drops the vote its consumer depends on. Caught in review of #2416. The file already documents the
// hazard at the `sopk_writes_scalar_data` exclusion note: "several SOPK ops (s_addk/s_mulk/s_cmovk/
// s_cmpk) READ or read-modify-write their dst via the implicit SIMM16".
//
// Only s_cmovk_i32 READS SCC in this space, verified rather than taken from a table --
// `llvm-mc -mcpu=gfx1030 -disassemble` over the SOPK opcode range gives 0x00 s_movk_i32,
// 0x02 s_cmovk_i32, 0x03.. s_cmpk_* (which WRITE SCC), 0x0f s_addk_i32, 0x10 s_mulk_i32,
// 0x12/0x13 s_getreg/s_setreg. Including the whole SOPK format would be conservative in the safe
// direction but costs every shader containing an s_movk -- which is nearly all of them -- a
// subgroup-size requirement it does not need.
inline bool shader_reads_scc(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        switch (in.fmt) {
            case Rdna2Format::SOP2:
                if (in.opcode == 0x04 || in.opcode == 0x05 ||
                    in.opcode == 0x0a || in.opcode == 0x0b) return true;
                break;
            case Rdna2Format::SOPP:
                if (in.opcode == 0x04 || in.opcode == 0x05) return true;
                break;
            case Rdna2Format::SOP1:
                if (in.opcode == 0x02 || in.opcode == 0x03) return true;
                break;
            case Rdna2Format::SOPK:
                if (in.opcode == 0x02) return true;   // s_cmovk_i32: conditional move ON SCC
                break;
            default: break;
        }
    }
    return false;
}

std::unordered_set<uint32_t> safe_execz_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> safe;
    // pc of the terminating s_endpgm — a forward execz whose target is here skips straight to the end.
    uint32_t end_pc = 0; bool have_end = false;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; have_end = true; break; }
    for (const auto& br : ins) {
        if (br.fmt != Rdna2Format::SOPP || br.opcode != 0x08 || br.simm16 <= 0) continue;
        const uint32_t target = br.pc + br.len_dwords + (uint32_t)br.simm16;
        bool ok = target > br.pc;
        bool target_found = false;
        for (const auto& in : ins) {
            if (in.pc == target) { target_found = true; break; }
        }
        if (!target_found) ok = false;
        // LOOP EXIT (#1183): if a backward branch that jumps to at-or-before this execz sits inside the
        // block [br, target), then this execz is a data-dependent loop's EXIT, not an if/guard-to-end —
        // even when its target happens to be s_endpgm. Linearizing it here would strand the loop's
        // back-edge as a straight-line reject; instead leave it OUT of `safe` so detect_divergent_loops
        // claims it and emit_divloop reconstructs the structured loop. Back-edges are the shapes that
        // detector recognizes: unconditional s_branch (0x02) or s_cbranch_execnz (0x09).
        bool is_loop_exit = false;
        for (const auto& bb : ins) {
            if (bb.fmt != Rdna2Format::SOPP || (bb.opcode != 0x02 && bb.opcode != 0x09)) continue;
            if (bb.pc <= br.pc || bb.pc >= target) continue;                 // must sit inside this block
            const int64_t bt = static_cast<int64_t>(bb.pc) + bb.len_dwords + bb.simm16;   // signed target
            if (bt <= static_cast<int64_t>(br.pc)) { is_loop_exit = true; break; }
        }
        if (is_loop_exit) continue;
        // Two shapes are safe to linearize (drop the branch, run the block under per-lane EXEC):
        //  * IF/ENDIF rejoining live code (target < end): only EXEC-predicated VGPR writes are safe,
        //    since scalar/VCC/memory writes past the merge would be observed by later code.
        //  * GUARD-TO-END (target == s_endpgm): the block's scalar/SGPR/VCC writes are DEAD (nothing
        //    runs after s_endpgm) and wave-uniform, memory loads are fault-free (robustBufferAccess) into
        //    predicated VGPRs, and memory stores are EXEC-predicated (conditional store). So anything is
        //    safe EXCEPT an EXP export (not EXEC-predicated → would export from inactive lanes).
        const bool guard_to_end = have_end && target >= end_pc;
        for (const auto& in : ins) {
            if (in.pc <= br.pc || in.pc >= target || in.is_end) continue;
            if (in.fmt == Rdna2Format::EXP) { ok = false; break; }   // exports are never EXEC-masked
            if (guard_to_end) continue;                              // dead-at-end / predicated / fault-free
            // IF/ENDIF rejoining live code: safe to linearize iff every write is EXEC-predicated. VGPR ALU
            // (VOP1/2/3) predicates its writes; memory ops (MIMG/MUBUF/DS) predicate their VGPR loads (and
            // are fault-free under robust access) and EXEC-predicate their stores. SGPR/VCC writes (SOP*/
            // VOPC/SMEM) are NOT predicated and would be observed past the merge -> still unsafe.
            // CARVE-OUTS: three op groups inside the "safe" VALU formats have UNPREDICATED scalar side
            // effects (emit_alu writes rs.sreg/rs.vcc/rs.sreg_bool for them, and predicate_write only
            // covers VGPRs) — on hardware the skipped block would have preserved VCC/the SGPR:
            //   VOP1 0x02 v_readfirstlane_b32 (writes an SGPR), VOP2 0x28-0x2A carry ops (write VCC),
            //   VOP3B 0x128-0x12A and 0x30F/0x310/0x319 (write the carry-out SGPR pair/VCC), and
            //   VOP3B v_mad_u64_u32 0x176 (its 65th-bit carry mask also lands in VCC/an SGPR pair
            //   unpredicated).
            const bool scalar_side_effect =
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02) ||
                (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A) ||
                vop3b_fresh_carry_output(in) ||
                (in.fmt == Rdna2Format::VOP3 &&
                 ((in.opcode >= 0x128 && in.opcode <= 0x12A) || in.opcode == 0x176));
            if (!scalar_side_effect &&
                (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3 ||
                 in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::MUBUF ||
                 in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::DS ||
                 in.fmt == Rdna2Format::FLAT ||
                 sopp_is_noop(in))) {
                continue;
            }
            if (scalar_side_effect) { ok = false; break; }
            // A pure scalar move (s_mov_b32/b64: no SCC/VCC/memory side effect) whose destination SGPR is
            // DEAD at the merge is also safe to linearize: the unconditional write is overwritten before any
            // later read, so masked-off lanes never observe it. (The tonemap/sRGB divergent-ifs load a scalar
            // constant used only within the block and reset it right after — see shader 033.)
            if (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x03 || in.opcode == 0x04) &&
                in.dst.kind == OperandKind::SGPR) {
                const bool b64 = (in.opcode == 0x04);
                // dst must be a plain SGPR (s0..s105). SOP1 destinations also decode EXEC/VCC/M0
                // (106/107/124/126/127) as SGPR-kind, but a move into those has wave-wide side effects read
                // implicitly (not via an SGPR operand) — they can never be proven dead, so exclude them.
                const int hi = in.dst.value + (b64 ? 1 : 0);
                if (hi <= 105 && sgpr_dead_at_merge(ins, target, in.dst.value) &&
                    (!b64 || sgpr_dead_at_merge(ins, target, in.dst.value + 1))) continue;
            }
            // A scalar LOAD (s_load/s_buffer_load) inside the block is likewise safe iff every SGPR it
            // writes is DEAD at the merge: the load itself is wave-uniform and fault-free, and a result
            // overwritten before any read is never observed by post-merge code. This is the divergent
            // lighting/fog block's `s_buffer_load_dwordx2 vcc, …` scratch-load shape (DOLL VS, #273);
            // sgpr_dead_at_merge understands VCC's implicit readers, so vcc-targeted loads qualify.
            if (in.fmt == Rdna2Format::SMEM) {
                uint32_t n = 0;
                switch (in.opcode) {
                    case 0x0: case 0x8: n = 1;  break;   case 0x1: case 0x9: n = 2;  break;
                    case 0x2: case 0xA: n = 4;  break;   case 0x3: case 0xB: n = 8;  break;
                    case 0x4: case 0xC: n = 16; break;   default: break;
                }
                if (n) {
                    bool all_dead = true;
                    for (uint32_t k = 0; k < n && all_dead; k++)
                        all_dead = sgpr_dead_at_merge(ins, target, in.dst.value + (int)k);
                    if (all_dead) continue;
                }
            }
            ok = false;
            break;
        }
        if (ok) safe.insert(br.pc);
    }
    return safe;
}

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
}

// FRAGMENT alpha-test / clip() discard via a SCALAR BRANCH. A per-lane condition (v_cmp -> VCC) is folded
// into a saved-EXEC survivor mask by a 64-bit wave-mask op (s_and/s_andn2_b64 sDST,sDST,vcc — which on HW
// sets SCC = "any lane survives"), then `s_cbranch_scc0 <fwd>` skips the shading when NO lane survives; the
// block then narrows EXEC (s_wqm exec, sDST) and shades, and the export lowers to an OpKill of the failed
// lanes. Per-invocation the wave early-out is a pure optimization — running the block for a lane that will
// be OpKill'd at export is harmless — so the branch is safe to LINEARIZE (drop it, run the block straight-
// line) exactly like a forward s_cbranch_execz. Recognize it by the mask op IMMEDIATELY preceding the
// branch (hints ignored). Returns the pc of each such branch. This is the shape of every Unity cutout /
// text draw; rejecting the branch dropped all of them (The Messenger's missing cutscene text, #102).
std::unordered_set<uint32_t> mask_test_branches(const std::vector<Rdna2Inst>& ins,
                                                bool allow_b32_masks = false) {
    std::unordered_set<uint32_t> out;
    std::unordered_set<uint32_t> b32_mask_writer_pcs;
    std::unordered_set<uint32_t> block_entries;
    size_t active_count = 0;
    while (active_count < ins.size() && !ins[active_count].is_end) ++active_count;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < active_count; ++i) index_by_pc.emplace(ins[i].pc, i);
    auto is_scalar_branch = [](const Rdna2Inst& candidate) {
        return candidate.fmt == Rdna2Format::SOPP &&
            (candidate.opcode == 0x02 ||
             (candidate.opcode >= 0x04 && candidate.opcode <= 0x09));
    };
    auto branch_target_pc = [](const Rdna2Inst& candidate) -> int64_t {
        return static_cast<int64_t>(candidate.pc) + candidate.len_dwords + candidate.simm16;
    };
    for (size_t i = 0; i < active_count; ++i) {
        if (!is_scalar_branch(ins[i])) continue;
        const int64_t target = branch_target_pc(ins[i]);
        if (target >= 0 && target <= UINT32_MAX)
            block_entries.insert(static_cast<uint32_t>(target));
    }

    // Transfer one instruction's one-word Wave32-mask provenance. The domain is a MUST fact: an
    // SGPR is present only when every path reaching this instruction proves that it contains a
    // complete Wave32 lane mask. This preserves masks through a conditional fall-through (the live
    // Astro alpha-test preamble) while intersecting away a compare performed on only one predecessor.
    auto transfer_b32_masks = [&](const Rdna2Inst& in,
                                  const std::unordered_set<int>& incoming,
                                  bool* mask_writer) {
        std::unordered_set<int> masks = incoming;
        auto tracked_mask_source = [&](const Operand& source) {
            return source.value == 126 ||
                   ((source.kind == OperandKind::SGPR ||
                     source.kind == OperandKind::Special) &&
                    incoming.contains(source.value));
        };
        bool writes_b32_mask = false;
        int mask_dst = -1;
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            mask_dst = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
            writes_b32_mask = true;
        } else if (in.fmt == Rdna2Format::SOP1 &&
                   (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                    sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
            writes_b32_mask = tracked_mask_source(in.src[0]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::SOP2 &&
                   (in.opcode == 0x0a ||
                    (in.opcode >= 0x0e && in.opcode <= 0x1c &&
                     (in.opcode & 1u) == 0))) {
            const bool real_mask_source = tracked_mask_source(in.src[0]) ||
                                          tracked_mask_source(in.src[1]);
            auto representable = [&](const Operand& source) {
                return tracked_mask_source(source) || source.kind == OperandKind::InlineInt;
            };
            writes_b32_mask = real_mask_source && representable(in.src[0]) &&
                              representable(in.src[1]) && in.dst.value != 127;
            mask_dst = in.dst.value;
        } else if (in.fmt == Rdna2Format::VOP3 &&
                   in.opcode >= 0x128 && in.opcode <= 0x12a &&
                   in.sdst.kind == OperandKind::SGPR) {
            writes_b32_mask = tracked_mask_source(in.src[2]);
            mask_dst = in.sdst.value;
        } else if (vop3b_fresh_carry_output(in)) {
            writes_b32_mask = true;
            mask_dst = in.sdst.value;
        }

        auto erase_written_words = [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                masks.erase(base + static_cast<int>(word));
        };
        const uint32_t dst_width = scalar_write_width(in);
        if (dst_width) erase_written_words(in.dst.value, dst_width);
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
            erase_written_words(in.dst.value, 1);
        if (vop3_writes_mask_sdst(in))
            erase_written_words(in.sdst.value,
                ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
                 vop3b_fresh_carry_output(in)) ? 1u : 2u);
        if (writes_b32_mask && mask_dst >= 0 && mask_dst != 126)
            masks.insert(mask_dst);
        if (mask_writer) *mask_writer = writes_b32_mask && mask_dst >= 0 && mask_dst != 126;
        return masks;
    };

    if (allow_b32_masks && active_count) {
        std::vector<std::unordered_set<int>> incoming(active_count);
        std::vector<bool> reachable(active_count, false);
        std::vector<size_t> worklist{0};
        reachable[0] = true;
        auto merge_into = [&](size_t successor, const std::unordered_set<int>& masks) {
            if (!reachable[successor]) {
                incoming[successor] = masks;
                reachable[successor] = true;
                worklist.push_back(successor);
                return;
            }
            std::unordered_set<int> intersection;
            for (int reg : incoming[successor])
                if (masks.contains(reg)) intersection.insert(reg);
            if (intersection != incoming[successor]) {
                incoming[successor] = std::move(intersection);
                worklist.push_back(successor);
            }
        };
        for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
            const size_t i = worklist[cursor];
            const auto outgoing = transfer_b32_masks(ins[i], incoming[i], nullptr);
            auto merge_pc = [&](int64_t pc) {
                if (pc < 0 || pc > UINT32_MAX) return;
                const auto found = index_by_pc.find(static_cast<uint32_t>(pc));
                if (found != index_by_pc.end()) merge_into(found->second, outgoing);
            };
            if (is_scalar_branch(ins[i])) {
                merge_pc(branch_target_pc(ins[i]));
                if (ins[i].opcode != 0x02 && i + 1 < active_count)
                    merge_into(i + 1, outgoing);
            } else if (i + 1 < active_count) {
                merge_into(i + 1, outgoing);
            }
        }
        for (size_t i = 0; i < active_count; ++i) {
            if (!reachable[i]) continue;
            bool writes_mask = false;
            (void)transfer_b32_masks(ins[i], incoming[i], &writes_mask);
            if (writes_mask) b32_mask_writer_pcs.insert(ins[i].pc);
        }
    }

    const Rdna2Inst* prev = nullptr;
    for (const auto& in : ins) {
        if (block_entries.contains(in.pc)) prev = nullptr;
        if (in.is_end) break;
        if (sopp_is_noop(in)) continue;                         // hints don't break the mask->branch pairing
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05) && in.simm16 > 0) {
            // scc0/scc1 FORWARD branch whose SCC was set by a 64-bit wave-mask op: SOP2 s_and_b64(0x0f) /
            // s_or_b64(0x11) / s_xor_b64(0x13) / s_andn2_b64(0x15), or SOP1 s_and/or_saveexec_b64 (0x24/0x25),
            // writing a plain SGPR-pair kill mask. (A branch on a v_cmp/s_cmp SCC is a REAL uniform-if and is
            // NOT matched — prev would be a SOPC/ALU, not a mask op.)
            if (prev) {
                const bool b64_mask_sop2 = prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0f || prev->opcode == 0x11 ||
                     prev->opcode == 0x13 || prev->opcode == 0x15);
                // The same compiler alpha-test shape uses the one-word logical family when
                // SPI_PS_IN_CONTROL proves Wave32. The B32 result is then the complete wave mask,
                // and SCC is the same whole-wave early-out vote as for the B64 form.
                const bool b32_mask_sop2 = allow_b32_masks && prev->fmt == Rdna2Format::SOP2 &&
                    (prev->opcode == 0x0e || prev->opcode == 0x10 ||
                     prev->opcode == 0x12 || prev->opcode == 0x14) &&
                    b32_mask_writer_pcs.contains(prev->pc);
                const bool mask_sop2 = b64_mask_sop2 || b32_mask_sop2;
                bool mask_saveexec = prev->fmt == Rdna2Format::SOP1 &&
                                     (prev->opcode == 0x24 || prev->opcode == 0x25 ||
                                      prev->opcode == 0x37);
                // The kill mask may live in a plain SGPR pair (s0..s105) OR in VCC itself — DOLL's
                // alpha-cull PS does `s_andn2_b64 vcc, exec, vcc; s_cbranch_scc0 <null-export>` then
                // `s_mov_b64 exec, vcc; export`. The SOP2 dst field decodes VCC_LO as SGPR 106, and
                // emit_alu's mask ops route a 106/107 dst to rs.vcc, so the same linearization holds
                // (the branch is a whole-wave early-out; per-invocation the export's OpKill covers it).
                if ((mask_sop2 || mask_saveexec) && prev->dst.kind == OperandKind::SGPR && prev->dst.value <= 106)
                    out.insert(in.pc);
            }
        }

        prev = is_scalar_branch(in) ? nullptr : &in;
    }
    return out;
}

// Some GFX10 pixel shaders leave scheduled 64-bit mask operations whose VCC/SCC results feed only
// other dead mask operations and are overwritten before an observable read. Astro Bot's SSAO shader does this with
// `s_and_b64 vcc, s[0:1], vcc`, where s[0:1] is also a live T# descriptor; attempting to reinterpret
// the descriptor bits as a per-lane mask is both impossible in descriptor-backed SPIR-V and pointless.
// Elide only the mechanically proven dead form: the shader has no SCC consumer anywhere, and CFG
// liveness proves the VCC pair cannot reach a non-mask read before redefinition. This deliberately
// does not become a general scalar-pair-to-wave-mask fallback.
std::unordered_set<uint32_t> dead_wave_mask_writes(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        const bool reads_scc =
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05)) ||
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == 0x04 || in.opcode == 0x05 ||
              in.opcode == 0x0a || in.opcode == 0x0b)) ||
            (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x05 || in.opcode == 0x06)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (reads_scc) return {};
    }
    auto is_mask = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 ||
             in.opcode == 0x15 || in.opcode == 0x17 || in.opcode == 0x19 ||
             in.opcode == 0x1b || in.opcode == 0x1d) &&
            (in.dst.value == 106 || in.dst.value == 107);
    };
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) by_pc[ins[i].pc] = i;
    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03;
        if (!branch || in.opcode != 0x02) {
            if (i + 1 < ins.size()) succ[i].push_back(i + 1);
        }
        if (branch) {
            const uint32_t target = in.pc + in.len_dwords +
                                    static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
            auto it = by_pc.find(target);
            if (it == by_pc.end()) return {}; // malformed/unbounded CFG: make no dead-write claim
            succ[i].push_back(it->second);
        }
    }
    auto uses = [&](const Rdna2Inst& in) -> uint8_t {
        uint8_t bits = 0;
        for (uint8_t k = 0; k < in.n_src; ++k) {
            if (in.src[k].kind != OperandKind::SGPR &&
                in.src[k].kind != OperandKind::Special) continue;
            if (in.src[k].value == 106) bits |= 1;
            if (in.src[k].value == 107) bits |= 2;
        }
        if (is_mask(in) && bits) bits = 3; // every modeled mask logical reads a full B64 pair
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a))) bits |= 3;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07)) bits |= 3;
        return bits;
    };
    auto defs = [&](const Rdna2Inst& in) -> uint8_t {
        if (is_mask(in)) return 3;
        // A cmpx writes EXEC and has NO VCC destination, so it must NOT count as defining VCC —
        // the decoder gives every VOPC e32 dst = 106 (VCC_LO), so without this exclusion a cmpx
        // would satisfy both conjuncts and record a phantom definition. A private copy of the
        // windows here listed three of the six, so every v_cmpx_*_f64/_i64/_u64/_u16 was recorded
        // as defining VCC; a preceding live `s_and_b64 vcc` then looked overwritten before use,
        // was classified dead and ELIDED, leaving stale VCC at the real consumer with no
        // diagnostic. Kernel 32r13v pins it. Use the one shared predicate (#2120).
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            (in.dst.value == 106 || in.dst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a)
            return 3;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.sdst.value == 106 || in.sdst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::SOP1) {
            if (in.dst.value == 106) return in.opcode == 0x04 ? 3 : 1;
            if (in.dst.value == 107) return 2;
        }
        if (in.fmt == Rdna2Format::SMEM) {
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n=1; break; case 0x1: case 0x9: n=2; break;
                case 0x2: case 0xa: n=4; break; case 0x3: case 0xb: n=8; break;
                case 0x4: case 0xc: n=16; break; default: break;
            }
            uint8_t bits = 0;
            if (n && in.dst.value <= 106 && 106 < in.dst.value + static_cast<int>(n)) bits |= 1;
            if (n && in.dst.value <= 107 && 107 < in.dst.value + static_cast<int>(n)) bits |= 2;
            return bits;
        }
        return 0;
    };
    // Least-fixed-point dataflow rooted only in observable (non-candidate) VCC reads. A mask
    // candidate propagates liveness to its B64 input only when its output is itself live; this also
    // removes dead self-dependent mask chains inside loops without mistaking the cycle for a use.
    std::vector<uint8_t> live_in(ins.size(), 0), live_out(ins.size(), 0);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ri = ins.size(); ri-- > 0;) {
            uint8_t out = 0;
            for (size_t s : succ[ri]) out |= live_in[s];
            const uint8_t def = defs(ins[ri]), use = uses(ins[ri]);
            const uint8_t propagated_use = is_mask(ins[ri]) && !(out & def) ? 0 : use;
            const uint8_t in = propagated_use | (out & static_cast<uint8_t>(~def));
            if (out != live_out[ri] || in != live_in[ri]) {
                live_out[ri] = out; live_in[ri] = in; changed = true;
            }
        }
    }
    std::unordered_set<uint32_t> dead;
    for (size_t i = 0; i < ins.size(); ++i)
        if (is_mask(ins[i]) && !(live_out[i] & 3)) dead.insert(ins[i].pc);
    return dead;
}

// WATERFALL loops (#273 — DOLL's skinned scene VS): the readfirstlane-uniformize idiom
//   L: v_readfirstlane s4, vIDX ; v_cmpx_eq_u32 s4, vIDX ; s_mov m0, s4 ; v_movrels …
//      s_andn2_b64 REMAIN, REMAIN, exec ; s_mov_b64 exec, REMAIN ; s_cbranch_scc1 L
// iterates once per DISTINCT per-lane index, with EXEC selecting the matching lanes. In the
// per-invocation model each lane's iteration IS its own: readfirstlane returns MY value, the cmpx
// keeps me active, and the andn2 leaves MY remaining-mask empty — so the loop body runs EXACTLY
// ONCE per invocation and the backward branch (whose SCC = "any lane remains", a cross-lane
// reduction) is never taken. A BACKWARD s_cbranch_scc0/scc1 whose SCC was produced by a 64-bit
// wave-mask op is therefore safe to DROP (linearize), exactly like the forward kill-mask branches.
// The SCC producer may sit a few instructions back (the shape interposes `s_mov_b64 exec, REMAIN`,
// which does not write SCC) — walk back over non-SCC-writing instructions to find it.
// CONFIDENCE: MED — exec-diff kernel T19 locks the once-through semantics.
std::unordered_set<uint32_t> waterfall_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> out;
    for (size_t i = 0; i < ins.size(); i++) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || (in.opcode != 0x04 && in.opcode != 0x05) || in.simm16 >= 0)
            continue;
        // Walk back to the most recent SCC-writing instruction, skipping ops known not to touch SCC
        // (scalar moves, all vector ALU incl. VOPC — it writes VCC/EXEC, not SCC — memory, hints).
        for (size_t j = i; j-- > 0;) {
            const Rdna2Inst& p = ins[j];
            bool skip_ok =
                (p.fmt == Rdna2Format::SOP1 && (p.opcode == 0x03 || p.opcode == 0x04)) ||   // s_mov_b32/b64
                p.fmt == Rdna2Format::VOP1 || p.fmt == Rdna2Format::VOP2 || p.fmt == Rdna2Format::VOP3 ||
                p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::MIMG ||
                p.fmt == Rdna2Format::MUBUF || p.fmt == Rdna2Format::MTBUF ||
                p.fmt == Rdna2Format::FLAT ||
                sopp_is_noop(p);
            if (skip_ok) continue;
            // The SCC producer: a 64-bit wave-mask logical op (s_and/or/xor/andn2_b64, SCC = result!=0).
            if (p.fmt == Rdna2Format::SOP2 &&
                (p.opcode == 0x0f || p.opcode == 0x11 || p.opcode == 0x13 || p.opcode == 0x15) &&
                p.dst.kind == OperandKind::SGPR && p.dst.value <= 106)
                out.insert(in.pc);
            break;   // any other SCC producer (s_cmp etc.): a REAL loop condition — not a waterfall
        }
    }
    return out;
}

// A recognized COUNTED loop (the game's MSAA-resolve / accumulation shape). Two accepted shapes:
//   (a) TOP-tested: a single backward UNCONDITIONAL s_branch (the back-edge) to a header, with
//       exactly one forward s_cbranch_scc0/scc1 exit inside the body — `for (...) {...}`.
//   (b) BOTTOM-tested (do-while): the back-edge is ITSELF a backward s_cbranch_scc0/scc1; the loop
//       continues while its SCC condition holds and exits by falling through — `do {...} while(...)`.
//       This is what the compiler emits for a fixed-trip texture-accumulation loop whose whole body
//       (including the s_cmp that sets the exit SCC) runs before the test. Alex Kidd's (PPSA02664)
//       light/blur-accumulation pixel shaders use it, and rejecting it dropped their draws (#320).
// Both are lowered by ONE emitter: shape (b) is modeled as exit_branch_pc == backedge_pc, so the
// emitter emits the entire body as the "condition region" (runs every iteration incl. the exiting
// one — exactly do-while), tests SCC at the bottom, and its "body region" (exit_branch+1..backedge)
// is empty. Anything more complex (nested loops, VCC/EXECNZ exits, multiple back-edges, mid-loop
// s_branch) is rejected — the recompiler then falls back to the straight-line path (as before).
struct CountedLoop {
    bool found = false;
    uint32_t header_pc = 0;       // loop header (target of the back-edge; condition eval starts here)
    uint32_t exit_branch_pc = 0;  // the s_cbranch_scc0/scc1 exit test (== backedge_pc for a do-while)
    uint32_t backedge_pc = 0;     // the backward branch (unconditional s_branch, or the scc back-edge)
    uint32_t exit_pc = 0;         // first pc after the loop (== backedge_pc + its length)
    bool exit_on_scc0 = true;     // exit-test polarity: true ⇔ the loop continues while SCC==1 (exits on SCC==0)
};
inline uint32_t branch_target(const Rdna2Inst& in) { return in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16; }

CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins) {
    CountedLoop L;
    // The single backward branch that closes the loop. A conditional scc0/scc1 back-edge is a
    // bottom-tested do-while; an unconditional s_branch is a top-tested loop.
    // A backward s_cbranch_scc0/scc1 whose SCC is a 64-bit wave-mask reduction is a WATERFALL
    // (once-through per invocation), NOT a loop back-edge — its cross-lane SCC has no representable
    // per-lane value, and every emit path already consumes it via safe_branches/the linearizer.
    // Waterfalls are excluded from the back-edge count entirely: SELECTING one would feed the loop
    // emitter a poisoned SCC exit and mis-lower the divergent body (T19), and merely COUNTING one
    // would reject a counted loop that legitimately coexists with a waterfall elsewhere in the same
    // shader — a previously-recompilable combination (kernel 44d locks it).
    const std::unordered_set<uint32_t> wf = waterfall_branches(ins);
    const Rdna2Inst* back = nullptr; int nback = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.simm16 >= 0) continue;
        if (in.opcode == 0x02 ||
            ((in.opcode == 0x04 || in.opcode == 0x05) && !wf.count(in.pc))) { back = &in; nback++; }
    }
    if (nback != 1) return L;                       // 0 -> no loop; >1 -> nested/multiple (unhandled)
    const bool do_while = (back->opcode != 0x02);   // conditional back-edge => bottom-tested
    const uint32_t header = branch_target(*back);
    bool header_ok = false;
    for (const auto& in : ins) if (in.pc == header && in.pc < back->pc) { header_ok = true; break; }
    if (!header_ok) return L;
    const uint32_t exit_pc = back->pc + back->len_dwords;
    const Rdna2Inst* exitbr = nullptr; int nexit = 0;
    for (const auto& in : ins) {                    // scan the loop body [header, back-edge]
        if (in.pc < header || in.pc > back->pc || in.fmt != Rdna2Format::SOPP) continue;
        if (&in == back) continue;                  // the back-edge is classified above, not here
        switch (in.opcode) {
            case 0x04: case 0x05:                   // s_cbranch_scc0 / scc1 — must be the single exit
                if (branch_target(in) != exit_pc) return L;
                exitbr = &in; nexit++; break;
            case 0x02:                              // any other s_branch -> reject
                return L;
            case 0x06: case 0x07: case 0x09:        // vcc / execnz branches -> reject
                return L;
            case 0x08: default: break;              // execz (forward, predication-handled) / hints ok
        }
    }
    if (do_while) {
        // Bottom-tested: the conditional back-edge IS the exit test; no separate forward exit is
        // allowed (a do-while with an inner uniform-if stays unsupported and rejects, conservatively).
        // The back-edge is TAKEN (loops) while its SCC condition holds. exit_on_scc0 drives the
        // emitter's loop_cond = (exit_on_scc0 ? SCC : !SCC), i.e. "continue when this is true": an
        // s_cbranch_scc1 back-edge continues while SCC==1 (exit_on_scc0=true); scc0 continues while
        // SCC==0 (exit_on_scc0=false).
        if (nexit != 0) return L;
        L.found = true; L.header_pc = header; L.exit_branch_pc = back->pc; L.backedge_pc = back->pc;
        L.exit_pc = exit_pc; L.exit_on_scc0 = (back->opcode == 0x05);
        return L;
    }
    if (nexit != 1) return L;
    L.found = true; L.header_pc = header; L.exit_branch_pc = exitbr->pc; L.backedge_pc = back->pc;
    L.exit_pc = exit_pc; L.exit_on_scc0 = (exitbr->opcode == 0x04);
    return L;
}

// MULTI-LOOP EXEC/VCC/SCC EXITS (#273/#615/#1554 — post-process and light-accumulation PSes,
// title-composite content producers, and compute histogram loops). The divergent compiled shape:
//     header: <exec recompute: v_cmpx_.. counter,bound  |  v_cmp..;s_andn2_b64 exec,exec,vcc>
//             s_cbranch_execz EXIT                      ; leave when no lane remains
//     body:   ... (nested forward-execz if regions, saveexec/restore, scalar counter++) ...
//             s_branch header                           ; backward unconditional back-edge
//     EXIT:   s_mov_b64 exec, <saved>
// Fragment semantics are exact: a forced 64-lane native subgroup vote keeps the complete guest wave
// in the loop until every EXEC/VCC bit clears. Cleared lanes remain synchronization participants while
// EXEC predicates their vector writes, so scalar state advances exactly as on RDNA2. Vertex and the
// narrow structured-compute path retain the older per-invocation lowering and its uniformity guards.
//
// A second flavor (DOLL's scalar-indexed unroll): the back-edge is a backward s_cbranch_EXECNZ and
// the header IS the execz exit (empty condition region); the body may carry an extra forward
// vccz/execz BREAK to the exit. A break lowers as a plain forward IF over the remainder of the body
// (skip-to-backedge): the broken lane's EXEC bit is already clear (the compiled break condition
// mirrors the exec recompute), so the next header check exits it. Fragment wave64 additionally
// supports a VCCZ break with an unconditional back-edge by carrying the uniform continue vote to the
// loop latch and branching the complete wave directly to the loop merge.
struct DivLoop {
    enum class Condition : uint8_t { Exec, Vcc, Scc };
    uint32_t header_pc = 0;        // back-edge target; condition region = [header_pc, exit_branch_pc)
    uint32_t exit_branch_pc = 0;   // canonical forward execz/vccz/scc branch whose target is exit_pc
    uint32_t backedge_pc = 0;      // backward s_branch (unconditional) or s_cbranch_execnz
    uint32_t exit_pc = 0;          // backedge_pc + its length (first pc after the loop)
    std::vector<uint32_t> break_pcs;   // extra forward vccz/execz -> exit_pc (lowered as body ifs)
    bool direct_exec_breaks = false;   // unconditional back-edge: an interior execz exits directly
    bool direct_wave_breaks = false;   // fragment wave64: an interior vccz exits the complete wave
    bool bottom_tested = false;        // s_cbranch_execnz back-edge is the condition (do-while)
    Condition condition = Condition::Exec;
    // EXECZ/VCCZ and SCC0 all continue while their represented predicate is set. SCC1 is the one
    // admitted opposite-polarity canonical exit: it leaves the loop while SCC is set.
    bool continue_on_set = true;
};

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
static bool cannot_write_vcc(const Rdna2Inst& in) {
    auto sgpr_dst_misses_vcc = [&](uint32_t n_dwords) {
        if (in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) return true;
        const int lo = in.dst.value, hi = in.dst.value + (int)n_dwords - 1;
        return hi < 106 || lo > 107;
    };
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VINTRP:
        case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
        case Rdna2Format::DS:    case Rdna2Format::FLAT:
        case Rdna2Format::EXP:                                     // VGPR/memory dsts only
        case Rdna2Format::SOPP:                                    // branches/hints/waitcnt: no dst
        case Rdna2Format::SOPC:  return true;                      // writes SCC only
        case Rdna2Format::VOP2:  return in.opcode < 0x28 || in.opcode > 0x2a;
        case Rdna2Format::VOP3P: return true;                      // packed VALU: VGPR dst, no carry
        case Rdna2Format::VOP3:
            // gfx10 v_writelane_b32 writes only its VGPR lane slot. v_readlane_b32 writes the
            // scalar destination encoded in the decoder's dst field, so it is VCC-preserving as
            // long as that destination does not overlap s106:s107. UE4 schedules long scalar-spill
            // sequences of these two ops between a uniform VOPC and its s_cbranch_vccz.
            if (in.opcode == 0x361) return true;
            if (in.opcode == 0x360) return sgpr_dst_misses_vcc(1);
            if (in.opcode == 0x176)
                return in.sdst.value != 106 && in.sdst.value != 107; // v_mad_u64_u32 carry-out
            // The remaining VOP3B ops inside the plain-VALU window — v_div_scale_f32/f64 (0x16D/
            // 0x16E) and v_mad_i64_i32 (0x177) — write a scalar carry/flag SDST that may be VCC;
            // their VGPR dst would otherwise satisfy sgpr_dst_misses_vcc. Stop the walk for them.
            if (in.opcode == 0x16D || in.opcode == 0x16E || in.opcode == 0x177) return false;
            return in.opcode >= 0x140 && in.opcode < 0x300 &&
                   sgpr_dst_misses_vcc(1);                         // plain-VALU window only
        case Rdna2Format::SOP1: case Rdna2Format::SOP2: case Rdna2Format::SOPK:
            return sgpr_dst_misses_vcc(2);                         // conservative 64-bit pair
        case Rdna2Format::SMEM: {
            uint32_t n = 0;
            switch (in.opcode & 7) { case 0: n = 1; break; case 1: n = 2; break; case 2: n = 4; break;
                                     case 3: n = 8; break; case 4: n = 16; break; default: return false; }
            return sgpr_dst_misses_vcc(n);
        }
        default: return false;                                     // VOPC / unknown: stops the walk
    }
}

bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    const Rdna2Inst* compare = nullptr;
    for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
        if (it->pc >= branch_pc) continue;
        if (sopp_is_noop(*it)) continue;
        if (it->fmt == Rdna2Format::VOPC) { compare = &*it; break; }
        if (cannot_write_vcc(*it)) continue;   // (#590) look past provably-VCC-preserving instrs —
        break;                                 // VCC's value is frozen at the compare; only a real
                                               // rewrite between compare and branch invalidates it
    }
    if (!compare || compare->fmt != Rdna2Format::VOPC || vopc_is_cmpx(compare->opcode)) return false;

    std::function<bool(const Operand&, uint32_t)> uniform_operand_at;
    uniform_operand_at = [&](const Operand& operand, uint32_t use_pc) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special ||
            operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true; // scalar-register sources (including VCC halves) broadcast one wave value
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= use_pc) continue;
            if (operand.value == rdna2_tfe_status_vgpr(*it)) return false;
            const uint32_t writes = rdna2_vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes) continue;
            if (it->dst.value != operand.value || it->has_dpp) return false;
            // The write makes every currently active lane uniform. Keep the original conservative
            // EXEC rule: any later mask change could expose a lane that retained an older value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    rdna2_instruction_may_change_exec(between)) return false;

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

// Returns the loops in header-pc order, or {} when any backward branch doesn't fit the shape.
// `safe` carries the linearized branches (waterfalls etc.) which are not loop back-edges.
//
// An empty result has TWO causes that the caller cannot tell apart, and neither can the log: no
// eligible backward branch was found at all (the `out.empty()` early return below, which reaches
// none of the reject sites), or a shape check refused. Callers conflate them deliberately --
// `(cf_rejected || Ls.empty())` -- so this distinction matters only to a reader of the diagnostic,
// which is why the reject reporter below prints on exactly one of those two paths.
std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe,
                                            bool exact_fragment_wave_breaks = false,
                                            const RecompileDiagnosticContext& diagnostic = {},
                                            const char* caller_role = "emit") {
    std::vector<DivLoop> out;
    // PROSPER_DBG: name WHICH rejection discards the loops. Every `return {}` here throws away
    // EVERY loop in the shader, not just the one that failed. What the caller then does with an
    // empty list is NOT fixed -- see the note on `caller_role` below -- but one of the outcomes is
    // the whole-stream CFG dispatcher, which EMULATES the loop instead of emitting a real SPIR-V
    // one. So one unhandled shape silently changes how an entire program executes. Static reading can
    // narrow a symptom to this function in minutes and then stall, because the sites are
    // indistinguishable from outside; this makes them distinguishable.
    //
    // Do NOT read an absent line as "this function was never called for that program". Absence has
    // three other causes: the `out.empty()` early return (no eligible backward branch, so no site is
    // reached), a caller that supplies no diagnostic context (suppressed below), and PROSPER_DBG
    // itself, which desyncs a timing-dependent pad script badly enough that a routed run may never
    // reach the dispatch you wanted to observe (see the note at the top of this file). Whether a
    // given caller consults this function at all is a question for the CALL SITE, not for the log.
    // Reported ONCE per (program, cause, pc). A program is recompiled or re-looked-up many times
    // over a run, so an unfiltered print repeats the same rejection -- measured on a GTA V gameplay
    // run, the busiest single site emitted 70 lines. A log whose line count far exceeds its finding
    // count invites exactly one mistake: quoting the line count as a finding count. De-duplicating
    // here means the raw count is the honest statistic.
    //
    // Reported ONLY when the caller supplied a real program address. Both graphics stages have no
    // diagnostic parameter at all and so stay silent: they would report `program=0x0`, which is
    // wrong twice over -- the address is not a program, and the de-dup key would merge genuinely
    // different shaders rejecting at the same cause and pc into one line, making a count of
    // distinct programs read LOW. An unattributed line is not a weaker finding, it is a wrong one.
    //
    // The message states ONLY what is certain at this point, and deliberately says nothing about
    // what happens next. It is tempting to add "and the stream then falls back to the CFG
    // dispatcher" -- an earlier version did, and so did the first attempt at THIS comment, which
    // asserted the opposite outcome just as confidently and was equally wrong.
    //
    // The honest statement is that the outcome is not determined here, and this comment will not
    // enumerate the ways it can go -- deliberately, because two drafts already got an enumeration
    // wrong in opposite directions.
    //
    // What is checkable and sufficient: an empty list is ONE input among several to a decision taken
    // in `emit_body`, whose branches carry predicates over values not visible here (whether the
    // dispatcher is permitted at all for this sub-stream, whether a guest barrier made it unsafe,
    // the branch count, the presence of particular ops). At least one caller withholds the
    // dispatcher outright, so "refused loop" and "loop emulated" are not the same claim. That is as
    // far as this function can see, and it is far enough to know the consequence is not its to
    // state. Plumbing those values in purely to keep a sentence would be the wrong trade: the
    // sentence is what should go.
    //
    // (Found reviewing #2695, which fixed the same over-assertion one level up. Counting both
    // earlier drafts of this comment -- one asserting the dispatcher, one asserting outright
    // rejection, one asserting a false universal over the entries -- the shared shape is that the
    // clause gets written where the CAUSE is detected rather than where the EFFECT is decided, and
    // those are different functions. Writing a *narrower* claim is not a fix if it is still a claim
    // about the other function.)
    //
    // `caller_role` is in the message AND in the de-dup key, and both halves are load-bearing.
    // Two callers reach this function with the same `safe` set and the same program, and their
    // consequences are NOT the same:
    //
    //   emit            -- the recompile. Every loop in the shader is discarded; what the stream
    //                      does next is decided by the caller, not here (see the note above).
    //   multiwave-probe -- `compute_shader_prefers_native_multiwave`, which only tests
    //                      `loops.empty()`. Nothing is discarded and no fallback happens; the
    //                      preference analysis simply proceeds as though the shader were loop-free.
    //
    // Without the role in the KEY the probe silences the recompile, because the probe runs first
    // (`gpu_executor.cpp` calls it before the recompile). Without the role in the MESSAGE the
    // surviving line states the recompile's consequence over the probe's cause -- which reads as
    // proof that a recompile discarded the loops when no recompile has happened yet. That is
    // precisely the misreading this diagnostic exists to prevent, so it must not manufacture it.
    auto divloop_reject = [&](const char* why, uint32_t pc) -> std::vector<DivLoop> {
        if (getenv("PROSPER_DBG") && diagnostic.program_address != 0) {
            static std::mutex seen_mu;
            // Keyed by string VALUE, not by pointer: every caller passes a literal today, so
            // pointer identity would happen to work, and would break silently the first time
            // someone passes a computed name.
            static std::set<std::tuple<uint64_t, std::string, uint32_t, std::string>> seen;
            std::lock_guard<std::mutex> lk(seen_mu);
            // The key space is (programs x causes x pcs x roles) and a long session keeps loading
            // new shaders, so leave nothing unbounded even on a diagnostic path. Clearing rather
            // than refusing to insert means the reporter degrades into REPEATING lines, never into
            // dropping them -- for a diagnostic, losing tidiness beats losing findings. Measured at
            // 14 tuples on a full routed GTA V run, so the cap is not expected to be reached.
            if (seen.size() >= 4096u) seen.clear();
            if (seen.emplace(diagnostic.program_address, why, pc, caller_role).second)
                std::fprintf(stderr,
                             "[divloop-reject] program=0x%llx role=%s %s at pc=%u (%s)\n",
                             (unsigned long long)diagnostic.program_address, caller_role, why, pc,
                             std::strcmp(caller_role, "emit") == 0
                                 ? "all loops discarded; what the stream does next is the caller's"
                                 : "probe only: nothing discarded, analysis proceeds as loop-free");
        }
        return {};
    };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Pass 1: each backward s_branch / s_cbranch_execnz is a candidate back-edge.
    std::vector<bool> backedge_execnz;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if ((in.opcode != 0x02 && in.opcode != 0x09) || in.simm16 >= 0) continue;
        if (safe.count(in.pc)) continue;
        DivLoop L;
        L.header_pc = branch_target(in);
        L.backedge_pc = in.pc;
        L.exit_pc = in.pc + in.len_dwords;
        if (L.header_pc >= L.backedge_pc) return divloop_reject("degenerate-loop-bounds", L.header_pc);
        if (L.exit_pc > end_pc) return divloop_reject("exit-past-end-of-program", L.header_pc);
        bool hdr_ok = false;
        for (const auto& h : ins) { if (h.pc == L.header_pc) { hdr_ok = true; break; } if (h.pc > L.header_pc) break; }
        if (!hdr_ok) return divloop_reject("header-pc-is-not-an-instruction", L.header_pc);
        out.push_back(L);
        backedge_execnz.push_back(in.opcode == 0x09);
    }
    if (out.empty()) return out;
    // Loops may be strictly DISJOINT (sequential) or PROPERLY NESTED (#590 — DOLL's post-process
    // kernel iterates an inner table loop inside an outer row loop). Pass 1 collected loops in
    // BACK-EDGE pc order (an inner back-edge precedes its outer's); re-sort by header so the list is
    // in emission order — emit_structured consumes loops by header pc, and emitting an outer body
    // recursively reaches the inner loop first because its header lies inside that range. Then
    // validate every pair as either disjoint-in-order or properly nested; partial overlap (an
    // unstructured region) still rejects loudly.
    {
        std::vector<size_t> order(out.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t c) { return out[a].header_pc < out[c].header_pc; });
        std::vector<DivLoop> sorted_loops; std::vector<bool> sorted_execnz;
        sorted_loops.reserve(out.size()); sorted_execnz.reserve(out.size());
        for (size_t i : order) { sorted_loops.push_back(out[i]); sorted_execnz.push_back(backedge_execnz[i]); }
        out.swap(sorted_loops); backedge_execnz.swap(sorted_execnz);
    }
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++) {
            const DivLoop& A = out[i]; const DivLoop& B = out[j];   // A.header_pc <= B.header_pc
            if (A.header_pc == B.header_pc) return divloop_reject("shared-header", A.header_pc); // shared header: not modeled
            const bool disjoint = B.header_pc >= A.exit_pc;
            const bool nested   = B.exit_pc <= A.backedge_pc;       // B entirely inside A's body
            if (!disjoint && !nested) return divloop_reject("partial-overlap", B.header_pc);    // partial overlap: unstructured
        }
    // Pass 2: validate each loop's interior branches and find the canonical exit + breaks. A branch
    // inside a NESTED child loop belongs to the child (which validates itself in its own pass-2
    // iteration) — skip it here, including the child's backward back-edge, which is exactly the
    // "second back-edge inside" that used to reject nesting outright.
    auto inside_nested_child = [&out](const DivLoop& L, uint32_t pc) {
        for (const auto& C : out)
            if (C.header_pc > L.header_pc && C.exit_pc <= L.backedge_pc &&
                pc >= C.header_pc && pc <= C.backedge_pc) return true;
        return false;
    };
    for (size_t li = 0; li < out.size(); li++) {
        DivLoop& L = out[li];
        const bool execnz = backedge_execnz[li];
        for (const auto& in : ins) {
            if (in.is_end || in.pc >= L.backedge_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (inside_nested_child(L, in.pc)) continue;
            switch (in.opcode) {
                case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: continue;   // hints
            }
            if (in.simm16 < 0) return divloop_reject("nested-backedge-in-body", in.pc);         // second back-edge inside -> nested loop
            if (safe.count(in.pc)) continue;                    // linearized (kill-mask / safe-execz)
            uint32_t tgt = branch_target(in);
            if (in.opcode == 0x02) {                            // forward s_branch: an else-arm terminator
                if (tgt > L.backedge_pc) return divloop_reject("else-arm-branch-leaves-body", in.pc); // may not leave the body
                continue;                                       // (validated by detect_forward_ifs)
            }
            if (tgt > L.exit_pc) return divloop_reject("conditional-branch-past-loop", in.pc);      // conditional jumping past the loop
            if (tgt == L.exit_pc) {                             // an exit test
                if (!L.exit_branch_pc) {                        // first one = the canonical exit
                    if (in.opcode == 0x08) {                    // execz -> EXIT
                        L.condition = DivLoop::Condition::Exec;
                    } else if (in.opcode == 0x06 && !execnz &&
                               (exact_fragment_wave_breaks ||
                                vcc_exit_is_wave_uniform(ins, in.pc))) {
                        // Fragment reduces the real VCC mask through a forced wave64 vote. Other
                        // structured stages require the compare to be proven uniform first.
                        L.condition = DivLoop::Condition::Vcc;
                    } else if ((in.opcode == 0x04 || in.opcode == 0x05) && !execnz) {
                        // A scalar SCC exit with an unconditional back-edge is the same canonical
                        // top-tested while loop accepted by CountedLoop. Keep it in this multi-loop
                        // region form when several disjoint/nested loops coexist in one shader.
                        // scc0 exits while SCC==0 (continue on set); scc1 has the opposite polarity.
                        L.condition = DivLoop::Condition::Scc;
                        L.continue_on_set = in.opcode == 0x04;
                    } else if (in.opcode == 0x06 && !execnz) {
                        // Reached the vccz arm but the compare was not proven wave-uniform.
                        return divloop_reject("vcc-exit-not-proven-wave-uniform", in.pc);
                    } else if (execnz) {
                        // A scalar/vcc exit test combined with an execnz back-edge: the bottom-tested
                        // lowering owns that back-edge, so this exit shape is not the one modeled.
                        return divloop_reject("scalar-exit-with-execnz-backedge", in.pc);
                    } else {
                        return divloop_reject("exit-branch-opcode-not-modeled", in.pc);
                    }
                    L.exit_branch_pc = in.pc;
                } else {                                        // later ones = breaks
                    if (in.opcode != 0x06 && in.opcode != 0x08) return divloop_reject("break-not-vccz-or-execz", in.pc); // vccz/execz only
                    // With an execnz back-edge, a cleared lane skips to the next header check. With
                    // an unconditional back-edge that could re-enable EXEC, only an EXECZ branch is
                    // exact: the emitter sends the cleared lane directly to the loop merge instead.
                    // A VCCZ break does not itself clear EXEC and remains unsupported for that shape.
                    if (!execnz && in.opcode != 0x08) {
                        // A VCCZ break does not clear per-lane EXEC, so the old lane-local loop
                        // approximation could not carry it to the merge. The fragment shell now
                        // reduces VCC over an enforced wave64 subgroup and can branch the complete
                        // guest wave directly at the loop latch. Other stages remain conservative.
                        if (!exact_fragment_wave_breaks || in.opcode != 0x06) return divloop_reject("vccz-break-unsupported-stage", in.pc);
                        L.direct_wave_breaks = true;
                    } else if (!execnz) {
                        L.direct_exec_breaks = true;
                    }
                    L.break_pcs.push_back(in.pc);
                }
            }
            // (tgt <= backedge_pc: an interior forward if — validated by detect_forward_ifs.)
        }
        if (!L.exit_branch_pc) {
            // Bottom-tested EXEC loop: the back-edge itself is `s_cbranch_execnz HEADER`, so the
            // complete body executes once before the first condition. Require the instruction
            // immediately preceding the branch to update EXEC; accepting a stale entry mask here
            // would turn an arbitrary backward branch into a do-while (and commonly an infinite
            // one). GTA V's nested lighting loop ends in V_CMPX; the regression uses S_MOV_B64
            // EXEC,0. Nested child branches were already validated and skipped above.
            if (!execnz) return divloop_reject("bottom-tested-needs-execnz-backedge", L.backedge_pc);
            const Rdna2Inst* condition_writer = nullptr;
            for (const auto& in : ins) {
                if (in.pc < L.header_pc) continue;
                if (in.pc >= L.backedge_pc) break;
                condition_writer = &in;
            }
            if (!condition_writer ||
                !rdna2_instruction_may_change_exec(*condition_writer)) return divloop_reject("bottom-tested-no-exec-writer", L.backedge_pc);
            L.exit_branch_pc = L.backedge_pc;
            L.condition = DivLoop::Condition::Exec;
            L.continue_on_set = true;
            L.bottom_tested = true;
        }
        if (!L.bottom_tested) {
            // A top-tested loop's canonical exit must be the FIRST branch, so its condition region
            // [header, exit_branch) is branch-free and can be emitted straight-line. A bottom-tested
            // loop instead puts its complete, already-validated structured body in this region.
            for (const auto& in : ins) {
                if (in.pc >= L.exit_branch_pc) break;
                if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
                if (in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03 &&
                    !safe.count(in.pc)) return divloop_reject("condition-region-not-branch-free", in.pc);
            }
        }
        // The execnz flavor's unconditional-continue lowering requires the header check to
        // immediately re-test EXEC (empty condition region) — see the shape comment.
        if (execnz && !L.bottom_tested && L.exit_branch_pc != L.header_pc) return divloop_reject("execnz-nonempty-condition-region", L.header_pc);
    }
    // A nested child must lie entirely within its parent's BODY: after the parent's canonical exit
    // test (the condition region [header, exit_branch) stays branch-free) and before its back-edge.
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++)
            if (out[j].exit_pc <= out[i].backedge_pc &&        // nested per the classification above
                !out[i].bottom_tested &&
                out[j].header_pc <= out[i].exit_branch_pc) return divloop_reject("child-header-inside-parent-condition-region", out[j].header_pc);
    // Pass 3: no branch from OUTSIDE a loop may target its interior (an unstructured entry edge).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if (in.opcode < 0x02 || in.opcode > 0x09 || in.opcode == 0x03) continue;
        uint32_t tgt = branch_target(in);
        for (const auto& L : out) {
            const bool inside_br = in.pc >= L.header_pc && in.pc <= L.backedge_pc;
            const bool inside_tgt = tgt > L.header_pc && tgt < L.exit_pc;
            if (!inside_br && inside_tgt) return divloop_reject("entry-edge-from-outside-loop", in.pc);
        }
    }
    return out;
}

// A single structured uniform IF: exactly ONE control-flow branch in the whole stream, and it is a
// FORWARD s_cbranch_scc0/scc1 (a scalar-uniform conditional — all lanes take the same path). The
// conditional block is [branch_pc+1, target_pc). Anything more (any s_branch/vcc/execz/execnz branch,
// a backward branch/loop, or a target outside the stream) is rejected → falls back to straight-line
// (which then rejects the scc branch, as before). Deliberately conservative: covers the common single
// forward uniform-if without a general structurizer.
struct ForwardIf {
    bool found = false;
    uint32_t branch_pc = 0, target_pc = 0;
    bool on_scc0 = true;   // s_cbranch_scc0/vccz/execz (branch taken == skip block when flag==0) vs scc1/vccnz
    bool on_vcc  = false;  // condition register: false = SCC, true = VCC (both are wave-uniform branches)
    bool early_out = false;// branch target was clamped to end_pc: the conditional block ENDS the shader
    // Divergent-region if (#273): the branch is a forward s_cbranch_execz over an EXEC-narrowed block
    // (v_cmpx … s_cbranch_execz — DOLL's FXAA PS). Fragment reduces EXEC across an enforced wave64;
    // other structured stages use the guarded per-invocation model. EXEC is phi'd at the merge.
    bool on_exec = false;
    // Compute only (#1554): this VCC branch was PROVED to decide identically for every wave in the
    // workgroup, so its region may contain guest barriers and nested wave votes. Left false for every
    // branch whose uniformity is not proved, which keeps the conservative per-wave treatment.
    bool uniform_workgroup = false;
    // IF/ELSE (#273): the then-arm [branch_pc+1, sb_pc) is terminated by `s_branch merge_pc` at sb_pc
    // (the instruction immediately before target_pc); the else-arm starts at target_pc and runs to
    // merge_pc (or to the enclosing region's end when merge_pc is the shared OUTER merge — DOLL's
    // color-grade if/else-if cascade, where every arm's s_branch jumps to the outermost merge).
    bool has_else = false;
    uint32_t sb_pc = 0, merge_pc = 0;
};
// allow_vcc: also accept a forward s_cbranch_vccz/vccnz. Fragment reduces the per-lane bool through an
// enforced native wave64 subgroup; vertex retains the guarded per-invocation representation. Compute
// routes accepted VCC/EXEC branches through the CFG dispatcher, whose workgroup scratch reduction spans
// the configured 32/64-lane guest wave independently of the implementation-defined host subgroup width.
// code/dwords: the raw stream, so a branch target past the first s_endpgm can be decoded and
// verified to be a genuine early-out (see below) instead of blanket-clamped.
ForwardIf detect_forward_if(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                            const uint32_t* code, size_t dwords,
                            const std::unordered_set<uint32_t>* skip = nullptr) {
    ForwardIf F; const Rdna2Inst* br = nullptr; int nbranch = 0; uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        switch (in.opcode) {
            case 0x02: case 0x08: case 0x09:                         // s_branch / execz / execnz -> reject
                return F;
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (!allow_vcc) return F;                            // compute: needs a wave-uniform VCC test
                br = &in; nbranch++; break;
            case 0x04: case 0x05:                                    // scc0 / scc1 (SCC is scalar/wave-uniform)
                if (skip && skip->count(in.pc)) break;               // alpha-test kill-mask branch: handled by
                                                                     // straight-line linearization, not a struct-if
                br = &in; nbranch++; break;
            default: break;                                          // hints (nop/waitcnt/…) are fine
        }
    }
    if (nbranch != 1 || !br) return F;
    uint32_t tgt = branch_target(*br);
    // Early-out: a forward branch PAST s_endpgm skips the rest of the shader — but only if execution
    // at the target immediately terminates. Decode from tgt and require the first non-s_nop
    // instruction to be s_endpgm (the compiled early-out shape `s_cbranch L; work; s_endpgm;
    // L: s_endpgm`). Anything else is REAL reachable code (an else-block) that the old blanket clamp
    // silently discarded — valid SPIR-V, wrong semantics (#129) — so reject instead (the caller's
    // straight-line fallback also rejects the branch, loudly). When verified, clamp the merge to
    // end_pc so the conditional block is [branch_pc+1, end_pc) and s_endpgm is emitted after the merge.
    bool early = false;
    if (tgt > end_pc && tgt != UINT32_MAX) {
        if (!code || tgt >= dwords) return F;    // target outside the decode window: can't verify
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + tgt, dwords - tgt, tail);
        bool ends_immediately = false;
        for (const auto& ti : tail) {
            if (ti.is_end) { ends_immediately = true; break; }
            if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;   // s_nop padding
            break;                               // real instruction at the target -> not an early-out
        }
        if (!ends_immediately) return F;
        tgt = end_pc; early = true;
    }
    if (tgt <= br->pc || tgt > end_pc) return F;                     // must be forward, within the stream
    F.found = true; F.branch_pc = br->pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
    F.on_scc0 = (br->opcode == 0x04 || br->opcode == 0x06);         // scc0/vccz: skip block when flag==0
    F.on_vcc  = (br->opcode == 0x06 || br->opcode == 0x07);
    return F;
}

// MULTIPLE structured uniform IFs — the generalization of detect_forward_if to N forward
// s_cbranch_scc*/vcc* branches that form a properly-NESTED (or disjoint sequential) region tree.
// Real shaders routinely carry several sequential uniform ifs (DOLL's color-grade PS: two
// `s_cbranch_vccz` blocks) or a nested guard (`vccz L1; …; vccz L2` with L2 < L1). Same per-branch
// validity rules as detect_forward_if (forward target inside the stream, or a VERIFIED early-out
// clamped to s_endpgm; scc branches in `skip` are alpha-test kill-masks handled by linearization);
// plus a structural check: blocks must nest or be disjoint — a partial overlap (branch into the
// middle of another block) is not an if-tree and rejects. Returns branches in pc order; empty =
// no/unsupported control flow (the caller falls back to straight-line, which rejects loudly at the
// branch). Compute execz uses a subgroup vote; execnz and unclaimed s_branch still reject wholesale.
// CONFIDENCE: HIGH on the structure (guarded by the phi machinery shared with the single-if path).
std::vector<ForwardIf> detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                                          const uint32_t* code, size_t dwords,
                                          const std::unordered_set<uint32_t>* skip = nullptr,
                                          const std::vector<DivLoop>* loops = nullptr,
                                          bool* rejected = nullptr,
                                          bool compute_wave_branches = false,
                                          RecompileDiagnosticContext diagnostic = {}) {
    std::vector<ForwardIf> out;
    if (rejected) *rejected = false;
    auto reject = [&]() -> std::vector<ForwardIf> { if (rejected) *rejected = true; return {}; };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    // Loop-claimed pcs (#273): back-edges + canonical exits are consumed by the loop emitter; a break
    // lowers as a forward if over the remainder of its loop's body (target clamped to the back-edge).
    auto loop_backedge = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.backedge_pc == pc) return true; return false; };
    auto loop_exit     = [&](uint32_t pc) { if (loops) for (const auto& L : *loops) if (L.exit_branch_pc == pc) return true; return false; };
    auto loop_break    = [&](uint32_t pc) -> const DivLoop* {
        if (loops) for (const auto& L : *loops)
            for (uint32_t bp : L.break_pcs) if (bp == pc) return &L;
        return nullptr;
    };
    // Pass 1: collect the conditional branches (else-arm s_branch terminators are claimed in pass 2).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        // Set when a compute vccz/vccnz if is accepted for exact guest-wave control flow. Its region
        // must remain free of operations that require every workgroup invocation to reach one common
        // synchronized phase (checked after the target/merge is known).
        bool compute_uniform_vcc = false;
        switch (in.opcode) {
            case 0x02: continue;                                     // s_branch: validated in pass 2
            case 0x09:                                               // execnz: only as a loop back-edge
                if (loop_backedge(in.pc)) continue;
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "unclaimed execnz pc=%u", in.pc);
                return reject();
            case 0x08:                                               // execz: safe-linearized predication
                if (skip && skip->count(in.pc)) continue;            // branch -> emit_alu no-ops it; else a
                if (loop_exit(in.pc)) continue;                      // loop exit test: the loop emitter's
                if (!allow_vcc && !compute_wave_branches) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "unsupported execz pc=%u", in.pc);
                    return reject();
                }
                break;                                               // compute lowers the wave-empty test with
                                                                     // OpGroupNonUniformAny(EXEC)
            case 0x06: case 0x07:                                    // vccz / vccnz
                if (loop_exit(in.pc)) continue;                      // canonical loop condition: loop emitter owns it
                if (!allow_vcc) {
                    // Compute VCC is a per-lane bool in the SPIR-V shell. The scalar branch consumes
                    // the architectural whole-wave VCCZ/VCCNZ flag. Accept it for the exact guest-wave
                    // CFG dispatcher; emit_body must never lower it through a native subgroup vote.
                    if (!compute_wave_branches) {
                        log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                                 "unsupported vcc branch pc=%u", in.pc);
                        return reject();
                    }
                    compute_uniform_vcc = true;
                }
                break;
            case 0x04: case 0x05:                                    // scc0 / scc1
                if (loop_exit(in.pc)) continue;                      // scalar loop condition: loop emitter owns it
                if (skip && skip->count(in.pc)) continue;            // kill-mask branch: linearized, not an if
                break;
            default: continue;                                       // hints
        }
        // A loop BREAK (validated vccz/execz -> exit_pc inside an execnz-back-edge loop): a plain
        // forward if skipping the REST of the body — the exec-governed continue then exits the lane.
        // A compute-uniform-accepted vcc branch can never be a break (compute loops require the
        // s_branch back-edge Vcc shape, which has no breaks) — reject the combination defensively.
        if (compute_uniform_vcc && loop_break(in.pc)) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "uniform vcc loop break pc=%u", in.pc);
            return reject();
        }
        if (const DivLoop* L = loop_break(in.pc)) {
            ForwardIf F;
            F.found = true; F.branch_pc = in.pc; F.target_pc = L->backedge_pc;
            F.on_scc0 = true;                                        // vccz/execz: skip block when flag==0
            F.on_vcc  = (in.opcode == 0x06);
            F.on_exec = (in.opcode == 0x08);
            out.push_back(F);
            continue;
        }
        uint32_t tgt = branch_target(in);
        bool early = false;
        if (tgt > end_pc && tgt != UINT32_MAX) {                     // possible early-out past s_endpgm:
            if (!code || tgt >= dwords) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "target out of range pc=%u target=%u", in.pc, tgt);
                return reject();                                     // verify the target terminates (see
            }
            std::vector<Rdna2Inst> tail;                             // detect_forward_if for the rationale)
            rdna2_walk(code + tgt, dwords - tgt, tail);
            bool ends_immediately = false;
            for (const auto& ti : tail) {
                if (ti.is_end) { ends_immediately = true; break; }
                if (ti.fmt == Rdna2Format::SOPP && ti.opcode == 0x00) continue;
                break;
            }
            if (!ends_immediately) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "nonterminal early target pc=%u target=%u", in.pc, tgt);
                return reject();
            }
            tgt = end_pc; early = true;
        }
        if (tgt <= in.pc || tgt > end_pc) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "nonforward target pc=%u target=%u", in.pc, tgt);
            return reject();                                         // must be forward, within the stream
        }
        ForwardIf F;
        F.found = true; F.branch_pc = in.pc; F.target_pc = tgt; F.early_out = (early || tgt == end_pc);
        F.on_scc0 = (in.opcode == 0x04 || in.opcode == 0x06 || in.opcode == 0x08);
        F.on_vcc  = (in.opcode == 0x06 || in.opcode == 0x07);
        F.on_exec = (in.opcode == 0x08);
        // IF/ELSE detection: the instruction immediately preceding target_pc is a FORWARD s_branch —
        // the then-arm's terminator jumping to the merge; the else-arm is [target_pc, merge). (Skip
        // for early-outs: the arm ends the shader.)
        if (!early && tgt != end_pc) {
            for (const auto& sb : ins) {
                if (sb.pc <= in.pc || sb.pc >= tgt) continue;
                if (sb.fmt != Rdna2Format::SOPP || sb.opcode != 0x02) continue;
                if (sb.pc + sb.len_dwords != tgt) continue;          // must be the arm's LAST instruction
                if (loop_backedge(sb.pc)) continue;                  // a loop's back-edge, not an else-jump
                if (sb.simm16 <= 0) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "backward else pc=%u branch=%u target=%u",
                                             in.pc, sb.pc, tgt);
                    return reject();                                 // backward else-jump: a loop, not an if
                }
                uint32_t lm = branch_target(sb);
                if (lm <= tgt || lm > end_pc) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "invalid else merge pc=%u branch=%u target=%u merge=%u",
                                             in.pc, sb.pc, tgt, lm);
                    return reject();                                 // merge must be forward, in-stream
                }
                F.has_else = true; F.sb_pc = sb.pc; F.merge_pc = lm;
                break;
            }
        }
        if (!allow_vcc && compute_wave_branches && F.on_exec) {
            // A compute execz condition is subgroup-uniform, so live scalar writes and LDS are safe
            // inside the structured arm. A workgroup barrier is not: different waves may take
            // different arms. Scalar writes into VCC_LO/HI also have an implicit mask-domain side
            // effect which the scalar scratch model cannot reconstruct; accept those only when the
            // mask is provably overwritten before any post-merge implicit read. The captured UE4
            // kernel writes VCC_LO as an integer scratch in the arm and overwrites VCC with a VOPC at
            // the merge, which this deliberately narrow proof accepts.
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                if (r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "execz pc=%u contains barrier pc=%u", in.pc, r.pc);
                    return reject();
                }

                uint32_t scalar_width = 0;
                if (r.fmt == Rdna2Format::SOP1) scalar_width = r.opcode == 0x04 ? 2u : 1u;
                else if (r.fmt == Rdna2Format::SOP2) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SOPK && sopk_writes_scalar_data(r.opcode)) scalar_width = 1;
                else if (r.fmt == Rdna2Format::SMEM) {
                    switch (r.opcode) {
                        case 0x0: case 0x8: scalar_width = 1; break;
                        case 0x1: case 0x9: scalar_width = 2; break;
                        case 0x2: case 0xA: scalar_width = 4; break;
                        case 0x3: case 0xB: scalar_width = 8; break;
                        case 0x4: case 0xC: scalar_width = 16; break;
                        default: break;
                    }
                }
                if (!scalar_width ||
                    (r.dst.kind != OperandKind::SGPR && r.dst.kind != OperandKind::Special)) continue;
                for (int vcc_half = 106; vcc_half <= 107; ++vcc_half) {
                    if (vcc_half < r.dst.value || vcc_half >= r.dst.value + (int)scalar_width) continue;
                    // PROSPER_VCC_SCALAR_DATA_MERGE=1 widens the proof from "not read" to "not read
                    // as a lane mask". The guard this loop implements is about the MASK domain --
                    // its own comment says so -- but the test it applies is total liveness, so a
                    // half that survives the merge only to be read as a 32-bit scalar dword is
                    // rejected even though the scalar model reproduces that dword exactly.
                    //
                    // Opt-in rather than default because the same physical pair being a mask on one
                    // predecessor and data on another has no runtime type tag, and this file already
                    // rejects that join deliberately. Keeping it behind a switch also makes the A/B
                    // have a lever that can be shown to have moved, which a byte-identical module
                    // has already cost this investigation once.
                    static const bool relax_vcc_scalar_data =
                        std::getenv("PROSPER_VCC_SCALAR_DATA_MERGE") != nullptr;
                    const ScalarMergeProof proof = relax_vcc_scalar_data
                        ? ScalarMergeProof::MaskDomainOnly : ScalarMergeProof::AnyRead;
                    ScalarMergeBlocker blocker;
                    if (!sgpr_dead_at_merge(ins, region_end, vcc_half, proof, &blocker)) {
                        log_recompile_diagnostic(
                            diagnostic, "compute-struct-reject", "route-decline",
                            "execz pc=%u scalar write pc=%u leaves vcc-half s%d live at merge pc=%u "
                            "blocked-by pc=%u kind=%s",
                            in.pc, r.pc, vcc_half, region_end, blocker.pc, blocker.kind);
                        return reject();
                    }
                }
            }
        }
        // A branch whose mask is built only from launch data over a full EXEC decides identically for
        // every wave of the workgroup (#1554), so its region may synchronize. Prove it before the
        // per-wave restriction below applies.
        if (compute_uniform_vcc && vcc_branch_is_workgroup_uniform(ins, in.pc)) {
            F.uniform_workgroup = true;
            compute_uniform_vcc = false;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr, "[compute-cfg] workgroup-uniform vcc branch pc=%u\n", in.pc);
        }
        if (compute_uniform_vcc) {
            // The exact branch decision is per-WAVE, so different waves in one workgroup may take
            // different arms. Ordinary LDS loads/stores/atomics are valid under that control flow:
            // they are per-invocation memory effects and impose no reconvergence requirement. A
            // guest barrier is not valid there, and synthesized wave collectives need the CFG
            // dispatcher's common phase rather than the compact structured emitter. Keep those
            // synchronized operations fail-visible here; raw LDS memory effects can remain in-arm.
            const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
            for (const auto& r : ins) {
                if (r.is_end || r.pc >= region_end) break;
                if (r.pc <= in.pc) continue;
                const bool ds_wave_collective = r.fmt == Rdna2Format::DS &&
                    (r.opcode == 0x35 || r.opcode == 0x3d || r.opcode == 0x3e ||
                     r.opcode == kDsOpcodeBpermuteB32);
                if ((r.fmt == Rdna2Format::SOPP && r.opcode == 0x0a) ||
                    ds_wave_collective ||
                    (r.fmt == Rdna2Format::VOP3 &&
                     (r.opcode == 0x365 || r.opcode == 0x366))) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-struct-reject", "route-decline",
                        "vcc branch pc=%u contains synchronized op pc=%u fmt=%d op=0x%x",
                        in.pc, r.pc, static_cast<int>(r.fmt), r.opcode);
                    return reject();
                }
            }
        }
        out.push_back(F);
    }
    // Pass 2: every s_branch in the stream must be a claimed else-arm terminator or a loop back-edge —
    // any other unconditional branch is control flow this structurizer doesn't model.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        if (loop_backedge(in.pc)) continue;
        bool claimed = false;
        for (const auto& F : out) if (F.has_else && F.sb_pc == in.pc) { claimed = true; break; }
        if (!claimed) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "unclaimed s_branch pc=%u target=%u",
                                     in.pc, branch_target(in));
            return reject();
        }
    }
    // Structural nesting check (branches are already in pc order): each region must lie entirely
    // inside the innermost still-open region, or start after it closed. A region spans to its merge
    // (if/else) or target (plain if); a LOOP is an opaque region [header_pc, exit_pc) that must nest
    // strictly. EXCEPTION (the shared-outer-merge cascade): an if/else whose merge_pc escapes the
    // innermost open region is accepted here and re-validated during emission (emit_structured
    // requires the escaping merge to equal the enclosing merge chain's continuation, i.e. no
    // instructions are skipped) — a violation rejects the shader there, fail-visible.
    std::vector<uint32_t> open;
    {
        size_t fi = 0, li = 0;
        const size_t nl = loops ? loops->size() : 0;
        while (fi < out.size() || li < nl) {
            const bool take_loop = li < nl &&
                (fi >= out.size() || (*loops)[li].header_pc < out[fi].branch_pc);
            const uint32_t start = take_loop ? (*loops)[li].header_pc : out[fi].branch_pc;
            uint32_t span_end    = take_loop ? (*loops)[li].exit_pc
                                 : (out[fi].has_else ? out[fi].merge_pc : out[fi].target_pc);
            const bool clampable = !take_loop && out[fi].has_else;
            while (!open.empty() && open.back() <= start) open.pop_back();
            if (!open.empty() && span_end > open.back()) {
                if (!clampable) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-struct-reject", "route-decline",
                        "overlapping region start=%u end=%u parent-end=%u",
                        start, span_end, open.back());
                    return reject();                     // plain if / loop escaping its region: not a tree
                }
                span_end = open.back();                  // cascade if/else: clamped to the enclosing
            }                                            // region for nesting purposes (see above)
            open.push_back(span_end);
            if (take_loop) li++; else fi++;
        }
    }
    return out;
}

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == kSop1OpcodeGetpcB64) return UINT32_MAX; // s_getpc has no source
            if (in.opcode == kSop1OpcodeBcnt1I32B64 ||
                in.opcode == kSop1OpcodeFf1I32B64 ||
                in.opcode == kSop1OpcodeFlbitI32B64 ||
                in.opcode == 0x18 || in.opcode == kSop1OpcodeQuadmaskB64)
                return 2;                    // bcnt/ff1/flbit consume a B64 pair
            return scalar_write_width(in) == 1 ? 1u : 2u;
        case Rdna2Format::SOP2:
            if (in.opcode == 0x1f || in.opcode == 0x21)
                return source == 1 ? 1u : 2u; // B64 shifts use a B32 shift count
            if (in.opcode == kSop2OpcodeBfmB64) return 1;  // BFM width and offset are B32
            if (in.opcode == kSop2OpcodeBfeU64)
                return source == 1 ? 1u : 2u;
            return scalar_write_width(in) == 1 ? 1u : 2u;
        case Rdna2Format::SOPC:
            return in.opcode == 0x12 || in.opcode == 0x13 ? 2u : 1u;
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
            return 1;
        case Rdna2Format::VOPC: {
            const uint32_t effective = vopc_is_cmpx(in.opcode)
                ? in.opcode - 0x10u : in.opcode;
            const bool integer64 =
                (effective >= 0xa1u && effective <= 0xa6u) ||
                (effective >= 0xe1u && effective <= 0xe6u);
            return integer64 ? 2u : 1u;
        }
        case Rdna2Format::VOP3:
            if (in.opcode == 0x360 || in.opcode == 0x361 ||
                in.opcode == 0x365 || in.opcode == 0x366 || in.opcode == 0x11f)
                return source < 2 ? 1u : UINT32_MAX;
            if (in.opcode == kVop3OpcodeLshlrevB64 ||
                in.opcode == kVop3OpcodeLshrrevB64)
                return source == 0 ? 1u : (source == 1 ? 2u : UINT32_MAX);
            if (in.opcode == 0x101)
                return source < 2 ? 1u : 2u; // cndmask data is B32; condition is a pair
            if (in.opcode == 0x141 || in.opcode == 0x143 || in.opcode == 0x347 ||
                in.opcode == 0x36f || in.opcode == kVop3OpcodeAdd3U32 ||
                in.opcode == kVop3OpcodeAndOrB32)
                return 1;
            return 2;
        default:
            return 0;
    }
}

// Scalar instructions whose encoded destination is also an implicit source. These reads must be
// observed before writer transfer functions invalidate the old lifetime. SOPK has no decoded src
// operands at all; conditional moves preserve the old destination on one SCC outcome, comparisons
// consume SDST without writing it, and ADDK/MULK are ordinary read-modify-write operations. SOP1
// conditional moves and bitset operations have an explicit source too, but it is the replacement
// value/bit index rather than the old destination.
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::SGPR) return 0;
    if (in.fmt == Rdna2Format::SOPK) {
        if (in.opcode == 0x02 ||                         // s_cmovk_i32
            (in.opcode >= 0x03 && in.opcode <= 0x10))   // s_cmpk_*, s_addk, s_mulk
            return 1;
        return 0;
    }
    if (in.fmt != Rdna2Format::SOP1) return 0;
    switch (in.opcode) {
        case kSop1OpcodeCmovB32: return 1;
        case kSop1OpcodeCmovB64: return 2;
        case kSop1OpcodeBitset0B32: return 1;
        case 0x1c: return 2; // s_bitset0_b64
        case kSop1OpcodeBitset1B32: return 1;
        case 0x1e: return 2; // s_bitset1_b64
        default: return 0;
    }
}

// S_MOV_B64 from VCC publishes both an exact Bool-domain saved mask and its two ballot words under
// native Wave64. The compact structured emitter preserves both views through its SSA/PHI machinery,
// but unlike the dispatcher it previously had no lifetime tag telling S_FF1/S_BCNT which view owns
// the pair at an exact consumer. Compute a small forward MUST analysis over the decoded scalar CFG:
// a saved-mask fact is generated by an exact VCC copy reached from a proved mask-domain VCC or by a
// SAVEEXEC destination (which receives OLD_EXEC independently of its logical source). Every
// overlapping scalar write kills it, and joins retain it only when every reachable predecessor
// agrees. Indirect PC updates have no successor, so they cannot manufacture a dominance fact beyond
// an unknown transfer.
std::unordered_set<uint32_t> proven_structured_wave64_mask_reduction_pcs(
        const std::vector<Rdna2Inst>& ins) {
    size_t count = 0;
    while (count < ins.size() && !ins[count].is_end) ++count;
    if (!count) return {};

    std::unordered_map<uint32_t, size_t> index_for_pc;
    for (size_t i = 0; i < count; ++i) index_for_pc.emplace(ins[i].pc, i);

    std::vector<std::set<int>> incoming(count);
    std::vector<bool> reachable(count, false);
    std::vector<size_t> pending{0};
    reachable[0] = true;
    while (!pending.empty()) {
        const size_t index = pending.back();
        pending.pop_back();
        std::set<int> saved = incoming[index];
        const Rdna2Inst& in = ins[index];

        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (auto it = saved.begin(); it != saved.end();) {
                const int root = *it;
                if (base < root + 2 && root < base + static_cast<int>(width))
                    it = saved.erase(it);
                else
                    ++it;
            }
        }, /*wave32_one_word_masks=*/false);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
            in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 106 &&
            saved.contains(106))
            saved.insert(in.dst.value);
        // Every SAVEEXEC variant writes OLD_EXEC to SDST before replacing architectural EXEC with
        // its logical result. OLD_EXEC is always one complete Wave64 mask, so an ordinary pair or
        // VCC destination becomes an exact mask producer without requiring source provenance. Do
        // not seed an EXEC destination: its architectural state after the instruction is the new
        // logical result, not the saved value.
        const bool saves_old_exec = in.fmt == Rdna2Format::SOP1 &&
            ((in.opcode >= kSop1OpcodeAndSaveexecB64 &&
              in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
             in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
             in.opcode == kSop1OpcodeOrn1SaveexecB64);
        if (saves_old_exec && in.dst.kind == OperandKind::SGPR &&
            in.dst.value >= 0 && in.dst.value <= 106)
            saved.insert(in.dst.value);
        // A non-CMPX vector compare writes a fresh architectural VCC predicate. This deliberately
        // stays narrower than the dispatcher's general mask transfer: the structured override only
        // needs to certify the captured VOPC -> S_MOV_B64 -> reduction chain, and an ordinary scalar
        // VCC write above kills the fact before it can seed a saved pair.
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            const int mask_destination =
                in.dst.kind == OperandKind::SGPR && in.dst.value <= 105
                    ? in.dst.value : 106;
            saved.insert(mask_destination);
        }

        std::vector<size_t> successors;
        auto add_successor = [&](uint32_t pc) {
            const auto found = index_for_pc.find(pc);
            if (found != index_for_pc.end() &&
                std::find(successors.begin(), successors.end(), found->second) == successors.end())
                successors.push_back(found->second);
        };
        const bool indirect_pc = in.fmt == Rdna2Format::SOP1 &&
            in.opcode >= kSop1OpcodeSetpcB64 && in.opcode <= kSop1OpcodeRfeB64;
        if (!indirect_pc && in.fmt == Rdna2Format::SOPP &&
            sopp_opcode_is_direct_branch(in.opcode)) {
            add_successor(branch_target(in));
            if (in.opcode != kSoppOpcodeBranch && index + 1 < count)
                successors.push_back(index + 1);
        } else if (!indirect_pc && in.fmt == Rdna2Format::SOPP && in.opcode == 0x12u) {
            // S_TRAP transfers control outside the decoded shader stream.
        } else if (!indirect_pc && index + 1 < count) {
            successors.push_back(index + 1);
        }

        for (size_t successor : successors) {
            if (!reachable[successor]) {
                reachable[successor] = true;
                incoming[successor] = saved;
                pending.push_back(successor);
                continue;
            }
            std::set<int> joined;
            std::set_intersection(
                incoming[successor].begin(), incoming[successor].end(),
                saved.begin(), saved.end(), std::inserter(joined, joined.end()));
            if (joined != incoming[successor]) {
                incoming[successor] = std::move(joined);
                pending.push_back(successor);
            }
        }
    }

    std::unordered_set<uint32_t> proven;
    for (size_t i = 0; i < count; ++i) {
        const Rdna2Inst& in = ins[i];
        if (!reachable[i] || in.fmt != Rdna2Format::SOP1 ||
            (in.opcode != kSop1OpcodeBcnt1I32B64 &&
             in.opcode != kSop1OpcodeFf1I32B64) ||
            in.src[0].kind != OperandKind::SGPR)
            continue;
        if (incoming[i].contains(in.src[0].value)) proven.insert(in.pc);
    }
    return proven;
}

// Prove S_LOAD_DWORDX2 descriptor-fragment shapes. The load supplies one or two live words of a
// four-dword V#; scalar code fills or replaces the other words before MUBUF, MTBUF, or S_BUFFER_LOAD
// consumes the complete live descriptor. The front half has already read the guest words and
// published the resulting resource at that exact consumer PC, so the loaded fragment is
// provenance-only in SPIR-V and can be represented by zero placeholders. Immediate descriptor-table
// offsets are admitted by this use proof rather than a title-specific offset inventory.
//
// This is deliberately a whole-CFG use proof, not opcode-wide admission. Every reachable path is
// followed until the loaded words are overwritten or execution ends. A loaded word may only be read
// through an exact-PC, key-less buffer descriptor; every ordinary scalar/address read and every
// unresolved control edge rejects the candidate. CONFIDENCE: HIGH for the admitted shape.
std::unordered_set<uint32_t> proven_smem_x2_descriptor_fragment_loads(
        const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt,
        uint32_t wave_size) {
    std::unordered_set<uint32_t> proven;
    if (!rt || ins.empty()) return proven;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t index = 0; index < ins.size(); ++index)
        index_by_pc.emplace(ins[index].pc, index);

    auto is_scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR ||
               (operand.kind == OperandKind::Special &&
                operand.value >= 106 && operand.value <= 124);
    };
    auto exact_buffer_resource = [&](const Rdna2Inst& consumer, bool scalar_buffer) {
        const ShaderResource* resource = rt->by_fetch_pc(consumer.pc);
        if (!resource || resource->fetch_pc != consumer.pc ||
            resource->srt_offset != 0xffffffffu ||
            resource->sgpr_base != 0xffffffffu || resource->table_index_count != 0)
            return false;
        if (scalar_buffer) return resource->cls == ResourceClass::ConstantBuffer;
        return resource->cls == ResourceClass::ConstantBuffer ||
               resource->cls == ResourceClass::VertexBuffer;
    };
    auto scalar_register_is = [&](const Operand& operand, int reg) {
        return is_scalar_operand(operand) && operand.value == reg;
    };
    auto instruction_reads_scc = [](const Rdna2Inst& in) {
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253)
                return true;
        if (in.fmt == Rdna2Format::SOP2)
            return in.opcode == kSop2OpcodeAddcU32 || in.opcode == 0x05u ||
                   in.opcode == kSop2OpcodeCselectB32 ||
                   in.opcode == kSop2OpcodeCselectB64;
        if (in.fmt == Rdna2Format::SOPP)
            return in.opcode == kSoppOpcodeCbranchScc0 || in.opcode == 0x05u;
        if (in.fmt == Rdna2Format::SOP1)
            return in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64;
        return in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32;
    };

    std::unordered_set<uint32_t> direct_branch_targets;
    for (const Rdna2Inst& in : ins)
        if (!in.is_end && in.fmt == Rdna2Format::SOPP &&
            sopp_opcode_is_direct_branch(in.opcode))
            direct_branch_targets.insert(branch_target(in));

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        const bool register_offset =
            load.src[1].kind == OperandKind::SGPR ||
            (load.src[1].kind == OperandKind::Special &&
             load.src[1].value >= 106 && load.src[1].value <= 123);
        const bool optional_null_immediate =
            load.src[1].kind == OperandKind::Special && load.src[1].value == 125 &&
            load.literal == kGtaOptionalBufferPointerOffset;
        const bool aligned_immediate =
            load.src[1].kind == OperandKind::Special && load.src[1].value == 125 &&
            static_cast<int32_t>(load.literal) >= 0 && (load.literal & 3u) == 0;
        if (load.is_end || load.fmt != Rdna2Format::SMEM ||
            load.opcode != kSmemOpcodeLoadDwordX2 ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            (load.dst.value > 104 && load.dst.value != 106) ||
            (!register_offset && !aligned_immediate))
            continue;

        const int base = load.dst.value;
        struct DescriptorRelocation {
            size_t low_index = SIZE_MAX;
            size_t high_index = SIZE_MAX;
            int destination_base = -1;
        } relocation;

        // Some descriptor builders relocate a loaded 64-bit address with an ordinary carry pair:
        //   s_add_u32  dst.lo, loaded.lo, base.lo
        //   s_addc_u32 dst.hi, loaded.hi, base.hi
        // Admit only the exact pair in one straight-line SCC lifetime. Interposed scalar moves are
        // SCC-transparent, and the produced final carry must have no later observer. This keeps the
        // loaded address as descriptor provenance without treating general scalar arithmetic as a
        // descriptor transformation.
        for (size_t low_index = load_index + 1;
             low_index < ins.size() && relocation.low_index == SIZE_MAX; ++low_index) {
            const Rdna2Inst& low = ins[low_index];
            if (low.is_end ||
                (low.fmt == Rdna2Format::SOPP && !sopp_is_noop(low)) ||
                (low.fmt == Rdna2Format::SOP1 &&
                 low.opcode >= kSop1OpcodeSetpcB64 && low.opcode <= kSop1OpcodeRfeB64))
                break;
            if (low.fmt != Rdna2Format::SOP2 || low.opcode != kSop2OpcodeAddU32 ||
                low.dst.kind != OperandKind::SGPR || low.dst.value < 0 ||
                low.dst.value + 3 > 105 || low.n_src != 2)
                continue;
            if (low.dst.value <= base + 1 && base <= low.dst.value + 1)
                continue;
            int loaded_source = -1;
            if (scalar_register_is(low.src[0], base)) loaded_source = 0;
            if (scalar_register_is(low.src[1], base)) {
                if (loaded_source >= 0) continue;
                loaded_source = 1;
            }
            if (loaded_source < 0 ||
                !is_scalar_operand(low.src[static_cast<uint32_t>(1 - loaded_source)]))
                continue;
            const int other_base = low.src[static_cast<uint32_t>(1 - loaded_source)].value;
            if (other_base < 0 || other_base + 1 > 105 ||
                (other_base <= base + 1 && base <= other_base + 1) ||
                (other_base <= low.dst.value + 1 &&
                 low.dst.value <= other_base + 1))
                continue;

            for (size_t high_index = low_index + 1; high_index < ins.size(); ++high_index) {
                const Rdna2Inst& high = ins[high_index];
                if (direct_branch_targets.contains(high.pc)) break;
                if (high.fmt == Rdna2Format::SOP2 &&
                    high.opcode == kSop2OpcodeAddcU32 && high.n_src == 2 &&
                    high.dst.kind == OperandKind::SGPR &&
                    high.dst.value == low.dst.value + 1) {
                    const bool matched_sources =
                        (scalar_register_is(high.src[0], base + 1) &&
                         scalar_register_is(high.src[1], other_base + 1)) ||
                        (scalar_register_is(high.src[1], base + 1) &&
                         scalar_register_is(high.src[0], other_base + 1));
                    if (!matched_sources) break;
                    bool carry_observed = false;
                    for (size_t later = high_index + 1; later < ins.size(); ++later)
                        if (instruction_reads_scc(ins[later])) {
                            carry_observed = true;
                            break;
                        }
                    if (!carry_observed)
                        relocation = {low_index, high_index, low.dst.value};
                    break;
                }
                if (high.fmt != Rdna2Format::SOP1 ||
                    high.opcode != kSop1OpcodeMovB32)
                    break;
                if (instruction_reads_scc(high)) break;
                if (high.dst.value == base || high.dst.value == base + 1 ||
                    high.dst.value == low.dst.value ||
                    high.dst.value == low.dst.value + 1)
                    break;
            }
        }

        auto overlap = [base, &relocation](uint8_t live, int first, uint32_t words) {
            if (first < 0 || !words) return false;
            for (uint32_t word = 0; word < words; ++word) {
                const int reg = first + static_cast<int>(word);
                const int original_relative = reg - base;
                if (original_relative >= 0 && original_relative < 2 &&
                    (live & (1u << original_relative)))
                    return true;
                const int relocated_relative = reg - relocation.destination_base;
                if (relocation.destination_base >= 0 && relocated_relative >= 0 &&
                    relocated_relative < 2 && (live & (1u << (relocated_relative + 2))))
                    return true;
            }
            return false;
        };
        auto clear_written = [base, &relocation](uint8_t& live, int first, uint32_t words) {
            if (first < 0) return;
            for (uint32_t word = 0; word < words; ++word) {
                const int reg = first + static_cast<int>(word);
                const int original_relative = reg - base;
                if (original_relative >= 0 && original_relative < 2)
                    live &= static_cast<uint8_t>(~(1u << original_relative));
                const int relocated_relative = reg - relocation.destination_base;
                if (relocation.destination_base >= 0 && relocated_relative >= 0 &&
                    relocated_relative < 2)
                    live &= static_cast<uint8_t>(~(1u << (relocated_relative + 2)));
            }
        };

        struct PendingState { size_t index; uint8_t live; };
        std::vector<PendingState> pending;
        if (load_index + 1 < ins.size()) pending.push_back({load_index + 1, 0x3u});
        std::vector<uint16_t> visited(ins.size(), 0);
        bool consumed = false;
        bool valid = true;

        auto enqueue_pc = [&](int64_t target, uint8_t live) {
            if (!live || !valid) return;
            if (target < 0) { valid = false; return; }
            const auto found = index_by_pc.find(static_cast<uint32_t>(target));
            if (found != index_by_pc.end()) {
                pending.push_back({found->second, live});
                return;
            }
            // A forward target beyond the decoded program is a terminating early-out. Every other
            // unresolved edge could re-enter code whose scalar reads are unknown to this proof.
            if (target <= static_cast<int64_t>(ins.back().pc)) valid = false;
        };

        while (valid && !pending.empty()) {
            PendingState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size()) continue;
            const uint16_t state_bit = static_cast<uint16_t>(1u << state.live);
            if (visited[state.index] & state_bit) continue;
            visited[state.index] |= state_bit;

            const Rdna2Inst& in = ins[state.index];
            if (in.is_end) continue;
            uint8_t live = state.live;

            // Captured builders patch descriptor control bits in the loaded second word before the
            // V# is consumed. Their results remain descriptor provenance: retain that word's live
            // marker so every later use is still checked, but do not mistake these exact RMW patches
            // for ordinary scalar observations of guest data.
            const bool bitset_descriptor_patch =
                in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeBitset1B32 &&
                in.dst.kind == OperandKind::SGPR && overlap(live, in.dst.value, 1) &&
                in.n_src == 1 && in.src[0].kind == OperandKind::InlineInt;
            const bool or_descriptor_patch =
                optional_null_immediate && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == kSop2OpcodeOrB32 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == base + 1 &&
                in.n_src == 2 && in.literal == kGtaOptionalBufferStrideWord &&
                ((in.src[0].kind == OperandKind::SGPR && in.src[0].value == in.dst.value &&
                  in.src[1].kind == OperandKind::Literal) ||
                 (in.src[1].kind == OperandKind::SGPR && in.src[1].value == in.dst.value &&
                  in.src[0].kind == OperandKind::Literal));
            constexpr uint32_t kGtavBufferDescriptorHighControlBits = 0x000c0000u;
            const bool high_control_descriptor_patch =
                register_offset && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == kSop2OpcodeOrB32 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == base + 1 &&
                in.n_src == 2 && in.src[0].kind == OperandKind::SGPR &&
                in.src[0].value == in.dst.value && in.src[1].kind == OperandKind::Literal &&
                in.literal == kGtavBufferDescriptorHighControlBits;
            const bool relocation_low = state.index == relocation.low_index;
            const bool relocation_high = state.index == relocation.high_index;
            const bool descriptor_patch = bitset_descriptor_patch || or_descriptor_patch ||
                high_control_descriptor_patch || relocation_low || relocation_high;

            // A load overlapping physical VCC also creates a mask lifetime. Reject every implicit
            // mask observation until a real VCC writer replaces the live half or pair; explicit
            // scalar observations remain covered by the ordinary operand walk below.
            const bool implicit_vcc_read =
                overlap(live, 106, wave_size == 32 ? 1u : 2u) &&
                ((in.fmt == Rdna2Format::SOPP &&
                  (in.opcode == 0x06u || in.opcode == 0x07u)) ||
                 (in.fmt == Rdna2Format::VOP2 &&
                  (in.opcode == 0x01u ||
                   (in.opcode >= 0x28u && in.opcode <= 0x2au))));
            if (implicit_vcc_read) {
                valid = false;
                break;
            }

            bool branch = false;
            bool fallthrough = true;
            if (in.fmt == Rdna2Format::SOPP) {
                if (sopp_opcode_is_direct_branch(in.opcode)) {
                    branch = true;
                    fallthrough = in.opcode != kSoppOpcodeBranch;
                } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::SOP1 &&
                       in.opcode >= kSop1OpcodeSetpcB64 &&
                       in.opcode <= kSop1OpcodeRfeB64) {
                valid = false;
                break;
            }

            bool descriptor_read = false;
            if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                in.src[1].kind == OperandKind::SGPR &&
                overlap(live, in.src[1].value, 4)) {
                if ((is_scalar_operand(in.src[2]) && overlap(live, in.src[2].value, 1)) ||
                    !exact_buffer_resource(in, false)) {
                    valid = false;
                    break;
                }
                descriptor_read = true;
                consumed = true;
            } else if (in.fmt == Rdna2Format::SMEM &&
                       smem_opcode_is_buffer_load(in.opcode) &&
                       is_scalar_operand(in.src[0]) && overlap(live, in.src[0].value, 4)) {
                if ((is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 1)) ||
                    !exact_buffer_resource(in, true)) {
                    valid = false;
                    break;
                }
                descriptor_read = true;
                consumed = true;
            }

            if (in.fmt == Rdna2Format::SOPP) {
                // Branch operands are implicit architectural condition codes, never ordinary SGPR
                // data. Their control edges were validated above.
            } else if (in.fmt == Rdna2Format::SMEM) {
                const uint32_t base_words = smem_opcode_is_buffer_load(in.opcode) ? 4u : 2u;
                if ((!descriptor_read && is_scalar_operand(in.src[0]) &&
                     overlap(live, in.src[0].value, base_words)) ||
                    (is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 1))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
                if ((!descriptor_read && in.src[1].kind == OperandKind::SGPR &&
                     overlap(live, in.src[1].value, 4)) ||
                    (is_scalar_operand(in.src[2]) && overlap(live, in.src[2].value, 1))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MIMG) {
                if ((in.src[1].kind == OperandKind::SGPR &&
                     overlap(live, in.src[1].value, 8)) ||
                    (in.src[2].kind == OperandKind::SGPR &&
                     overlap(live, in.src[2].value, 4))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::FLAT) {
                if (is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 2)) {
                    valid = false;
                    break;
                }
            } else {
                const uint32_t implicit_read = scalar_implicit_destination_read_width(in);
                if (!descriptor_patch && implicit_read &&
                    overlap(live, in.dst.value, implicit_read)) {
                    valid = false;
                    break;
                }
                if (!descriptor_patch) {
                    for (uint32_t source = 0; source < in.n_src; ++source) {
                        if (!is_scalar_operand(in.src[source])) continue;
                        uint32_t words = scalar_alu_source_words(in, source);
                        if (words == UINT32_MAX) continue;
                        if (!words) words = 1; // decoded non-ALU scalar operands remain fail-closed
                        if (overlap(live, in.src[source].value, words)) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (!valid) break;
            }

            if (relocation_low || relocation_high) {
                const uint8_t destination_bit = relocation_low ? 0x4u : 0x8u;
                const bool source_live = relocation_low
                    ? (live & 0x1u) != 0
                    : (live & (0x2u | 0x4u)) != 0;
                clear_written(live,
                              relocation.destination_base + (relocation_high ? 1 : 0), 1);
                if (source_live) live |= destination_bit;
            } else if (!descriptor_patch) {
                for_each_scalar_write(in, [&](int first, uint32_t words) {
                    clear_written(live, first, words);
                }, wave_size == 32);
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.kind == OperandKind::SGPR ||
                     in.dst.kind == OperandKind::Special) && in.dst.value == 106)
                    clear_written(live, 106, wave_size == 32 ? 1u : 2u);
            }
            if (!live) continue;

            if (branch) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                    static_cast<int64_t>(in.len_dwords) + static_cast<int64_t>(in.simm16);
                enqueue_pc(target, live);
            }
            if (fallthrough) {
                if (state.index + 1 < ins.size()) pending.push_back({state.index + 1, live});
                else valid = false; // live descriptor data fell off an unterminated stream
            }
        }
        if (valid && consumed) proven.insert(load.pc);
    }
    return proven;
}

// Prove the narrow S_LOAD_DWORDX16 descriptor-bundle shape used by GTA V compute kernels. A wide
// scalar load is not intrinsically a descriptor fetch: accepting every x16 as two T#s would replace
// real scalar data with zero placeholders. For each candidate, follow all sixteen loaded words to
// overwrite/end and require BOTH aligned eight-word halves to be consumed as MIMG SRSRCs. Every such
// consumer must have its own key-less exact-PC image resource; an SRT tag cannot name two descriptors
// packed under the load's one immediate offset.
//
// The only scalar transformation admitted is the captured compiler's T#.word3 patch (with at most
// one independent VOP scheduled between the two scalar operations):
//
//   s_and_b32 tmp, tword3, 0x0fffffff
//   s_or_b32  tword3, tmp, 0xd0000000
//
// All ordinary scalar/vector reads, partial descriptor writes, samplers overlapping the bundle, and
// scalar control flow reject the candidate. This is deliberately a use proof, not an opcode-wide
// declaration that x16 loads are descriptors. CONFIDENCE: HIGH for the admitted shape: the front-half
// snapshots both halves and publishes the live descriptor at each exact MIMG PC.
bool smem_x16_patch_gap_reads_implicit_state(const Rdna2Inst& in, int temporary) {
    // The AND's SCC is descriptor-derived until the matching OR overwrites it. Vector ALU can name
    // SCC as the scalar source encoding even though it is outside the ordinary SGPR/special range.
    for (uint32_t source = 0; source < in.n_src; ++source)
        if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253)
            return true;

    // E32 cndmask and carry forms consume architectural VCC without exposing it in n_src. This is
    // observable descriptor-derived data when the compiler chose VCC_LO as its word3 temporary.
    return (temporary == 106 || temporary == 107) && in.fmt == Rdna2Format::VOP2 &&
           (in.opcode == 0x01u || (in.opcode >= 0x28u && in.opcode <= 0x2au));
}

std::unordered_set<uint32_t> proven_smem_x16_descriptor_loads(
        const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt) {
    std::unordered_set<uint32_t> proven;
    if (!rt || ins.empty()) return proven;

    // Alternate entries would require path-sensitive lifetime/provenance joins. Keep this first
    // admission linear: hints, waits and barriers are transparent; every real scalar branch or
    // indirect PC transfer makes the whole candidate ineligible.
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            return proven;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) && in.opcode != 0x0au)
            return proven;
    }

    auto scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR ||
               (operand.kind == OperandKind::Special &&
                operand.value >= 106 && operand.value <= 124);
    };
    auto literal_is = [](const Rdna2Inst& in, const Operand& operand, uint32_t value) {
        return operand.kind == OperandKind::Literal && in.literal == value;
    };

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        if (load.is_end || load.fmt != Rdna2Format::SMEM || load.opcode != 0x04u ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + 15 > 105 ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0)
            continue;

        const int base = load.dst.value;
        uint16_t live = 0xffffu;
        bool consumed[2] = {false, false};
        bool valid = true;
        auto live_overlap = [&](int first, uint32_t words) {
            if (first < 0 || !words) return false;
            for (uint32_t word = 0; word < words; ++word) {
                const int relative = first + static_cast<int>(word) - base;
                if (relative >= 0 && relative < 16 &&
                    (live & static_cast<uint16_t>(1u << relative)))
                    return true;
            }
            return false;
        };
        auto clear_written = [&](int first, uint32_t words) {
            if (first < 0) return;
            for (uint32_t word = 0; word < words; ++word) {
                const int relative = first + static_cast<int>(word) - base;
                if (relative >= 0 && relative < 16)
                    live &= static_cast<uint16_t>(~(1u << relative));
            }
        };

        for (size_t index = load_index + 1; valid && index < ins.size(); ++index) {
            const Rdna2Inst& in = ins[index];
            if (in.is_end) break;

            // Recognize the complete word3 patch as one unit. One captured variant schedules an
            // independent VOP between the two scalar instructions; admit that exact one-instruction
            // gap only when it cannot observe or replace either the descriptor bundle or temporary.
            bool patched = false;
            if (index + 1 < ins.size() && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == 0x0eu && in.dst.kind == OperandKind::SGPR &&
                in.src[0].kind == OperandKind::SGPR &&
                literal_is(in, in.src[1], 0x0fffffffu)) {
                const int descriptor_word = in.src[0].value;
                const int half = descriptor_word == base + 3 ? 0
                               : descriptor_word == base + 11 ? 1 : -1;
                // The retained GTA V shape uses VCC_LO exactly. Do not generalize this to M0 or
                // other architectural scalar registers: their implicit consumers are not all in the
                // ordinary SGPR liveness inventory (DS observes M0 without a decoded scalar source).
                const bool exact_patch_temporary = in.dst.value == 106;
                size_t join_index = index + 1;
                const Rdna2Inst& possible_gap = ins[join_index];
                if (possible_gap.fmt != Rdna2Format::SOP2 || possible_gap.opcode != 0x10u) {
                    bool transparent_gap = possible_gap.fmt == Rdna2Format::VOP1 ||
                                           possible_gap.fmt == Rdna2Format::VOP2 ||
                                           possible_gap.fmt == Rdna2Format::VOP3;
                    if (transparent_gap &&
                        smem_x16_patch_gap_reads_implicit_state(possible_gap, in.dst.value))
                        transparent_gap = false;
                    for (uint32_t source = 0;
                         transparent_gap && source < possible_gap.n_src; ++source) {
                        if (!scalar_operand(possible_gap.src[source])) continue;
                        if (possible_gap.src[source].value == in.dst.value ||
                            live_overlap(possible_gap.src[source].value, 2))
                            transparent_gap = false;
                    }
                    for_each_scalar_write(possible_gap, [&](int first, uint32_t words) {
                        if (live_overlap(first, words) ||
                            (in.dst.value >= first &&
                             in.dst.value < first + static_cast<int>(words)))
                            transparent_gap = false;
                    });
                    if (transparent_gap) ++join_index;
                }
                const Rdna2Inst& join = join_index < ins.size() ? ins[join_index] : in;
                const bool temporary_unobserved = join_index + 1 >= ins.size() ||
                    sgpr_dead_at_merge(ins, ins[join_index + 1].pc, in.dst.value);
                const bool exact_join = half >= 0 && exact_patch_temporary &&
                    join_index < ins.size() && join.fmt == Rdna2Format::SOP2 &&
                    join.opcode == 0x10u && join.dst.kind == OperandKind::SGPR &&
                    join.dst.value == descriptor_word && scalar_operand(join.src[0]) &&
                    join.src[0].value == in.dst.value &&
                    literal_is(join, join.src[1], 0xd0000000u) && temporary_unobserved;
                const uint16_t word_bit = half >= 0
                    ? static_cast<uint16_t>(1u << (descriptor_word - base)) : 0u;
                if (exact_join && (live & word_bit)) {
                    patched = true;
                    index = join_index;
                }
            }
            if (patched) continue;

            if (in.fmt == Rdna2Format::MIMG) {
                const bool t_is_scalar = in.src[1].kind == OperandKind::SGPR;
                const int half = t_is_scalar && in.src[1].value == base ? 0
                               : t_is_scalar && in.src[1].value == base + 8 ? 1 : -1;
                const bool touches_t = t_is_scalar && live_overlap(in.src[1].value, 8);
                const bool touches_sampler = scalar_operand(in.src[2]) &&
                                             live_overlap(in.src[2].value, 4);
                if (touches_sampler) { valid = false; break; }
                if (half >= 0) {
                    const uint16_t half_mask = static_cast<uint16_t>(0xffu << (half * 8));
                    const ShaderResource* resource = rt->by_fetch_pc(in.pc);
                    const bool exact_image = resource && resource->fetch_pc == in.pc &&
                        resource->srt_offset == 0xffffffffu &&
                        resource->sgpr_base == 0xffffffffu &&
                        (resource->cls == ResourceClass::Texture ||
                         resource->cls == ResourceClass::StorageImage);
                    if ((live & half_mask) != half_mask || !exact_image) {
                        valid = false;
                        break;
                    }
                    consumed[half] = true;
                    continue;
                }
                if (touches_t) { valid = false; break; }
            }

            // Bound implicit descriptor/address reads before the generic decoded operands. These
            // packet formats encode a base SGPR while consuming a wider range.
            if (in.fmt == Rdna2Format::SMEM) {
                const uint32_t base_words = in.opcode >= 0x08u ? 4u : 2u;
                if (scalar_operand(in.src[0]) && live_overlap(in.src[0].value, base_words)) {
                    valid = false;
                    break;
                }
                if (scalar_operand(in.src[1]) && live_overlap(in.src[1].value, 1)) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
                if ((in.src[1].kind == OperandKind::SGPR &&
                     live_overlap(in.src[1].value, 4)) ||
                    (scalar_operand(in.src[2]) && live_overlap(in.src[2].value, 1))) {
                    valid = false;
                    break;
                }
            } else {
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if (!scalar_operand(in.src[source])) continue;
                    const uint32_t words =
                        in.fmt == Rdna2Format::SOP1 ||
                        in.fmt == Rdna2Format::SOP2 ||
                        in.fmt == Rdna2Format::SOPC ||
                        in.fmt == Rdna2Format::VOP3 ? 2u : 1u;
                    if (live_overlap(in.src[source].value, words)) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }

            // SOPK read/modify/write forms do not expose their implicit destination read through
            // n_src. Treat every live overlap conservatively; the target descriptor bundles use no
            // SOPK writes, so widening this is unnecessary.
            if (in.fmt == Rdna2Format::SOPK && in.dst.kind == OperandKind::SGPR &&
                live_overlap(in.dst.value, 1)) {
                valid = false;
                break;
            }
            for_each_scalar_write(in, clear_written);
        }
        if (valid && consumed[0] && consumed[1]) proven.insert(load.pc);
    }
    return proven;
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
            if (found != index_of.end()) succ[i].push_back(found->second);
            else if (target <= ins.back().pc) return {};   // an edge this walk cannot resolve
            // A target past the decoded stream is an early-out that terminates the wave; it
            // contributes no successor inside this function.
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02) continue;   // s_branch: no fallthrough
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
            if (gen(ins[i])) out = 1;
            else if (kill(ins[i])) out = 0;
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
bool exec_write_sets_full_mask(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.n_src != 1) return false;
    if ((in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) ||
        in.dst.value != 126)
        return false;
    return in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1;
}

bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    size_t branch_index = ins.size();
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].pc == branch_pc) { branch_index = i; break; }
    if (branch_index == ins.size() || branch_index == 0) return false;

    // Obligation (1): EXEC is full at the branch along every path.
    const std::vector<uint8_t> exec_full = must_fact_at(
        ins, exec_write_sets_full_mask,
        [](const Rdna2Inst& in) { return rdna2_instruction_may_change_exec(in); }, true);
    if (exec_full.empty() || !exec_full[branch_index]) return false;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };
    std::set<uint32_t> targets;
    for (const auto& in : ins)
        if (is_branch(in)) targets.insert(branch_target(in));

    // The straight-line region ending at the branch. Every definition consumed from inside it has a
    // unique reaching definition, which is what lets the backwards slice below be a textual walk.
    size_t block_start = branch_index;
    while (block_start > 0) {
        if (targets.count(ins[block_start].pc)) break;      // another edge enters here
        const Rdna2Inst& prev = ins[block_start - 1];
        if (prev.is_end || is_branch(prev)) break;
        --block_start;
    }

    auto writes_reg = [](const Rdna2Inst& in, int reg, uint32_t words) {
        bool hit = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (reg < base + static_cast<int>(width) && base < reg + static_cast<int>(words))
                hit = true;
        });
        // A VOPC without an explicit scalar destination writes architectural VCC implicitly; the
        // writer inventory only reports the explicit SGPR form.
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::Special && in.dst.value == 106 &&
            reg < 108 && 106 < reg + static_cast<int>(words))
            hit = true;
        return hit;
    };

    auto last_writer = [&](int reg, uint32_t words, size_t before) -> size_t {
        for (size_t i = before; i-- > block_start;)
            if (writes_reg(ins[i], reg, words)) return i;
        return ins.size();
    };

    // A register with no definition inside the region is accepted only when it provably still holds
    // its wave-entry value at the branch. Entry SGPRs are launch data, hence workgroup-uniform.
    std::map<int, bool> entry_value_cache;
    auto still_entry_value = [&](int reg) {
        auto cached = entry_value_cache.find(reg);
        if (cached != entry_value_cache.end()) return cached->second;
        const std::vector<uint8_t> fact = must_fact_at(
            ins, [](const Rdna2Inst&) { return false; },
            [&](const Rdna2Inst& in) { return writes_reg(in, reg, 1); }, true);
        const bool proved = !fact.empty() && fact[branch_index] != 0;
        entry_value_cache[reg] = proved;
        return proved;
    };

    std::function<bool(int, size_t, uint32_t)> uniform_scalar;
    std::function<bool(size_t, uint32_t)> uniform_scc;
    std::function<bool(int, size_t, uint32_t)> uniform_mask;

    auto uniform_operand = [&](const Operand& op, uint32_t words, size_t before,
                               uint32_t depth) -> bool {
        switch (op.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal:
                return true;
            case OperandKind::Special:
                if (op.value == 125) return true;                       // SGPR_NULL
                if (op.value == 126 || op.value == 127 || op.value == 253)
                    return false;                                       // EXEC and SCC are not data
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!uniform_scalar(op.value + static_cast<int>(word), before, depth + 1))
                        return false;
                return true;
            default:
                return false;                                           // VGPR and everything else
        }
    };

    uniform_scalar = [&](int reg, size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        const size_t w = last_writer(reg, 1, before);
        if (w == ins.size()) return reg >= 0 && reg <= 105 && still_entry_value(reg);
        const Rdna2Inst& in = ins[w];
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                if (in.opcode == 0x1f) return true;                     // s_getpc_b64
                if (rdna2_instruction_may_change_exec(in) || in.n_src != 1) return false;
                return uniform_operand(in.src[0], scalar_write_width(in) == 2 ? 2u : 1u, w, depth);
            case Rdna2Format::SOP2: {
                // Carry and cselect forms consume SCC as well as their decoded operands; the mask
                // slice models the cselect shape explicitly and nothing else may use them as data.
                if (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a ||
                    in.opcode == 0x0b)
                    return false;
                const uint32_t words = scalar_write_width(in) == 2 ? 2u : 1u;
                return in.n_src == 2 && uniform_operand(in.src[0], words, w, depth) &&
                       uniform_operand(in.src[1], words, w, depth);
            }
            case Rdna2Format::SMEM: {
                // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads a base pair.
                if (!scalar_write_width(in) || in.n_src < 1) return false;
                const uint32_t base_words = in.opcode >= 0x8 ? 4u : 2u;
                if (!uniform_operand(in.src[0], base_words, w, depth)) return false;
                return in.n_src < 2 || uniform_operand(in.src[1], 1, w, depth);
            }
            case Rdna2Format::SOPK:
                return in.opcode == 0x00;                               // s_movk_i32 is a literal
            default:
                return false;         // VOPC masks, v_readfirstlane, v_readlane, everything else
        }
    };

    uniform_scc = [&](size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        for (size_t i = before; i-- > block_start;) {
            const Rdna2Inst& in = ins[i];
            if (sopp_is_noop(in)) continue;
            if (in.fmt == Rdna2Format::SOPC) {
                if (in.n_src == 0) return false;
                const uint32_t words = in.opcode == 0x12 || in.opcode == 0x13 ? 2u : 1u;
                for (uint32_t s = 0; s < in.n_src; ++s)
                    if (!uniform_operand(in.src[s], words, i, depth)) return false;
                return true;
            }
            // Any scalar ALU or SOPK instruction may write SCC. Stop rather than accidentally
            // consuming an older compare through an unmodeled SCC writer.
            if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                in.fmt == Rdna2Format::SOPK)
                return false;
        }
        return false;                 // SCC entering the region from outside is not proved
    };

    uniform_mask = [&](int reg, size_t before, uint32_t depth) -> bool {
        if (depth > 64) return false;
        const size_t w = last_writer(reg, 2, before);
        if (w == ins.size()) return false;      // a mask entering from outside is not proved
        const Rdna2Inst& in = ins[w];
        if (in.fmt != Rdna2Format::SOP2 || in.dst.value != reg || scalar_write_width(in) != 2 ||
            in.n_src != 2)
            return false;
        auto is_exec = [](const Operand& op) {
            return (op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) &&
                   op.value == 126;
        };
        auto is_zero = [](const Operand& op) {
            return op.kind == OperandKind::InlineInt && op.value == 0;
        };
        if (in.opcode == 0x0b) {                // s_cselect_b64 dst, EXEC, 0  (either polarity)
            const bool shape = (is_exec(in.src[0]) && is_zero(in.src[1])) ||
                               (is_zero(in.src[0]) && is_exec(in.src[1]));
            return shape && uniform_scc(w, depth + 1);
        }
        if (in.opcode == 0x0f || in.opcode == 0x11) {          // s_and_b64 / s_or_b64
            for (uint32_t s = 0; s < 2; ++s) {
                const Operand& op = in.src[s];
                if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
                if (op.value == 126 || op.value == 127) return false;   // raw EXEC is per-wave
                if (!uniform_mask(op.value, w, depth + 1)) return false;
            }
            return true;
        }
        return false;
    };

    // Obligation (2): architectural VCC at the branch is a select over workgroup-uniform SCCs.
    return uniform_mask(106, branch_index, 0);
}

// Scalar registers that MAY be overwritten while a loop executes. This is deliberately separate
// from loop_written_regs: mask-pair destinations overwrite physical SGPRs (and therefore descriptor
// provenance) but their values live in sreg_bool rather than the scalar-data SSA domain.
void loop_scalar_may_writes(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                            std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                sregs.insert(base + static_cast<int>(word));
        });
    }
}

void invalidate_loop_descriptor_provenance(RegState& rs, const std::set<int>& sregs) {
    for (int reg : sregs) {
        rs.sreg_written.insert(reg);
        rs.sreg_input.erase(reg);
        rs.sreg_srt.erase(reg);
    }
}

// Complex CFG dispatch and the narrow loop structurizers persist B64 mask values, but not the
// separate one-word-validity state required by Wave32 aliases. Conservatively find any B32 mask
// copy that the region could create. The source set is deliberately path-insensitive: a pair made a
// mask on any path may reach a later copy on another dispatcher edge, and rejecting an impossible
// ordering is safer than silently restoring a stale Boolean.
bool has_unpersisted_b32_mask_lifetime(const std::vector<Rdna2Inst>& ins,
                                      uint32_t lo, uint32_t hi,
                                      const RegState& entry) {
    std::set<int> possible_mask_sources{106, 126}; // VCC_LO and EXEC_LO
    for (const auto& kv : entry.sreg_bool) possible_mask_sources.insert(kv.first);
    for (int reg : entry.sreg_bool_b32) possible_mask_sources.insert(reg);
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            if (scalar_write_is_b64_mask(in, base)) possible_mask_sources.insert(base);
        });
    }
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi || in.fmt != Rdna2Format::SOP1)
            continue;
        if (in.opcode == 0x09) return true; // s_wqm_b32 creates/consumes the same width state
        if (in.opcode != 0x03 && in.opcode != 0x07) continue;
        const bool register_source = in.src[0].kind == OperandKind::SGPR ||
                                     in.src[0].kind == OperandKind::Special;
        if ((register_source && possible_mask_sources.contains(in.src[0].value)) ||
            in.dst.value == 126)
            return true;
    }
    return false;
}

} // namespace

int shader_max_vgpr(const std::vector<Rdna2Inst>& ins) {
    int highest = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const uint32_t source_span = rdna2_vgpr_source_span(in, source);
            if (source_span)
                highest = std::max(highest,
                    in.src[source].value + static_cast<int>(source_span) - 1);
        }
        const uint32_t destination_span = rdna2_vgpr_destination_span(in);
        if (destination_span)
            highest = std::max(highest,
                in.dst.value + static_cast<int>(destination_span) - 1);
    }
    return highest;
}

// Registers WRITTEN in the pc range [lo, hi): candidates for an OpPhi at the loop header. Over-
// approximation is safe (an extra phi for a non-carried value merges equal values). Mirrors emit_alu's
// write targets, INCLUDING multi-register writes (MIMG dmask -> N consecutive VGPRs, SMEM -> N SGPRs) so
// no genuinely-carried register is missed (a missing phi = an undominated use = invalid SPIR-V).
void loop_written_regs(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                       std::set<int>& vregs, std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        switch (in.fmt) {
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) sregs.insert(in.dst.value);        // v_readfirstlane -> SGPR
                else if (in.opcode == kVop1OpcodeMovreldB32)             // v_movreld: any observable
                    for (int reg = in.dst.value; reg <= shader_max_vgpr(ins); ++reg)
                        vregs.insert(reg);                               // VDST+M0 target
                else vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP2: case Rdna2Format::VOP3P:
                vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360) sregs.insert(in.dst.value);       // v_readlane -> SGPR
                else {
                    for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                        vregs.insert(in.dst.value + (int)k);
                }
                break;                                                    // (writelane: slots, not SSA)
            case Rdna2Format::DS:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
            case Rdna2Format::FLAT:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                if (const int tfe_status = rdna2_tfe_status_vgpr(in); tfe_status >= 0)
                    vregs.insert(tfe_status);
                break;
            case Rdna2Format::SOP1:
                sregs.insert(in.dst.value); break;
            case Rdna2Format::SOPK:
                if (sopk_writes_scalar_data(in.opcode)) sregs.insert(in.dst.value);
                break;
            case Rdna2Format::SOP2:
                // s_lshr_b64 -> EXEC is modeled only in the per-lane mask domain. It does not
                // produce scalar SGPR data, so carrying a scalar value through a loop/if merge is
                // both unnecessary and semantically wrong.
                if (in.opcode != 0x21 || (in.dst.value != 126 && in.dst.value != 127)) {
                    uint32_t words = 1;
                    if (in.opcode == 0x0b)
                        words = scalar_write_width(in);
                    else if (in.opcode == 0x1f || in.opcode == 0x21)
                        words = 2;
                    for (uint32_t word = 0; word < words; ++word)
                        sregs.insert(in.dst.value + static_cast<int>(word));
                }
                break;
            case Rdna2Format::SMEM: {                                      // s_load/s_buffer_load: N consecutive SGPRs
                uint32_t n = 1; switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; }
                for (uint32_t k = 0; k < n; k++) sregs.insert(in.dst.value + (int)k); break;
            }
            default: break;                          // VOPC/SOPC write VCC/SCC — handled by their own phis
        }
    }
}

// PC-relative EMBEDDED-TABLE detection (#273). Compilers put small constant lookup tables (a 4x4
// ordered-dither matrix, etc.) directly after the shader code and address them with an inline-built
// V#: `s_getpc_b64 s[a:a+1]; s_add_u32 sa, <off>, sa; s_addc_u32 sa+1, 0, sa+1; s_mov_b32 sa+2,
// <num_records>; s_mov_b32 sa+3, <cfg>; buffer_load_dwordx4 v, v, s[a:a+3], 0 offen`. The table
// BYTES are inside the code blob we are recompiling, so the load can be folded to a compile-time
// constant lookup. This pass does forward constant propagation over the LINEAR stream (facts are
// killed on any other write; a fact is only used when no branch target enters between its def and
// the load) and returns: mubuf pc -> the table's dwords copied out of the blob.
// CONFIDENCE: MED-HIGH — exact for the compiler idiom (verified against DOLL's dither PS); every
// guard failure falls back to the old reject.
struct PcrelTables {
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem;
};

PcrelTables detect_pcrel_tables(
        const std::vector<Rdna2Inst>& ins, const uint32_t* code, size_t dwords,
        size_t* required_dwords = nullptr) {
    PcrelTables out;
    std::unordered_map<int, uint64_t> pcoff;   // reg -> byte offset into the code blob (pair LO half)
    std::unordered_set<int> pchi;              // regs holding the matching HI half
    std::unordered_map<int, uint32_t> kconst;  // reg -> compile-time constant (s_mov_b32 literal/inline)
    std::unordered_map<int, uint32_t> fact_pc; // reg -> pc where its current fact was established
    std::vector<uint32_t> br_targets;          // all branch targets (for the entry-soundness check)
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03)
            br_targets.push_back(in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16);
    }
    auto kill = [&](int r) { pcoff.erase(r); pchi.erase(r); kconst.erase(r); fact_pc.erase(r); };
    auto imm_of = [&](const Operand& o, const Rdna2Inst& in, uint32_t* v) -> bool {
        if (o.kind == OperandKind::Literal)   { *v = in.literal; return true; }
        if (o.kind == OperandKind::InlineInt) { *v = (uint32_t)o.value; return true; }
        if (o.kind == OperandKind::SGPR)      { auto it = kconst.find(o.value);
                                                if (it != kconst.end()) { *v = it->second; return true; } }
        return false;
    };
    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                kill(in.dst.value);
                if (in.opcode == 0x1f) {                     // s_getpc_b64: pair = address of NEXT inst
                    kill(in.dst.value + 1);
                    pcoff[in.dst.value] = (uint64_t)(in.pc + in.len_dwords) * 4u;
                    pchi.insert(in.dst.value + 1);
                    fact_pc[in.dst.value] = in.pc; fact_pc[in.dst.value + 1] = in.pc;
                } else if (in.opcode == 0x03) {              // s_mov_b32 <const>
                    uint32_t v;
                    if (imm_of(in.src[0], in, &v)) { kconst[in.dst.value] = v; fact_pc[in.dst.value] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_mov_b64 (pair write)
                    kill(in.dst.value + 1);
                }
                break;
            case Rdna2Format::SOP2: {
                int d = in.dst.value;
                if (in.opcode == 0x00) {                     // s_add_u32: pcrel-lo + immediate
                    uint32_t imm; uint64_t base; bool got = false;
                    if (in.src[0].kind == OperandKind::SGPR && pcoff.count(in.src[0].value) &&
                        imm_of(in.src[1], in, &imm)) { base = pcoff[in.src[0].value]; got = true; }
                    else if (in.src[1].kind == OperandKind::SGPR && pcoff.count(in.src[1].value) &&
                             imm_of(in.src[0], in, &imm)) { base = pcoff[in.src[1].value]; got = true; }
                    kill(d);
                    if (got) { pcoff[d] = base + imm; fact_pc[d] = in.pc; }
                } else if (in.opcode == 0x04) {              // s_addc_u32: pcrel-hi + 0 (+carry; a real
                    // carry needs the lo word to wrap over a <4 KB offset — not a shader-blob shape)
                    uint32_t imm;
                    bool hi_ok = (in.src[0].kind == OperandKind::SGPR && pchi.count(in.src[0].value) &&
                                  imm_of(in.src[1], in, &imm) && imm == 0) ||
                                 (in.src[1].kind == OperandKind::SGPR && pchi.count(in.src[1].value) &&
                                  imm_of(in.src[0], in, &imm) && imm == 0);
                    kill(d);
                    if (hi_ok) { pchi.insert(d); fact_pc[d] = in.pc; }
                } else {
                    kill(d);
                    if (in.opcode == 0x29) kill(d + 1);      // s_bfe_u64 writes a pair
                }
                break;
            }
            case Rdna2Format::SOPK:
                // s_movk_i32 is also a common way to fill V# word2 (num_records). Astro Bot's
                // fullscreen-table VS uses it immediately before s_getpc_b64; treating every SOPK
                // as an unknown write prevented the otherwise-proven embedded table from folding.
                // The immediate is sign-extended by the ISA, matching the scalar register value.
                if (sopk_writes_scalar_data(in.opcode)) kill(in.dst.value);
                if (in.opcode == 0x00) {
                    kconst[in.dst.value] = static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
                    fact_pc[in.dst.value] = in.pc;
                }
                break;
            case Rdna2Format::SMEM: {
                uint32_t n = 1;
                switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; default: break; }
                // Scalar-buffer form of the same compiler idiom: SBASE is the getpc-built V#, while
                // SOFFSET selects a row from the embedded table. Inspect it before killing SDATA facts.
                const int sb = in.src[0].value;
                if (in.opcode >= 0x8 && in.opcode <= 0xC && pcoff.count(sb) &&
                    pchi.count(sb + 1) && kconst.count(sb + 2) && kconst.count(sb + 3)) {
                    const uint32_t nrec = kconst[sb + 2];
                    const uint64_t off = pcoff[sb];
                    uint32_t newest = 0;
                    for (int r : {sb, sb + 1, sb + 2, sb + 3}) {
                        auto fact = fact_pc.find(r);
                        if (fact != fact_pc.end() && fact->second > newest) newest = fact->second;
                    }
                    bool entered = false;
                    for (uint32_t target : br_targets)
                        if (target > newest && target <= in.pc) { entered = true; break; }
                    const bool supported_soffset =
                        (in.src[1].kind == OperandKind::Special && in.src[1].value == 125) ||
                        in.src[1].kind == OperandKind::SGPR ||
                        (in.src[1].kind == OperandKind::Special && in.src[1].value >= 106 &&
                         in.src[1].value <= 123) ||
                        (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0);
                    if (!entered && supported_soffset && static_cast<int32_t>(in.literal) >= 0 &&
                        !(off & 3u) && !(nrec & 3u) && nrec && nrec <= 1024 &&
                        off / 4 + nrec / 4 <= dwords) {
                        if (required_dwords)
                            *required_dwords = std::max(*required_dwords,
                                static_cast<size_t>(off / 4 + nrec / 4));
                        out.smem[in.pc] = std::vector<uint32_t>(
                            code + off / 4, code + off / 4 + nrec / 4);
                    }
                }
                for (uint32_t k = 0; k < n; k++) kill(in.dst.value + (int)k);
                break;
            }
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) kill(in.dst.value);   // v_readfirstlane -> SGPR
                break;
            case Rdna2Format::VOPC:
                if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)
                    { kill(in.dst.value); kill(in.dst.value + 1); }   // e64 compare -> SGPR pair
                break;
            case Rdna2Format::VOP3:
                if (in.sdst.kind == OperandKind::SGPR) { kill(in.sdst.value); kill(in.sdst.value + 1); }
                break;
            case Rdna2Format::MUBUF: {
                if (in.opcode < 0xCu || in.opcode > 0xFu) break;   // raw loads only
                const int sb = in.src[1].value;                    // SRSRC base SGPR of the V# quad
                if (!pcoff.count(sb) || !pchi.count(sb + 1) || !kconst.count(sb + 2)) break;
                const uint32_t nrec = kconst[sb + 2];              // V# word2 = num_records (bytes here)
                const bool offen = (in.literal >> 12) & 1u, idxen = (in.literal >> 13) & 1u;
                const bool soff0 = (in.src[2].kind == OperandKind::Special && in.src[2].value == 125) ||
                                   (in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0);
                const uint64_t off = pcoff[sb];
                // Entry soundness: no branch may enter between the newest contributing fact and here.
                uint32_t newest = 0;
                for (int r : {sb, sb + 1, sb + 2}) { auto it = fact_pc.find(r); if (it != fact_pc.end() && it->second > newest) newest = it->second; }
                bool entered = false;
                for (uint32_t t : br_targets) if (t > newest && t <= in.pc) { entered = true; break; }
                if (entered) break;
                if (idxen || !soff0 || (off & 3u) || (nrec & 3u) || nrec == 0 || nrec > 1024 ||
                    off / 4 + nrec / 4 > dwords) break;            // unprovable / out of window -> reject
                if (required_dwords)
                    *required_dwords = std::max(*required_dwords,
                                                static_cast<size_t>(off / 4 + nrec / 4));
                out.mubuf[in.pc] = std::vector<uint32_t>(code + off / 4, code + off / 4 + nrec / 4);
                (void)offen;                                       // offen just adds the runtime index
                break;
            }
            default: break;   // formats that don't write SGPRs
        }
    }
    return out;
}

namespace {

const Rdna2Inst* last_scalar_writer(const std::vector<Rdna2Inst>& ins, uint32_t before_pc,
                                    int reg) {
    const Rdna2Inst* result = nullptr;
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= before_pc) break;
        bool writes_reg = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            writes_reg |= reg >= base && reg < base + static_cast<int>(width);
        });
        if (writes_reg) result = &in;
    }
    return result;
}

bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

bool immediate_operand(const Operand& operand, const Rdna2Inst& in, uint32_t& value) {
    if (operand.kind == OperandKind::Literal) { value = in.literal; return true; }
    if (operand.kind == OperandKind::InlineInt) {
        value = static_cast<uint32_t>(operand.value);
        return true;
    }
    return false;
}

bool binary_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return (reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate)) ||
           (reg_operand(in.src[1], reg) && immediate_operand(in.src[0], in, immediate));
}

bool ordered_reg_immediate(const Rdna2Inst& in, int reg, uint32_t& immediate) {
    return reg_operand(in.src[0], reg) && immediate_operand(in.src[1], in, immediate);
}

bool binary_regs(const Rdna2Inst& in, int a, int b) {
    return (reg_operand(in.src[0], a) && reg_operand(in.src[1], b)) ||
           (reg_operand(in.src[0], b) && reg_operand(in.src[1], a));
}

uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

PcrelDispatchInfo detect_pcrel_dispatch(const std::vector<Rdna2Inst>& ins,
                                        const uint32_t* code, size_t dwords,
                                        size_t program_dwords) {
    PcrelDispatchInfo out;
    if (!code || ins.empty()) return out;

    std::unordered_set<uint32_t> instruction_pcs;
    std::vector<uint32_t> branch_targets;
    for (const auto& in : ins) {
        instruction_pcs.insert(in.pc);
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03)
            branch_targets.push_back(scalar_branch_target(in));
    }

    for (const auto& setpc : ins) {
        if (setpc.is_end || setpc.fmt != Rdna2Format::SOP1 || setpc.opcode != 0x20) continue;
        const int jump_lo = setpc.src[0].value;
        const Rdna2Inst* jump_add = last_scalar_writer(ins, setpc.pc, jump_lo);
        const Rdna2Inst* jump_addc = last_scalar_writer(ins, setpc.pc, jump_lo + 1);
        if (!jump_add || !jump_addc || jump_add->fmt != Rdna2Format::SOP2 ||
            jump_add->opcode != 0x00 || jump_addc->fmt != Rdna2Format::SOP2 ||
            jump_addc->opcode != 0x04 || jump_add->pc >= jump_addc->pc) continue;

        int table_lo = -1;
        for (const Operand& source : jump_add->src) {
            if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
                source.value != jump_lo) table_lo = source.value;
        }
        if (table_lo < 0 || !binary_regs(*jump_add, jump_lo, table_lo) ||
            !binary_regs(*jump_addc, jump_lo + 1, table_lo + 1)) continue;
        const Rdna2Inst* target_getpc = last_scalar_writer(ins, jump_add->pc, jump_lo);
        if (!target_getpc || target_getpc->fmt != Rdna2Format::SOP1 ||
            target_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* table_load = last_scalar_writer(ins, jump_add->pc, table_lo);
        if (!table_load || table_load != last_scalar_writer(ins, jump_add->pc, table_lo + 1) ||
            table_load->fmt != Rdna2Format::SMEM || table_load->opcode != 0x01 ||
            table_load->literal != 0 || table_load->src[0].kind != OperandKind::SGPR) continue;
        const int table_base_lo = table_load->src[0].value;
        const int selector = table_load->src[1].value;

        const Rdna2Inst* table_add = last_scalar_writer(ins, table_load->pc, table_base_lo);
        const Rdna2Inst* table_addc = last_scalar_writer(ins, table_load->pc, table_base_lo + 1);
        if (!table_add || !table_addc || table_add->fmt != Rdna2Format::SOP2 ||
            table_add->opcode != 0x00 || table_addc->fmt != Rdna2Format::SOP2 ||
            table_addc->opcode != 0x04 || table_add->pc >= table_addc->pc) continue;
        uint32_t table_delta = 0, high_zero = 1;
        if (!binary_reg_immediate(*table_add, table_base_lo, table_delta) ||
            !binary_reg_immediate(*table_addc, table_base_lo + 1, high_zero) || high_zero != 0)
            continue;
        const Rdna2Inst* table_getpc = last_scalar_writer(ins, table_add->pc, table_base_lo);
        if (!table_getpc || table_getpc->fmt != Rdna2Format::SOP1 ||
            table_getpc->opcode != 0x1f) continue;

        const Rdna2Inst* shift = last_scalar_writer(ins, table_load->pc, selector);
        if (!shift || shift->fmt != Rdna2Format::SOP2 || shift->opcode != 0x1e) continue;
        uint32_t shift_amount = 0;
        // s_lshl_b32 is ordered: selector << 3 scales the qword index, while 3 << selector
        // is a different program and must not be admitted by the commutative matcher.
        if (!ordered_reg_immediate(*shift, selector, shift_amount) || shift_amount != 3) continue;
        const Rdna2Inst* clamp = last_scalar_writer(ins, shift->pc, selector);
        if (!clamp || clamp->fmt != Rdna2Format::SOP2 || clamp->opcode != 0x07) continue;
        uint32_t selector_max = 0;
        if (!binary_reg_immediate(*clamp, selector, selector_max) || selector_max > 63) continue;

        const Rdna2Inst* adjust = last_scalar_writer(ins, clamp->pc, selector);
        int32_t selector_addend = 0;
        if (adjust && adjust->fmt == Rdna2Format::SOP2 &&
            (adjust->opcode == 0x00 || adjust->opcode == 0x02)) {
            uint32_t addend = 0;
            if (!binary_reg_immediate(*adjust, selector, addend)) continue;
            selector_addend = static_cast<int32_t>(addend);
        } else {
            adjust = nullptr;
        }
        const Rdna2Inst* selector_load = last_scalar_writer(
            ins, adjust ? adjust->pc : clamp->pc, selector);
        if (!selector_load || selector_load->fmt != Rdna2Format::SMEM ||
            selector_load->opcode != 0x08 || selector_load->src[0].kind != OperandKind::SGPR ||
            selector_load->src[1].kind != OperandKind::Special ||
            selector_load->src[1].value != 125) continue;

        const uint64_t table_byte =
            static_cast<uint64_t>(table_getpc->pc + table_getpc->len_dwords) * 4u + table_delta;
        const size_t entry_count = static_cast<size_t>(selector_max) + 1;
        if ((table_byte & 7u) || table_byte / 4 < program_dwords ||
            table_byte / 4 + entry_count * 2 > dwords) continue;

        std::vector<uint32_t> targets;
        targets.reserve(entry_count);
        const int64_t target_pc_byte =
            static_cast<int64_t>(target_getpc->pc + target_getpc->len_dwords) * 4;
        bool table_ok = true;
        for (size_t index = 0; index < entry_count; ++index) {
            const size_t word = static_cast<size_t>(table_byte / 4) + index * 2;
            const int64_t relative = static_cast<int64_t>(
                (static_cast<uint64_t>(code[word + 1]) << 32) | code[word]);
            const int64_t target_byte = target_pc_byte + relative;
            if (target_byte < 0 || (target_byte & 3) ||
                target_byte / 4 > static_cast<int64_t>(UINT32_MAX) ||
                !instruction_pcs.contains(static_cast<uint32_t>(target_byte / 4))) {
                table_ok = false;
                break;
            }
            targets.push_back(static_cast<uint32_t>(target_byte / 4));
        }
        if (!table_ok || targets.empty()) continue;
        const uint32_t merge_pc = *std::max_element(targets.begin(), targets.end());
        if (merge_pc <= setpc.pc) continue;
        for (uint32_t target : targets) if (target < setpc.pc + setpc.len_dwords || target > merge_pc)
            table_ok = false;
        if (!table_ok) continue;

        const std::vector<uint32_t> setup = {
            selector_load->pc,
            adjust ? adjust->pc : UINT32_MAX,
            clamp->pc, shift->pc, table_getpc->pc, table_add->pc, table_addc->pc,
            table_load->pc, target_getpc->pc, jump_add->pc, jump_addc->pc, setpc.pc,
        };
        const uint32_t setup_first = selector_load->pc;
        for (uint32_t target : branch_targets) {
            if (target > setup_first && target <= setpc.pc) { table_ok = false; break; }
        }
        if (!table_ok) continue;

        out.valid = true;
        out.selector_sgpr_base = static_cast<uint32_t>(selector_load->src[0].value);
        out.selector_byte_offset = selector_load->literal;
        out.selector_addend = selector_addend;
        out.selector_max = selector_max;
        out.setpc_pc = setpc.pc;
        out.merge_pc = merge_pc;
        out.required_dwords = static_cast<size_t>(table_byte / 4) + entry_count * 2;
        out.target_pcs = std::move(targets);
        for (uint32_t pc : setup) if (pc != UINT32_MAX) out.setup_pcs.push_back(pc);
        return out;
    }
    return out;
}

bool specialize_pcrel_dispatch(std::vector<Rdna2Inst>& ins, const PcrelDispatchInfo& info,
                               uint32_t selected_target) {
    if (!info.valid || std::find(info.target_pcs.begin(), info.target_pcs.end(), selected_target) ==
                           info.target_pcs.end()) return false;
    std::unordered_set<uint32_t> remove(info.setup_pcs.begin(), info.setup_pcs.end());

    // A compiler may jump over an alternate entry prologue before it starts the dispatch setup. Fold
    // only forward unconditional branches wholly contained in that prelude; any external entry into a
    // skipped range makes the specialization unprovable.
    for (const auto& branch : ins) {
        if (branch.pc >= info.setpc_pc || branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x02)
            continue;
        const uint32_t target = scalar_branch_target(branch);
        if (target <= branch.pc || target > info.setpc_pc) return false;
        for (const auto& other : ins) {
            if (other.fmt != Rdna2Format::SOPP || other.pc == branch.pc || other.opcode < 0x02 ||
                other.opcode > 0x09 || other.opcode == 0x03) continue;
            const uint32_t entered = scalar_branch_target(other);
            if (entered > branch.pc + branch.len_dwords && entered < target) return false;
        }
        remove.insert(branch.pc);
        for (const auto& skipped : ins)
            if (skipped.pc >= branch.pc + branch.len_dwords && skipped.pc < target)
                remove.insert(skipped.pc);
    }

    uint32_t selected_end = info.merge_pc;
    for (const auto& in : ins) {
        if (in.pc < selected_target || in.pc >= info.merge_pc) continue;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        // Internal forward branches and loop back-edges remain in the selected routine and are
        // validated/structured by emit_body. Only the compiler's route terminator jumps to the
        // common merge and can be removed as a now-redundant branch.
        if (scalar_branch_target(in) != info.merge_pc) continue;
        selected_end = in.pc;
        remove.insert(in.pc);
        break;
    }

    std::vector<Rdna2Inst> specialized;
    specialized.reserve(ins.size());
    for (const auto& in : ins) {
        const bool prelude = in.pc < info.setpc_pc;
        const bool selected = in.pc >= selected_target && in.pc < selected_end;
        const bool merge = in.pc >= info.merge_pc;
        if ((prelude || selected || merge) && !remove.contains(in.pc)) specialized.push_back(in);
    }
    if (specialized.empty()) return false;

    // No surviving branch may enter an omitted alternative. The ordinary structurizer performs the
    // remaining detailed CFG checks after this coarse specialization boundary check.
    std::unordered_set<uint32_t> retained;
    uint32_t end_pc = 0;
    for (const auto& in : specialized) retained.insert(in.pc);
    for (const auto& in : specialized) if (in.is_end) { end_pc = in.pc; break; }
    for (const auto& in : specialized) {
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03 || in.simm16 < 0) continue;
        const uint32_t target = scalar_branch_target(in);
        // Existing forward-if validation accepts a compiler early-out just beyond the primary
        // S_ENDPGM only after proving that raw target terminates immediately. Preserve that case for
        // the detailed validator; targets into an omitted alternative still fail here.
        if (!retained.contains(target) && target <= end_pc) return false;
    }
    ins = std::move(specialized);
    return true;
}

struct ShaderConstantValue {
    bool known = false;
    uint32_t value = 0;
};

ShaderConstantValue shader_constant_operand(
        const std::vector<Rdna2Inst>& ins, size_t block_first, size_t use_index,
        const Rdna2Inst& use, const Operand& operand, uint32_t depth) {
    if (depth > 16) return {};
    if (operand.kind == OperandKind::InlineInt)
        return {true, static_cast<uint32_t>(operand.value)};
    if (operand.kind == OperandKind::Literal)
        return use.has_literal ? ShaderConstantValue{true, use.literal} : ShaderConstantValue{};
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special)
        return {};

    const int reg = operand.value;
    for (size_t i = use_index; i-- > block_first;) {
        const Rdna2Inst& writer = ins[i];
        // VCC_LO/HI can temporarily hold ordinary scalar data, as in Astro's exact branch setup,
        // but any intervening vector ALU may replace the architectural VCC pair implicitly. The
        // shared scalar-writer inventory intentionally models that in the Bool domain, so reject it
        // explicitly here instead of walking past a VOPC to an obsolete scalar definition.
        if ((reg == 106 || reg == 107) &&
            (writer.fmt == Rdna2Format::VOP1 || writer.fmt == Rdna2Format::VOP2 ||
             writer.fmt == Rdna2Format::VOPC || writer.fmt == Rdna2Format::VOP3 ||
             writer.fmt == Rdna2Format::VOP3P))
            return {};
        bool writes = false;
        uint32_t width = 0;
        int base = -1;
        for_each_scalar_write(writer, [&](int candidate_base, uint32_t candidate_width) {
            if (reg >= candidate_base &&
                reg < candidate_base + static_cast<int>(candidate_width)) {
                writes = true;
                base = candidate_base;
                width = candidate_width;
            }
        });
        if (!writes) continue;

        // Only one-dword pure scalar data writers participate. A pair write, memory result, lane
        // read, or wave-mask producer is intentionally not a shader-constant proof.
        if (base != reg || width != 1) return {};
        if (writer.fmt == Rdna2Format::SOP1 && writer.opcode == 0x03) { // s_mov_b32
            return shader_constant_operand(
                ins, block_first, i, writer, writer.src[0], depth + 1);
        }
        if (writer.fmt != Rdna2Format::SOP2) return {};

        const ShaderConstantValue lhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[0], depth + 1);
        const ShaderConstantValue rhs = shader_constant_operand(
            ins, block_first, i, writer, writer.src[1], depth + 1);
        if (!lhs.known || !rhs.known) return {};

        switch (writer.opcode) {
            case 0x00: return {true, lhs.value + rhs.value}; // s_add_u32
            case 0x01: return {true, lhs.value - rhs.value}; // s_sub_u32
            case 0x02: return {true, lhs.value + rhs.value}; // s_add_i32
            case 0x03: return {true, lhs.value - rhs.value}; // s_sub_i32
            case 0x0e: return {true, lhs.value & rhs.value}; // s_and_b32
            case 0x10: return {true, lhs.value | rhs.value}; // s_or_b32
            case 0x12: return {true, lhs.value ^ rhs.value}; // s_xor_b32
            // Shifts take the amount from S1[4:0] -- RDNA2 ISA -- so the `& 31` is required.
            //
            // It is deliberately UNTESTED, because on this host it is untestable: x86 `shl` already
            // masks its count to 5 bits, so an unmasked `lhs.value << rhs.value` yields the SAME
            // answer here (measured: naive `5u << 32` prints 5). The mask is therefore not fixing a
            // wrong result on x86; it is removing undefined behaviour that a compiler is entitled to
            // exploit, and that would diverge on a host whose shift does not wrap. Do not add a
            // regression arm claiming to prove it -- such an arm passes with the mask removed, which
            // makes it a control that cannot fail. The arms below cover only what is observable:
            // that these opcodes fold at all.
            //
            // Opcodes verified against llvm-mc (gfx1030), not against this file's own tables: the
            // decoder that produces the listing you would otherwise check them with is upstream of
            // them, so it cannot check them. #2481 records a mnemonic error that survived three
            // internally consistent anchors and inverted a frontier conclusion.
            case 0x1e: return {true, lhs.value << (rhs.value & 31u)};  // s_lshl_b32
            case 0x20: return {true, lhs.value >> (rhs.value & 31u)};  // s_lshr_b32
            case 0x22: return {true, static_cast<uint32_t>(            // s_ashr_i32
                std::bit_cast<int32_t>(lhs.value) >> (rhs.value & 31u))};
            case 0x26: return {true, lhs.value * rhs.value}; // s_mul_i32
            default: return {};
        }
    }
    // No in-block writer means an entry/user SGPR or other runtime state. Never specialize it.
    return {};
}

bool shader_constant_compare(const std::vector<Rdna2Inst>& ins, size_t block_first,
                             size_t compare_index, bool& result) {
    const Rdna2Inst& compare = ins[compare_index];
    if (compare.fmt != Rdna2Format::SOPC || compare.opcode > 0x0b) return false;
    const ShaderConstantValue lhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[0], 0);
    const ShaderConstantValue rhs = shader_constant_operand(
        ins, block_first, compare_index, compare, compare.src[1], 0);
    if (!lhs.known || !rhs.known) return false;
    const int32_t signed_lhs = std::bit_cast<int32_t>(lhs.value);
    const int32_t signed_rhs = std::bit_cast<int32_t>(rhs.value);

    switch (compare.opcode) {
        case 0x00: result = signed_lhs == signed_rhs; break;
        case 0x01: result = signed_lhs != signed_rhs; break;
        case 0x02: result = signed_lhs >  signed_rhs; break;
        case 0x03: result = signed_lhs >= signed_rhs; break;
        case 0x04: result = signed_lhs <  signed_rhs; break;
        case 0x05: result = signed_lhs <= signed_rhs; break;
        case 0x06: result = lhs.value == rhs.value; break;
        case 0x07: result = lhs.value != rhs.value; break;
        case 0x08: result = lhs.value >  rhs.value; break;
        case 0x09: result = lhs.value >= rhs.value; break;
        case 0x0a: result = lhs.value <  rhs.value; break;
        case 0x0b: result = lhs.value <= rhs.value; break;
        default: return false;
    }
    return true;
}

bool scalar_cfg_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
        sopp_opcode_is_direct_branch(in.opcode);
}

void prune_scalar_cfg_reachability(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::vector<bool> reachable(ins.size(), false);
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t i = pending.back();
        pending.pop_back();
        if (i >= ins.size() || reachable[i]) continue;
        reachable[i] = true;
        const Rdna2Inst& in = ins[i];
        if (in.is_end || (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) continue;
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) pending.push_back(target->second);
            if (in.opcode == 0x02) continue;
        }
        if (i + 1 < ins.size()) pending.push_back(i + 1);
    }

    std::vector<Rdna2Inst> retained;
    retained.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
        if (reachable[i]) retained.push_back(ins[i]);
    // Once a taken branch's omitted arm is gone, its target is often the next retained instruction.
    // Turn that now-redundant edge into a no-op so the ordinary straight-line/structured paths do not
    // need to reconstruct an empty branch region.
    for (size_t i = 0; i + 1 < retained.size(); ++i) {
        Rdna2Inst& in = retained[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02 &&
            scalar_branch_target(in) == retained[i + 1].pc) {
            in.opcode = 0x00;
            in.simm16 = 0;
            in.words[0] = 0xbf800000u;
        }
    }
    ins = std::move(retained);
}

size_t specialize_shader_constant_branches(std::vector<Rdna2Inst>& ins) {
    if (ins.empty()) return 0;
    // An indirect PC transfer can enter code outside the explicit SOPP graph. Keep the whole shader
    // unspecialized rather than treating its lexical successor as the only possible destination.
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::SOP1 &&
                   in.opcode >= 0x20 && in.opcode <= 0x22;
        })) return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    std::set<size_t> block_starts{0};
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (scalar_cfg_branch(in)) {
            const auto target = index_by_pc.find(scalar_branch_target(in));
            if (target != index_by_pc.end()) block_starts.insert(target->second);
        }
        if ((scalar_cfg_branch(in) || in.is_end ||
             (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12)) &&
            i + 1 < ins.size())
            block_starts.insert(i + 1);
    }

    size_t specialized_count = 0;
    for (size_t i = 1; i < ins.size(); ++i) {
        Rdna2Inst& branch = ins[i];
        if (branch.fmt != Rdna2Format::SOPP ||
            (branch.opcode != 0x04 && branch.opcode != 0x05) || branch.simm16 <= 0)
            continue; // shader-constant SCC only; VCCZ/EXECZ remain runtime wave conditions
        if (!index_by_pc.contains(scalar_branch_target(branch)))
            continue; // do not specialize an exit beyond the decoded instruction graph
        const Rdna2Inst& compare = ins[i - 1];
        if (compare.pc + compare.len_dwords != branch.pc ||
            compare.fmt != Rdna2Format::SOPC)
            continue;
        const auto block = block_starts.upper_bound(i - 1);
        const size_t block_first = block == block_starts.begin() ? 0 : *std::prev(block);
        if (block_first > i - 1) continue;

        bool scc = false;
        if (!shader_constant_compare(ins, block_first, i - 1, scc)) continue;
        const bool taken = branch.opcode == 0x05 ? scc : !scc;
        if (taken) {
            branch.opcode = 0x02; // s_branch: retain the exact immediate target
            branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
        } else {
            branch.opcode = 0x00; // s_nop 0: retain fallthrough
            branch.simm16 = 0;
            branch.words[0] = 0xbf800000u;
        }
        ++specialized_count;
    }
    if (!specialized_count) return 0;

    // Entry-rooted reachability after replacing proven conditions. Unknown conditional branches
    // retain both successors, so an unresolved resource remains in the instruction stream whenever
    // any runtime path can execute it.
    prune_scalar_cfg_reachability(ins);
    return specialized_count;
}

size_t specialize_proven_null_bvh_exits(std::vector<Rdna2Inst>& ins,
                                        const ShaderResourceTable* rt,
                                        uint32_t wave_size) {
    if (!rt || (wave_size != 32 && wave_size != 64) || ins.empty()) return 0;
    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);

    size_t specialized = 0;
    for (const ShaderResource& resource : rt->resources) {
        if (!is_proven_null_bvh(resource)) continue;
        const auto found = index_by_pc.find(resource.fetch_pc);
        if (found == index_by_pc.end()) continue;
        const size_t ray_index = found->second;
        if (ray_index + 5 >= ins.size()) continue;
        const Rdna2Inst& ray = ins[ray_index];
        const Rdna2Inst& wait = ins[ray_index + 1];
        const Rdna2Inst& compare = ins[ray_index + 2];
        const Rdna2Inst& scalar_compare = ins[ray_index + 3];
        const Rdna2Inst& exec_copy = ins[ray_index + 4];
        Rdna2Inst& exit = ins[ray_index + 5];

        // Exact Wave32 no-hit idiom:
        //   image_bvh_intersect_ray vN..vN+3, NULL_BVH
        //   s_waitcnt ...
        //   v_cmp_ne_u32 sM, -1, vN
        //   s_cmp_lg_u32 0, sM
        //   s_mov_b32 exec_lo, vcc_lo
        //   s_cbranch_scc0 EXIT
        // The Wave64 sibling uses the complete two-word mask instead:
        //   v_cmp_ne_u32 s[M:M+1], -1, vN
        //   s_cmp_lg_u64 s[M:M+1], 0
        //   s_mov_b64 exec, vcc
        // The null lowering writes -1 to every ray result for each active lane. The saved compare
        // mask is therefore exactly zero, making SCC zero and the exit unconditional. Requiring the
        // exact width-specific compare and EXEC copy avoids assuming anything about an unobserved
        // mask half. Adjacency and exact register links prevent a nearby unrelated compare from
        // proving the branch.
        const bool ray_shape = ray.fmt == Rdna2Format::MIMG && ray.opcode == 0xe6u &&
            ray.mimg_dmask == 0xfu && ray.dst.kind == OperandKind::VGPR;
        const bool wait_shape = wait.pc == ray.pc + ray.len_dwords &&
            wait.fmt == Rdna2Format::SOPP && wait.opcode == 0x0cu &&
            wait.words[0] == 0xbf8c3f70u; // s_waitcnt vmcnt(0)
        const bool compare_shape = compare.pc == wait.pc + wait.len_dwords &&
            compare.fmt == Rdna2Format::VOPC && compare.opcode == 0xc5u &&
            compare.dst.kind == OperandKind::SGPR &&
            compare.dst.value <= (wave_size == 32 ? 105 : 104) &&
            compare.src[0].kind == OperandKind::InlineInt && compare.src[0].value == -1 &&
            compare.src[1].kind == OperandKind::VGPR && compare.src[1].value == ray.dst.value;
        const bool scalar_shape = scalar_compare.pc == compare.pc + compare.len_dwords &&
            scalar_compare.fmt == Rdna2Format::SOPC &&
            (wave_size == 32
                ? scalar_compare.opcode == 0x07u &&
                  scalar_compare.src[0].kind == OperandKind::InlineInt &&
                  scalar_compare.src[0].value == 0 &&
                  scalar_compare.src[1].kind == OperandKind::SGPR &&
                  scalar_compare.src[1].value == compare.dst.value
                : scalar_compare.opcode == 0x13u &&
                  scalar_compare.src[0].kind == OperandKind::SGPR &&
                  scalar_compare.src[0].value == compare.dst.value &&
                  scalar_compare.src[1].kind == OperandKind::InlineInt &&
                  scalar_compare.src[1].value == 0);
        const bool copy_shape = exec_copy.pc == scalar_compare.pc + scalar_compare.len_dwords &&
            exec_copy.fmt == Rdna2Format::SOP1 &&
            exec_copy.opcode == (wave_size == 32 ? 0x03u : 0x04u) &&
            exec_copy.dst.kind == OperandKind::SGPR && exec_copy.dst.value == 126 &&
            exec_copy.src[0].kind == OperandKind::Special && exec_copy.src[0].value == 106;
        const bool exit_shape = exit.pc == exec_copy.pc + exec_copy.len_dwords &&
            exit.fmt == Rdna2Format::SOPP && exit.opcode == 0x04u && exit.simm16 > 0 &&
            index_by_pc.contains(scalar_branch_target(exit));
        if (!ray_shape || !wait_shape || !compare_shape || !scalar_shape || !copy_shape ||
            !exit_shape)
            continue;

        // Some compiler-generated traversal loops enter with an empty scalar stack, visit one
        // root ray, then pop work written by that ray.  When the dispatch-scoped root is proven
        // null, the exact no-hit branch above reaches the empty-stack test without writing the
        // depth.  That makes the pop/back-edge arm unreachable, which in turn proves that the
        // loop-selected ray sites cannot seed themselves.  Keep this deliberately narrower than
        // ordinary scalar constant propagation: every entry, register relationship, and branch on
        // the first path must match the observed stack idiom.
        auto specialize_empty_stack = [&]() {
            const uint32_t null_exit_pc = scalar_branch_target(exit);
            const auto tail_found = index_by_pc.find(null_exit_pc);
            if (tail_found == index_by_pc.end() || tail_found->second + 1 >= ins.size())
                return false;
            const size_t tail_index = tail_found->second;
            const Rdna2Inst& stack_compare = ins[tail_index];
            Rdna2Inst& stack_exit = ins[tail_index + 1];
            if (stack_compare.fmt != Rdna2Format::SOPC || stack_compare.opcode != 0x07u ||
                stack_compare.src[0].kind != OperandKind::InlineInt ||
                stack_compare.src[0].value != 0 ||
                stack_compare.src[1].kind != OperandKind::SGPR ||
                stack_compare.src[1].value < 0 || stack_compare.src[1].value > 105 ||
                stack_exit.pc != stack_compare.pc + stack_compare.len_dwords ||
                stack_exit.fmt != Rdna2Format::SOPP || stack_exit.opcode != 0x04u ||
                stack_exit.simm16 <= 0 ||
                !index_by_pc.contains(scalar_branch_target(stack_exit)))
                return false;
            // The compiler allocates this scalar stack depth opportunistically (the observed
            // traversal kernels use both s41 and s45). Derive the physical word from the exact
            // empty-stack comparison, then require the initializer and every write check below to
            // agree with it. This broadens only register allocation, not the proven control/data
            // relationship.
            const int stack_reg = stack_compare.src[1].value;
            const uint32_t stack_exit_pc = scalar_branch_target(stack_exit);

            // The first-iteration selector branch targets the block that initializes all four
            // ray results to invalid before the guarded root query.  Requiring this complete
            // eight-instruction prefix prevents a nearby null ray from being mistaken for the
            // traversal root that owns the scalar stack below.
            if (ray_index < 8) return false;
            const size_t root_block = ray_index - 8;
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const Rdna2Inst& init = ins[root_block + lane];
                if (init.fmt != Rdna2Format::VOP1 || init.opcode != 0x01u ||
                    init.dst.kind != OperandKind::VGPR ||
                    init.dst.value != ray.dst.value + static_cast<int>(lane) ||
                    init.src[0].kind != OperandKind::InlineInt || init.src[0].value != -1)
                    return false;
            }
            const Rdna2Inst& mask_copy = ins[root_block + 4];
            const Rdna2Inst& empty_guard = ins[root_block + 5];
            const Rdna2Inst& root_index = ins[root_block + 6];
            const Rdna2Inst& root_nop = ins[root_block + 7];
            const bool mask_copy_shape = wave_size == 32
                ? mask_copy.fmt == Rdna2Format::SOP1 &&
                  mask_copy.opcode == kSop1OpcodeAndSaveexecB32 &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::Special &&
                  mask_copy.src[0].value == 107
                : mask_copy.fmt == Rdna2Format::SOP1 && mask_copy.opcode == 0x24u &&
                  mask_copy.dst.kind == OperandKind::SGPR && mask_copy.dst.value == 106 &&
                  mask_copy.src[0].kind == OperandKind::SGPR;
            if (!mask_copy_shape ||
                empty_guard.fmt != Rdna2Format::SOPP || empty_guard.opcode != 0x08u ||
                scalar_branch_target(empty_guard) != scalar_compare.pc ||
                root_index.fmt != Rdna2Format::VOP1 || root_index.opcode != 0x01u ||
                root_index.dst.kind != OperandKind::VGPR ||
                root_index.dst.value != ray.dst.value ||
                root_index.src[0].kind != OperandKind::SGPR ||
                root_nop.fmt != Rdna2Format::SOPP || root_nop.opcode != 0x00u)
                return false;

            size_t selector_branch_index = SIZE_MAX;
            size_t stack_init_index = SIZE_MAX;
            bool selector_scc = false;
            for (size_t i = root_block; i-- > 2;) {
                const Rdna2Inst& branch = ins[i];
                if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x05u ||
                    scalar_branch_target(branch) != ins[root_block].pc ||
                    ins[i - 1].pc + ins[i - 1].len_dwords != branch.pc ||
                    ins[i - 1].fmt != Rdna2Format::SOPC)
                    continue;
                for (size_t j = i - 1; j-- > 1;) {
                    const Rdna2Inst& selector_init = ins[j];
                    if (selector_init.fmt != Rdna2Format::SOP1 ||
                        selector_init.opcode != 0x03u ||
                        selector_init.dst.kind != OperandKind::SGPR ||
                        selector_init.src[0].kind != OperandKind::InlineInt ||
                        selector_init.src[0].value != 37 ||
                        root_index.src[0].value != selector_init.dst.value)
                        continue;
                    size_t candidate_stack_init = SIZE_MAX;
                    if (wave_size == 32) {
                        const Rdna2Inst& stack_init = ins[j - 1];
                        if (stack_init.pc + stack_init.len_dwords == selector_init.pc &&
                            stack_init.fmt == Rdna2Format::SOP1 &&
                            stack_init.opcode == 0x03u &&
                            stack_init.dst.kind == OperandKind::SGPR &&
                            stack_init.dst.value == stack_reg &&
                            stack_init.src[0].kind == OperandKind::InlineInt &&
                            stack_init.src[0].value == 0)
                            candidate_stack_init = j - 1;
                    } else {
                        // The Wave64 sibling schedules independent scalar address setup between
                        // `stack=0` and `selector=37`. Accept only the nearest in-block write to the
                        // derived stack register, and require it to be the exact zero initializer.
                        for (size_t k = j; k-- > 0;) {
                            bool writes_candidate = false;
                            for_each_scalar_write(ins[k], [&](int base, uint32_t width) {
                                writes_candidate |= base <= stack_reg &&
                                    stack_reg < base + static_cast<int>(width);
                            });
                            if (!writes_candidate) {
                                if (scalar_cfg_branch(ins[k]) || ins[k].is_end) break;
                                continue;
                            }
                            const Rdna2Inst& stack_init = ins[k];
                            if (stack_init.fmt == Rdna2Format::SOP1 &&
                                stack_init.opcode == 0x03u &&
                                stack_init.dst.kind == OperandKind::SGPR &&
                                stack_init.dst.value == stack_reg &&
                                stack_init.src[0].kind == OperandKind::InlineInt &&
                                stack_init.src[0].value == 0)
                                candidate_stack_init = k;
                            break;
                        }
                    }
                    if (candidate_stack_init == SIZE_MAX) continue;
                    bool has_branch = false;
                    for (size_t k = candidate_stack_init + 1; k < i; ++k)
                        has_branch |= scalar_cfg_branch(ins[k]);
                    if (has_branch ||
                        !shader_constant_compare(ins, j, i - 1, selector_scc) ||
                        !selector_scc)
                        continue;
                    selector_branch_index = i;
                    stack_init_index = candidate_stack_init;
                    break;
                }
                if (selector_branch_index != SIZE_MAX) break;
            }
            if (selector_branch_index == SIZE_MAX) return false;

            auto writes_stack = [&](const Rdna2Inst& instruction) {
                bool writes = false;
                for_each_scalar_write(instruction, [&](int base, uint32_t width) {
                    writes |= base <= stack_reg &&
                        stack_reg < base + static_cast<int>(width);
                }, /*wave32_one_word_masks=*/wave_size == 32);
                return writes;
            };
            // Only the selector setup, root/no-hit block, and empty-stack comparison are reachable
            // before the proven exits.  None may alter the initialized stack depth.
            for (size_t i = stack_init_index + 1; i <= selector_branch_index; ++i)
                if (writes_stack(ins[i])) return false;
            for (size_t i = root_block; i <= ray_index + 5; ++i)
                if (writes_stack(ins[i])) return false;
            if (writes_stack(stack_compare)) return false;

            const uint32_t proof_begin = ins[stack_init_index].pc;
            // An indirect transfer or an external edge into the middle of the proof could bypass
            // the zero initializer.  Re-entry at the initializer itself is harmless: it resets the
            // invariant before the selector is evaluated again.
            for (const Rdna2Inst& instruction : ins) {
                if (instruction.fmt == Rdna2Format::SOP1 &&
                    instruction.opcode >= 0x20u && instruction.opcode <= 0x22u)
                    return false;
                if (!scalar_cfg_branch(instruction)) continue;
                const uint32_t target = scalar_branch_target(instruction);
                const bool source_inside = instruction.pc >= proof_begin &&
                    instruction.pc < stack_exit_pc;
                if (!source_inside && target > proof_begin && target < stack_exit_pc)
                    return false;
            }

            Rdna2Inst& selector_branch = ins[selector_branch_index];
            selector_branch.opcode = 0x02u;
            selector_branch.words[0] = 0xbf820000u |
                static_cast<uint16_t>(selector_branch.simm16);
            stack_exit.opcode = 0x02u;
            stack_exit.words[0] = 0xbf820000u |
                static_cast<uint16_t>(stack_exit.simm16);
            return true;
        };

        (void)specialize_empty_stack();
        exit.opcode = 0x02;
        exit.words[0] = 0xbf820000u | static_cast<uint16_t>(exit.simm16);
        ++specialized;
    }
    if (specialized) prune_scalar_cfg_reachability(ins);
    return specialized;
}

bool valid_scalar_pair_base(int base) {
    return base >= 0 && !(base & 1) &&
        (base <= 104 || base == 106 ||
         (base >= 108 && base <= 122) || base == 126);
}

bool scalar_pair_operand(const Operand& operand, int& base) {
    if ((operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) ||
        !valid_scalar_pair_base(operand.value))
        return false;
    base = operand.value;
    return true;
}

bool valid_buffer_resource_base(const Operand& operand) {
    if (operand.kind != OperandKind::SGPR || operand.value < 0 ||
        (operand.value & 3))
        return false;
    return operand.value <= 100 ||
        (operand.value >= 108 && operand.value <= 120);
}

bool zero_record_load_shape(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::MUBUF &&
        in.opcode == kMubufOpcodeLoadDword && in.len_dwords == 2u &&
        in.dst.kind == OperandKind::VGPR && in.dst.value >= 0 &&
        in.src[0].kind == OperandKind::VGPR &&
        valid_buffer_resource_base(in.src[1]) &&
        in.src[2].kind == OperandKind::InlineInt && in.src[2].value == 0 &&
        !in.mubuf_glc && !in.mubuf_dlc && !in.mubuf_lds && !in.mubuf_tfe;
}

bool scalar_instruction_writes_anything(const Rdna2Inst& in) {
    bool writes = false;
    for_each_scalar_write(in, [&](int, uint32_t) { writes = true; });
    return writes;
}

bool dynamic_vgpr_destination(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
}

bool writes_vgpr(const Rdna2Inst& in, int vgpr) {
    if (dynamic_vgpr_destination(in)) return true;
    const uint32_t words = rdna2_vgpr_write_count(in);
    if (words && in.dst.kind == OperandKind::VGPR && in.dst.value <= vgpr &&
        vgpr < in.dst.value + static_cast<int>(words))
        return true;
    return rdna2_tfe_status_vgpr(in) == vgpr;
}

// Prove that every lane active at `and_index` was also active when `load_index` wrote zero. The
// scalar compiler idioms in the live shader elect one lane and later restore a saved mask; tracking
// only subset lineage is enough and deliberately cannot prove an unrelated mask or an expanding
// EXEC write. Conditional skips inside the interval must contain no proof-relevant writes so the
// lexical transfer below represents both paths.
bool zero_load_reaches_and_under_exec_subset(const std::vector<Rdna2Inst>& ins,
                                             size_t load_index, size_t and_index,
                                             const std::unordered_map<uint32_t, size_t>& index_by_pc) {
    const int zero_vgpr = ins[load_index].dst.value;

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& branch = ins[i];
        if (!scalar_cfg_branch(branch)) continue;
        const auto target = index_by_pc.find(scalar_branch_target(branch));
        if (target == index_by_pc.end()) return false;

        const bool source_inside = i > load_index && i < and_index;
        // The compare and branch are part of the proof too: an external edge to either would
        // bypass the zero-reaching definition just as surely as one into the transfer interval.
        const bool target_enters_after_load = target->second > load_index &&
            target->second <= and_index + 2u;
        if (!source_inside && target_enters_after_load) return false;
        if (!source_inside) continue;
        if (branch.opcode == kSoppOpcodeBranch || target->second <= i ||
            target->second > and_index)
            return false;
        for (size_t skipped = i + 1; skipped < target->second; ++skipped) {
            const Rdna2Inst& candidate = ins[skipped];
            if (rdna2_instruction_may_change_exec(candidate) ||
                scalar_instruction_writes_anything(candidate) ||
                writes_vgpr(candidate, zero_vgpr))
                return false;
        }
    }

    std::array<bool, 128> mask_subset{};
    bool exec_subset = true; // the load's EXEC is the reference set

    auto pair_is_subset = [&](const Operand& operand) {
        if (operand.value == 126 &&
            (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special))
            return exec_subset;
        int base = -1;
        return scalar_pair_operand(operand, base) &&
            mask_subset[base] && mask_subset[base + 1];
    };

    for (size_t i = load_index + 1; i < and_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (writes_vgpr(in, zero_vgpr)) return false;

        bool derived_mask = false;
        int mask_dst = -1;
        bool next_exec_subset = exec_subset;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
            if (in.dst.value == 126) {
                next_exec_subset = pair_is_subset(in.src[0]);
                if (!next_exec_subset) return false;
            } else if (in.dst.value == 127) {
                // A B64 destination pair must be even-aligned. Reject the invalid EXEC_HI-rooted
                // packet here rather than letting it carry proof state through this exact analysis.
                return false;
            } else if (pair_is_subset(in.src[0])) {
                if (!valid_scalar_pair_base(in.dst.value))
                    return false;
                derived_mask = true;
                mask_dst = in.dst.value;
            }
        }
        if (in.fmt == Rdna2Format::SOP1 &&
            in.opcode == kSop1OpcodeAndSaveexecB64) {
            // The destination receives OLD_EXEC and the new EXEC is OLD_EXEC & src.
            if (!valid_scalar_pair_base(in.dst.value))
                return false;
            derived_mask = exec_subset;
            mask_dst = in.dst.value;
            next_exec_subset = exec_subset;
        } else if (rdna2_instruction_may_change_exec(in)) {
            // CMPX only removes lanes from the current mask. Every other unhandled EXEC writer may
            // activate a lane whose destination was preserved by the zero-record load.
            if (!(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) &&
                !(in.fmt == Rdna2Format::SOP1 &&
                  (in.opcode == kSop1OpcodeMovB64 ||
                   in.opcode == kSop1OpcodeAndSaveexecB64)))
                return false;
        }

        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = base + static_cast<int>(word);
                if (reg < 0 || reg >= 128) continue;
                mask_subset[reg] = false;
            }
        });
        if (derived_mask && mask_dst >= 0 && mask_dst + 1 < 128) {
            mask_subset[mask_dst] = true;
            mask_subset[mask_dst + 1] = true;
        }
        exec_subset = next_exec_subset;
    }
    return exec_subset;
}

size_t specialize_zero_record_execz_exits(std::vector<Rdna2Inst>& ins,
                                          const ShaderResourceTable* rt,
                                          uint32_t wave_size) {
    if (!rt || wave_size != 64 || ins.empty()) return 0;
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return (in.fmt == Rdna2Format::SOP1 &&
                    in.opcode >= kSop1OpcodeSetpcB64 &&
                    in.opcode <= kSop1OpcodeRfeB64) ||
                   (in.fmt == Rdna2Format::SOPK &&
                    (in.opcode == kSopkOpcodeCallB64 ||
                     in.opcode == kSopkOpcodeSubvectorLoopBegin ||
                     in.opcode == kSopkOpcodeSubvectorLoopEnd));
        }))
        return 0;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t i = 0; i < ins.size(); ++i) index_by_pc.emplace(ins[i].pc, i);
    std::set<uint32_t> specialized_branches;

    for (const ShaderResource& resource : rt->resources) {
        if (!is_zero_record_raw_buffer(resource)) continue;
        // Translation resolves an instruction-scoped buffer through the table's first matching
        // fetch PC. A later zero marker must not override an earlier ordinary resource here.
        if (rt->by_fetch_pc(resource.fetch_pc) != &resource) continue;
        const auto load_found = index_by_pc.find(resource.fetch_pc);
        if (load_found == index_by_pc.end()) continue;
        const size_t load_index = load_found->second;
        const Rdna2Inst& load = ins[load_index];
        if (!zero_record_load_shape(load)) continue;

        for (size_t i = load_index + 1; i + 2 < ins.size(); ++i) {
            const Rdna2Inst& bit_and = ins[i];
            const Rdna2Inst& compare = ins[i + 1];
            const Rdna2Inst& branch = ins[i + 2];
            if (writes_vgpr(bit_and, load.dst.value)) break;
            const bool and_shape = bit_and.fmt == Rdna2Format::VOP2 &&
                bit_and.opcode == kVop2OpcodeAndB32 &&
                bit_and.len_dwords == 1u && !bit_and.has_sdwa &&
                !bit_and.has_modifier && !bit_and.has_dpp &&
                bit_and.dst.kind == OperandKind::VGPR &&
                bit_and.src[0].kind == OperandKind::InlineInt &&
                bit_and.src[0].value == 7 &&
                bit_and.src[1].kind == OperandKind::VGPR &&
                bit_and.src[1].value == load.dst.value;
            if (!and_shape) continue;
            const bool compare_shape = compare.pc == bit_and.pc + bit_and.len_dwords &&
                compare.fmt == Rdna2Format::VOPC &&
                compare.opcode == kVopcOpcodeCmpxEqU32 &&
                compare.len_dwords == 1u && !compare.has_sdwa &&
                !compare.has_modifier && !compare.has_dpp &&
                compare.src[0].kind == OperandKind::InlineInt &&
                compare.src[0].value == 5 &&
                compare.src[1].kind == OperandKind::VGPR &&
                compare.src[1].value == bit_and.dst.value;
            const bool branch_shape = branch.pc == compare.pc + compare.len_dwords &&
                branch.fmt == Rdna2Format::SOPP &&
                branch.opcode == kSoppOpcodeCbranchExecz && branch.simm16 > 0 &&
                index_by_pc.contains(scalar_branch_target(branch));
            if (!compare_shape || !branch_shape ||
                !zero_load_reaches_and_under_exec_subset(
                    ins, load_index, i, index_by_pc))
                continue;
            specialized_branches.insert(branch.pc);
            break;
        }
    }

    for (uint32_t pc : specialized_branches) {
        Rdna2Inst& branch = ins[index_by_pc.at(pc)];
        branch.opcode = kSoppOpcodeBranch;
        branch.words[0] = 0xbf820000u | static_cast<uint16_t>(branch.simm16);
    }
    if (!specialized_branches.empty()) prune_scalar_cfg_reachability(ins);
    return specialized_branches.size();
}

} // namespace

bool rdna2_specialize_pcrel_dispatch(std::vector<Rdna2Inst>& instructions,
                                     const PcrelDispatchInfo& info,
                                     uint32_t selected_target) {
    return specialize_pcrel_dispatch(instructions, info, selected_target);
}

size_t rdna2_specialize_shader_constant_branches(
        std::vector<Rdna2Inst>& instructions) {
    return specialize_shader_constant_branches(instructions);
}

size_t rdna2_specialize_proven_null_bvh_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_proven_null_bvh_exits(instructions, resources, wave_size);
}

size_t rdna2_specialize_zero_record_execz_paths(
        std::vector<Rdna2Inst>& instructions, const ShaderResourceTable* resources,
        uint32_t wave_size) {
    return specialize_zero_record_execz_exits(instructions, resources, wave_size);
}

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords) {
    PcrelDispatchInfo out;
    if (!code || !dwords) return out;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    return detect_pcrel_dispatch(ins, code, dwords, program_dwords);
}

size_t rdna2_recompile_code_span(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    size_t required = program_dwords;
    std::vector<Rdna2Inst> terminating_cfg = ins;
    size_t terminating_span = program_dwords;
    if (extend_terminating_if_else(code, dwords, terminating_cfg, &terminating_span))
        required = std::max(required, terminating_span);
    // Detection both proves the compiler idiom and bounds every referenced table. Do not retain an
    // arbitrary post-ENDPGM trailer: only bytes that can affect the generated SPIR-V belong in the key.
    (void)detect_pcrel_tables(terminating_cfg, code, dwords, &required);
    const PcrelDispatchInfo dispatch = detect_pcrel_dispatch(ins, code, dwords, program_dwords);
    if (dispatch.valid) required = std::max(required, dispatch.required_dwords);
    return std::min(required, dwords);
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
// Re-reading costs three getenv calls per emitted loop during RECOMPILATION only (never per draw or
// per dispatch — cf. #2214, which removed per-resource-per-draw getenv from the live renderer), and
// recompiles are cache-warm after the first. CONFIDENCE: HIGH.
ComputeTripBoundSettings compute_trip_bound_settings() {
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
    return settings;
}

bool compute_trip_witness_active(uint64_t program_address) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    // Mirrors every arming rule that does not need the program's bytes. The GDS-use refusal is the
    // one exception and lives with the caller that already decodes the program (see gpu_executor's
    // uses_gds), so this must not be treated as the complete predicate.
    if (!settings.bound) return false;
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) return false;
    if (!settings.only_program) return true;
    return program_address == settings.only_program;
}

// True when the guest program itself reads or writes GDS. The witness lives in the internal GDS
// buffer, which is guest-addressable, so instrumenting such a program would change its INPUT -- and
// a diagnostic that perturbs the state it measures can manufacture or suppress the behaviour under
// test. Decoded from the whole program rather than one phase: a GDS access in any phase disqualifies
// the program, and `ins` here is only the current phase's slice.
bool program_touches_guest_gds(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return false;
    std::vector<Rdna2Inst> decoded;
    if (!rdna2_walk(code, dwords, decoded)) return true;   // undecodable: refuse, fail closed
    return std::any_of(decoded.begin(), decoded.end(), [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && in.ds_gds;
    });
}

uint32_t emitted_loop_trip_bound(uint64_t program_address, uint32_t phase,
                                uint32_t start_pc, uint32_t end_pc,
                                const uint32_t* code, size_t dwords) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    const uint32_t bound = settings.bound;
    if (!bound) return 0u;
    if (settings.only_program && program_address != settings.only_program) return 0u;

    // A PHASE SELECTOR IS REQUIRED, and this is a coherence requirement rather than ergonomics.
    //
    // A barrier-phased program emits one dispatcher per phase, and each phase has its OWN dispatch
    // table -- ordinal 9 means different guest pcs in phase 0 and phase 2. The witness is a single
    // record: if two phases can hit during one dispatch, its phase field is whichever invocation
    // stored last and its ordinal extrema are a mixture of two incompatible maps, which the host then
    // prints as one phase's range. There is no reading of that record that is true.
    //
    // Discovery still works with the bound armed and no phase chosen: every phase prints its dispatch
    // map (see the caller), so one run tells you how many phases exist and what each covers. Only the
    // emission is withheld.
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) {
        static std::once_flag once;
        std::call_once(once, [] {
            fprintf(stderr,
                    "[cfg-trip-bound] PROSPER_CFG_TRIP_BOUND_PHASE is REQUIRED and is unset: no "
                    "bound emitted. One witness record cannot describe two phases -- their dispatch "
                    "ordinals index different tables. The dispatch maps below list every phase; "
                    "re-run with PROSPER_CFG_TRIP_BOUND_PHASE=<k>.\n");
        });
        return 0u;
    }
    if (phase != settings.only_phase) return 0u;

    // Refuse to instrument a program that uses GDS itself; see program_touches_guest_gds.
    if (program_touches_guest_gds(code, dwords)) {
        static std::mutex refused_mutex;
        static std::set<uint64_t> refused;
        bool first = false;
        {
            std::lock_guard lock(refused_mutex);
            first = refused.insert(program_address).second;
        }
        if (first)
            fprintf(stderr,
                    "[cfg-trip-bound] program 0x%llx REFUSED: it accesses GDS itself, and the "
                    "witness would overwrite its data. Not instrumented.\n",
                    static_cast<unsigned long long>(program_address));
        return 0u;
    }

    static std::mutex announce_mutex;
    static std::set<std::pair<uint64_t, uint32_t>> announced;
    bool first = false;
    {
        std::lock_guard lock(announce_mutex);
        first = announced.insert({program_address, phase}).second;
    }
    if (first)
        fprintf(stderr,
                "[cfg-trip-bound] program 0x%llx phase %u (guest pc %u..<%u, end-exclusive) "
                "bounded at %u iterations (DIAGNOSTIC: truncates guest control flow)\n",
                static_cast<unsigned long long>(program_address), phase, start_pc, end_pc, bound);
    return bound;
}

// Does THIS module write the trip-bound witness?
//
// Derived from the compiled artifact, not from process history. An earlier revision kept a global
// set of program addresses that had ever emitted one, which cannot express the contract the host
// needs: the set was monotonic and keyed only by address, so once a program emitted under one phase,
// recompiling the SAME address under a phase it does not have still answered "instrumented" -- and
// the host would then read and clear guest-visible dwords no shader in the current module writes.
//
// Reading the module removes the whole class: the answer is a property of the bytes the backend is
// about to run, so it cannot be stale, cannot be defeated by a shader-cache hit, and needs no
// invalidation. The witness's first field is published by an atomic through an OpAccessChain onto the
// internal GDS binding at kComputeTripWitnessDword, which nothing else emits.
bool spirv_writes_trip_witness(const std::vector<uint32_t>& spirv) {
    // FAIL CLOSED, and "well formed" means the exact signature this function relies on -- not merely
    // that the words parse. The result authorizes the host to write guest-visible GDS, so every step
    // that could be true by accident has to be pinned:
    //
    //   * a decorated ID must actually name an OpVariable (a decoration can outlive its target);
    //   * a candidate must not carry conflicting DescriptorSet/Binding values;
    //   * the instructions consumed must have their EXACT operand counts -- a truncated OpAtomicUMax
    //     whose declared length happens to end at the module boundary passes any `word + len` check;
    //   * the access chain must have the shape the builder emits, not merely enough operands.
    //
    // This predicate consumes prosper's own generator output, so accepting exactly that canonical
    // form and refusing everything else is both sufficient and the conservative choice. It does not
    // attempt general SPIR-V validation, and it is not a substitute for spirv-val.
    constexpr uint32_t kSpirvMagic = 0x07230203u;
    if (spirv.size() < 5 || spirv[0] != kSpirvMagic) return false;

    constexpr uint32_t OpDecorate = 71, OpVariable = 59, OpConstant = 43, OpAccessChain = 65,
                       OpAtomicUMax = 239;
    constexpr uint32_t DecorationBinding = 33, DecorationDescriptorSet = 34;
    // Exact word counts for the forms consumed below (opcode word included).
    constexpr uint32_t kDecorateLiteralWords = 4;      // OpDecorate target decoration literal
    constexpr uint32_t kVariableWords = 4;             // result-type result storage-class, NO init
    constexpr uint32_t kAccessChainWords = 6;          // result-type result base member-0 slot
    constexpr uint32_t kAtomicUMaxWords = 7;           // result-type result pointer scope sem value

    std::set<uint32_t> variables;
    std::map<uint32_t, uint32_t> descriptor_set, binding;
    std::set<uint32_t> conflicting;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;   // truncated stream: fail closed
        // EXACT, like every other instruction consumed as proof. The builder emits the internal-GDS
        // variable as four words with no initializer; a longer OpVariable is a different declaration
        // and this predicate has no business reasoning about it.
        if (op == OpVariable && len == kVariableWords) {
            variables.insert(spirv[word + 2]);
        } else if (op == OpDecorate && len == kDecorateLiteralWords) {
            const uint32_t target = spirv[word + 1], value = spirv[word + 3];
            auto record = [&](std::map<uint32_t, uint32_t>& into) {
                const auto existing = into.find(target);
                if (existing == into.end()) into.emplace(target, value);
                else if (existing->second != value) conflicting.insert(target);
            };
            if (spirv[word + 2] == DecorationDescriptorSet) record(descriptor_set);
            else if (spirv[word + 2] == DecorationBinding) record(binding);
        }
        word += len;
    }

    std::set<uint32_t> witness_variables;
    for (uint32_t id : variables) {
        if (conflicting.count(id)) continue;
        const auto set_it = descriptor_set.find(id);
        const auto binding_it = binding.find(id);
        if (set_it != descriptor_set.end() && set_it->second == 0u &&
            binding_it != binding.end() && binding_it->second == kComputeInternalGdsBinding)
            witness_variables.insert(id);
    }
    if (witness_variables.empty()) return false;

    std::set<uint32_t> zero_constants, slot_constants, witness_pointers;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t op = spirv[word] & 0xffffu, len = spirv[word] >> 16;
        if (!len || word + len > spirv.size()) return false;
        if (op == OpConstant && len == 4) {
            if (spirv[word + 3] == 0u) zero_constants.insert(spirv[word + 2]);
            else if (spirv[word + 3] == kComputeTripWitnessDword)
                slot_constants.insert(spirv[word + 2]);
        } else if (op == OpAccessChain && len == kAccessChainWords &&
                   witness_variables.count(spirv[word + 3]) &&
                   zero_constants.count(spirv[word + 4]) &&
                   slot_constants.count(spirv[word + 5])) {
            witness_pointers.insert(spirv[word + 2]);
        } else if (op == OpAtomicUMax && len == kAtomicUMaxWords &&
                   witness_pointers.count(spirv[word + 3])) {
            return true;
        }
        word += len;
    }
    return false;
}

bool emit_cfg_state_machine(
    SpirvCompute& b, RegState& initial, const std::vector<Rdna2Inst>& ins,
    const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
    bool allow_exec_update, bool allow_smem,
    const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
    const uint32_t* code, size_t dwords, uint32_t initial_active = 0,
    bool synchronize_lds_fminmax = false) {
    const bool graphics = b.is_fragment || b.is_vertex;
    auto reject_cfg = [&](uint32_t pc, const char* reason) {
        log_recompile_diagnostic(b.diagnostic,
                                 b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                                 "terminal", "pc=%u reason=%s", pc, reason);
        return false;
    };
    if ((!b.is_compute && !graphics) || ins.empty()) return false;
    // `safe` branches have already been proven equivalent to straight-line predication by the
    // stage-specific analysis (fragment alpha-test wave early-outs, safe EXECZ regions, and the
    // bounded NGG terminal export gate).  The compact SSA emitter feeds them to emit_alu, which
    // deliberately no-ops the scalar branch while retaining the per-invocation EXEC effect.  Do the
    // same in the CFG fallback: treating one as a basic-block terminator makes a kill-mask SCC look
    // like an ordinary scalar boolean, even though the mask lowering intentionally poisons that
    // cross-lane SCC.  Astro Bot's complex material PS combines both shapes and was rejected there.
    auto linearized_branch = [&](const Rdna2Inst& in) {
        return graphics && in.fmt == Rdna2Format::SOPP && safe.contains(in.pc);
    };
    auto cfg_terminator = [&](const Rdna2Inst& in) {
        if (in.is_end) return true;
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP) return false;
        const bool branch = in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03;
        return branch || in.opcode == 0x12; // s_trap terminates the guest wave
    };

    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    if (end_pc == UINT32_MAX) return false;
    auto proven_exit_target = [&](uint32_t target) {
        if (target <= end_pc || !code || target >= dwords) return false;
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + target, dwords - target, tail);
        for (const auto& in : tail) {
            if (in.is_end) return true;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x00) continue;
            break;
        }
        return false;
    };
    if (b.is_compute &&
        ((b.wave_size != 32 && b.wave_size != 64) || !b.local_count || b.local_count > 1024))
        return false;
    const bool has_synchronized_lds_store = synchronize_lds_fminmax &&
        !b.atomicized_lds_store_pcs.empty();
    const bool has_synchronized_lds_fminmax = synchronize_lds_fminmax &&
        std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::DS && !in.ds_gds &&
                (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32);
        });
    // The float-atomic event is bracketed by Workgroup barriers. Keep even an exact native subgroup
    // on the common dispatcher path so lane zero's ordinary initializer stores are published before
    // the first CAS and the final CAS completes before the later gather. Admission below limits this
    // synthesized ordering to one guest wave, so ended/trapped lanes can safely remain participants.
    const bool direct_dispatch = (graphics || b.native_subgroup_size) &&
        !has_synchronized_lds_store && !has_synchronized_lds_fminmax;
    const bool proven_wave32_masks = b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32));
    const bool compute_scalar_vcc_bridge = allows_compute_scalar_vcc_bridge(b);
    const uint32_t wave_count = b.is_compute
        ? (b.local_count + b.wave_size - 1) / b.wave_size : 0;
    const uint32_t padded_lanes = wave_count * b.wave_size;

    // Discover every scalar pair that lives in the per-lane mask domain.  Besides defining the
    // dispatcher variables below, this identifies s_cmp_{eq,lg}_u64 mask,0: its SCC result is a
    // whole-wave reduction and therefore must be lowered in the common synchronized phase.
    std::set<int> static_mask_keys;
    for (const auto& kv : initial.sreg_bool) static_mask_keys.insert(kv.first);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            const bool wave32_one_word_mask = proven_wave32_masks &&
                ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) ||
                 (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                  in.opcode <= 0x12a && base == in.sdst.value) ||
                 (vop3b_fresh_carry_output(in) && base == in.sdst.value) ||
                 (compute_scalar_vcc_bridge &&
                  is_scalar_cselect_b32_to_vcc_lo(in) && base == 106));
            if (wave32_one_word_mask)
                static_mask_keys.insert(base);
            else if (base <= 105 && scalar_write_is_b64_mask(in, base))
                static_mask_keys.insert(base);
        }, proven_wave32_masks);
    }
    // Keep block discovery independent of the mask-domain dataflow below.  In particular, a
    // physical Wave32 SGPR can hold a mask in one lifetime and ordinary scalar data in another.
    // Conservatively split every syntactically eligible zero comparison here, then consult the
    // block-entry RegState while emitting it to decide which lifetime is actually live.
    auto mask_zero_compare_candidate_source = [&](const Rdna2Inst& in) -> int {
        if (in.fmt != Rdna2Format::SOPC) return -1;
        auto zero = [](const Operand& o) {
            return o.kind == OperandKind::InlineInt && o.value == 0;
        };
        auto possible_mask = [](const Operand& o) {
            return o.kind == OperandKind::SGPR || o.kind == OperandKind::Special;
        };
        // B64 EQ/LG and these B32 unsigned zero comparisons depend only on whether any mask bit is
        // set. In particular, Wave32 code commonly uses `s_cmp_gt_u32 vcc_lo, 0` after VOPC.
        const bool b64_compare = in.opcode == 0x12 || in.opcode == 0x13;
        const bool b32_mask_first = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x08 || in.opcode == 0x0b);
        const bool b32_mask_second = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x09 || in.opcode == 0x0a);
        if (possible_mask(in.src[0]) && zero(in.src[1]) &&
            (b64_compare || b32_mask_first))
            return in.src[0].value;
        if (possible_mask(in.src[1]) && zero(in.src[0]) &&
            (b64_compare || b32_mask_second))
            return in.src[1].value;
        return -1;
    };
    auto mask_zero_compare_inverts = [&](const Rdna2Inst& in) {
        if (in.opcode == 0x12 || in.opcode == 0x06) return true; // EQ mask,0 / 0,mask
        const bool mask_first =
            (in.src[0].kind == OperandKind::SGPR ||
             in.src[0].kind == OperandKind::Special) &&
            mask_zero_compare_candidate_source(in) == in.src[0].value;
        return mask_first ? in.opcode == 0x0b                 // mask <= 0
                          : in.opcode == 0x09;                // 0 >= mask
    };

    // Syberia's fullscreen compute pass compares EXEC with a mask written by an explicit Wave64
    // VOPC destination (`s_cmp_lg_u64 exec,s[16:17]`).  Both values live only in the per-lane Bool
    // domain, so their scalar inequality is exactly ANY(EXEC xor saved_mask) over the guest wave.
    // Keep this deliberately narrower than a general B64 comparison: one operand must be the
    // architectural EXEC pair and the other an ordinary saved-mask SGPR pair whose live RegState
    // value is checked while emitting the dispatcher case.
    auto exec_saved_mask_compare_source = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::SOPC ||
            (in.opcode != 0x12 && in.opcode != 0x13))
            return -1;
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR ||
                    operand.kind == OperandKind::Special) &&
                   operand.value == 126;
        };
        auto saved_mask = [](const Operand& operand) {
            return operand.kind == OperandKind::SGPR &&
                   operand.value >= 0 && operand.value <= 105;
        };
        if (is_exec(in.src[0]) && saved_mask(in.src[1])) return in.src[1].value;
        if (saved_mask(in.src[0]) && is_exec(in.src[1])) return in.src[0].value;
        return -1;
    };

    // Unity fragment programs also compare two ordinary saved B64 mask pairs.  Keep candidate
    // discovery syntactic so block splitting does not depend on a path-local register lifetime;
    // the MUST dataflow below separately proves that BOTH pairs are masks at the exact compare.
    // A numeric pair, an EXEC/VCC special pair, or a path-dependent mask/data join therefore does
    // not enter this lowering and falls back to the ordinary scalar emitter (or rejects visibly).
    auto saved_mask_pair_compare_sources = [&](const Rdna2Inst& in)
            -> std::array<int, 2> {
        if (in.fmt != Rdna2Format::SOPC ||
            (in.opcode != 0x12 && in.opcode != 0x13) ||
            in.src[0].kind != OperandKind::SGPR ||
            in.src[1].kind != OperandKind::SGPR ||
            in.src[0].value < 0 || in.src[0].value > 105 ||
            in.src[1].value < 0 || in.src[1].value > 105)
            return {-1, -1};
        return {in.src[0].value, in.src[1].value};
    };

    // House of the Dead 2 reduces an unsigned value in place across one architectural DPP row
    // immediately after the two saved-mask comparisons above. Keep this dispatcher escape hatch
    // exact: fragment V_MIN_U32, unbounded ROW_SHR, full-mask/no-modifier DPP16 (proved by the
    // decoder's has_dpp contract), and one physical VGPR used as VDST/SRC0/SRC1. The subgroup
    // operation itself is emitted in the common phase below so lanes taking other CFG cases still
    // participate without supplying a false neighbor.
    auto fragment_dpp_min_row_shr = [&](const Rdna2Inst& in) {
        return b.is_fragment && in.fmt == Rdna2Format::VOP2 &&
            in.opcode == 0x13 && in.has_dpp && !in.dpp_bound_ctrl &&
            in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
            in.dst.kind == OperandKind::VGPR &&
            in.src[0].kind == OperandKind::VGPR &&
            in.src[1].kind == OperandKind::VGPR &&
            in.dst.value == in.src[0].value &&
            in.dst.value == in.src[1].value;
    };

    // GTA V's compute reductions use an in-place V_ADD_NC_U32 ladder over each architectural
    // DPP16 row. Keep this contract as narrow as the observed packets: unbounded ROW_SHR with no
    // modifier/mask (the decoder's has_dpp contract), and one physical VGPR as VDST/SRC0/SRC1.
    // A native exact-wave dispatcher can execute the shuffle in its uniform switch case. The
    // portable dispatcher publishes it as an event below because a host subgroup may be narrower
    // than Wave64 and another guest wave can be parked at a different static instruction.
    auto compute_dpp_add_row_shr = [&](const Rdna2Inst& in) {
        return b.is_compute && is_inplace_vadd_nc_u32_dpp_row_shr(in);
    };

    // GTA V's MOV/MIN/MAX ROW_ROR:8 family has the same synchronization requirement as the add
    // ladder: exact native waves can shuffle in the uniform dispatcher case, while portable waves
    // publish an event-tagged source through workgroup scratch in the common phase.
    auto compute_dpp_row_ror8 = [&](const Rdna2Inst& in) {
        return b.is_compute && dpp_row_ror8_op(in) != DppRowRor8Op::None;
    };

    // The row reduction is followed by an identity QUAD_PERM whose partial ROW_MASK selects rows
    // 1 and 3. No value crosses lanes: selected EXEC-active lanes add their current VDST/SRC0 to a
    // distinct SRC1, while masked rows preserve VDST. Keeping this a dedicated dispatcher case
    // avoids granting arbitrary partial DPP masks to the generic ALU emitter.
    auto compute_dpp_add_row_mask = [&](const Rdna2Inst& in) {
        return b.is_compute && is_vadd_nc_u32_dpp_partial_row(in);
    };

    // VOPC e64 can compare a complete 64-bit scalar mask as integer data. Generated Wave64 code
    // uses `v_cmp_gt_u64 vcc,vcc,0` to broadcast (old VCC != 0) back into every active VCC lane.
    // The per-invocation mask representation has no 64-bit scalar payload, but the comparison is
    // exactly one guest-wave ANY vote. Keep the admission deliberately narrow: unsigned B64,
    // architectural VCC destination, one proven mask source, literal zero, and predicates whose
    // result is either ANY or !ANY. Other B64 arithmetic comparisons remain fail-visible.
    auto vopc_mask_zero_compare_source = [&](const Rdna2Inst& in) -> int {
        if (in.fmt != Rdna2Format::VOPC || vopc_is_cmpx(in.opcode) ||
            in.dst.kind != OperandKind::SGPR || in.dst.value != 106 ||
            in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1])
            return -1;
        auto zero = [](const Operand& operand) {
            return operand.kind == OperandKind::InlineInt && operand.value == 0;
        };
        auto possible_mask = [](const Operand& operand) {
            return operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special;
        };
        const bool mask_first = in.opcode == 0xe2 || in.opcode == 0xe3 ||
                                in.opcode == 0xe4 || in.opcode == 0xe5;
        const bool mask_second = in.opcode == 0xe1 || in.opcode == 0xe2 ||
                                 in.opcode == 0xe5 || in.opcode == 0xe6;
        if (mask_first && possible_mask(in.src[0]) && zero(in.src[1]))
            return in.src[0].value;
        if (mask_second && zero(in.src[0]) && possible_mask(in.src[1]))
            return in.src[1].value;
        return -1;
    };
    auto vopc_mask_zero_compare_inverts = [&](const Rdna2Inst& in) {
        const bool mask_first = in.src[0].kind != OperandKind::InlineInt;
        return mask_first ? in.opcode == 0xe2 || in.opcode == 0xe3 // mask ==/<= 0
                          : in.opcode == 0xe2 || in.opcode == 0xe6; // 0 ==/>= mask
    };

    // A Wave32 B32 logical writes SCC=any(result mask). The ordinary lane-local emitter poisons
    // that SCC because it cannot reduce a guest wave by itself. Inside an exact native dispatcher,
    // however, every lane reaches the same switch case and an immediately consuming SCC branch can
    // use one exact subgroup vote. Restrict the vote to the last architectural SCC writer before
    // the branch; generated traversal kernels often chain several mask intersections and only the
    // final result is live, so voting after every intermediate AND would add needless hot-loop work.
    std::unordered_set<uint32_t> native_b32_mask_scc_vote_pcs;
    if (b.native_subgroup_size && proven_wave32_masks) {
        for (size_t i = 0; i + 1 < ins.size(); ++i) {
            const Rdna2Inst& producer = ins[i];
            const Rdna2Inst& consumer = ins[i + 1];
            if (producer.fmt == Rdna2Format::SOP2 &&
                sop2_is_b32_logical(producer.opcode) &&
                consumer.fmt == Rdna2Format::SOPP &&
                (consumer.opcode == 0x04 || consumer.opcode == 0x05) &&
                producer.pc + producer.len_dwords == consumer.pc)
                native_b32_mask_scc_vote_pcs.insert(producer.pc);
        }
    }

    // A B64 mask logical writes SCC=(result mask != 0). Find the producer only when that SCC is
    // actually consumed by a later scalar branch, walking backwards across instructions that are
    // architecturally SCC-preserving. This avoids a synchronized vote after every intermediate
    // mask operation in branch-heavy kernels while retaining fail-closed behavior across another
    // scalar ALU/control-flow instruction whose SCC effect is not proven here.
    auto b64_mask_logical_opcode = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
               (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 ||
                in.opcode == 0x15 || in.opcode == 0x17 || in.opcode == 0x19 ||
                in.opcode == 0x1b || in.opcode == 0x1d);
    };
    std::unordered_set<uint32_t> scalar_block_starts{ins.front().pc};
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& terminator = ins[i];
        if (!cfg_terminator(terminator)) continue;
        if (terminator.fmt == Rdna2Format::SOPP && terminator.opcode >= 0x02 &&
            terminator.opcode <= 0x09 && terminator.opcode != 0x03) {
            const uint32_t target = scalar_branch_target(terminator);
            if (target <= end_pc) scalar_block_starts.insert(target);
        }
        if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
            scalar_block_starts.insert(ins[i + 1].pc);
    }
    std::unordered_set<uint32_t> b64_mask_scc_vote_pcs;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& consumer = ins[i];
        if (consumer.is_end) break;
        if (consumer.fmt != Rdna2Format::SOPP ||
            (consumer.opcode != 0x04 && consumer.opcode != 0x05))
            continue;
        // SCC at a block entry may come from more than one predecessor. Do not associate a linear
        // producer across that join: this narrow proof owns only the branch's current basic block.
        size_t block_begin = i;
        while (block_begin > 0 && !scalar_block_starts.contains(ins[block_begin].pc))
            --block_begin;
        for (size_t j = i; j-- > block_begin;) {
            const Rdna2Inst& candidate = ins[j];
            if (b64_mask_logical_opcode(candidate)) {
                b64_mask_scc_vote_pcs.insert(candidate.pc);
                break;
            }
            const bool preserves_scc =
                candidate.fmt == Rdna2Format::VOP1 ||
                candidate.fmt == Rdna2Format::VOP2 ||
                candidate.fmt == Rdna2Format::VOP3 ||
                candidate.fmt == Rdna2Format::VOP3P ||
                candidate.fmt == Rdna2Format::VOPC ||
                candidate.fmt == Rdna2Format::SMEM ||
                candidate.fmt == Rdna2Format::MUBUF ||
                candidate.fmt == Rdna2Format::MTBUF ||
                candidate.fmt == Rdna2Format::MIMG ||
                candidate.fmt == Rdna2Format::DS ||
                candidate.fmt == Rdna2Format::FLAT ||
                candidate.fmt == Rdna2Format::EXP ||
                candidate.fmt == Rdna2Format::VINTRP ||
                (candidate.fmt == Rdna2Format::SOP1 &&
                 candidate.opcode == kSop1OpcodeMovB64) ||
                sopp_is_noop(candidate);
            if (!preserves_scc) break;
        }
    }

    // GTA V scans one physical dword of a saved Wave64 predicate with V_FFBH_U32. In the portable
    // dispatcher the predicate exists only as one Bool per guest lane; no host subgroup-width
    // contract exists from which operand_bits could form the complete SGPR word. Split every plain
    // SGPR-fed candidate syntactically, then use the path-filtered B64 mask state at emission time to
    // select the synchronized scratch phase below. Ordinary scalar-data inputs still use emit_alu.
    auto portable_mask_ffbh_candidate = [&](const Rdna2Inst& in) {
        return b.is_compute && b.wave_size == 64 && !b.native_subgroup_size &&
            in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeFfbhU32 &&
            in.src[0].kind == OperandKind::SGPR && !in.has_sdwa && !in.has_dpp;
    };

    // Split at every branch target/fallthrough and around every cross-lane operation. Case values are
    // dense block indices, not guest PCs. A cross-lane op must end its block so the common synchronized
    // phase can publish its result before any invocation advances to the following guest instruction.
    std::set<uint32_t> start_set{ins.front().pc};
    std::unordered_map<uint32_t, uint32_t> mbcnt_event_for_pc;
    std::unordered_map<uint32_t, uint32_t> append_event_for_pc;
    bool has_gds_append = false;
    bool has_lds_append = false;
    std::unordered_set<uint32_t> swizzle_pcs;
    std::unordered_map<uint32_t, uint32_t> bpermute_event_for_pc;
    std::unordered_set<uint32_t> fragment_dpp_min_row_shr_pcs;
    std::unordered_map<uint32_t, uint32_t> fragment_dpp_min_event_for_pc;
    std::set<int> fragment_dpp_min_row_shr_dsts;
    std::unordered_set<uint32_t> compute_dpp_add_row_shr_pcs;
    std::unordered_map<uint32_t, uint32_t> compute_dpp_add_event_for_pc;
    std::set<int> compute_dpp_add_row_shr_dsts;
    std::unordered_set<uint32_t> compute_dpp_row_ror8_pcs;
    std::unordered_map<uint32_t, uint32_t> compute_dpp_ror8_event_for_pc;
    std::set<int> compute_dpp_row_ror8_dsts;
    uint32_t next_compute_dpp_event = 1;
    std::unordered_set<uint32_t> compute_dpp_add_row_mask_pcs;
    std::unordered_map<uint32_t, uint32_t> portable_mask_ffbh_event_for_pc;
    std::set<int> portable_mask_ffbh_dsts;
    std::unordered_set<uint32_t> synchronized_lds_store_pcs;
    std::unordered_set<uint32_t> lds_fminmax_pcs;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.opcode == 0x365 || in.opcode == 0x366)) {
            mbcnt_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(mbcnt_event_for_pc.size()));
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
            append_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(append_event_for_pc.size()));
            if (in.ds_gds) has_gds_append = true;
            else has_lds_append = true;
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
            swizzle_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && in.opcode == kDsOpcodeBpermuteB32) {
            if (!b.is_compute || in.ds_gds || !b.native_subgroup_size ||
                b.native_subgroup_size != b.wave_size)
                return reject_cfg(in.pc, "ds-bpermute-native-wave-contract");
            bpermute_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(bpermute_event_for_pc.size() + 1));
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (fragment_dpp_min_row_shr(in)) {
            fragment_dpp_min_row_shr_pcs.insert(in.pc);
            fragment_dpp_min_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(fragment_dpp_min_event_for_pc.size() + 1));
            fragment_dpp_min_row_shr_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_add_row_shr(in)) {
            compute_dpp_add_row_shr_pcs.insert(in.pc);
            compute_dpp_add_event_for_pc.emplace(
                in.pc, next_compute_dpp_event++);
            compute_dpp_add_row_shr_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_row_ror8(in)) {
            compute_dpp_row_ror8_pcs.insert(in.pc);
            compute_dpp_ror8_event_for_pc.emplace(
                in.pc, next_compute_dpp_event++);
            compute_dpp_row_ror8_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_add_row_mask(in)) {
            compute_dpp_add_row_mask_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (portable_mask_ffbh_candidate(in)) {
            portable_mask_ffbh_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(portable_mask_ffbh_event_for_pc.size() + 1));
            portable_mask_ffbh_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (synchronize_lds_fminmax && b.atomicized_lds_store_pcs.contains(in.pc)) {
            if (!b.is_compute || in.fmt != Rdna2Format::DS || in.ds_gds ||
                b.local_count > b.wave_size)
                return reject_cfg(in.pc, "lds-store-common-phase-contract");
            synchronized_lds_store_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (synchronize_lds_fminmax && in.fmt == Rdna2Format::DS &&
            (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32)) {
            if (!b.is_compute || in.ds_gds || b.local_count > b.wave_size ||
                (in.words[1] & 0xffff0000u) != 0u ||
                b.compute_pgm_rsrc1 == UINT32_MAX)
                return reject_cfg(in.pc, "lds-fminmax-common-phase-contract");
            lds_fminmax_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (mask_zero_compare_candidate_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (exec_saved_mask_compare_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (saved_mask_pair_compare_sources(in)[0] >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (vopc_mask_zero_compare_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (b64_mask_scc_vote_pcs.contains(in.pc)) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
            continue;
        }
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP ||
            in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        const uint32_t target = branch_target(in);
        if (target <= end_pc) start_set.insert(target);
        if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc) start_set.insert(ins[i + 1].pc);
    }
    const bool has_portable_compute_dpp_add =
        !b.native_subgroup_size && !compute_dpp_add_row_shr_pcs.empty();
    const bool has_portable_compute_dpp_ror8 =
        !b.native_subgroup_size && !compute_dpp_row_ror8_pcs.empty();
    const bool has_portable_compute_dpp =
        has_portable_compute_dpp_add || has_portable_compute_dpp_ror8;
    // Portable DPP needs a full-width value beside an event/EXEC word for every invocation. The
    // first plane remains reusable by MBCNT/votes after DPP's trailing barrier; only shaders that
    // actually contain this event pay for the second plane.
    const uint32_t dpp_value_base = 0;
    const uint32_t dpp_metadata_base = padded_lanes;
    const uint32_t wave_result_base = padded_lanes +
        (has_portable_compute_dpp ? padded_lanes : 0u);
    const uint32_t group_active_slot = wave_result_base + wave_count;
    if (b.is_compute && !direct_dispatch &&
        !b.declare_cfg_scratch(group_active_slot + 1))
        return reject_cfg(ins.front().pc, "cfg-scratch-too-small");
    start_set.insert(end_pc);
    std::vector<uint32_t> starts(start_set.begin(), start_set.end());
    if (starts.empty() || starts.front() != ins.front().pc) return false;
    std::unordered_map<uint32_t, uint32_t> block_for_pc;
    for (uint32_t i = 0; i < starts.size(); ++i) block_for_pc[starts[i]] = i;

    // Persist only registers that the stream reads or writes, plus the caller's initialized inputs.
    std::set<int> vregs, sregs;
    loop_written_regs(ins, 0, end_pc, vregs, sregs);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            const Operand& src = in.src[k];
            if (src.kind == OperandKind::VGPR) vregs.insert(src.value);
            else if (src.kind == OperandKind::SGPR ||
                     (src.kind == OperandKind::Special && src.value >= 106 && src.value <= 124))
                sregs.insert(src.value);
        }
        // 64-bit scalar compares encode only the low SGPR of each pair in their two source fields.
        if (in.fmt == Rdna2Format::SOPC && (in.opcode == 0x12 || in.opcode == 0x13))
            for (uint32_t k = 0; k < 2; ++k)
                if (in.src[k].kind == OperandKind::SGPR) sregs.insert(in.src[k].value + 1);
        if (in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x1f || in.opcode == 0x21) &&
            (in.src[0].kind == OperandKind::SGPR ||
             (in.src[0].kind == OperandKind::Special &&
              in.src[0].value >= 106 && in.src[0].value < 124)))
            sregs.insert(in.src[0].value + 1);
    }
    for (const auto& kv : initial.vreg) vregs.insert(kv.first);
    for (const auto& kv : initial.sreg) sregs.insert(kv.first);

    // Most vector destinations are one static consecutive range. V_MOVRELD is different: M0 can
    // select any observable VGPR at or above VDST, exactly the range emit_alu updates. Analyses that
    // protect v_writelane spill lifetimes must invalidate that full range as well, or a dynamic
    // ordinary write can leave a stale mask-half proof attached to an erased slot.
    const int max_observable_vgpr = shader_max_vgpr(ins);
    auto for_each_possible_vector_write = [&](const Rdna2Inst& in, const auto& callback) {
        if (in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32) {
            for (int reg = in.dst.value; reg <= max_observable_vgpr; ++reg) callback(reg);
            return;
        }
        const uint32_t writes = rdna2_vgpr_write_count(in);
        for (uint32_t word = 0; word < writes; ++word)
            callback(in.dst.value + static_cast<int>(word));
    };

    // Direct user-data descriptors are deliberately absent from the initial scalar SSA map: their
    // identity lives in the resource table, and presence in sreg means shader code overwrote them.
    // The dispatcher persists every referenced SGPR in a Function variable, so recover that
    // distinction per block until the first static write; otherwise a direct V# such as UE4's s[8:11]
    // looks rewritten merely because the dispatcher loaded its zero-initialized backing variable.
    std::set<int> direct_descriptor_sregs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            const uint32_t words = (resource.cls == ResourceClass::Texture ||
                                    resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; ++word)
                direct_descriptor_sregs.insert(static_cast<int>(resource.sgpr_base + word));
        }
    }

    // The dispatcher reloads scalar values from Function variables, so map membership cannot say
    // whether shader code has overwritten an entry-time descriptor. Compute a forward MAY-write set
    // for every reachable basic-block entry instead. MAY is intentional: if one reachable predecessor
    // overwrote the descriptor, falling back to entry metadata after the join would be wrong on that
    // path. Entry-rooted reachability excludes writes in dead blocks, while backedges participate in
    // the fixed point and prevent stale fallback on later loop iterations.
    std::vector<std::unordered_set<int>> scalar_writes(starts.size());
    std::vector<std::set<int>> vector_writes(starts.size());
    std::vector<std::set<int>> vector_reads(starts.size());
    std::vector<std::vector<uint32_t>> successors(starts.size());
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        std::set<int> ignored_scalar_writes;
        loop_written_regs(ins, lo, hi, vector_writes[block], ignored_scalar_writes);
        vector_reads[block] = vector_writes[block];
        bool reads_dynamic_vector_range = false;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const uint32_t words = rdna2_vgpr_source_span(in, source);
                if (words) {
                    for (uint32_t word = 0; word < words; ++word)
                        vector_reads[block].insert(
                            in.src[source].value + static_cast<int>(word));
                }
            }
            // Buffer/image packets have format- and dmask-dependent implicit register ranges (and
            // stores encode VDATA in the decoded destination field). Relative VGPR reads similarly
            // select from a runtime range. Keep those uncommon blocks fully conservative; ordinary
            // ALU/branch blocks can still avoid loading the rest of the vector register file.
            reads_dynamic_vector_range |=
                in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF ||
                in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::FLAT ||
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43);
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                const bool wave32_vop3b = proven_wave32_masks &&
                    ((in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                      in.opcode <= 0x12a) || vop3b_fresh_carry_output(in)) &&
                    base == in.sdst.value;
                const uint32_t effective_width = wave32_vop3b ? 1u : width;
                for (uint32_t word = 0; word < effective_width; ++word)
                    scalar_writes[block].insert(base + static_cast<int>(word));
            }, proven_wave32_masks);
            if (b.allow_b32_masks && in.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::SGPR &&
                (in.dst.value == 106 || in.dst.value == 107))
                scalar_writes[block].insert(in.dst.value);
        }

        const Rdna2Inst* terminator = nullptr;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            if (cfg_terminator(in)) {
                terminator = &in;
                break;
            }
        }
        auto add_successor = [&](uint32_t pc) {
            auto next = block_for_pc.find(pc);
            if (pc <= end_pc && next != block_for_pc.end() &&
                std::find(successors[block].begin(), successors[block].end(), next->second) ==
                    successors[block].end())
                successors[block].push_back(next->second);
        };
        if (!terminator) {
            add_successor(hi);
        } else if (!terminator->is_end && terminator->opcode != 0x12) {
            add_successor(branch_target(*terminator));
            if (terminator->opcode != 0x02)
                add_successor(terminator->pc + terminator->len_dwords);
        }
        if (reads_dynamic_vector_range) vector_reads[block] = vregs;
    }

    // Wave32 saved masks can be replaced by ordinary scalar-data lifetimes. The dispatcher reloads
    // a statically-shaped register file at every case, so carry both the one-word B32 mask domain
    // and B64 saved-mask domain as compile-time properties of each basic-block entry. This is exact
    // whenever all reachable predecessors agree. A disagreement means that the same physical SGPR
    // is a mask on one edge and scalar data on another; keep that genuinely dynamic type join
    // fail-visible.
    //
    // This is deliberately a MUST/equality analysis rather than a union: loading a stale Boolean on
    // the scalar-data edge would be a silent miscompile. Compiler-generated save/restore regions,
    // including Astro Bot's large Wave32 material loop, have identical domains at their joins.
    std::vector<std::set<int>> b32_mask_in(starts.size());
    std::vector<std::set<int>> b32_mask_ambiguous_in(starts.size());
    std::vector<std::set<int>> b64_mask_in(starts.size());
    std::vector<std::set<int>> b64_mask_ambiguous_in(starts.size());
    std::vector<bool> b32_mask_reachable(starts.size(), false);
    std::unordered_set<uint32_t> proven_saved_mask_pair_compare_pcs;
    if (b.allow_b32_masks && !starts.empty()) {
        b32_mask_in.front().insert(
            initial.sreg_bool_b32.begin(), initial.sreg_bool_b32.end());
        for (const auto& mask : initial.sreg_bool) {
            if (!initial.sreg_bool_b32.contains(mask.first) &&
                mask.first != 106 && mask.first != 107)
                b64_mask_in.front().insert(mask.first);
        }
        // `vcc` is the implicit VCC_LO mask even when no instruction has needed an explicit
        // sreg_bool[106] alias yet. Preserve that valid entry lifetime in the same physical-domain
        // analysis as explicit Wave32 masks. A zero SSA id instead means the caller currently has
        // ordinary scalar data in the VCC words, so it deliberately does not seed the mask domain.
        if (initial.vcc) b32_mask_in.front().insert(106);
        b32_mask_reachable.front() = true;
        auto implicit_vcc_mask_source = [](const Rdna2Inst& in) -> int {
            if (in.fmt == Rdna2Format::SOPP &&
                (in.opcode == 0x06 || in.opcode == 0x07))
                return 106; // s_cbranch_vccz/nz
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 ||
                 (in.opcode >= 0x28 && in.opcode <= 0x2a)))
                return 106; // e32 cndmask and carry-in/out forms use implicit VCC
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x101 ||
                 (in.opcode >= 0x128 && in.opcode <= 0x12a))) {
                const Operand& mask = in.src[2];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x365 || in.opcode == 0x366)) {
                const Operand& mask = in.src[0];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            return -1;
        };
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = b32_mask_in[block];
            std::set<int> ambiguous = b32_mask_ambiguous_in[block];
            std::set<int> b64_masks = b64_mask_in[block];
            std::set<int> b64_ambiguous = b64_mask_ambiguous_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;

                const auto compare_sources = saved_mask_pair_compare_sources(in);
                if (compare_sources[0] >= 0) {
                    const bool proven =
                        b64_masks.contains(compare_sources[0]) &&
                        b64_masks.contains(compare_sources[1]) &&
                        !b64_ambiguous.contains(compare_sources[0]) &&
                        !b64_ambiguous.contains(compare_sources[1]);
                    if (proven)
                        proven_saved_mask_pair_compare_pcs.insert(in.pc);
                    else
                        proven_saved_mask_pair_compare_pcs.erase(in.pc);
                }

                // The dispatcher may use a false backing value for a physical VCC lifetime that is
                // provably dead. Do not let that implementation placeholder become observable: every
                // instruction whose ISA encoding requires a VCC mask must see one on all incoming
                // paths. Explicit SGPR operands remain governed by the data/mask-domain checks below.
                const int implicit_vcc = implicit_vcc_mask_source(in);
                if (implicit_vcc >= 0 &&
                    (!masks.contains(implicit_vcc) || ambiguous.contains(implicit_vcc))) {
                    log_recompile_diagnostic(
                        b.diagnostic,
                        b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=missing-wave32-vcc-mask", in.pc);
                    return false;
                }

                bool reads_ambiguous = false;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if ((in.src[source].kind == OperandKind::SGPR ||
                         in.src[source].kind == OperandKind::Special)) {
                        const int reg = in.src[source].value;
                        reads_ambiguous |= ambiguous.contains(reg);
                        for (int base : b64_ambiguous)
                            reads_ambiguous |= reg == base || reg == base + 1;
                    }
                }
                if (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x06 || in.opcode == 0x07) &&
                    ambiguous.contains(106))
                    reads_ambiguous = true;
                if (reads_ambiguous) {
                    log_recompile_diagnostic(
                        b.diagnostic,
                        b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=wave32-ambiguous-mask-read", in.pc);
                    return false;
                }

                bool writes_b32_mask = false;
                auto register_mask = [&](const Operand& source) {
                    // EXEC_HI is the architectural zero dword in Wave32, not the second half of
                    // a live mask. Keep this transfer classification aligned with emit_alu's
                    // scalar materialization so an ordinary value derived from EXEC_HI survives
                    // dispatcher save/reload boundaries as scalar data.
                    return (source.kind == OperandKind::Special && source.value == 126) ||
                        ((source.kind == OperandKind::SGPR ||
                          source.kind == OperandKind::Special) &&
                         (masks.contains(source.value) ||
                          b64_masks.contains(source.value)));
                };
                auto source_mask = [&](const Operand& source) {
                    return register_mask(source) || source.kind == OperandKind::InlineInt ||
                        source.kind == OperandKind::Literal;
                };
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
                    const bool source_is_mask = register_mask(in.src[0]) ||
                        (in.dst.value == 126 && in.src[0].kind == OperandKind::InlineInt);
                    writes_b32_mask = source_is_mask && in.dst.value != 127 &&
                        (!sop1_opcode_is_emitted_saveexec_b32(in.opcode) ||
                         in.dst.value != 126);
                }
                if (in.fmt == Rdna2Format::SOP2 &&
                    (in.opcode == 0x0a || (in.opcode >= 0x0e &&
                                           in.opcode <= 0x1c && (in.opcode & 1u) == 0))) {
                    const bool scalar_vcc_bridge = compute_scalar_vcc_bridge &&
                        is_scalar_cselect_b32_to_vcc_lo(in);
                    const bool scalar_vcchi_packet = compute_scalar_vcc_bridge &&
                        b.native_subgroup_size == 32 &&
                        is_gtav_wave32_vcchi_scalar_packet(in);
                    const bool mask_domain = in.dst.value == 126 || in.src[0].value == 126 ||
                        in.src[1].value == 126 ||
                        (((in.src[0].kind == OperandKind::SGPR ||
                           in.src[0].kind == OperandKind::Special) &&
                          masks.contains(in.src[0].value))) ||
                        (((in.src[1].kind == OperandKind::SGPR ||
                           in.src[1].kind == OperandKind::Special) &&
                          masks.contains(in.src[1].value)));
                    writes_b32_mask = !scalar_vcchi_packet &&
                        (scalar_vcc_bridge ||
                         (mask_domain && source_mask(in.src[0]) &&
                          source_mask(in.src[1]) && in.dst.value != 127));
                }

                int b32_write_reg = writes_b32_mask ? in.dst.value : -1;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                    in.opcode <= 0x12a && in.sdst.kind == OperandKind::SGPR) {
                    const Operand& carry_in = in.src[2];
                    if (register_mask(carry_in)) b32_write_reg = in.sdst.value;
                } else if (vop3b_fresh_carry_output(in)) {
                    b32_write_reg = in.sdst.value;
                }

                // B64 mask-shaped instructions are not sufficient to prove a live mask lifetime:
                // s_mov_b64 and the logical family also move ordinary 64-bit scalar data. Require
                // their sources to be masks in the current path state. Saveexec always writes OLD
                // EXEC to its explicit destination, while BFM and VOP3B construct fresh masks.
                bool writes_b64_mask = false;
                if (in.fmt == Rdna2Format::SOP1 && in.dst.value <= 105) {
                    if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a)
                        writes_b64_mask = source_mask(in.src[0]);
                    else if ((in.opcode >= 0x24 && in.opcode <= 0x2b) ||
                             in.opcode == 0x37 || in.opcode == 0x38)
                        writes_b64_mask = true;
                }
                if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 105) {
                    if (in.opcode == 0x25)
                        writes_b64_mask = true;
                    else if (in.opcode == 0x0b ||
                             (in.opcode >= 0x0f && in.opcode <= 0x1d &&
                              (in.opcode & 1u) == 1))
                        writes_b64_mask = source_mask(in.src[0]) && source_mask(in.src[1]);
                }
                const bool vop3_b32_carry = in.fmt == Rdna2Format::VOP3 &&
                    ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
                     vop3b_fresh_carry_output(in));
                // rdna2_decode populates `sdst` only for its explicit ten-opcode VOP3B whitelist:
                // add/sub carry, div-scale flag, and 64-bit multiply-add carry outputs. Each SDST
                // is a per-lane scalar mask/flag; ordinary VOP3A scalar data never appears here
                // (v_readlane uses `dst`, not `sdst`).
                if (vop3_writes_mask_sdst(in) && !vop3_b32_carry &&
                    in.sdst.value <= 105)
                    writes_b64_mask = true;
                int b64_write_reg = writes_b64_mask ?
                    (in.fmt == Rdna2Format::VOP3 ? in.sdst.value : in.dst.value) : -1;

                auto erase_b64_overlapping = [&](int base, uint32_t width) {
                    auto erase = [&](std::set<int>& values) {
                        for (auto it = values.begin(); it != values.end();) {
                            const int mask_base = *it;
                            if (base < mask_base + 2 && mask_base < base + static_cast<int>(width))
                                it = values.erase(it);
                            else
                                ++it;
                        }
                    };
                    erase(b64_masks);
                    erase(b64_ambiguous);
                };

                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    const bool one_word_write = base == b32_write_reg;
                    const uint32_t effective_width = one_word_write ? 1u : width;
                    erase_b64_overlapping(base, effective_width);
                    for (uint32_t word = 0; word < effective_width; ++word) {
                        const int reg = base + static_cast<int>(word);
                        if (!one_word_write || reg != b32_write_reg) {
                            masks.erase(reg);
                            ambiguous.erase(reg);
                        }
                    }
                    if (one_word_write && b32_write_reg != 126) {
                        masks.insert(b32_write_reg);
                        ambiguous.erase(b32_write_reg);
                    }
                }, proven_wave32_masks);

                if (b64_write_reg >= 0) {
                    for (int word = 0; word < 2; ++word) {
                        masks.erase(b64_write_reg + word);
                        ambiguous.erase(b64_write_reg + word);
                    }
                    b64_masks.insert(b64_write_reg);
                    b64_ambiguous.erase(b64_write_reg);
                }

                // Every non-CMPX VOPC destination is a one-word mask in proven Wave32, including
                // explicit ordinary SGPRs. CMPX writes EXEC only and establishes no VCC lifetime.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                    const int destination = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
                    erase_b64_overlapping(destination, 1);
                    masks.insert(destination);
                    ambiguous.erase(destination);
                }
            }

            for (uint32_t successor : successors[block]) {
                if (!b32_mask_reachable[successor]) {
                    b32_mask_reachable[successor] = true;
                    b32_mask_in[successor] = masks;
                    b32_mask_ambiguous_in[successor] = ambiguous;
                    b64_mask_in[successor] = b64_masks;
                    b64_mask_ambiguous_in[successor] = b64_ambiguous;
                    pending.push_back(successor);
                } else {
                    std::set<int> joined_masks;
                    std::set<int> joined_ambiguous = b32_mask_ambiguous_in[successor];
                    std::set_intersection(
                        b32_mask_in[successor].begin(), b32_mask_in[successor].end(),
                        masks.begin(), masks.end(),
                        std::inserter(joined_masks, joined_masks.end()));
                    for (int reg : b32_mask_in[successor])
                        if (!masks.contains(reg)) joined_ambiguous.insert(reg);
                    for (int reg : masks)
                        if (!b32_mask_in[successor].contains(reg)) joined_ambiguous.insert(reg);
                    joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                    for (int reg : joined_ambiguous) joined_masks.erase(reg);
                    std::set<int> joined_b64_masks;
                    std::set<int> joined_b64_ambiguous = b64_mask_ambiguous_in[successor];
                    std::set_intersection(
                        b64_mask_in[successor].begin(), b64_mask_in[successor].end(),
                        b64_masks.begin(), b64_masks.end(),
                        std::inserter(joined_b64_masks, joined_b64_masks.end()));
                    for (int reg : b64_mask_in[successor])
                        if (!b64_masks.contains(reg)) joined_b64_ambiguous.insert(reg);
                    for (int reg : b64_masks)
                        if (!b64_mask_in[successor].contains(reg)) joined_b64_ambiguous.insert(reg);
                    joined_b64_ambiguous.insert(
                        b64_ambiguous.begin(), b64_ambiguous.end());
                    for (int reg : joined_b64_ambiguous) joined_b64_masks.erase(reg);
                    if (joined_masks != b32_mask_in[successor] ||
                        joined_ambiguous != b32_mask_ambiguous_in[successor] ||
                        joined_b64_masks != b64_mask_in[successor] ||
                        joined_b64_ambiguous != b64_mask_ambiguous_in[successor]) {
                        b32_mask_in[successor] = std::move(joined_masks);
                        b32_mask_ambiguous_in[successor] = std::move(joined_ambiguous);
                        b64_mask_in[successor] = std::move(joined_b64_masks);
                        b64_mask_ambiguous_in[successor] = std::move(joined_b64_ambiguous);
                        pending.push_back(successor);
                    }
                }
            }
        }
        for (const auto& masks : b32_mask_in)
            for (int reg : masks)
                if (reg <= 107) static_mask_keys.insert(reg);
        for (const auto& masks : b64_mask_in)
            for (int reg : masks)
                if (reg <= 105) static_mask_keys.insert(reg);
    }

    // Wave64 dispatcher Bool variables hold values, not lifetime tags. A scalar overwrite stores
    // false for a dead mask, and an unfiltered load at a later case can therefore make mere map
    // membership look like a valid saved mask. Prove the B64 mask domain separately at every
    // Wave64 whole-mask comparisons (mask-vs-zero, EXEC-vs-saved or saved-vs-saved): mask producers
    // generate the fact, every overlapping half/pair scalar write kills it, and joins retain it only
    // when all reachable predecessors agree. Architectural VCC lives in `state.vcc`, not in the
    // saved-mask map, so this proof is also the lifetime tag that lets the dispatcher distinguish a
    // live VCC mask from a physical VCC pair that has been recycled as scalar data.
    std::unordered_set<uint32_t> proven_wave64_mask_zero_compare_pcs;
    std::unordered_set<uint32_t> proven_exec_saved_mask_compare_pcs;
    std::unordered_set<uint32_t> proven_wave64_mask_reduction_pcs;
    std::unordered_map<uint32_t, int> proven_wave64_mbcnt_mask_root_for_pc;
    // Retain the entry facts beyond the consumer-specialization pass below. Function Bool
    // variables persist values only; this MUST set is the separate lifetime tag load_state needs
    // before reconstructing a saved-mask RegState entry in each dispatcher case. Domain conflicts
    // at joins remain ambiguous until a definite overwrite and reject on their first read.
    std::vector<std::set<int>> wave64_b64_mask_in(starts.size());
    std::vector<std::set<int>> wave64_b64_ambiguous_in(starts.size());
    // A mask/scalar join poisons the physical pair, but a later definite B32 write still makes the
    // addressed scalar word safe to consume. Track those post-conflict scalar definitions with a
    // separate MUST fact: the other half remains ambiguous until it too is definitely replaced.
    // This matters for GTA V's scalar scratch in VCC/ordinary mask pairs, where readfirstlane,
    // SMEM, or a B32 scalar ALU defines one half before a one-dword VALU/SALU consumer.
    std::vector<std::set<int>> wave64_scalar_word_in(starts.size());
    // Dispatcher Function variables persist SCC's Boolean value but not whether that value is an
    // architectural SCC or the false placeholder stored for an unrepresentable wave-mask result.
    // Carry a separate CFG MUST-validity bit and use it both for scalar-word provenance and when
    // reconstructing RegState at each dispatcher case.
    std::vector<bool> wave64_scalar_scc_valid_in(starts.size(), false);
    std::vector<bool> wave64_b64_reachable(starts.size(), false);
    auto wave64_mask_reduction_source = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
            (in.opcode != 0x10 && in.opcode != 0x14))
            return -1;
        if (in.src[0].kind == OperandKind::SGPR &&
            in.src[0].value >= 0 && in.src[0].value <= 105)
            return in.src[0].value;
        if (in.src[0].kind == OperandKind::Special && in.src[0].value == 106)
            return 106;
        // Architectural EXEC is a complete B64 mask source. emit_alu already materializes this
        // reduction from EXEC directly, but the MUST dataflow below has to agree, or the result is
        // a scalar the dispatcher erases at the next block entry -- see the note at the
        // exact_mask_reduction use. The exactness precondition is emit_alu's own: the ballot equals
        // the guest wave mask only when the native subgroup IS the guest wave.
        //
        // This covers BOTH opcodes admitted above, not only the population count: 0x10 is
        // s_bcnt1_i32_b64 and 0x14 is s_ff1_i32_b64, and `s_ff1_i32_b64 <- exec` (find-first-set over
        // the active mask) reaches this same return. The committed fixture exercises it at dword 336,
        // immediately before the bcnt at 337, so the wider coverage is tested rather than incidental.
        //
        // CONFIDENCE: HIGH — EXEC as a B64 mask source for these two reductions is architectural
        // (RDNA2 ISA 70648 §5.3/§12.2: both take a 64-bit scalar source, and EXEC is a legal SSRC),
        // and the wave-size guard is the same precondition emit_alu already relies on rather than a
        // new assumption. What is NOT covered is a 32-wide native subgroup, where the ballot is not
        // the guest wave: that returns -1 and the dispatcher declines, loudly, via
        // [subgroup-width] ... DISABLED. See #2429.
        if (in.src[0].kind == OperandKind::Special && in.src[0].value == 126 &&
            b.native_subgroup_size == b.wave_size)
            return 126;
        return -1;
    };
    auto wave64_mbcnt_mask_root = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::VOP3 ||
            (in.opcode != 0x365 && in.opcode != 0x366) ||
            in.src[0].kind != OperandKind::SGPR)
            return -1;
        // LOW names the mask pair's low word. HIGH names its high word, while the Bool-domain
        // value remains keyed by the low word. Architectural EXEC/VCC use Special operands and
        // continue through mbcnt_source_bit without this saved-SGPR lifetime proof.
        const int root = in.opcode == 0x366 ? in.src[0].value - 1 : in.src[0].value;
        return root >= 0 && root <= 105 ? root : -1;
    };
    if ((b.is_compute || b.is_fragment) && b.wave_size == 64 && !starts.empty()) {
        for (const auto& mask : initial.sreg_bool)
            if (!initial.sreg_bool_b32.contains(mask.first) && mask.first <= 105)
                wave64_b64_mask_in.front().insert(mask.first);
        if (initial.vcc) wave64_b64_mask_in.front().insert(106);
        for (const auto& value : initial.sreg)
            if (value.first <= 124) wave64_scalar_word_in.front().insert(value.first);
        for (const auto& value : initial.sreg_input)
            if (value.first <= 124) wave64_scalar_word_in.front().insert(value.first);
        // Direct descriptors intentionally live outside RegState's ordinary scalar map, but their
        // entry words are real scalar data until shader code overwrites them. Seed the MUST facts
        // from the same resource-table ranges used by the emitter's direct-descriptor fallback.
        for (int reg : direct_descriptor_sregs)
            if (reg <= 124) wave64_scalar_word_in.front().insert(reg);
        wave64_scalar_scc_valid_in.front() = initial.scc != 0;
        wave64_b64_reachable.front() = true;

        enum class ScalarSourceRead : uint8_t {
            None = 0, B32 = 1, Pair = 2, Quad = 4, Oct = 8,
        };
        auto scalar_source_read = [&](const Rdna2Inst& in, uint32_t source)
                -> ScalarSourceRead {
            const uint32_t alu_words = scalar_alu_source_words(in, source);
            if (alu_words == UINT32_MAX) return ScalarSourceRead::None;
            if (alu_words)
                return static_cast<ScalarSourceRead>(alu_words);
            switch (in.fmt) {
                case Rdna2Format::SMEM:
                    // Ordinary scalar memory uses a two-word base. S_BUFFER_* opcodes use a
                    // complete four-word descriptor; SOFFSET remains one scalar dword.
                    if (source == 1) return ScalarSourceRead::B32;
                    return in.opcode >= 0x08
                        ? ScalarSourceRead::Quad : ScalarSourceRead::Pair;
                case Rdna2Format::MUBUF:
                case Rdna2Format::MTBUF:
                    // The four-word V# starts at SRC1 while optional SOFFSET is one dword.
                    return source == 1 ? ScalarSourceRead::Quad : ScalarSourceRead::B32;
                case Rdna2Format::MIMG: {
                    // T# is eight words; the exact BVH form uses a four-word descriptor instead.
                    // Storage/load/resinfo/BVH packets have no sampler, while sampled operations
                    // consume the complete four-word S#.
                    if (source == 1)
                        return in.opcode == 0xe6
                            ? ScalarSourceRead::Quad : ScalarSourceRead::Oct;
                    const bool storage_only_op = in.opcode == 0x08 || in.opcode == 0x09 ||
                        in.opcode == 0x0f ||
                        (in.opcode >= 0x11 && in.opcode <= 0x1a && in.opcode != 0x13);
                    if (source == 2)
                        return in.opcode == 0x00 || in.opcode == 0x01 ||
                                   in.opcode == 0x0e || in.opcode == 0xe6 || storage_only_op
                            ? ScalarSourceRead::None : ScalarSourceRead::Quad;
                    return ScalarSourceRead::B32;
                }
                default:
                    return ScalarSourceRead::Pair;
            }
        };

        auto advance_wave64_b64_masks = [&](std::set<int>& masks,
                                            std::set<int>& ambiguous,
                                            std::set<int>& scalar_words,
                                            bool& scalar_scc,
                                            const Rdna2Inst& in,
                                            bool record_compare) {
            auto source_is_scalar_word = [&](const Operand& source) {
                switch (source.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::SGPR:
                        return scalar_words.contains(source.value);
                    case OperandKind::Special:
                        if (source.value == 125) return true; // SGPR_NULL
                        if (source.value == 253) return scalar_scc;
                        return source.value >= 106 && source.value <= 124 &&
                               scalar_words.contains(source.value);
                    default:
                        return false;
                }
            };
            auto reads_scc = [](const Rdna2Inst& candidate) {
                return (candidate.fmt == Rdna2Format::SOP2 &&
                        (candidate.opcode == 0x04u || candidate.opcode == 0x05u ||
                         candidate.opcode == kSop2OpcodeCselectB32 ||
                         candidate.opcode == 0x0bu)) ||
                       (candidate.fmt == Rdna2Format::SOP1 &&
                        (candidate.opcode == kSop1OpcodeCmovB32 ||
                         candidate.opcode == kSop1OpcodeCmovB64)) ||
                       (candidate.fmt == Rdna2Format::SOPK &&
                        candidate.opcode == kSopkOpcodeCmovkI32);
            };
            const bool valid_scc_read = !reads_scc(in) || scalar_scc;
            const bool b32_vcc_scalar_write =
                is_gtav_wave64_vcc_lo_scalar_cselect(in) ||
                (in.fmt == Rdna2Format::SOP2 &&
                 (in.dst.value == 106 || in.dst.value == 107) &&
                 in.opcode >= 0x0e && in.opcode <= 0x1c &&
                 (in.opcode & 1u) == 0);
            const bool b32_vcc_scalar_result = b32_vcc_scalar_write &&
                source_is_scalar_word(in.src[0]) && source_is_scalar_word(in.src[1]);
            const int b32_vcc_sibling = in.dst.value == 106 ? 107 : 106;
            const bool b32_vcc_complete_scalar_pair =
                b32_vcc_scalar_result && scalar_words.contains(b32_vcc_sibling);
            if (record_compare && b32_vcc_scalar_write) {
                if (b32_vcc_scalar_result)
                    b.vcc_b32_scalar_result_pcs.insert(in.pc);
                else
                    b.vcc_b32_scalar_result_pcs.erase(in.pc);
                if (b32_vcc_complete_scalar_pair)
                    b.vcc_b32_scalar_pair_pcs.insert(in.pc);
                else
                    b.vcc_b32_scalar_pair_pcs.erase(in.pc);
            }
            // A block-entry join where the same physical pair is a mask on one predecessor and
            // scalar data on another has no runtime type tag. Reject the first observable read;
            // loading either the Bool's false placeholder or the scalar variable's zero placeholder
            // would silently choose one predecessor's domain for both paths.
            bool reads_ambiguous = false;
            auto source_is_mask = [&](const Operand& source) {
                if (source.kind == OperandKind::InlineInt) return true;
                if (source.kind != OperandKind::SGPR &&
                    source.kind != OperandKind::Special)
                    return false;
                if (source.value == 126 || source.value == 127) return true;
                return masks.contains(source.value);
            };
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const Operand& operand = in.src[source];
                if (operand.kind != OperandKind::SGPR &&
                    operand.kind != OperandKind::Special)
                    continue;
                // The mask-only emitter for B64 logical operations consumes a proved Bool-domain
                // source directly. A fresh VCC pair can overlap the high word of an older odd-rooted
                // ambiguity without becoming scalar data; only the scalar-pair operand below needs
                // per-word resolution. MBCNT likewise consumes one proved mask half, not a scalar
                // pair beginning at its encoded SGPR.
                const bool b64_logical_mask_source =
                    in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
                    in.opcode <= 0x1d && (in.opcode & 1u) && source_is_mask(operand);
                const int mbcnt_root = in.opcode == 0x366
                    ? operand.value - 1 : operand.value;
                const bool mbcnt_mask_source = in.fmt == Rdna2Format::VOP3 && source == 0 &&
                    (in.opcode == 0x365 || in.opcode == 0x366) &&
                    (operand.value == (in.opcode == 0x366 ? 127 : 126) ||
                     operand.value == (in.opcode == 0x366 ? 107 : 106) ||
                     masks.contains(mbcnt_root));
                if (b64_logical_mask_source || mbcnt_mask_source) continue;
                const ScalarSourceRead read = scalar_source_read(in, source);
                if (read == ScalarSourceRead::None) continue;
                const int first = operand.value;
                const int last = first + static_cast<int>(read);
                for (int base : ambiguous) {
                    const int overlap_first = std::max(first, base);
                    const int overlap_last = std::min(last, base + 2);
                    for (int word = overlap_first; word < overlap_last; ++word)
                        if (!scalar_words.contains(word)) reads_ambiguous = true;
                }
            }
            const bool implicit_vcc_read =
                (in.fmt == Rdna2Format::SOPP &&
                 (in.opcode == 0x06 || in.opcode == 0x07)) ||
                (in.fmt == Rdna2Format::VOP2 &&
                 (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)));
            reads_ambiguous |= implicit_vcc_read && ambiguous.contains(106);
            const uint32_t implicit_scalar_words =
                scalar_implicit_destination_read_width(in);
            if (implicit_scalar_words) {
                const int first = in.dst.value;
                const int last = first + static_cast<int>(implicit_scalar_words);
                for (int word = first; word < last; ++word)
                    for (int mask_base : ambiguous)
                        if ((word == mask_base || word == mask_base + 1) &&
                            !scalar_words.contains(word))
                            reads_ambiguous = true;
            }
            if (reads_ambiguous)
                return reject_cfg(in.pc, "wave64-ambiguous-mask-read");

            const int reduction_source = wave64_mask_reduction_source(in);
            // EXEC needs no saved-mask lifetime: it is architectural state that always holds a
            // live mask, so it is exact wherever the source helper admits it.
            const bool exact_mask_reduction =
                reduction_source == 126 ||
                (reduction_source >= 0 && masks.contains(reduction_source));
            if (record_compare && exact_mask_reduction)
                proven_wave64_mask_reduction_pcs.insert(in.pc);
            const int mbcnt_root = wave64_mbcnt_mask_root(in);
            if (record_compare && mbcnt_root >= 0 && masks.contains(mbcnt_root))
                proven_wave64_mbcnt_mask_root_for_pc.emplace(in.pc, mbcnt_root);
            const int zero_compare_source = mask_zero_compare_candidate_source(in);
            if (record_compare && zero_compare_source >= 0 &&
                masks.contains(zero_compare_source))
                proven_wave64_mask_zero_compare_pcs.insert(in.pc);
            const int compare_source = exec_saved_mask_compare_source(in);
            if (record_compare && compare_source >= 0 && masks.contains(compare_source))
                proven_exec_saved_mask_compare_pcs.insert(in.pc);
            const auto pair_compare = saved_mask_pair_compare_sources(in);
            if (record_compare && pair_compare[0] >= 0 &&
                masks.contains(pair_compare[0]) && masks.contains(pair_compare[1]))
                proven_saved_mask_pair_compare_pcs.insert(in.pc);
            const bool writes_exact_wave_scc =
                (zero_compare_source >= 0 && masks.contains(zero_compare_source)) ||
                (compare_source >= 0 && masks.contains(compare_source)) ||
                (pair_compare[0] >= 0 && masks.contains(pair_compare[0]) &&
                 masks.contains(pair_compare[1]));

            // Scalar-data presence and SCC validity share the same provenance contract. SOPK
            // exposes its old SDST only as an implicit source; accepting ADDK, CMPK, or CMOVK after
            // a dispatcher reload therefore requires that exact word to be a MUST scalar value.
            // Check every dword of B64 inputs before the destination transfer ends the old lifetime.
            auto source_is_scalar_range = [&](const Operand& source, uint32_t width) {
                if (source.kind == OperandKind::InlineInt ||
                    source.kind == OperandKind::InlineFloat ||
                    source.kind == OperandKind::Literal)
                    return true;
                if (source.kind != OperandKind::SGPR &&
                    source.kind != OperandKind::Special)
                    return false;
                if (source.kind == OperandKind::Special && source.value == 125)
                    return true; // SGPR_NULL
                if (source.kind == OperandKind::Special && source.value == 253)
                    return width == 1 && scalar_scc;
                if (source.kind == OperandKind::Special &&
                    (source.value == 126 || source.value == 127))
                    return width == 1 && b.is_compute && b.wave_size == 64 &&
                        b.native_subgroup_size == 64;
                for (uint32_t word = 0; word < std::max(width, 1u); ++word)
                    if (!scalar_words.contains(source.value + static_cast<int>(word)))
                        return false;
                return true;
            };
            bool scalar_sources = true;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const uint32_t width = scalar_alu_source_words(in, source);
                if (width != UINT32_MAX)
                    scalar_sources &= source_is_scalar_range(in.src[source], width);
            }
            bool implicit_scalar_source = true;
            if (implicit_scalar_words) {
                implicit_scalar_source = in.dst.kind == OperandKind::SGPR;
                for (uint32_t word = 0; word < implicit_scalar_words; ++word)
                    implicit_scalar_source &=
                        scalar_words.contains(in.dst.value + static_cast<int>(word));
            }
            const bool scalar_alu_result = scalar_sources && implicit_scalar_source;

            const bool wave64_vcc_b32_mask_not =
                in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeNotB32 &&
                (in.dst.value == 106 || in.dst.value == 107) &&
                in.src[0].value == in.dst.value && masks.contains(106) &&
                !scalar_words.contains(in.src[0].value);
            const bool exact_quadmask =
                in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeQuadmaskB64 &&
                b.wave_size == 64 && source_is_mask(in.src[0]) &&
                (b.is_fragment || (b.is_compute && b.native_subgroup_size == 64));

            int mask_write = -1;
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                mask_write = in.dst.kind == OperandKind::SGPR && in.dst.value <= 105
                    ? in.dst.value : 106;
            } else if (in.fmt == Rdna2Format::SOP1 && in.dst.value <= 107) {
                if (wave64_vcc_b32_mask_not)
                    mask_write = 106;
                else if ((in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a) &&
                    source_is_mask(in.src[0]))
                    mask_write = in.dst.value;
                else if ((in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                          in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                         in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                         in.opcode == kSop1OpcodeOrn1SaveexecB64 || exact_quadmask)
                    mask_write = in.dst.value;
            } else if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 107) {
                if (b32_vcc_complete_scalar_pair)
                    mask_write = 106;
                else if (in.opcode == kSop2OpcodeBfmB64)
                    mask_write = in.dst.value;
                else if (in.opcode >= 0x0f && in.opcode <= 0x1d &&
                         (in.opcode & 1u) == 1)
                    // Every B64 logical has a Bool-domain result. Exact native Wave64 additionally
                    // materializes that result's ballot words below, but its mask lifetime remains
                    // the primary classification here.
                    mask_write = in.dst.value;
                else if (valid_scc_read && in.opcode == 0x0b && in.dst.value == 106 &&
                         !b.cselect_b64_low_only_pcs.contains(in.pc))
                    // A complete scalar-data pair selected into VCC has a dual lifetime: emit_alu
                    // derives its per-lane predicate even though neither input is a mask. The one
                    // incomplete GTA form deliberately has no predicate and is excluded here.
                    mask_write = 106;
                else if (valid_scc_read && in.opcode == 0x0b &&
                         source_is_mask(in.src[0]) && source_is_mask(in.src[1]))
                    mask_write = in.dst.value;
            } else if (vop3_writes_mask_sdst(in) && in.sdst.value <= 107) {
                mask_write = in.sdst.value;
            }

            auto erase_overlapping = [&](int base, uint32_t width) {
                for (auto it = masks.begin(); it != masks.end();) {
                    const int mask_base = *it;
                    if (base < mask_base + 2 &&
                        mask_base < base + static_cast<int>(width))
                        it = masks.erase(it);
                    else
                        ++it;
                }
                // A one-word scalar write kills the definite-mask fact for the physical pair, but
                // it resolves only the addressed half of an ambiguous pair. Keep the ambiguity
                // until one definite write covers both halves; otherwise the untouched word could
                // be reloaded from the wrong Function-variable domain. The separate scalar-word
                // MUST facts below can validate exactly the overwritten word while complete-pair
                // consumers remain fail-closed until both halves are definitely scalar.
                for (auto it = ambiguous.begin(); it != ambiguous.end();) {
                    const int mask_base = *it;
                    if (base <= mask_base &&
                        base + static_cast<int>(width) >= mask_base + 2)
                        it = ambiguous.erase(it);
                    else
                        ++it;
                }
            };
            std::vector<std::pair<int, uint32_t>> scalar_writes;
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                scalar_writes.emplace_back(base, width);
                erase_overlapping(base, width);
            }, /*wave32_one_word_masks*/false);
            // This exact B32 write resolves VCC_LO to scalar data while the whole-stream proof
            // guarantees the untouched high half cannot be observed before a complete replacement.
            // It is therefore safe to clear the pair-domain ambiguity for this path; a later join
            // with a mask path will recreate the ambiguity in the ordinary MUST merge below.
            if (b.is_compute && b.vcc_b32_low_only_pcs.contains(in.pc) &&
                b32_vcc_scalar_result)
                ambiguous.erase(106);
            // An implicit VOPC destination is architectural VCC and is absent from the explicit
            // scalar-writer inventory.
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)) {
                erase_overlapping(106, 2);
                // VOPC writes a complete Wave64 mask even though it has no explicit scalar
                // destination in the generic write inventory. End both physical words' old
                // scalar lifetimes: retaining them can falsely classify a later one-word VCC_LO
                // overwrite as a complete scalar pair after a dispatcher reload.
                scalar_words.erase(106);
                scalar_words.erase(107);
            }
            if (mask_write >= 0) {
                masks.insert(mask_write);
                ambiguous.erase(mask_write);
                // An inline S_MOV_B64 is represented in both domains by emit_alu: its Bool
                // view is an exact wave mask and its two scalar words are the same architectural bit
                // pattern. GTA joins `s_mov_b64 s[16:17], 0` against an SMEM load, then consumes the
                // pair as scalar data. Preserve that dual definition; other mask writes still erase
                // scalar facts because their Boolean value cannot be materialized as two SGPR words.
                const bool mov_dual_domain = in.fmt == Rdna2Format::SOP1 &&
                    in.opcode == 0x04 &&
                    (in.src[0].kind == OperandKind::InlineInt ||
                     scalar_alu_result ||
                     (b.is_compute && b.wave_size == 64 &&
                      b.native_subgroup_size == 64 && source_is_mask(in.src[0])));
                const bool cselect_scalar_branch = valid_scc_read &&
                    in.fmt == Rdna2Format::SOP2 &&
                    in.opcode == 0x0b && in.dst.value == 106 &&
                    !b.cselect_b64_low_only_pcs.contains(in.pc) &&
                    !(source_is_mask(in.src[0]) && source_is_mask(in.src[1])) &&
                    scalar_alu_result;
                const bool logical_native_ballot = in.fmt == Rdna2Format::SOP2 &&
                    in.opcode >= 0x0f && in.opcode <= 0x1d &&
                    (in.opcode & 1u) == 1 && b.is_compute && b.wave_size == 64 &&
                    b.native_subgroup_size == 64;
                const bool quadmask_native_ballot = exact_quadmask;
                const bool dual_domain_scalar_write =
                    mov_dual_domain || cselect_scalar_branch || logical_native_ballot ||
                    quadmask_native_ballot || b32_vcc_complete_scalar_pair;
                if (dual_domain_scalar_write && valid_scc_read) {
                    for (const auto& [base, width] : scalar_writes)
                        for (uint32_t word = 0; word < width; ++word)
                            scalar_words.insert(base + static_cast<int>(word));
                } else {
                    for (const auto& [base, width] : scalar_writes)
                        for (uint32_t word = 0; word < width; ++word)
                            scalar_words.erase(base + static_cast<int>(word));
                }
            } else if (valid_scc_read &&
                       ((in.fmt != Rdna2Format::SOP1 &&
                         in.fmt != Rdna2Format::SOP2 &&
                         in.fmt != Rdna2Format::SOPK) || scalar_alu_result ||
                        exact_mask_reduction)) {
                for (const auto& [base, width] : scalar_writes)
                    for (uint32_t word = 0; word < width; ++word)
                        scalar_words.insert(base + static_cast<int>(word));
            } else {
                for (const auto& [base, width] : scalar_writes)
                    for (uint32_t word = 0; word < width; ++word)
                        scalar_words.erase(base + static_cast<int>(word));
            }
            // B32 VCC logicals have a separate mask-domain lowering. If either input lacks a MUST
            // scalar word, the emitter may take that path and erase its uint result. Never publish
            // a scalar fact merely because the architectural destination is one dword: dispatcher
            // Function variables contain zero placeholders for absent domains.
            if (b32_vcc_scalar_write && !b32_vcc_scalar_result)
                scalar_words.erase(in.dst.value);

            // SCC is persisted through a dispatcher Function variable without a runtime validity
            // tag. A scalar SOPC establishes a real Boolean, while the three exact whole-wave mask
            // comparisons above establish one through their synchronized vote phase. Other scalar
            // ALU writers need fully scalar sources or kill the fact. The resulting validity is a
            // CFG MUST property, so a placeholder from any incoming edge still poisons the join.
            if (in.fmt == Rdna2Format::SOPC) {
                scalar_scc = scalar_sources;
            } else if (in.fmt == Rdna2Format::SOP2) {
                const bool preserves_scc =
                    in.opcode == kSop2OpcodeCselectB32 || in.opcode == 0x0bu ||
                    in.opcode == kSop2OpcodeBfmB32 ||
                    in.opcode == kSop2OpcodeBfmB64 || in.opcode == 0x26u ||
                    (in.opcode >= 0x32u && in.opcode <= 0x36u);
                if (in.opcode == 0x04u || in.opcode == 0x05u)
                    scalar_scc = scalar_scc && scalar_sources;
                else if (in.opcode == kSop2OpcodeBfeU64)
                    scalar_scc = scalar_sources &&
                        in.dst.value != 106 && in.dst.value != 107 &&
                        in.dst.value != 126 && in.dst.value != 127;
                else if (!preserves_scc)
                    scalar_scc = scalar_sources && mask_write < 0;
            } else if (in.fmt == Rdna2Format::SOP1) {
                const bool preserves_scc =
                    in.opcode == kSop1OpcodeMovB32 ||
                    in.opcode == kSop1OpcodeMovB64 ||
                    in.opcode == kSop1OpcodeCmovB32 ||
                    in.opcode == kSop1OpcodeCmovB64 ||
                    in.opcode == kSop1OpcodeBrevB32 ||
                    in.opcode == kSop1OpcodeBcnt1I32B64 ||
                    in.opcode == kSop1OpcodeFf1I32B64 ||
                    in.opcode == kSop1OpcodeFlbitI32B32 ||
                    in.opcode == kSop1OpcodeFlbitI32B64 ||
                    in.opcode == kSop1OpcodeBitset0B32 ||
                    in.opcode == kSop1OpcodeBitset1B32 ||
                    in.opcode == kSop1OpcodeGetpcB64;
                const bool saveexec =
                    (in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                     in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                    in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                    in.opcode == kSop1OpcodeOrn1SaveexecB64;
                if (in.opcode == kSop1OpcodeNotB32 ||
                    in.opcode == kSop1OpcodeAbsI32)
                    scalar_scc = scalar_alu_result && !wave64_vcc_b32_mask_not;
                else if (saveexec)
                    scalar_scc = b.is_fragment && initial.reads_scc &&
                        source_is_mask(in.src[0]);
                else if (in.opcode == kSop1OpcodeQuadmaskB64)
                    scalar_scc = exact_quadmask;
                else if (!preserves_scc)
                    scalar_scc = false;
            } else if (in.fmt == Rdna2Format::SOPK) {
                const bool writes_scc =
                    (in.opcode >= kSopkOpcodeCmpkFirst &&
                     in.opcode <= kSopkOpcodeCmpkLast) ||
                    in.opcode == kSopkOpcodeAddkI32;
                const bool preserves_scc =
                    in.opcode == kSopkOpcodeMovkI32 ||
                    in.opcode == kSopkOpcodeCmovkI32 ||
                    in.opcode == kSopkOpcodeMulkI32 ||
                    in.opcode == kSopkOpcodeSetregB32 ||
                    (in.opcode >= kSopkOpcodeWaitcntVscnt &&
                     in.opcode <= kSopkOpcodeWaitcntLgkmcnt);
                if (writes_scc)
                    scalar_scc = implicit_scalar_source;
                else if (!preserves_scc)
                    scalar_scc = false;
            }
            if (b64_mask_scc_vote_pcs.contains(in.pc) ||
                native_b32_mask_scc_vote_pcs.contains(in.pc) ||
                writes_exact_wave_scc)
                scalar_scc = true;
            return true;
        };

        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = wave64_b64_mask_in[block];
            std::set<int> ambiguous = wave64_b64_ambiguous_in[block];
            std::set<int> scalar_words = wave64_scalar_word_in[block];
            bool scalar_scc = wave64_scalar_scc_valid_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (!advance_wave64_b64_masks(
                        masks, ambiguous, scalar_words, scalar_scc, in,
                        /*record_compare*/false))
                    return false;
            }
            for (uint32_t successor : successors[block]) {
                if (!wave64_b64_reachable[successor]) {
                    wave64_b64_reachable[successor] = true;
                    wave64_b64_mask_in[successor] = masks;
                    wave64_b64_ambiguous_in[successor] = ambiguous;
                    wave64_scalar_word_in[successor] = scalar_words;
                    wave64_scalar_scc_valid_in[successor] = scalar_scc;
                    pending.push_back(successor);
                    continue;
                }
                std::set<int> joined;
                std::set<int> joined_ambiguous = wave64_b64_ambiguous_in[successor];
                std::set_intersection(
                    wave64_b64_mask_in[successor].begin(),
                    wave64_b64_mask_in[successor].end(),
                    masks.begin(), masks.end(),
                    std::inserter(joined, joined.end()));
                for (int base : wave64_b64_mask_in[successor])
                    if (!masks.contains(base)) joined_ambiguous.insert(base);
                for (int base : masks)
                    if (!wave64_b64_mask_in[successor].contains(base))
                        joined_ambiguous.insert(base);
                joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                for (int base : joined_ambiguous) joined.erase(base);
                std::set<int> joined_scalar_words;
                std::set_intersection(
                    wave64_scalar_word_in[successor].begin(),
                    wave64_scalar_word_in[successor].end(),
                    scalar_words.begin(), scalar_words.end(),
                    std::inserter(joined_scalar_words, joined_scalar_words.end()));
                const bool joined_scalar_scc =
                    wave64_scalar_scc_valid_in[successor] && scalar_scc;
                if (joined != wave64_b64_mask_in[successor] ||
                    joined_ambiguous != wave64_b64_ambiguous_in[successor] ||
                    joined_scalar_words != wave64_scalar_word_in[successor] ||
                    joined_scalar_scc != wave64_scalar_scc_valid_in[successor]) {
                    wave64_b64_mask_in[successor] = std::move(joined);
                    wave64_b64_ambiguous_in[successor] = std::move(joined_ambiguous);
                    wave64_scalar_word_in[successor] = std::move(joined_scalar_words);
                    wave64_scalar_scc_valid_in[successor] = joined_scalar_scc;
                    pending.push_back(successor);
                }
            }
        }
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!wave64_b64_reachable[block]) continue;
            std::set<int> masks = wave64_b64_mask_in[block];
            std::set<int> ambiguous = wave64_b64_ambiguous_in[block];
            std::set<int> scalar_words = wave64_scalar_word_in[block];
            bool scalar_scc = wave64_scalar_scc_valid_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (!advance_wave64_b64_masks(
                        masks, ambiguous, scalar_words, scalar_scc, in,
                        /*record_compare*/true))
                    return false;
            }
        }
    }

    // V_WRITELANE/V_READLANE scalar spills can carry one physical half of a Wave64 mask through a
    // loop. The Bool slot alone loses whether it was LO or HI, while a dispatcher uint placeholder
    // is not a validity tag. Track that identity as a CFG MUST fact and publish it only at exact
    // native-Wave64 readlane PCs. Joins retain equal facts; every ordinary overwrite kills them.
    std::vector<std::map<int, uint32_t>> wave64_mask_half_sreg_in(starts.size());
    std::vector<bool> wave64_mask_half_reachable(starts.size(), false);
    if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 && !starts.empty()) {
        struct MaskHalfState {
            std::map<int, uint32_t> sreg;
            std::map<std::pair<int, int>, uint32_t> slot;
            bool operator==(const MaskHalfState&) const = default;
        };
        auto special_half = [](const Operand& source) -> int {
            if (source.kind != OperandKind::Special) return -1;
            // EXEC is always a live mask in RegState. VCC_LO/HI may instead be scalar scratch, and
            // their physical encodings do not carry a runtime domain tag; treating those words as
            // masks here can turn a dispatcher placeholder into a ballot of false. Admit VCC only
            // after a future proof is explicitly tied to the Wave64 mask-domain MUST analysis.
            if (source.value == 126) return 0;
            if (source.value == 127) return 1;
            return -1;
        };
        auto meet = [](MaskHalfState& dst, const MaskHalfState& incoming) {
            for (auto it = dst.sreg.begin(); it != dst.sreg.end();) {
                const auto other = incoming.sreg.find(it->first);
                if (other == incoming.sreg.end() || other->second != it->second)
                    it = dst.sreg.erase(it);
                else
                    ++it;
            }
            for (auto it = dst.slot.begin(); it != dst.slot.end();) {
                const auto other = incoming.slot.find(it->first);
                if (other == incoming.slot.end() || other->second != it->second)
                    it = dst.slot.erase(it);
                else
                    ++it;
            }
        };
        auto transfer = [&](MaskHalfState& state, const Rdna2Inst& in, bool record) {
            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360) {
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    for (uint32_t word = 0; word < width; ++word)
                        state.sreg.erase(base + static_cast<int>(word));
                }, /*wave32_one_word_masks*/false);
                if (in.src[1].kind == OperandKind::InlineInt &&
                    in.src[1].value >= 0 && in.src[1].value <= 63) {
                    const std::pair<int, int> key{in.src[0].value, in.src[1].value};
                    const auto half = state.slot.find(key);
                    if (half != state.slot.end()) {
                        state.sreg[in.dst.value] = half->second;
                        if (record)
                            b.wave64_mask_readlane_half_for_pc[in.pc] = half->second;
                    }
                }
                return;
            }

            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t word = 0; word < width; ++word)
                    state.sreg.erase(base + static_cast<int>(word));
            }, /*wave32_one_word_masks*/false);

            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                // A dynamic selector may overwrite any lane and therefore kills every known slot
                // in this VGPR. A constant selector updates only its named slot.
                if (in.src[1].kind != OperandKind::InlineInt ||
                    in.src[1].value < 0 || in.src[1].value > 63) {
                    for (auto it = state.slot.begin(); it != state.slot.end();) {
                        if (it->first.first == in.dst.value) it = state.slot.erase(it);
                        else ++it;
                    }
                    return;
                }
                const std::pair<int, int> key{in.dst.value, in.src[1].value};
                int half = special_half(in.src[0]);
                if (half < 0 && in.src[0].kind == OperandKind::SGPR) {
                    const auto source = state.sreg.find(in.src[0].value);
                    if (source != state.sreg.end()) {
                        half = source->second;
                        if (record)
                            b.wave64_mask_writelane_alias_pcs.insert(in.pc);
                    }
                }
                if (half < 0) state.slot.erase(key);
                else state.slot[key] = static_cast<uint32_t>(half);
                return;
            }

            for_each_possible_vector_write(in, [&](int vgpr) {
                for (auto it = state.slot.begin(); it != state.slot.end();) {
                    if (it->first.first == vgpr) it = state.slot.erase(it);
                    else ++it;
                }
            });
        };

        std::vector<MaskHalfState> half_in(starts.size());
        std::vector<bool> half_reachable(starts.size(), false);
        half_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end)
                    transfer(state, in, /*record*/false);
            for (uint32_t successor : successors[block]) {
                if (!half_reachable[successor]) {
                    half_reachable[successor] = true;
                    half_in[successor] = state;
                    pending.push_back(successor);
                    continue;
                }
                MaskHalfState joined = half_in[successor];
                meet(joined, state);
                if (!(joined == half_in[successor])) {
                    half_in[successor] = std::move(joined);
                    pending.push_back(successor);
                }
            }
        }
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!half_reachable[block]) continue;
            wave64_mask_half_reachable[block] = true;
            wave64_mask_half_sreg_in[block] = half_in[block].sreg;
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end)
                    transfer(state, in, /*record*/true);
        }
    }

    auto mask_zero_compare_is_proven = [&](const Rdna2Inst& in) {
        // Wave32 has its own one-word MUST-domain filtering in load_state. Wave64 needs the
        // pair-width analysis above because a persisted Bool value is not itself a lifetime tag.
        return b.wave_size != 64 || proven_wave64_mask_zero_compare_pcs.contains(in.pc);
    };
    auto mask_zero_compare_value = [&](const RegState& state, int source) -> uint32_t {
        // The current GTA V sites name canonical VCC_LO as the low word of a B64 source. Its exact
        // per-lane value is stored separately from ordinary saved SGPR masks.
        if (b.wave_size == 64 && source == 106) return state.vcc;
        const auto found = state.sreg_bool.find(source);
        return found == state.sreg_bool.end() ? 0 : found->second;
    };

    // Fragment scalar PCs are lane-local in the dispatcher.  A saved-mask pair comparison must
    // nevertheless vote over every lane in the guest wave, including lanes currently executing a
    // different switch case.  Give every proven static comparison an identity and retain its exact
    // operands/polarity for the uniform common-phase votes emitted below.
    struct SavedMaskPairCompareEvent {
        int first = -1;
        int second = -1;
        uint32_t opcode = 0;
    };
    std::unordered_map<uint32_t, uint32_t> saved_mask_pair_event_for_pc;
    std::vector<SavedMaskPairCompareEvent> saved_mask_pair_events;
    if (b.is_fragment) {
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (!proven_saved_mask_pair_compare_pcs.contains(in.pc)) continue;
            const auto sources = saved_mask_pair_compare_sources(in);
            if (sources[0] < 0) continue;
            saved_mask_pair_events.push_back({sources[0], sources[1], in.opcode});
            saved_mask_pair_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(saved_mask_pair_events.size()));
        }
    }

    std::vector<std::unordered_set<int>> scalar_may_write_in(starts.size());
    std::vector<bool> scalar_reachable(starts.size(), false);
    if (!scalar_may_write_in.empty()) {
        scalar_may_write_in.front() = initial.sreg_written;
        scalar_reachable.front() = true;
    }
    bool provenance_changed = true;
    while (provenance_changed) {
        provenance_changed = false;
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!scalar_reachable[block]) continue;
            std::unordered_set<int> out = scalar_may_write_in[block];
            out.insert(scalar_writes[block].begin(), scalar_writes[block].end());
            for (uint32_t successor : successors[block]) {
                if (!scalar_reachable[successor]) {
                    scalar_reachable[successor] = true;
                    provenance_changed = true;
                }
                const size_t before = scalar_may_write_in[successor].size();
                scalar_may_write_in[successor].insert(out.begin(), out.end());
                provenance_changed |= scalar_may_write_in[successor].size() != before;
            }
        }
    }

    // The native exact-wave path does not need to return to a synchronized common phase after a
    // plain unconditional edge.  Fuse maximal forward chains whose successor has no other
    // predecessor; the successor cannot be entered from outside the chain, so keeping its register
    // state in SSA is equivalent to a dispatcher save/reload without duplicating guest code.  Keep
    // backward edges as dispatcher iterations (they form loops), and end a chain at every wave op
    // that must execute as one uniform switch case.
    std::vector<std::vector<uint32_t>> dispatch_blocks;
    std::vector<uint32_t> dispatch_for_block(starts.size(), UINT32_MAX);
    std::vector<uint32_t> predecessor_count(starts.size(), 0);
    for (const auto& edges : successors)
        for (uint32_t successor : edges)
            if (successor < predecessor_count.size()) ++predecessor_count[successor];
    std::vector<bool> synchronized_block(starts.size(), false);
    std::vector<bool> conditional_block(starts.size(), false);
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            synchronized_block[block] = synchronized_block[block] ||
                mbcnt_event_for_pc.contains(in.pc) || append_event_for_pc.contains(in.pc) ||
                swizzle_pcs.contains(in.pc) || bpermute_event_for_pc.contains(in.pc) ||
                fragment_dpp_min_row_shr_pcs.contains(in.pc) ||
                compute_dpp_add_row_shr_pcs.contains(in.pc) ||
                compute_dpp_row_ror8_pcs.contains(in.pc) ||
                compute_dpp_add_row_mask_pcs.contains(in.pc) ||
                lds_fminmax_pcs.contains(in.pc) ||
                mask_zero_compare_candidate_source(in) >= 0 ||
                exec_saved_mask_compare_source(in) >= 0 ||
                saved_mask_pair_compare_sources(in)[0] >= 0 ||
                vopc_mask_zero_compare_source(in) >= 0 ||
                b64_mask_scc_vote_pcs.contains(in.pc);
            conditional_block[block] = conditional_block[block] ||
                (!linearized_branch(in) && in.fmt == Rdna2Format::SOPP &&
                 in.opcode >= 0x04 && in.opcode <= 0x09 && in.opcode != 0x03);
        }
    }
    for (uint32_t first = 0; first < starts.size(); ++first) {
        if (dispatch_for_block[first] != UINT32_MAX) continue;
        const uint32_t dispatch = static_cast<uint32_t>(dispatch_blocks.size());
        dispatch_blocks.push_back({});
        uint32_t block = first;
        while (true) {
            dispatch_for_block[block] = dispatch;
            dispatch_blocks.back().push_back(block);
            if (!direct_dispatch || synchronized_block[block] || conditional_block[block] ||
                successors[block].size() != 1) break;
            const uint32_t successor = successors[block].front();
            if (successor <= block || predecessor_count[successor] != 1 ||
                dispatch_for_block[successor] != UINT32_MAX) break;
            block = successor;
        }
    }

    std::vector<std::set<int>> dispatch_vector_reads(dispatch_blocks.size());
    std::vector<std::set<int>> dispatch_vector_writes(dispatch_blocks.size());
    std::vector<std::unordered_set<int>> dispatch_scalar_writes(dispatch_blocks.size());
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        for (uint32_t block : dispatch_blocks[dispatch]) {
            dispatch_vector_reads[dispatch].insert(
                vector_reads[block].begin(), vector_reads[block].end());
            dispatch_vector_writes[dispatch].insert(
                vector_writes[block].begin(), vector_writes[block].end());
            dispatch_scalar_writes[dispatch].insert(
                scalar_writes[block].begin(), scalar_writes[block].end());
        }
    }

    // Saved mask pairs and scalar-spill lane slots have their own value domains.
    std::set<int> mask_keys = static_mask_keys;
    std::set<int> mask_half_alias_keys;
    for (uint32_t block = 0; block < starts.size(); ++block)
        if (wave64_mask_half_reachable[block])
            for (const auto& alias : wave64_mask_half_sreg_in[block])
                mask_half_alias_keys.insert(alias.first);
    std::set<std::pair<int, int>> lane_slots, mask_lane_slots;
    // A lane spill can precede another static definition of its mask in a back-edge block, hence the
    // complete discovery pass above occurs before classifying any spill slots here.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 &&
            in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value <= 63) {
            const std::pair<int, int> slot{in.dst.value, in.src[1].value};
            // A fragment Wave64 spill of a saved B64 mask consumes one physical ballot dword.
            // Persist it in the scalar-data lane domain; the Bool-slot representation cannot
            // distinguish LO from HI and would reload false at a later dispatcher case. Direct
            // architectural EXEC/VCC sources and other stages retain their mask-slot treatment.
            const bool fragment_physical_mask_word =
                b.is_fragment && b.wave_size == 64 &&
                in.src[0].kind == OperandKind::SGPR &&
                (static_mask_keys.count(in.src[0].value) ||
                 (in.src[0].value > 0 &&
                  static_mask_keys.count(in.src[0].value - 1)));
            const bool is_mask = !fragment_physical_mask_word &&
                (in.src[0].value == 106 || in.src[0].value == 107 ||
                 in.src[0].value == 126 || in.src[0].value == 127 ||
                 (in.src[0].kind == OperandKind::SGPR &&
                  (static_mask_keys.count(in.src[0].value) ||
                   b.wave64_mask_writelane_alias_pcs.contains(in.pc))));
            (is_mask ? mask_lane_slots : lane_slots).insert(slot);
        }
    }
    for (const auto& vg : initial.vgpr_lane_slots)
        for (const auto& slot : vg.second) lane_slots.emplace(vg.first, slot.first);
    for (const auto& vg : initial.vgpr_lane_mask_slots)
        for (const auto& slot : vg.second) mask_lane_slots.emplace(vg.first, slot.first);
    // A physical spill lane may be recycled between scalar-data and wave-mask lifetimes. Persist
    // both domains across dispatcher blocks; v_readlane retains both destination views when both
    // are present, and the statically typed consumer selects the representation it needs.

    // Map presence is the scalar-spill validity bit in RegState. Function variables cannot encode
    // that compile-time type state: saving an erased slot as zero and reconstructing it in the next
    // dispatcher case would turn an invalid lifetime into a valid scalar zero. Conservatively reject
    // any CFG path on which an ordinary VGPR write can invalidate a spill array before v_readlane;
    // a later v_writelane starts a fresh lifetime and clears the tombstone, matching emit_alu.
    std::set<int> spill_vgprs;
    for (const auto& slot : lane_slots) spill_vgprs.insert(slot.first);
    for (const auto& slot : mask_lane_slots) spill_vgprs.insert(slot.first);
    std::unordered_set<int> terminal_invalidated_vgpr_lane_slots =
        initial.invalidated_vgpr_lane_slots;
    if (!spill_vgprs.empty()) {
        std::vector<std::set<int>> invalidated_in(starts.size());
        std::vector<bool> invalidated_reachable(starts.size(), false);
        invalidated_in.front().insert(initial.invalidated_vgpr_lane_slots.begin(),
                                      initial.invalidated_vgpr_lane_slots.end());
        invalidated_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> invalidated = invalidated_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                    invalidated.erase(in.dst.value);
                    continue;
                }
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
                    invalidated.contains(in.src[0].value))
                    return reject_cfg(in.pc, "invalidated-vgpr-lane-slot");
                for_each_possible_vector_write(in, [&](int reg) {
                    if (spill_vgprs.contains(reg)) invalidated.insert(reg);
                });
                const int tfe_status = rdna2_tfe_status_vgpr(in);
                if (spill_vgprs.contains(tfe_status)) invalidated.insert(tfe_status);
            }
            for (uint32_t successor : successors[block]) {
                if (!invalidated_reachable[successor]) {
                    invalidated_reachable[successor] = true;
                    invalidated_in[successor] = invalidated;
                    pending.push_back(successor);
                    continue;
                }
                const size_t before = invalidated_in[successor].size();
                invalidated_in[successor].insert(invalidated.begin(), invalidated.end());
                if (invalidated_in[successor].size() != before) pending.push_back(successor);
            }
        }
        if (const auto terminal = block_for_pc.find(end_pc);
            terminal != block_for_pc.end() && invalidated_reachable[terminal->second])
            terminal_invalidated_vgpr_lane_slots = {
                invalidated_in[terminal->second].begin(),
                invalidated_in[terminal->second].end()};
    }

    uint32_t ptr_u32 = 0, ptr_bool = 0;
    std::map<int, uint32_t> vv, sv, mv, mhv;
    std::map<std::pair<int, int>, uint32_t> lv, lmv;
    for (int r : vregs) vv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : sregs) sv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : mask_keys) mv[r] = b.function_var(b.t_bool, ptr_bool);
    for (int r : mask_half_alias_keys) mhv[r] = b.function_var(b.t_bool, ptr_bool);
    for (const auto& slot : lane_slots) lv[slot] = b.function_var(b.t_u32, ptr_u32);
    for (const auto& slot : mask_lane_slots) lmv[slot] = b.function_var(b.t_bool, ptr_bool);
    const uint32_t scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vcc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t exec_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t pc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t active_var = b.function_var(b.t_bool, ptr_bool);
    // PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Force the dispatcher out after N iterations.
    //
    // A dispatcher loop that never terminates hangs the GPU into a driver reset, which costs the
    // whole process its compute backend and every later indirect draw. That failure is
    // indistinguishable, from outside, from a shader that is merely slow or from a defect anywhere
    // else in the submit. Bounding the loop turns "is this a non-terminating dispatcher?" into a
    // one-run yes/no: if the device survives with a bound and dies without one, it is a loop.
    //
    // Unset, nothing is emitted and the module is byte-identical. It is NOT a fix — truncating a
    // guest program's control flow produces wrong results by construction — which is why it is
    // opt-in and says so.
    const uint32_t cfg_phase = b.cfg_phase_ordinal++;
    // ONE end-exclusive boundary for this phase, derived once and used by every report below.
    //
    // The two consumers disagreed before: the announcement used `ins.back().pc` (right for a phase
    // closed by the emitter's synthetic terminator, wrong for a real tail, where it drops the final
    // instruction) while the dispatch map used `pc + len_dwords` (right for a real tail, wrong for a
    // synthetic one, where it re-includes a boundary marker that is not guest code). The fixture made
    // them contradict each other in adjacent lines: `phase 0 guest pc 0..<14` above `6:pc14..<15`.
    const uint32_t cfg_phase_end =
        ins.empty() ? 0u
                    : (ins.back().synthetic_terminator ? ins.back().pc
                                                       : ins.back().pc + ins.back().len_dwords);
    const uint32_t cfg_trip_bound =
        emitted_loop_trip_bound(b.diagnostic.program_address, cfg_phase,
                                ins.empty() ? 0u : ins.front().pc, cfg_phase_end,
                                code, dwords);
    const uint32_t trip_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    if (trip_var) b.store_function(trip_var, b.uconst(0));
    // Publish the ordinal -> guest pc map for this phase, once, when a bound arms.
    //
    // Every number the witness reports is a dispatch ordinal, and an ordinal means nothing on its
    // own. Leaving the reader to reconstruct the mapping is not a documentation gap, it is a defect
    // in the instrument: an ordinal that happens to fall inside the block COUNT reads as a plausible
    // block index, and mapping it by hand onto a guest pc range is exactly how a wrong conclusion got
    // published here (instrument trap 172). Emitting the map costs one line per ordinal, once.
    if (compute_trip_bound_settings().bound) {
        std::string map;
        char entry[64];
        for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
            if (dispatch_blocks[dispatch].empty()) continue;
            const uint32_t entry_block = dispatch_blocks[dispatch].front();
            const uint32_t last_block = dispatch_blocks[dispatch].back();
            const uint32_t lo = starts[entry_block];
            // The final ordinal has no following block start to bound it. Use the phase's own end
            // -- one dword past its last decoded instruction -- rather than UINT32_MAX, which is not
            // a pc and made the last map entry unreadable. `len_dwords` is the decoded length, so
            // this is exact for a variable-length ISA rather than assuming one dword.
            const uint32_t hi = last_block + 1 < starts.size() ? starts[last_block + 1]
                                                              : cfg_phase_end;
            const int written = snprintf(entry, sizeof(entry), "%s%u:pc%u..<%u", dispatch ? " " : "",
                                         dispatch, lo, hi);
            if (written > 0) map.append(entry, static_cast<size_t>(
                std::min<size_t>(static_cast<size_t>(written), sizeof(entry) - 1)));
        }
        fprintf(stderr, "[cfg-trip-bound] program 0x%llx phase %u dispatch map: %s\n",
                static_cast<unsigned long long>(b.diagnostic.program_address), cfg_phase,
                map.c_str());
    }
    // The span of DISPATCH ORDINALS the state machine actually visited, tracked only while a bound
    // is armed.
    //
    // `pc_var` holds a dispatcher switch-case ordinal (`dispatch_for_block`), NOT a guest pc — read
    // the store sites, not the variable's name. One ordinal at the instant the cap ran out is a
    // single sample and cannot separate "spinning here" from "passing through here"; the extremes
    // can, because a state machine confined to a few ordinals is cycling among them.
    //
    // Ordinals are only meaningful against the map this phase announces below, which is why that map
    // is printed rather than left to be reconstructed by hand. Hand-mapping one of these numbers onto
    // a guest pc range is precisely what produced a wrong published conclusion — instrument trap 172.
    const uint32_t dispatch_min_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    const uint32_t dispatch_max_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    if (dispatch_min_var) b.store_function(dispatch_min_var, b.uconst(0xffffffffu));
    if (dispatch_max_var) b.store_function(dispatch_max_var, b.uconst(0));
    const uint32_t vote_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_value_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_invert_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_to_scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_to_vcc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_taken_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t vote_next_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_mask_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_low_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_write_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_acc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_sum_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_consume_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_gds_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_idx_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_count_var = b.function_var(b.t_u32, ptr_u32);
    const bool has_synchronized_lds_store_event = !synchronized_lds_store_pcs.empty();
    const uint32_t synchronized_lds_store_pending_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t synchronized_lds_store_active_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t synchronized_lds_store_count_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    std::array<uint32_t, 4> synchronized_lds_store_idx_vars{};
    std::array<uint32_t, 4> synchronized_lds_store_value_vars{};
    if (has_synchronized_lds_store_event) {
        for (uint32_t& var : synchronized_lds_store_idx_vars)
            var = b.function_var(b.t_u32, ptr_u32);
        for (uint32_t& var : synchronized_lds_store_value_vars)
            var = b.function_var(b.t_u32, ptr_u32);
    }
    const bool has_lds_fminmax_event = !lds_fminmax_pcs.empty();
    const uint32_t lds_fminmax_pending_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_active_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_min_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_idx_var = has_lds_fminmax_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t lds_fminmax_value_var = has_lds_fminmax_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t swizzle_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_source_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_source_lane_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_dst_var = b.function_var(b.t_u32, ptr_u32);
    const bool has_bpermute = !bpermute_event_for_pc.empty();
    const uint32_t bpermute_pending_var = has_bpermute
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t bpermute_active_var = has_bpermute
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t bpermute_address_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_source_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_offset_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_event_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_dst_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_saved_mask_pair_events = !saved_mask_pair_events.empty();
    const uint32_t saved_mask_pair_pending_var = has_saved_mask_pair_events
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t saved_mask_pair_event_var = has_saved_mask_pair_events
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_dpp_min_row_shr = !fragment_dpp_min_row_shr_pcs.empty();
    const uint32_t dpp_min_pending_var = has_dpp_min_row_shr
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_min_active_var = has_dpp_min_row_shr
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_min_source_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_amount_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_dst_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_event_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_pending_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_add_active_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_add_source_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_amount_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_dst_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_event_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_pending_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_ror8_active_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_ror8_src0_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_src1_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_op_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_dst_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_event_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_portable_mask_ffbh = !portable_mask_ffbh_event_for_pc.empty();
    const uint32_t mask_ffbh_pending_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_mask_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_write_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_event_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t mask_ffbh_half_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t mask_ffbh_dst_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;

    const uint32_t zero = b.uconst(0), no = b.bfalse(), yes = b.btrue();
    for (const auto& kv : vv) {
        auto it = initial.vreg.find(kv.first);
        b.store_function(kv.second, it == initial.vreg.end() ? zero : it->second);
    }
    for (const auto& kv : sv) {
        auto it = initial.sreg.find(kv.first);
        b.store_function(kv.second, it == initial.sreg.end() ? zero : it->second);
    }
    for (const auto& kv : mv) {
        auto it = initial.sreg_bool.find(kv.first);
        b.store_function(kv.second, it == initial.sreg_bool.end() ? no : it->second);
    }
    for (const auto& kv : mhv) b.store_function(kv.second, no);
    for (const auto& kv : lv) {
        uint32_t value = zero;
        auto vg = initial.vgpr_lane_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    for (const auto& kv : lmv) {
        uint32_t value = no;
        auto vg = initial.vgpr_lane_mask_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_mask_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    b.store_function(scc_var, initial.scc ? initial.scc : no);
    // The dispatcher has no runtime type tag for a physical VCC word recycled as scalar data.
    // Wave32's compile-time mask-domain analysis above proves whether each implicit VCC consumer
    // sees a real mask. It is therefore safe to persist false while that lifetime is absent/dead;
    // load_state keeps the placeholder out of RegState on those entries. Other modes retain the
    // old fail-visible contract because they have no equivalent proof.
    if (!initial.vcc && !proven_wave32_masks)
        return reject_cfg(ins.front().pc, "missing-entry-vcc");
    b.store_function(vcc_var, initial.vcc ? initial.vcc : no);
    b.store_function(exec_var, initial.exec);
    b.store_function(pc_var, b.uconst(0));
    b.store_function(active_var, initial_active ? initial_active : yes);
    if (has_synchronized_lds_store_event) {
        b.store_function(synchronized_lds_store_count_var, zero);
        for (uint32_t var : synchronized_lds_store_idx_vars)
            b.store_function(var, zero);
        for (uint32_t var : synchronized_lds_store_value_vars)
            b.store_function(var, zero);
    }
    if (has_lds_fminmax_event) {
        b.store_function(lds_fminmax_min_var, no);
        b.store_function(lds_fminmax_idx_var, zero);
        b.store_function(lds_fminmax_value_var, zero);
    }
    b.store_function(swizzle_source_var, zero);
    b.store_function(swizzle_source_lane_var, zero);
    b.store_function(swizzle_dst_var, zero);
    if (has_bpermute) {
        b.store_function(bpermute_address_var, zero);
        b.store_function(bpermute_source_var, zero);
        b.store_function(bpermute_offset_var, zero);
        b.store_function(bpermute_event_var, zero);
        b.store_function(bpermute_dst_var, zero);
    }
    if (has_saved_mask_pair_events)
        b.store_function(saved_mask_pair_event_var, zero);
    if (has_dpp_min_row_shr) {
        b.store_function(dpp_min_source_var, zero);
        b.store_function(dpp_min_amount_var, zero);
        b.store_function(dpp_min_dst_var, zero);
        b.store_function(dpp_min_event_var, zero);
    }
    if (has_portable_compute_dpp_add) {
        b.store_function(dpp_add_source_var, zero);
        b.store_function(dpp_add_amount_var, zero);
        b.store_function(dpp_add_dst_var, zero);
        b.store_function(dpp_add_event_var, zero);
    }
    if (has_portable_compute_dpp_ror8) {
        b.store_function(dpp_ror8_src0_var, zero);
        b.store_function(dpp_ror8_src1_var, zero);
        b.store_function(dpp_ror8_op_var, zero);
        b.store_function(dpp_ror8_dst_var, zero);
        b.store_function(dpp_ror8_event_var, zero);
    }
    if (has_portable_mask_ffbh) {
        b.store_function(mask_ffbh_event_var, zero);
        b.store_function(mask_ffbh_half_var, zero);
        b.store_function(mask_ffbh_dst_var, zero);
    }

    auto load_state = [&](uint32_t dispatch = UINT32_MAX) {
        RegState state;
        state.scalar_presence_has_no_placeholders = false;
        state.max_vgpr = initial.max_vgpr;
        state.sreg_input = initial.sreg_input;
        state.smem_x16_descriptor_loads = initial.smem_x16_descriptor_loads;
        state.smem_x16_descriptor_analysis_done = initial.smem_x16_descriptor_analysis_done;
        state.smem_x2_descriptor_fragment_loads =
            initial.smem_x2_descriptor_fragment_loads;
        state.smem_x2_descriptor_fragment_analysis_done =
            initial.smem_x2_descriptor_fragment_analysis_done;
        state.reads_scc = initial.reads_scc;
        state.invalidated_vgpr_lane_slots = initial.invalidated_vgpr_lane_slots;
        for (const auto& kv : vv) {
            if (dispatch != UINT32_MAX &&
                !dispatch_vector_reads[dispatch].contains(kv.first)) continue;
            state.vreg[kv.first] = b.load_function(b.t_u32, kv.second);
        }
        for (const auto& kv : sv) state.sreg[kv.first] = b.load_function(b.t_u32, kv.second);
        const std::set<int>* entry_b32 = nullptr;
        const std::set<int>* entry_b64 = nullptr;
        const std::set<int>* entry_wave64_b64 = nullptr;
        const std::set<int>* entry_wave64_scalar_words = nullptr;
        const std::map<int, uint32_t>* entry_wave64_mask_halves = nullptr;
        uint32_t entry_block = UINT32_MAX;
        if (dispatch != UINT32_MAX)
            entry_block = dispatch_blocks[dispatch].front();
        else if (const auto terminal = block_for_pc.find(end_pc);
                 terminal != block_for_pc.end())
            entry_block = terminal->second;
        if (b.allow_b32_masks) {
            if (entry_block != UINT32_MAX && b32_mask_reachable[entry_block]) {
                entry_b32 = &b32_mask_in[entry_block];
                entry_b64 = &b64_mask_in[entry_block];
            }
        }
        const bool filters_wave64_b64 = (b.is_compute || b.is_fragment) &&
            b.wave_size == 64;
        if (filters_wave64_b64 && entry_block != UINT32_MAX &&
            wave64_b64_reachable[entry_block]) {
            entry_wave64_b64 = &wave64_b64_mask_in[entry_block];
            entry_wave64_scalar_words = &wave64_scalar_word_in[entry_block];
        }
        if (entry_block != UINT32_MAX &&
            wave64_mask_half_reachable[entry_block])
            entry_wave64_mask_halves = &wave64_mask_half_sreg_in[entry_block];
        for (const auto& kv : mv) {
            if (b.allow_b32_masks &&
                (!entry_b64 || !entry_b64->contains(kv.first)) &&
                (!entry_b32 || !entry_b32->contains(kv.first)))
                continue;
            if (filters_wave64_b64 &&
                (!entry_wave64_b64 || !entry_wave64_b64->contains(kv.first)))
                continue;
            state.sreg_bool[kv.first] = b.load_function(b.t_bool, kv.second);
            state.sreg_bool_narrowed[kv.first] = true;
        }
        if (entry_wave64_mask_halves)
            for (const auto& kv : mhv)
                if (const auto half = entry_wave64_mask_halves->find(kv.first);
                    half != entry_wave64_mask_halves->end()) {
                    state.sreg_wave64_mask_half[kv.first] =
                        b.load_function(b.t_bool, kv.second);
                    state.sreg_wave64_mask_half_index[kv.first] = half->second;
                }
        for (const auto& low : state.sreg_wave64_mask_half) {
            const int base = low.first;
            if (base & 1) continue;
            const auto high = state.sreg_wave64_mask_half.find(base + 1);
            const auto low_index = state.sreg_wave64_mask_half_index.find(base);
            const auto high_index = state.sreg_wave64_mask_half_index.find(base + 1);
            if (high == state.sreg_wave64_mask_half.end() ||
                low_index == state.sreg_wave64_mask_half_index.end() ||
                high_index == state.sreg_wave64_mask_half_index.end() ||
                low_index->second != 0 || high_index->second != 1)
                continue;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
            state.sreg_bool[base] = b.bsel(
                b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                high->second, low.second);
            state.sreg_bool_narrowed[base] = true;
        }
        for (const auto& kv : lv)
            state.vgpr_lane_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_u32, kv.second);
        for (const auto& kv : lmv)
            state.vgpr_lane_mask_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_bool, kv.second);
        const bool filters_wave64_scalar_scc =
            (b.is_compute || b.is_fragment) && b.wave_size == 64;
        const bool live_scalar_scc = !filters_wave64_scalar_scc ||
            (entry_block != UINT32_MAX &&
             entry_block < wave64_scalar_scc_valid_in.size() &&
             wave64_scalar_scc_valid_in[entry_block]);
        state.scc = live_scalar_scc
            ? b.load_function(b.t_bool, scc_var) : 0;
        const bool live_vcc = filters_wave64_b64
            ? entry_wave64_b64 && entry_wave64_b64->contains(106)
            : (!entry_b32 || entry_b32->contains(106));
        state.vcc = live_vcc ? b.load_function(b.t_bool, vcc_var) : 0;
        state.exec = b.load_function(b.t_bool, exec_var);
        // `sv` has a Function variable for every statically observed scalar lifetime, including
        // zero placeholders stored while the same physical pair carries only a Bool-domain mask.
        // Do not let those placeholders shadow the live mask after a dispatcher reload. A genuinely
        // dual-domain value (an inline/native-ballot S_MOV or scalar CSELECT-to-VCC) retains exact
        // scalar-word MUST facts and therefore keeps its scalar loads.
        if (entry_wave64_b64) {
            for (int base : *entry_wave64_b64) {
                for (int word = base; word < base + 2; ++word) {
                    if (entry_wave64_scalar_words &&
                        entry_wave64_scalar_words->contains(word))
                        continue;
                    state.sreg.erase(word);
                    state.sreg_input.erase(word);
                }
            }
        }
        // Every referenced scalar has a Function variable, initialized to zero even when the
        // architectural word has no value on this path. The Wave64 MUST analysis is its separate
        // validity tag: remove all absent ordinary/special scalar words before emitting a case so
        // an uninitialized M0/SGPR cannot become a valid zero merely by crossing the dispatcher.
        if (entry_wave64_scalar_words) {
            for (auto it = state.sreg.begin(); it != state.sreg.end();) {
                if (it->first <= 124 && !entry_wave64_scalar_words->contains(it->first))
                    it = state.sreg.erase(it);
                else
                    ++it;
            }
            for (auto it = state.sreg_input.begin(); it != state.sreg_input.end();) {
                if (it->first <= 124 && !entry_wave64_scalar_words->contains(it->first))
                    it = state.sreg_input.erase(it);
                else
                    ++it;
            }
        }
        if (entry_b32) {
            for (int reg : *entry_b32) {
                state.sreg.erase(reg);
                state.sreg_input.erase(reg);
                state.sreg_bool_b32.insert(reg);
                if (reg == 106) {
                    state.sreg_bool[106] = state.vcc;
                    state.sreg_bool_narrowed[106] = true;
                }
            }
        }
        state.exec_narrowed = true; // state-machine joins may carry a narrowed EXEC edge
        state.mubuf_pcrel_tables = initial.mubuf_pcrel_tables;
        state.smem_pcrel_tables = initial.smem_pcrel_tables;
        return state;
    };
    auto save_state = [&](const RegState& state, uint32_t dispatch) {
        // A dispatcher case starts from the persistent register file, executes exactly one guest
        // basic block, then returns to the common loop.  Only registers WRITTEN by that block need
        // to be copied back.  Saving every tracked VGPR/SGPR at every edge generated thousands of
        // redundant Function-memory stores for large kernels (Astro Bot's title compute shader has
        // around one hundred tracked registers and dozens of cases), creating severe register
        // pressure and scratch traffic in the host driver.  The conservative static writer sets
        // include multi-register loads and secondary scalar destinations; unchanged variables keep
        // the value loaded at the preceding dispatcher edge.
        for (const auto& kv : vv) {
            if (!dispatch_vector_writes[dispatch].contains(kv.first)) continue;
            auto it = state.vreg.find(kv.first);
            b.store_function(kv.second, it == state.vreg.end() ? zero : it->second);
        }
        for (const auto& kv : sv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg.find(kv.first);
            b.store_function(kv.second, it == state.sreg.end() ? zero : it->second);
        }
        for (const auto& kv : mv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg_bool.find(kv.first);
            b.store_function(kv.second, it == state.sreg_bool.end() ? no : it->second);
        }
        for (const auto& kv : mhv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg_wave64_mask_half.find(kv.first);
            b.store_function(kv.second,
                             it == state.sreg_wave64_mask_half.end() ? no : it->second);
        }
        for (const auto& kv : lv) {
            uint32_t value = zero;
            auto vg = state.vgpr_lane_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        for (const auto& kv : lmv) {
            uint32_t value = no;
            auto vg = state.vgpr_lane_mask_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_mask_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        // A poisoned (0) SCC degrades to bfalse at the block boundary: the same-block consumers
        // reject on the sentinel; cross-block staleness matches the pre-poison model.
        b.store_function(scc_var, state.scc ? state.scc : b.bfalse());
        b.store_function(vcc_var, state.vcc ? state.vcc : no);
        b.store_function(exec_var, state.exec);
    };

    const uint32_t loop_header = b.id(), switch_header = b.id(), switch_merge = b.id();
    const uint32_t loop_continue = b.id(), loop_merge = b.id(), fallback = b.id();
    std::vector<uint32_t> labels(dispatch_blocks.size());
    std::vector<std::pair<uint32_t, uint32_t>> switch_cases;
    for (uint32_t i = 0; i < dispatch_blocks.size(); ++i) {
        labels[i] = b.id();
        switch_cases.emplace_back(i, labels[i]);
    }
    b.emit_branch(loop_header);
    b.emit_label(loop_header);
    // Every live hardware wave executes one guest basic block per dispatcher iteration. Inactive
    // invocations remain in the loop as synchronization participants until all waves finish.
    // With an exact native subgroup, one host subgroup is one guest wave and its scalar PC is
    // uniform.  Cross-lane operations can therefore execute directly in their switch case.  The
    // old publish/merge phase remains necessary only for the portable workgroup-scratch fallback;
    // resetting all of its mailboxes on every native dispatcher iteration was a surprisingly large
    // SALU/function-memory tax for branch-heavy kernels.
    if (!direct_dispatch) {
        b.store_function(vote_pending_var, no);
        b.store_function(vote_value_var, no);
        b.store_function(vote_invert_var, no);
        b.store_function(vote_to_scc_var, no);
        b.store_function(vote_to_vcc_var, no);
        b.store_function(vote_taken_var, zero);
        b.store_function(vote_next_var, zero);
        b.store_function(mbcnt_pending_var, no);
        b.store_function(mbcnt_mask_var, no);
        b.store_function(mbcnt_low_var, no);
        b.store_function(mbcnt_write_var, no);
        b.store_function(mbcnt_event_var, zero);
        b.store_function(mbcnt_acc_var, zero);
        b.store_function(mbcnt_dst_var, zero);
        b.store_function(append_pending_var, no);
        b.store_function(append_active_var, no);
        b.store_function(append_event_var, zero);
        b.store_function(append_consume_var, no);
        b.store_function(append_gds_var, no);
        b.store_function(append_idx_var, zero);
        b.store_function(append_dst_var, zero);
    }
    if (has_lds_fminmax_event) {
        b.store_function(lds_fminmax_pending_var, no);
        b.store_function(lds_fminmax_active_var, no);
    }
    if (has_synchronized_lds_store_event) {
        b.store_function(synchronized_lds_store_pending_var, no);
        b.store_function(synchronized_lds_store_active_var, no);
        b.store_function(synchronized_lds_store_count_var, zero);
    }
    b.store_function(swizzle_pending_var, no);
    b.store_function(swizzle_active_var, no);
    if (has_bpermute) {
        b.store_function(bpermute_pending_var, no);
        b.store_function(bpermute_active_var, no);
        b.store_function(bpermute_event_var, zero);
    }
    if (has_saved_mask_pair_events) {
        b.store_function(saved_mask_pair_pending_var, no);
        b.store_function(saved_mask_pair_event_var, zero);
    }
    if (has_dpp_min_row_shr) {
        b.store_function(dpp_min_pending_var, no);
        b.store_function(dpp_min_active_var, no);
        b.store_function(dpp_min_event_var, zero);
    }
    if (has_portable_compute_dpp_add) {
        b.store_function(dpp_add_pending_var, no);
        b.store_function(dpp_add_active_var, no);
        b.store_function(dpp_add_event_var, zero);
    }
    if (has_portable_compute_dpp_ror8) {
        b.store_function(dpp_ror8_pending_var, no);
        b.store_function(dpp_ror8_active_var, no);
        b.store_function(dpp_ror8_event_var, zero);
    }
    if (has_portable_mask_ffbh) {
        b.store_function(mask_ffbh_pending_var, no);
        b.store_function(mask_ffbh_mask_var, no);
        b.store_function(mask_ffbh_write_var, no);
        b.store_function(mask_ffbh_event_var, zero);
    }
    b.emit_loopmerge(loop_merge, loop_continue);
    b.emit_branch(switch_header);
    b.emit_label(switch_header);
    const uint32_t selector = b.sel(b.load_function(b.t_bool, active_var),
                                    b.load_function(b.t_u32, pc_var), b.uconst(UINT32_MAX));
    b.emit_selmerge(switch_merge);
    b.emit_switch(selector, fallback, switch_cases);

    auto set_next = [&](uint32_t pc) {
        auto found = block_for_pc.find(pc);
        if (pc > end_pc) {
            if (!proven_exit_target(pc)) return false;
            b.store_function(active_var, no);
        }
        else if (found == block_for_pc.end()) {
            if (getenv("PROSPER_DBG")) {
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-successor-reject", "terminal",
                    "pc=%u end=%u blocks=%zu", pc, end_pc, block_for_pc.size());
                std::string successors;
                char successor[32];
                for (const auto& entry : block_for_pc) {
                    const int written = std::snprintf(successor, sizeof successor, " %u",
                                                      entry.first);
                    if (written > 0)
                        successors.append(successor, std::min<size_t>(
                            static_cast<size_t>(written), sizeof(successor) - 1));
                }
                log_recompile_diagnostic(b.diagnostic, "compute-cfg-successors", "terminal",
                                         "%s", successors.c_str());
            }
            return false;
        }
        else b.store_function(pc_var, b.uconst(dispatch_for_block[found->second]));
        return true;
    };
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        const uint32_t entry_block = dispatch_blocks[dispatch].front();
        b.emit_label(labels[dispatch]);
        RegState state = load_state(dispatch);
        state.sreg_written = scalar_may_write_in[entry_block];
        for (int reg : state.sreg_written) state.sreg_input.erase(reg);
        if (!direct_descriptor_sregs.empty()) {
            for (int reg : direct_descriptor_sregs)
                if (!state.sreg_written.count(reg)) state.sreg.erase(reg);
        }
        const Rdna2Inst* terminator = nullptr;
        const Rdna2Inst* mbcnt = nullptr;
        const Rdna2Inst* append = nullptr;
        const Rdna2Inst* synchronized_lds_store = nullptr;
        const Rdna2Inst* lds_fminmax = nullptr;
        const Rdna2Inst* swizzle = nullptr;
        const Rdna2Inst* bpermute = nullptr;
        const Rdna2Inst* dpp_min_row_shr = nullptr;
        const Rdna2Inst* dpp_add_row_shr = nullptr;
        const Rdna2Inst* dpp_row_ror8 = nullptr;
        const Rdna2Inst* dpp_add_row_mask = nullptr;
        const Rdna2Inst* mask_ffbh = nullptr;
        const Rdna2Inst* mask_compare = nullptr;
        const Rdna2Inst* exec_saved_mask_compare = nullptr;
        const Rdna2Inst* saved_mask_pair_compare = nullptr;
        const Rdna2Inst* vopc_mask_compare = nullptr;
        const Rdna2Inst* b64_mask_scc_vote = nullptr;
        const uint32_t final_block = dispatch_blocks[dispatch].back();
        const uint32_t final_hi = final_block + 1 < starts.size()
            ? starts[final_block + 1] : UINT32_MAX;
        for (size_t member = 0; member < dispatch_blocks[dispatch].size(); ++member) {
            const uint32_t block = dispatch_blocks[dispatch][member];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            const Rdna2Inst* block_terminator = nullptr;
            const Rdna2Inst* block_mbcnt = nullptr;
            const Rdna2Inst* block_append = nullptr;
            const Rdna2Inst* block_synchronized_lds_store = nullptr;
            const Rdna2Inst* block_lds_fminmax = nullptr;
            const Rdna2Inst* block_swizzle = nullptr;
            const Rdna2Inst* block_bpermute = nullptr;
            const Rdna2Inst* block_dpp_min_row_shr = nullptr;
            const Rdna2Inst* block_dpp_add_row_shr = nullptr;
            const Rdna2Inst* block_dpp_row_ror8 = nullptr;
            const Rdna2Inst* block_dpp_add_row_mask = nullptr;
            const Rdna2Inst* block_mask_ffbh = nullptr;
            const Rdna2Inst* block_mask_compare = nullptr;
            const Rdna2Inst* block_exec_saved_mask_compare = nullptr;
            const Rdna2Inst* block_saved_mask_pair_compare = nullptr;
            const Rdna2Inst* block_vopc_mask_compare = nullptr;
            const Rdna2Inst* block_b64_mask_scc_vote = nullptr;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi) continue;
                if (cfg_terminator(in)) {
                    block_terminator = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::VOP3 &&
                    (in.opcode == 0x365 || in.opcode == 0x366)) {
                    block_mbcnt = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
                    block_append = &in;
                    break;
                }
                if (synchronized_lds_store_pcs.contains(in.pc)) {
                    block_synchronized_lds_store = &in;
                    break;
                }
                if (lds_fminmax_pcs.contains(in.pc)) {
                    block_lds_fminmax = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
                    block_swizzle = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && in.opcode == kDsOpcodeBpermuteB32) {
                    block_bpermute = &in;
                    break;
                }
                if (fragment_dpp_min_row_shr_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[graphics-cfg-dpp-min-row-shr] "
                                     "pc=%u vgpr=v%d amount=%u\n",
                                     in.pc, in.dst.value,
                                     static_cast<uint32_t>(in.dpp_ctrl - 0x110u));
                    block_dpp_min_row_shr = &in;
                    break;
                }
                if (compute_dpp_add_row_shr_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-add-row-shr] "
                                     "pc=%u vgpr=v%d amount=%u\n",
                                     in.pc, in.dst.value,
                                     static_cast<uint32_t>(in.dpp_ctrl - 0x110u));
                    block_dpp_add_row_shr = &in;
                    break;
                }
                if (compute_dpp_row_ror8_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-row-ror8] "
                                     "pc=%u op=%u dst=v%d src0=v%d src1=v%d\n",
                                     in.pc, static_cast<uint32_t>(dpp_row_ror8_op(in)),
                                     in.dst.value, in.src[0].value,
                                     in.n_src > 1 ? in.src[1].value : -1);
                    block_dpp_row_ror8 = &in;
                    break;
                }
                if (compute_dpp_add_row_mask_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-add-row-mask] "
                                     "pc=%u dst=v%d src1=v%d row_mask=0x%x\n",
                                     in.pc, in.dst.value, in.src[1].value,
                                     in.dpp_row_mask);
                    block_dpp_add_row_mask = &in;
                    break;
                }
                if (portable_mask_ffbh_candidate(in) &&
                    !state.sreg.contains(in.src[0].value) &&
                    !state.sreg_input.contains(in.src[0].value)) {
                    const int source = in.src[0].value;
                    const bool source_is_b64_base = state.sreg_bool.contains(source) &&
                        !state.sreg_bool_b32.contains(source);
                    const bool previous_is_b64_base = source > 0 &&
                        state.sreg_bool.contains(source - 1) &&
                        !state.sreg_bool_b32.contains(source - 1);
                    if (source_is_b64_base || previous_is_b64_base) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[compute-cfg-mask-ffbh] pc=%u source=s%d half=%u\n",
                                         in.pc, source,
                                         source_is_b64_base ? 0u : 1u);
                        block_mask_ffbh = &in;
                        break;
                    }
                }
                const int mask_compare_source =
                    mask_zero_compare_candidate_source(in);
                if (mask_compare_source >= 0 &&
                    mask_zero_compare_is_proven(in) &&
                    mask_zero_compare_value(state, mask_compare_source)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-mask-compare] pc=%u source=s%d op=0x%x\n",
                                     in.pc, mask_compare_source, in.opcode);
                    block_mask_compare = &in;
                    break;
                }
                const int exec_saved_mask_source =
                    exec_saved_mask_compare_source(in);
                if (exec_saved_mask_source >= 0 &&
                    proven_exec_saved_mask_compare_pcs.contains(in.pc) &&
                    state.sreg_bool.contains(exec_saved_mask_source)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-exec-saved-mask-compare] "
                                     "pc=%u source=s%d op=0x%x\n",
                                     in.pc, exec_saved_mask_source, in.opcode);
                    block_exec_saved_mask_compare = &in;
                    break;
                }
                const auto saved_pair_sources =
                    saved_mask_pair_compare_sources(in);
                if (saved_pair_sources[0] >= 0 &&
                    proven_saved_mask_pair_compare_pcs.contains(in.pc) &&
                    state.sreg_bool.contains(saved_pair_sources[0]) &&
                    state.sreg_bool.contains(saved_pair_sources[1])) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[%s-saved-mask-pair-compare] "
                                     "pc=%u sources=s[%d:%d],s[%d:%d] op=0x%x\n",
                                     b.is_compute ? "compute" : "graphics",
                                     in.pc, saved_pair_sources[0], saved_pair_sources[0] + 1,
                                     saved_pair_sources[1], saved_pair_sources[1] + 1,
                                     in.opcode);
                    block_saved_mask_pair_compare = &in;
                    break;
                }
                const int vopc_mask_compare_source =
                    vopc_mask_zero_compare_source(in);
                if (vopc_mask_compare_source >= 0 &&
                    (state.sreg_bool.contains(vopc_mask_compare_source) ||
                     (vopc_mask_compare_source == 106 && state.vcc))) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-vopc-mask-compare] pc=%u source=s%d op=0x%x\n",
                                     in.pc, vopc_mask_compare_source, in.opcode);
                    block_vopc_mask_compare = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::EXP) {
                    if (!exp_fn(state, in)) return reject_cfg(in.pc, "export");
                    continue;
                }
                if (proven_wave64_mask_reduction_pcs.contains(in.pc)) {
                    const int source = wave64_mask_reduction_source(in);
                    const bool mask_available = source == 126
                        ? state.exec != 0
                        : source == 106
                        ? state.vcc != 0
                        : state.sreg_bool.contains(source);
                    if (source < 0 || !mask_available)
                        return reject_cfg(in.pc, "proven mask reduction missing mask state");
                    // The dispatcher persists the physical SGPR file and Bool-domain masks in
                    // separate function variables.  At this exact MUST-proven consumer, the u32
                    // pair is only a synthetic placeholder from an earlier scalar lifetime; leave
                    // the generic S_FF1/S_BCNT ambiguity guard intact and expose the proven mask by
                    // removing only those two stale data views.
                    for (int reg = source; reg <= source + 1; ++reg) {
                        state.sreg.erase(reg);
                        state.sreg_input.erase(reg);
                        state.sreg_srt.erase(reg);
                    }
                }
                bool ok = true;
                const bool handled = emit_alu(b, state, in, ok, allow_exec_update, &safe,
                                              allow_smem, rt, /*allow_wave*/false);
                if (handled && ok)
                    record_scalar_write(
                        state, in,
                        allows_compute_scalar_vcc_bridge(b));
                if (!handled || !ok) {
                    // NOT gated on PROSPER_DBG. log_recompile_diagnostic gates its own PRINTING
                    // on that variable and additionally RECORDS the reason for the unconditional
                    // `[compute] skip unsupported program 0x… reason=…` line. An outer gate here
                    // therefore suppressed the recording too, which is why five GTA V programs
                    // reported reason=unrecorded while their cause existed and was formatted.
                        // The raw INSTRUCTION WORDS, because without them this line cannot be acted
                        // on. `fmt` and `op` are our decoder's own labels, so they identify the
                        // instruction only if you already trust the decode -- and a reject is
                        // precisely the case where you should not. The word is ground truth and
                        // names the instruction in one command:
                        //     llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding
                        // assemble the candidate and compare, exactly as #2275 identified the image
                        // atomics and #2309 identified s_cbranch_vccz from `bf860051`.
                        // Its sibling at the ALU reject site already prints these; this one did not,
                        // so half the rejects from a run were unidentifiable and the two diagnostics
                        // could not be compared (#2309).
                        // `mode` as at the ALU reject site (#2412): `unknown-encoding` means no
                        // lowering exists and one must be written; `unresolved-operand` means the
                        // lowering ran and could not resolve an operand or a resource-table
                        // descriptor. Without it a census cannot tell "implement this" from
                        // "this instruction is fine, its descriptor is not".
                        log_recompile_diagnostic(
                            b.diagnostic, "cfg-recompile-reject", "terminal",
                            "mode=%s pc=%u words=%s len=%u fmt=%d op=0x%x",
                            handled ? "unresolved-operand" : "unknown-encoding",
                            in.pc, reject_words_text(in).c_str(), in.len_dwords,
                            static_cast<int>(in.fmt), in.opcode);
                    return false;
                }
                // Ordinary scalar B64 logicals already produced an exact nonzero SCC id above.
                // Only divert the mask-domain form, whose cross-wave SCC is deliberately poisoned.
                if (b64_mask_scc_vote_pcs.contains(in.pc) && !state.scc) {
                    block_b64_mask_scc_vote = &in;
                    break;
                }
                if (!state.scc && native_b32_mask_scc_vote_pcs.contains(in.pc)) {
                    const auto result = state.sreg_bool.find(in.dst.value);
                    if (result == state.sreg_bool.end() ||
                        !state.sreg_bool_b32.contains(in.dst.value))
                        return reject_cfg(in.pc, "missing-b32-mask-scc-source");
                    state.scc = b.native_wave_any(result->second);
                }
            }
            const bool last = member + 1 == dispatch_blocks[dispatch].size();
            if (!last) {
                // Group construction admits only one-successor plain blocks before the tail.
                if (block_mbcnt || block_append || block_synchronized_lds_store ||
                    block_lds_fminmax ||
                    block_swizzle || block_bpermute ||
                    block_dpp_min_row_shr || block_dpp_add_row_shr ||
                    block_dpp_row_ror8 ||
                    block_dpp_add_row_mask || block_mask_ffbh || block_mask_compare ||
                    block_exec_saved_mask_compare || block_saved_mask_pair_compare ||
                    block_vopc_mask_compare ||
                    block_b64_mask_scc_vote ||
                    (block_terminator && (block_terminator->is_end ||
                                          block_terminator->opcode != 0x02)))
                    return reject_cfg(starts[block], "invalid-fused-block");
                continue; // consume an unconditional guest branch without a dispatcher round-trip
            }
            terminator = block_terminator;
            mbcnt = block_mbcnt;
            append = block_append;
            synchronized_lds_store = block_synchronized_lds_store;
            lds_fminmax = block_lds_fminmax;
            swizzle = block_swizzle;
            bpermute = block_bpermute;
            dpp_min_row_shr = block_dpp_min_row_shr;
            dpp_add_row_shr = block_dpp_add_row_shr;
            dpp_row_ror8 = block_dpp_row_ror8;
            dpp_add_row_mask = block_dpp_add_row_mask;
            mask_ffbh = block_mask_ffbh;
            mask_compare = block_mask_compare;
            exec_saved_mask_compare = block_exec_saved_mask_compare;
            saved_mask_pair_compare = block_saved_mask_pair_compare;
            vopc_mask_compare = block_vopc_mask_compare;
            b64_mask_scc_vote = block_b64_mask_scc_vote;
        }
        if (mask_ffbh) {
            const int source = mask_ffbh->src[0].value;
            int mask_base = source;
            auto mask = state.sreg_bool.find(mask_base);
            if (mask == state.sreg_bool.end() || state.sreg_bool_b32.contains(mask_base)) {
                mask_base = source - 1;
                mask = state.sreg_bool.find(mask_base);
            }
            const auto event = portable_mask_ffbh_event_for_pc.find(mask_ffbh->pc);
            if (source < 0 || mask_base < 0 || source - mask_base < 0 ||
                source - mask_base > 1 || mask == state.sreg_bool.end() ||
                state.sreg_bool_b32.contains(mask_base) ||
                event == portable_mask_ffbh_event_for_pc.end())
                return reject_cfg(mask_ffbh->pc, "mask-ffbh-source");
            b.store_function(mask_ffbh_pending_var, yes);
            b.store_function(mask_ffbh_mask_var, mask->second);
            b.store_function(mask_ffbh_write_var, state.exec);
            b.store_function(mask_ffbh_event_var, b.uconst(event->second));
            b.store_function(mask_ffbh_half_var,
                b.uconst(static_cast<uint32_t>(source - mask_base)));
            b.store_function(mask_ffbh_dst_var,
                b.uconst(static_cast<uint32_t>(mask_ffbh->dst.value)));
        }
        if (mbcnt) {
            if (graphics) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=mbcnt-cross-lane", mbcnt->pc);
                return false;
            }
            bool operand_ok = true;
            uint32_t mask = 0;
            if (b.wave_size == 64 && mbcnt->src[0].kind == OperandKind::SGPR) {
                // Dispatcher Bool variables preserve values after a scalar overwrite by storing
                // false, so map membership is not a lifetime proof. Admit a saved SGPR mask only
                // at the exact consumer where the Wave64 MUST analysis proves its pair root live.
                const auto proof = proven_wave64_mbcnt_mask_root_for_pc.find(mbcnt->pc);
                if (proof == proven_wave64_mbcnt_mask_root_for_pc.end())
                    return reject_cfg(mbcnt->pc, "mbcnt-unproven-saved-mask");
                const auto live = state.sreg_bool.find(proof->second);
                if (live == state.sreg_bool.end())
                    return reject_cfg(mbcnt->pc, "mbcnt-proven-mask-missing-state");
                mask = live->second;
            } else {
                mask = mbcnt_source_bit(
                    b, state, mbcnt->src[0], mbcnt->opcode == 0x366);
            }
            const uint32_t acc = operand_bits(b, state, *mbcnt, mbcnt->src[1], &operand_ok);
            const auto event = mbcnt_event_for_pc.find(mbcnt->pc);
            if (!mask || !operand_ok || event == mbcnt_event_for_pc.end()) return false;
            if (b.native_subgroup_size) {
                // The switch selector is scalar within this exact-size subgroup, so every guest
                // lane reaches the same case.  Execute the wave prefix count here and retain masked-
                // off lanes exactly as an RDNA VGPR write does, instead of dynamically selecting a
                // destination across the entire persistent register file in the common phase.
                const uint32_t result = b.native_compute_mbcnt(
                    mask, acc, mbcnt->opcode == 0x365 ? yes : no);
                const int dst = mbcnt->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(mbcnt_pending_var, yes);
                b.store_function(mbcnt_mask_var, mask);
                b.store_function(mbcnt_low_var, mbcnt->opcode == 0x365 ? yes : no);
                // load_state conservatively marks EXEC narrowed. Writing under the current per-lane EXEC
                // is equivalent for a known-full mask and preserves inactive VGPR lanes for divergent code.
                b.store_function(mbcnt_write_var, state.exec);
                b.store_function(mbcnt_event_var, b.uconst(event->second));
                b.store_function(mbcnt_acc_var, acc);
                b.store_function(mbcnt_dst_var, b.uconst(static_cast<uint32_t>(mbcnt->dst.value)));
            }
        }
        if (append) {
            if (graphics) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=gds-cross-lane", append->pc);
                return false;
            }
            const auto m0 = state.sreg.find(124);
            const auto event = append_event_for_pc.find(append->pc);
            if (m0 == state.sreg.end() || event == append_event_for_pc.end()) return false;
            if (!append->ds_gds) b.declare_lds();
            const uint32_t idx = ds_append_consume_index(
                b, m0->second, append->literal, append->ds_gds);
            if (b.native_subgroup_size) {
                const uint32_t result = append->ds_gds
                    ? b.native_gds_append(idx, state.exec, append->opcode == 0x3d)
                    : b.native_wave_append(
                          idx, state.exec, append->opcode == 0x3d ? yes : no);
                const int dst = append->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(append_pending_var, yes);
                b.store_function(append_active_var, state.exec);
                b.store_function(append_event_var, b.uconst(event->second));
                b.store_function(append_consume_var, append->opcode == 0x3d ? yes : no);
                b.store_function(append_gds_var, append->ds_gds ? yes : no);
                b.store_function(append_idx_var, idx);
                b.store_function(append_dst_var,
                    b.uconst(static_cast<uint32_t>(append->dst.value)));
            }
        }
        if (synchronized_lds_store) {
            if (!synchronized_lds_store_pcs.contains(synchronized_lds_store->pc) ||
                synchronized_lds_store->fmt != Rdna2Format::DS ||
                synchronized_lds_store->ds_gds)
                return reject_cfg(synchronized_lds_store->pc,
                                  "lds-store-common-phase-contract");
            b.declare_lds();
            if (!b.lds_var)
                return reject_cfg(synchronized_lds_store->pc, "lds-store-common-phase-lds");
            auto vread = [&](int reg) {
                const auto value = state.vreg.find(reg);
                return value == state.vreg.end() ? zero : value->second;
            };
            std::vector<std::pair<uint32_t, uint32_t>> writes;
            auto append_write = [&](uint32_t idx, int value_reg) {
                writes.emplace_back(idx, vread(value_reg));
            };
            const Rdna2Inst& store = *synchronized_lds_store;
            if (store.opcode == 0xb0) {
                const auto m0 = state.sreg.find(124);
                if (m0 == state.sreg.end())
                    return reject_cfg(store.pc, "lds-store-common-phase-m0");
                const uint32_t base = b.ibin(
                    Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
                const uint32_t tid = b.ibin(
                    Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1u));
                const uint32_t byte_address = b.ibin(
                    Op_IAdd, b.ibin(Op_IAdd, base, b.uconst(store.literal)),
                    b.ibin(Op_ShiftLeftLogical, tid, b.uconst(2)));
                append_write(
                    b.ibin(Op_ShiftRightLogical, byte_address, b.uconst(2)),
                    store.src[1].value);
            } else if (store.opcode == 0x0e || store.opcode == 0x4e) {
                const uint32_t base = b.ibin(
                    Op_ShiftRightLogical, vread(store.src[0].value), b.uconst(2));
                const uint32_t width = store.opcode == 0x4e ? 2u : 1u;
                const uint32_t offset0 = (store.literal & 0xffu) * width;
                const uint32_t offset1 = ((store.literal >> 8u) & 0xffu) * width;
                const uint32_t idx0 = offset0
                    ? b.ibin(Op_IAdd, base, b.uconst(offset0)) : base;
                const uint32_t idx1 = offset1
                    ? b.ibin(Op_IAdd, base, b.uconst(offset1)) : base;
                for (uint32_t word = 0; word < width; ++word)
                    append_write(word ? b.ibin(Op_IAdd, idx0, b.uconst(word)) : idx0,
                                 store.src[1].value + static_cast<int>(word));
                if ((store.literal & 0xffu) != ((store.literal >> 8u) & 0xffu))
                    for (uint32_t word = 0; word < width; ++word)
                        append_write(word ? b.ibin(Op_IAdd, idx1, b.uconst(word)) : idx1,
                                     store.src[2].value + static_cast<int>(word));
            } else if (store.opcode == 0x0d || store.opcode == 0x4d ||
                       store.opcode == 0xde || store.opcode == 0xdf) {
                const uint32_t byte_address = b.ibin(
                    Op_IAdd, vread(store.src[0].value), b.uconst(store.literal));
                const uint32_t base = b.ibin(
                    Op_ShiftRightLogical, byte_address, b.uconst(2));
                const uint32_t width = store.opcode == 0x0d ? 1u :
                    store.opcode == 0x4d ? 2u : store.opcode == 0xde ? 3u : 4u;
                for (uint32_t word = 0; word < width; ++word)
                    append_write(word ? b.ibin(Op_IAdd, base, b.uconst(word)) : base,
                                 store.src[1].value + static_cast<int>(word));
            }
            if (writes.empty() || writes.size() > synchronized_lds_store_idx_vars.size())
                return reject_cfg(store.pc, "lds-store-common-phase-shape");
            b.store_function(synchronized_lds_store_pending_var, yes);
            b.store_function(synchronized_lds_store_active_var, state.exec);
            b.store_function(synchronized_lds_store_count_var,
                             b.uconst(static_cast<uint32_t>(writes.size())));
            for (size_t word = 0; word < writes.size(); ++word) {
                b.store_function(synchronized_lds_store_idx_vars[word], writes[word].first);
                b.store_function(synchronized_lds_store_value_vars[word], writes[word].second);
            }
        }
        if (lds_fminmax) {
            if (!lds_fminmax_pcs.contains(lds_fminmax->pc) || lds_fminmax->ds_gds ||
                (lds_fminmax->words[1] & 0xffff0000u) != 0u ||
                (lds_fminmax->opcode != kDsOpcodeMinF32 &&
                 lds_fminmax->opcode != kDsOpcodeMaxF32))
                return reject_cfg(lds_fminmax->pc, "lds-fminmax-common-phase-contract");
            b.declare_lds();
            if (!b.lds_var) return reject_cfg(lds_fminmax->pc, "lds-fminmax-lds");
            const auto address = state.vreg.find(lds_fminmax->src[0].value);
            const auto value = state.vreg.find(lds_fminmax->src[1].value);
            const uint32_t byte_address = b.ibin(
                Op_IAdd, address == state.vreg.end() ? zero : address->second,
                b.uconst(lds_fminmax->literal));
            b.store_function(lds_fminmax_pending_var, yes);
            b.store_function(lds_fminmax_active_var, state.exec);
            b.store_function(lds_fminmax_min_var,
                lds_fminmax->opcode == kDsOpcodeMinF32 ? yes : no);
            b.store_function(lds_fminmax_idx_var,
                b.ibin(Op_ShiftRightLogical, byte_address, b.uconst(2)));
            b.store_function(lds_fminmax_value_var,
                value == state.vreg.end() ? zero : value->second);
        }
        if (swizzle) {
            if (!swizzle_pcs.contains(swizzle->pc)) return false;
            uint32_t source_lane = 0;
            if (!b.ds_swizzle_source_lane(swizzle->literal, &source_lane)) return false;
            const auto source = state.vreg.find(swizzle->src[0].value);
            b.store_function(swizzle_pending_var, yes);
            b.store_function(swizzle_active_var, state.exec);
            b.store_function(swizzle_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(swizzle_source_lane_var, source_lane);
            b.store_function(swizzle_dst_var,
                b.uconst(static_cast<uint32_t>(swizzle->dst.value)));
        }
        if (bpermute) {
            const auto event = bpermute_event_for_pc.find(bpermute->pc);
            if (event == bpermute_event_for_pc.end() || !b.is_compute ||
                bpermute->ds_gds || !b.native_subgroup_size ||
                b.native_subgroup_size != b.wave_size)
                return reject_cfg(bpermute->pc, "ds-bpermute-native-wave-contract");
            // Publish lane-local operands in the selected case. The actual gathers run after the
            // switch merge, where every subgroup invocation participates in uniform control flow.
            const auto address = state.vreg.find(bpermute->src[0].value);
            const auto source = state.vreg.find(bpermute->src[1].value);
            b.store_function(bpermute_pending_var, yes);
            b.store_function(bpermute_active_var, state.exec);
            b.store_function(bpermute_address_var,
                address == state.vreg.end() ? zero : address->second);
            b.store_function(bpermute_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(bpermute_offset_var, b.uconst(bpermute->literal));
            b.store_function(bpermute_event_var, b.uconst(event->second));
            b.store_function(bpermute_dst_var,
                b.uconst(static_cast<uint32_t>(bpermute->dst.value)));
        }
        if (dpp_min_row_shr) {
            if (!fragment_dpp_min_row_shr(*dpp_min_row_shr))
                return reject_cfg(dpp_min_row_shr->pc, "dpp-min-row-shr-contract");
            const auto event = fragment_dpp_min_event_for_pc.find(dpp_min_row_shr->pc);
            if (event == fragment_dpp_min_event_for_pc.end())
                return reject_cfg(dpp_min_row_shr->pc, "dpp-min-row-shr-event");
            const auto source = state.vreg.find(dpp_min_row_shr->src[0].value);
            b.store_function(dpp_min_pending_var, yes);
            b.store_function(dpp_min_active_var, state.exec);
            b.store_function(dpp_min_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(dpp_min_amount_var,
                b.uconst(static_cast<uint32_t>(dpp_min_row_shr->dpp_ctrl - 0x110u)));
            b.store_function(dpp_min_dst_var,
                b.uconst(static_cast<uint32_t>(dpp_min_row_shr->dst.value)));
            b.store_function(dpp_min_event_var, b.uconst(event->second));
        }
        if (dpp_add_row_shr) {
            if (!compute_dpp_add_row_shr(*dpp_add_row_shr))
                return reject_cfg(dpp_add_row_shr->pc, "dpp-add-row-shr-contract");
            const auto event = compute_dpp_add_event_for_pc.find(dpp_add_row_shr->pc);
            if (event == compute_dpp_add_event_for_pc.end())
                return reject_cfg(dpp_add_row_shr->pc, "dpp-add-row-shr-event");
            const int dst = dpp_add_row_shr->dst.value;
            const auto source = state.vreg.find(dst);
            const uint32_t source_value =
                source == state.vreg.end() ? zero : source->second;
            const uint32_t amount = b.uconst(
                static_cast<uint32_t>(dpp_add_row_shr->dpp_ctrl - 0x110u));
            if (b.native_subgroup_size) {
                // One exact native subgroup is one guest wave, and the scalar dispatcher selector
                // is subgroup-uniform. The source EXEC bit still matters: FI=0/BOUND_CTRL=0
                // disables a destination whose shifted source lane is inactive.
                uint32_t valid_source = 0;
                const uint32_t shifted = b.subgroup_row_shr_dynamic(
                    source_value, state.exec, amount, 0, &valid_source);
                const uint32_t result = b.ibin(Op_IAdd, source_value, shifted);
                state.vreg[dst] = b.sel(
                    b.land(state.exec, valid_source), result, source_value);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(dpp_add_pending_var, yes);
                b.store_function(dpp_add_active_var, state.exec);
                b.store_function(dpp_add_source_var, source_value);
                b.store_function(dpp_add_amount_var, amount);
                b.store_function(dpp_add_dst_var,
                    b.uconst(static_cast<uint32_t>(dst)));
                b.store_function(dpp_add_event_var, b.uconst(event->second));
            }
        }
        if (dpp_row_ror8) {
            const DppRowRor8Op operation = dpp_row_ror8_op(*dpp_row_ror8);
            if (!compute_dpp_row_ror8(*dpp_row_ror8))
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-contract");
            const auto event = compute_dpp_ror8_event_for_pc.find(dpp_row_ror8->pc);
            if (event == compute_dpp_ror8_event_for_pc.end())
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-event");
            const int dst = dpp_row_ror8->dst.value;
            const auto old = state.vreg.find(dst);
            const auto src0 = state.vreg.find(dpp_row_ror8->src[0].value);
            const uint32_t old_value = old == state.vreg.end() ? zero : old->second;
            const uint32_t src0_value = src0 == state.vreg.end() ? zero : src0->second;
            uint32_t src1_value = zero;
            if (dpp_row_ror8->n_src > 1) {
                const auto src1 = state.vreg.find(dpp_row_ror8->src[1].value);
                src1_value = src1 == state.vreg.end() ? zero : src1->second;
            }
            if (b.native_subgroup_size) {
                // One exact native subgroup is one guest wave and this case is subgroup-uniform.
                // FI=0 makes an EXEC-inactive permuted source read as zero. Every stride in this
                // family XORs only bits 0..3, so the source is always an in-range lane of the same
                // 16-lane row and BOUND_CTRL does not decide this case.
                uint32_t valid_source = 0;
                const uint32_t rotated = b.subgroup_row_xor(
                    src0_value, state.exec,
                    [&]{ uint32_t st = 0; dpp_row_xor_ctrl(dpp_row_ror8->dpp_ctrl, &st); return st; }(),
                    &valid_source);
                const uint32_t bounded = b.sel(valid_source, rotated, zero);
                uint32_t result = bounded;
                if (operation == DppRowRor8Op::MinF32)
                    result = b.fext2(Glsl_NMin, bounded, src1_value);
                else if (operation == DppRowRor8Op::MaxF32)
                    result = b.fext2(Glsl_NMax, bounded, src1_value);
                state.vreg[dst] = b.sel(state.exec, result, old_value);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(dpp_ror8_pending_var, yes);
                b.store_function(dpp_ror8_active_var, state.exec);
                b.store_function(dpp_ror8_src0_var, src0_value);
                b.store_function(dpp_ror8_src1_var, src1_value);
                b.store_function(dpp_ror8_op_var,
                    b.uconst(static_cast<uint32_t>(operation)));
                b.store_function(dpp_ror8_dst_var,
                    b.uconst(static_cast<uint32_t>(dst)));
                b.store_function(dpp_ror8_event_var, b.uconst(event->second));
            }
        }
        if (dpp_add_row_mask) {
            if (!compute_dpp_add_row_mask(*dpp_add_row_mask))
                return reject_cfg(dpp_add_row_mask->pc, "dpp-add-row-mask-contract");
            const int dst = dpp_add_row_mask->dst.value;
            const auto old = state.vreg.find(dst);
            const auto addend = state.vreg.find(dpp_add_row_mask->src[1].value);
            const uint32_t old_value = old == state.vreg.end() ? zero : old->second;
            const uint32_t addend_value =
                addend == state.vreg.end() ? zero : addend->second;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1u));
            const uint32_t row = b.ibin(
                Op_ShiftRightLogical, lane, b.uconst(4));
            const uint32_t row_bit = b.ibin(
                Op_ShiftLeftLogical, b.uconst(1), row);
            const uint32_t row_selected = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd, row_bit,
                       b.uconst(dpp_add_row_mask->dpp_row_mask)),
                zero);
            const uint32_t result = b.ibin(Op_IAdd, old_value, addend_value);
            state.vreg[dst] = b.sel(
                b.land(state.exec, row_selected), result, old_value);
            for (auto& vg : state.vgpr_lane_slots)
                if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
            for (auto& vg : state.vgpr_lane_mask_slots)
                if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
        }
        if (mask_compare) {
            if (b.is_vertex) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=wave-mask-compare", mask_compare->pc);
                return false;
            }
            const int source = mask_zero_compare_candidate_source(*mask_compare);
            const uint32_t value = mask_zero_compare_value(state, source);
            if (!mask_zero_compare_is_proven(*mask_compare) || !value)
                return reject_cfg(mask_compare->pc, "missing-mask-compare-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(mask_compare->pc, "mask-vote");
                state.scc = mask_zero_compare_inverts(*mask_compare)
                    ? b.logical_not(wave_any) : wave_any;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var,
                    mask_zero_compare_inverts(*mask_compare) ? yes : no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        if (exec_saved_mask_compare) {
            const int source =
                exec_saved_mask_compare_source(*exec_saved_mask_compare);
            const auto saved_mask = state.sreg_bool.find(source);
            if (saved_mask == state.sreg_bool.end())
                return reject_cfg(exec_saved_mask_compare->pc,
                                  "missing-exec-saved-mask-compare-source");
            const uint32_t mismatch = b.bsel(
                state.exec, b.logical_not(saved_mask->second), saved_mask->second);
            if (b.native_subgroup_size) {
                const uint32_t different = b.native_wave_any(mismatch);
                if (!different)
                    return reject_cfg(exec_saved_mask_compare->pc,
                                      "exec-saved-mask-vote");
                state.scc = exec_saved_mask_compare->opcode == 0x12
                    ? b.logical_not(different) : different;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, mismatch);
                b.store_function(vote_invert_var,
                    exec_saved_mask_compare->opcode == 0x12 ? yes : no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        if (saved_mask_pair_compare) {
            if (b.is_fragment) {
                const auto event =
                    saved_mask_pair_event_for_pc.find(saved_mask_pair_compare->pc);
                if (event == saved_mask_pair_event_for_pc.end())
                    return reject_cfg(saved_mask_pair_compare->pc,
                                      "saved-mask-pair-event");
                b.store_function(saved_mask_pair_pending_var, yes);
                b.store_function(saved_mask_pair_event_var, b.uconst(event->second));
            } else {
                const auto sources =
                    saved_mask_pair_compare_sources(*saved_mask_pair_compare);
                const auto first = state.sreg_bool.find(sources[0]);
                const auto second = state.sreg_bool.find(sources[1]);
                if (first == state.sreg_bool.end() || second == state.sreg_bool.end())
                    return reject_cfg(saved_mask_pair_compare->pc,
                                      "missing-saved-mask-pair-compare-source");
                const uint32_t mismatch = b.bsel(
                    first->second, b.logical_not(second->second), second->second);
                if (b.native_subgroup_size) {
                    const uint32_t different = b.native_wave_any(mismatch);
                    if (!different)
                        return reject_cfg(saved_mask_pair_compare->pc,
                                          "saved-mask-pair-vote");
                    state.scc = saved_mask_pair_compare->opcode == 0x12
                        ? b.logical_not(different) : different;
                } else {
                    b.store_function(vote_pending_var, yes);
                    b.store_function(vote_value_var, mismatch);
                    b.store_function(vote_invert_var,
                        saved_mask_pair_compare->opcode == 0x12 ? yes : no);
                    b.store_function(vote_to_scc_var, yes);
                }
            }
        }
        if (vopc_mask_compare) {
            if (b.is_vertex) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=vopc-wave-mask-compare",
                                         vopc_mask_compare->pc);
                return false;
            }
            const int source = vopc_mask_zero_compare_source(*vopc_mask_compare);
            const auto saved_value = state.sreg_bool.find(source);
            const uint32_t value = source == 106 && state.vcc
                ? state.vcc
                : saved_value != state.sreg_bool.end() ? saved_value->second : 0;
            if (!value)
                return reject_cfg(vopc_mask_compare->pc,
                                  "missing-vopc-mask-compare-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(vopc_mask_compare->pc, "vopc-mask-vote");
                const uint32_t condition =
                    vopc_mask_zero_compare_inverts(*vopc_mask_compare)
                        ? b.logical_not(wave_any) : wave_any;
                state.vcc = b.land(state.exec, condition);
                state.sreg_bool[106] = state.vcc;
                state.sreg_bool_narrowed[106] = true;
                if (const auto saved_vcc = mv.find(106); saved_vcc != mv.end())
                    b.store_function(saved_vcc->second, state.vcc);
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var,
                    vopc_mask_zero_compare_inverts(*vopc_mask_compare) ? yes : no);
                b.store_function(vote_to_vcc_var, yes);
            }
        }
        if (b64_mask_scc_vote) {
            uint32_t value = 0;
            if (b64_mask_scc_vote->dst.value == 126 ||
                b64_mask_scc_vote->dst.value == 127) {
                value = state.exec;
            } else if (b64_mask_scc_vote->dst.value == 106 ||
                       b64_mask_scc_vote->dst.value == 107) {
                value = state.vcc;
            } else {
                const auto saved = state.sreg_bool.find(b64_mask_scc_vote->dst.value);
                if (saved != state.sreg_bool.end()) value = saved->second;
            }
            if (!value)
                return reject_cfg(b64_mask_scc_vote->pc, "missing-b64-mask-scc-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(b64_mask_scc_vote->pc, "b64-mask-scc-vote");
                state.scc = wave_any;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var, no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        save_state(state, dispatch);
        if (mask_ffbh) {
            if (!set_next(mask_ffbh->pc + mask_ffbh->len_dwords))
                return reject_cfg(mask_ffbh->pc, "mask-ffbh-successor");
        } else if (mbcnt) {
            if (!set_next(mbcnt->pc + mbcnt->len_dwords))
                return reject_cfg(mbcnt->pc, "mbcnt-successor");
        } else if (append) {
            if (!set_next(append->pc + append->len_dwords))
                return reject_cfg(append->pc, "append-successor");
        } else if (synchronized_lds_store) {
            if (!set_next(synchronized_lds_store->pc +
                          synchronized_lds_store->len_dwords))
                return reject_cfg(synchronized_lds_store->pc,
                                  "lds-store-successor");
        } else if (lds_fminmax) {
            if (!set_next(lds_fminmax->pc + lds_fminmax->len_dwords))
                return reject_cfg(lds_fminmax->pc, "lds-fminmax-successor");
        } else if (swizzle) {
            if (!set_next(swizzle->pc + swizzle->len_dwords))
                return reject_cfg(swizzle->pc, "swizzle-successor");
        } else if (bpermute) {
            if (!set_next(bpermute->pc + bpermute->len_dwords))
                return reject_cfg(bpermute->pc, "bpermute-successor");
        } else if (dpp_min_row_shr) {
            if (!set_next(dpp_min_row_shr->pc + dpp_min_row_shr->len_dwords))
                return reject_cfg(dpp_min_row_shr->pc,
                                  "dpp-min-row-shr-successor");
        } else if (dpp_add_row_shr) {
            if (!set_next(dpp_add_row_shr->pc + dpp_add_row_shr->len_dwords))
                return reject_cfg(dpp_add_row_shr->pc,
                                  "dpp-add-row-shr-successor");
        } else if (dpp_row_ror8) {
            if (!set_next(dpp_row_ror8->pc + dpp_row_ror8->len_dwords))
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-successor");
        } else if (dpp_add_row_mask) {
            if (!set_next(dpp_add_row_mask->pc + dpp_add_row_mask->len_dwords))
                return reject_cfg(dpp_add_row_mask->pc,
                                  "dpp-add-row-mask-successor");
        } else if (mask_compare) {
            if (!set_next(mask_compare->pc + mask_compare->len_dwords))
                return reject_cfg(mask_compare->pc, "mask-compare-successor");
        } else if (exec_saved_mask_compare) {
            if (!set_next(exec_saved_mask_compare->pc +
                          exec_saved_mask_compare->len_dwords))
                return reject_cfg(exec_saved_mask_compare->pc,
                                  "exec-saved-mask-compare-successor");
        } else if (saved_mask_pair_compare) {
            if (!set_next(saved_mask_pair_compare->pc +
                          saved_mask_pair_compare->len_dwords))
                return reject_cfg(saved_mask_pair_compare->pc,
                                  "saved-mask-pair-compare-successor");
        } else if (vopc_mask_compare) {
            if (!set_next(vopc_mask_compare->pc + vopc_mask_compare->len_dwords))
                return reject_cfg(vopc_mask_compare->pc,
                                  "vopc-mask-compare-successor");
        } else if (b64_mask_scc_vote) {
            if (!set_next(b64_mask_scc_vote->pc + b64_mask_scc_vote->len_dwords))
                return reject_cfg(b64_mask_scc_vote->pc,
                                  "b64-mask-scc-successor");
        } else if (!terminator) {
            if (!set_next(final_hi)) return reject_cfg(starts[final_block], "fallthrough-successor");
        } else if (terminator->is_end || terminator->opcode == 0x12) {
            // Vulkan has no compute-stage trap instruction. Ending this dispatcher invocation is
            // the closest fail-closed model of s_trap when no guest trap handler is emulated; the
            // overwhelmingly common release-shader shape branches around the trap on valid data.
            b.store_function(active_var, no);
        } else if (terminator->opcode == 0x02) {
            if (!set_next(branch_target(*terminator)))
                return reject_cfg(terminator->pc, "branch-successor");
        } else {
            uint32_t condition = 0;
            switch (terminator->opcode) {
                // state.scc == 0 marks SCC poisoned by a 64-bit mask op inside this block (its
                // hardware SCC is a cross-lane reduction) — reject rather than branch on a stale
                // value an older s_cmp produced.
                case 0x04: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = b.logical_not(state.scc); break; // s_cbranch_scc0
                case 0x05: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = state.scc; break;
                case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: return reject_cfg(terminator->pc, "branch-opcode");
            }
            const uint32_t target = branch_target(*terminator);
            const uint32_t fallthrough = terminator->pc + terminator->len_dwords;
            auto taken = block_for_pc.find(target), next = block_for_pc.find(fallthrough);
            const bool taken_exit = target > end_pc && proven_exit_target(target);
            const bool next_exit = fallthrough > end_pc && proven_exit_target(fallthrough);
            if ((!taken_exit && taken == block_for_pc.end()) ||
                (!next_exit && next == block_for_pc.end()))
                return reject_cfg(terminator->pc, "branch-target");
            if ((taken_exit || next_exit) && b.is_compute && !b.native_subgroup_size)
                return reject_cfg(terminator->pc, "portable-compute-exit");
            const uint32_t taken_dispatch = taken_exit ? 0 : dispatch_for_block[taken->second];
            const uint32_t next_dispatch = next_exit ? 0 : dispatch_for_block[next->second];
            auto route = [&](uint32_t branch_condition) {
                if (taken_exit || next_exit) {
                    const uint32_t remains_active = taken_exit
                        ? b.logical_not(branch_condition) : branch_condition;
                    b.store_function(active_var, remains_active);
                    b.store_function(pc_var, b.uconst(
                        taken_exit ? next_dispatch : taken_dispatch));
                } else {
                    b.store_function(pc_var,
                        b.sel(branch_condition, b.uconst(taken_dispatch),
                              b.uconst(next_dispatch)));
                }
            };
            if (terminator->opcode == 0x04 || terminator->opcode == 0x05) {
                route(condition);
            } else if (graphics) {
                const uint32_t lane_condition =
                    terminator->opcode <= 0x07 ? state.vcc : state.exec;
                const uint32_t branch_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(lane_condition) : lane_condition;
                route(branch_condition);
            } else if (b.native_subgroup_size) {
                const uint32_t wave_any = b.native_wave_any(
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                const uint32_t wave_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(wave_any) : wave_any;
                route(wave_condition);
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var,
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                b.store_function(vote_invert_var,
                    terminator->opcode == 0x06 || terminator->opcode == 0x08 ? yes : no);
                b.store_function(vote_taken_var, b.uconst(taken_dispatch));
                b.store_function(vote_next_var, b.uconst(next_dispatch));
            }
        }
        b.emit_branch(switch_merge);
    }
    b.emit_label(fallback);
    b.store_function(active_var, no);
    b.emit_branch(switch_merge);
    b.emit_label(switch_merge);
    b.emit_branch(loop_continue);
    b.emit_label(loop_continue);

    // DS_SWIZZLE common phase. The dispatcher cases only publish source data and a lane selector;
    // every invocation executes the actual subgroup gathers here in uniform control flow. This is
    // required even though the instruction does not touch LDS: subgroup operations in a divergent
    // switch arm would have undefined participation on Vulkan.
    if (!swizzle_pcs.empty()) {
        const uint32_t swizzle_pending = b.load_function(b.t_bool, swizzle_pending_var);
        const uint32_t swizzle_active = b.load_function(b.t_bool, swizzle_active_var);
        const uint32_t swizzle_lane = b.load_function(b.t_u32, swizzle_source_lane_var);
        const uint32_t swizzle_value = b.subgroup_shuffle(
            b.load_function(b.t_u32, swizzle_source_var), swizzle_lane);
        const uint32_t source_active_word = b.sel(
            b.land(swizzle_pending, swizzle_active), b.uconst(1), zero);
        const uint32_t source_active = b.ucmp(
            Op_INotEqual, b.subgroup_shuffle(source_active_word, swizzle_lane), zero);
        const uint32_t swizzle_result = b.sel(source_active, swizzle_value, zero);
        const uint32_t swizzle_dst = b.load_function(b.t_u32, swizzle_dst_var);
        const uint32_t swizzle_write = b.land(swizzle_pending, swizzle_active);
        for (const auto& kv : vv) {
            const uint32_t selected = b.land(
                swizzle_write,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, swizzle_result, old));
        }
        // An ordinary VGPR definition ends a scalar lane-spill lifetime at that physical register.
        for (const auto& kv : lv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // DS_BPERMUTE common phase. Even the exact native dispatcher publishes operands in its switch
    // case and performs subgroup gathers here: keeping all lanes at one structurally uniform merge
    // avoids implementation-dependent participation in a case arm. Static event tags accompany
    // DATA0 and EXEC so adjacent BPERMUTE sites cannot consume one another's mailbox values.
    if (has_bpermute) {
        const uint32_t pending = b.load_function(b.t_bool, bpermute_pending_var);
        const uint32_t active = b.load_function(b.t_bool, bpermute_active_var);
        const uint32_t event = b.load_function(b.t_u32, bpermute_event_var);
        const uint32_t result = b.ds_bpermute_b32(
            b.load_function(b.t_u32, bpermute_address_var),
            b.load_function(b.t_u32, bpermute_source_var),
            b.land(pending, active),
            b.load_function(b.t_u32, bpermute_offset_var), event);
        const uint32_t dst = b.load_function(b.t_u32, bpermute_dst_var);
        const uint32_t write = b.land(pending, active);
        for (const auto& kv : vv) {
            const uint32_t selected = b.land(
                write, b.ucmp(Op_IEqual, dst,
                              b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, result, old));
        }
        // A physical VGPR definition ends any scalar lane-spill lifetime even when EXEC suppresses
        // this lane's data write, matching predicate_write and the DS_SWIZZLE phase above.
        for (const auto& kv : lv) {
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // Fragment saved-mask pair comparisons execute their subgroup votes here, outside the
    // lane-divergent dispatcher switch.  Every invocation evaluates every static event from the
    // persistent mask register file, so a mismatch bit remains visible even when that lane is
    // currently parked at another guest PC.  The publishing lane's event tag selects the exact
    // operand pair and EQ/LG polarity whose result updates its SCC.
    if (has_saved_mask_pair_events) {
        const uint32_t pending =
            b.load_function(b.t_bool, saved_mask_pair_pending_var);
        const uint32_t event =
            b.load_function(b.t_u32, saved_mask_pair_event_var);
        for (size_t index = 0; index < saved_mask_pair_events.size(); ++index) {
            const auto& compare = saved_mask_pair_events[index];
            const auto first_var = mv.find(compare.first);
            const auto second_var = mv.find(compare.second);
            if (first_var == mv.end() || second_var == mv.end())
                return reject_cfg(0, "missing-saved-mask-pair-common-source");
            const uint32_t first = b.load_function(b.t_bool, first_var->second);
            const uint32_t second = b.load_function(b.t_bool, second_var->second);
            const uint32_t mismatch = b.bsel(
                first, b.logical_not(second), second);
            const uint32_t different = b.fragment_wave_any(mismatch);
            if (!different) return reject_cfg(0, "saved-mask-pair-common-vote");
            const uint32_t result = compare.opcode == 0x12
                ? b.logical_not(different) : different;
            const uint32_t selected = b.land(
                pending,
                b.ucmp(Op_IEqual, event,
                       b.uconst(static_cast<uint32_t>(index + 1))));
            const uint32_t old = b.load_function(b.t_bool, scc_var);
            b.store_function(scc_var, b.bsel(selected, result, old));
        }
    }

    // Fragment DPP V_MIN_U32 common phase. A graphics dispatcher's PC is lane-local, so a source
    // lane is eligible only when it published this same static DPP event and its EXEC bit is active.
    // The source's event tag is shuffled beside its value and compared with the destination tag. The
    // unbounded ROW_SHR contract disables out-of-row/inactive-source lanes and preserves VDST;
    // active destination lanes reduce the shifted neighbor with their unchanged local value.
    if (!fragment_dpp_min_row_shr_pcs.empty()) {
        const uint32_t pending = b.load_function(b.t_bool, dpp_min_pending_var);
        const uint32_t active = b.load_function(b.t_bool, dpp_min_active_var);
        const uint32_t source = b.load_function(b.t_u32, dpp_min_source_var);
        uint32_t valid_source = 0;
        const uint32_t shifted = b.subgroup_row_shr_dynamic(
            source, b.land(pending, active),
            b.load_function(b.t_u32, dpp_min_amount_var),
            b.load_function(b.t_u32, dpp_min_event_var), &valid_source);
        const uint32_t result = b.uext2(Glsl_UMin, shifted, source);
        const uint32_t write = b.land(b.land(pending, active), valid_source);
        const uint32_t dst = b.load_function(b.t_u32, dpp_min_dst_var);
        for (int reg : fragment_dpp_min_row_shr_dsts) {
            const auto kv = vv.find(reg);
            if (kv == vv.end()) return reject_cfg(0, "missing-dpp-min-row-shr-dst");
            const uint32_t selected = b.land(
                write, b.ucmp(Op_IEqual, dst,
                              b.uconst(static_cast<uint32_t>(reg))));
            const uint32_t old = b.load_function(b.t_u32, kv->second);
            b.store_function(kv->second, b.sel(selected, result, old));
        }
        // A physical VGPR definition ends scalar lane-spill lifetimes even when EXEC suppresses
        // this lane's data write, matching predicate_write and the DS_SWIZZLE common phase above.
        for (const auto& kv : lv) {
            if (!fragment_dpp_min_row_shr_dsts.contains(kv.first.first)) continue;
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            if (!fragment_dpp_min_row_shr_dsts.contains(kv.first.first)) continue;
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // Apply the diagnostic trip bound to whichever back-edge this dispatcher actually emits.
    //
    // This used to live only in the portable branch below, while the arm was announced before the
    // split — so a DIRECT dispatcher printed "bounded" and then emitted an unbounded loop. An
    // announced-but-inert lever is worse than no lever: it invites exactly the reading that the
    // instrument was applied, which is how a null result gets published as evidence.
    auto apply_trip_bound = [&](uint32_t keep_going) {
        if (!trip_var) return keep_going;
        const uint32_t next_trip =
            b.ibin(Op_IAdd, b.load_function(b.t_u32, trip_var), b.uconst(1));
        b.store_function(trip_var, next_trip);
        // Updated on EVERY back-edge traversal, not only on the hit, so the extremes describe the
        // whole run rather than its final instant.
        const uint32_t dispatch_now = b.load_function(b.t_u32, pc_var);
        const uint32_t old_min = b.load_function(b.t_u32, dispatch_min_var);
        b.store_function(dispatch_min_var,
                         b.sel(b.ucmp(Op_ULessThan, dispatch_now, old_min), dispatch_now, old_min));
        const uint32_t old_max = b.load_function(b.t_u32, dispatch_max_var);
        b.store_function(dispatch_max_var,
                         b.sel(b.ucmp(Op_UGreaterThan, dispatch_now, old_max), dispatch_now,
                               old_max));
        const uint32_t under_bound = b.ucmp(Op_ULessThan, next_trip, b.uconst(cfg_trip_bound));
        // WITNESS. A cap that is armed but never reached proves nothing, so record the hit where the
        // host can read it: the internal GDS buffer's top five dwords (kComputeTripWitnessDword),
        // which the host prepares before this dispatch and only touches while armed.
        // Predicated on "still nominally running AND the bound just ran out", so a loop that exits
        // normally writes nothing and the ABSENCE of a record is itself an answer.
        const uint32_t hit = b.land(keep_going, b.logical_not(under_bound));
        // EVERY field is published with a device-scope atomic, including the two that are
        // invocation-invariant. Concurrent non-atomic stores of the same value are still a data race
        // by the memory model, and "they happen to agree" is not a publication protocol -- it is the
        // same reasoning that made the last-writer range look coherent. Max is idempotent for a flag
        // and for a compile-time constant, so this costs nothing and removes the exception.
        b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 0),
                                    b.uconst(1), hit);
        b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 1),
                                    b.uconst(cfg_phase), hit);
        // Fields 2..4 are per-invocation and must be REDUCED, not overwritten. The deleted field
        // here was the dispatcher ordinal at the instant one invocation hit the cap: a single
        // sample, unusable for the question ("is it cycling?"), and the field whose label was
        // published wrongly twice. The span it belonged to is what actually answers that, so only
        // the span survives -- and as true extremes over every invocation rather than whichever
        // wrote last.
        b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 2),
                                    next_trip, hit);
        b.compute_gds_atomic_minmax(Op_AtomicUMin, b.uconst(kComputeTripWitnessDword + 3),
                                    b.load_function(b.t_u32, dispatch_min_var), hit);
        b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 4),
                                    b.load_function(b.t_u32, dispatch_max_var), hit);
        return b.land(keep_going, under_bound);
    };
    if (direct_dispatch) {
        // Native wave operations and votes were already resolved in the selected case.  PC and
        // ACTIVE are scalar guest-wave state, so every invocation in this exact-size subgroup has
        // the same value; another subgroup in the workgroup may still leave independently because
        // this path contains no workgroup barriers.
        b.emit_condbranch(apply_trip_bound(b.load_function(b.t_bool, active_var)),
                          loop_header, loop_merge);
    } else {
    // Atomicized LDS-store common phase. Each guest DS_WRITE packet is one dispatcher event. The
    // trailing barrier completes every lane's complete packet before another guest store packet can
    // begin, preserving RDNA wave instruction order even when two packets' address footprints alias.
    if (has_synchronized_lds_store_event) {
        b.barrier();
        const uint32_t pending_and_active = b.land(
            b.load_function(b.t_bool, synchronized_lds_store_pending_var),
            b.load_function(b.t_bool, synchronized_lds_store_active_var));
        const uint32_t count = b.load_function(
            b.t_u32, synchronized_lds_store_count_var);
        for (uint32_t word = 0; word < synchronized_lds_store_idx_vars.size(); ++word) {
            const uint32_t word_block = b.id(), word_merge = b.id();
            const uint32_t perform_word = b.land(
                pending_and_active,
                b.ucmp(Op_UGreaterThan, count, b.uconst(word)));
            b.emit_selmerge(word_merge);
            b.emit_condbranch(perform_word, word_block, word_merge);
            b.emit_label(word_block);
            b.lds_atomic(
                Op_AtomicExchange,
                b.load_function(b.t_u32, synchronized_lds_store_idx_vars[word]),
                b.load_function(b.t_u32, synchronized_lds_store_value_vars[word]),
                false, yes);
            b.emit_branch(word_merge);
            b.emit_label(word_merge);
            // Wide/write2 DS instructions have ordered component effects. Keep each component's
            // exchange globally complete before the following one can begin; the barrier remains
            // outside the conditional so every workgroup invocation participates.
            b.barrier();
        }
    }

    // DS_MIN/MAX_F32 common phase. Dispatcher cases publish one lane-local atomic request without
    // touching LDS. Every invocation, including lanes whose guest wave trapped or ended, reaches the
    // publication barrier here. The trailing barrier completes this event before the next dispatcher
    // iteration can execute another atomic or the lane-zero gather. Admission requires a single guest
    // wave whenever ordinary initializer stores precede the atomics, so this does not invent ownership
    // between independently executing guest waves.
    if (has_lds_fminmax_event) {
        b.barrier();
        const uint32_t perform = b.land(
            b.load_function(b.t_bool, lds_fminmax_pending_var),
            b.load_function(b.t_bool, lds_fminmax_active_var));
        const uint32_t atomic_block = b.id(), atomic_merge = b.id();
        b.emit_selmerge(atomic_merge);
        b.emit_condbranch(perform, atomic_block, atomic_merge);
        b.emit_label(atomic_block);
        const uint32_t min_block = b.id(), max_block = b.id(), operation_merge = b.id();
        const uint32_t is_min = b.load_function(b.t_bool, lds_fminmax_min_var);
        b.emit_selmerge(operation_merge);
        b.emit_condbranch(is_min, min_block, max_block);
        b.emit_label(min_block);
        b.lds_atomic_fminmax(
            b.load_function(b.t_u32, lds_fminmax_idx_var),
            b.load_function(b.t_u32, lds_fminmax_value_var), true, false, yes);
        b.emit_branch(operation_merge);
        b.emit_label(max_block);
        b.lds_atomic_fminmax(
            b.load_function(b.t_u32, lds_fminmax_idx_var),
            b.load_function(b.t_u32, lds_fminmax_value_var), false, false, yes);
        b.emit_branch(operation_merge);
        b.emit_label(operation_merge);
        b.emit_branch(atomic_merge);
        b.emit_label(atomic_merge);
        b.barrier();
    }

    // Portable compute DPP V_ADD_NC_U32 common phase. Host subgroup shuffles cannot model a
    // Wave64 guest on a subgroup32 device, and different guest waves can reach different static
    // DPP sites in one dispatcher iteration. Publish a full source value plus an event/EXEC word
    // for every workgroup invocation, then address the shifted lane directly inside the same guest
    // 16-lane row. Two barriers bracket the scratch lifetime so later MBCNT/vote phases can reuse
    // the value plane without observing a peer's previous dispatcher iteration.
    if (has_portable_compute_dpp_add) {
    const uint32_t dpp_pending = b.load_function(b.t_bool, dpp_add_pending_var);
    const uint32_t dpp_active = b.load_function(b.t_bool, dpp_add_active_var);
    const uint32_t dpp_source = b.load_function(b.t_u32, dpp_add_source_var);
    const uint32_t dpp_event = b.load_function(b.t_u32, dpp_add_event_var);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), b.linear_localid), dpp_source);
    const uint32_t dpp_metadata = b.sel(
        dpp_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
               b.sel(dpp_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), b.linear_localid),
        dpp_metadata);
    b.barrier();

    const uint32_t dpp_amount = b.load_function(b.t_u32, dpp_add_amount_var);
    const uint32_t dpp_row_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(15));
    const uint32_t dpp_in_bounds = b.ucmp(
        Op_UGreaterThanEqual, dpp_row_lane, dpp_amount);
    // Keep even the disabled lane's scratch address valid. BOUND_CTRL=0 uses the validity gate
    // below to preserve old VDST rather than consuming this self-addressed placeholder.
    const uint32_t dpp_source_index = b.sel(
        dpp_in_bounds,
        b.ibin(Op_ISub, b.linear_localid, dpp_amount),
        b.linear_localid);
    const uint32_t dpp_shifted = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), dpp_source_index));
    const uint32_t dpp_source_metadata = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), dpp_source_index));
    const uint32_t dpp_source_event = b.ibin(
        Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active = b.ucmp(
        Op_INotEqual,
        b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    uint32_t dpp_valid_source = b.land(
        dpp_in_bounds, dpp_source_active);
    dpp_valid_source = b.land(
        dpp_valid_source, b.ucmp(Op_IEqual, dpp_source_event, dpp_event));
    const uint32_t dpp_result = b.ibin(Op_IAdd, dpp_source, dpp_shifted);
    const uint32_t dpp_write = b.land(
        b.land(dpp_pending, dpp_active), dpp_valid_source);
    const uint32_t dpp_dst = b.load_function(b.t_u32, dpp_add_dst_var);
    for (int reg : compute_dpp_add_row_shr_dsts) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return reject_cfg(0, "missing-dpp-add-row-shr-dst");
        const uint32_t selected = b.land(
            dpp_write, b.ucmp(Op_IEqual, dpp_dst,
                              b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // The physical VGPR definition invalidates scalar lane-spill aliases even when EXEC or the
    // shifted source suppresses this invocation's data write.
    for (const auto& kv : lv) {
        if (!compute_dpp_add_row_shr_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!compute_dpp_add_row_shr_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // Portable compute DPP ROW_ROR:8 common phase. This is deliberately separate from the ROW_SHR
    // add phase above: each phase publishes its own pending state, consumes it between two workgroup
    // barriers, and only then permits the other operation to reuse the scratch planes. Event IDs
    // distinguish static sites; the operation tag distinguishes MOV/MIN/MAX semantics per invocation.
    if (has_portable_compute_dpp_ror8) {
    const uint32_t dpp_pending = b.load_function(b.t_bool, dpp_ror8_pending_var);
    const uint32_t dpp_active = b.load_function(b.t_bool, dpp_ror8_active_var);
    const uint32_t dpp_src0 = b.load_function(b.t_u32, dpp_ror8_src0_var);
    const uint32_t dpp_src1 = b.load_function(b.t_u32, dpp_ror8_src1_var);
    const uint32_t dpp_operation = b.load_function(b.t_u32, dpp_ror8_op_var);
    const uint32_t dpp_event = b.load_function(b.t_u32, dpp_ror8_event_var);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), b.linear_localid), dpp_src0);
    const uint32_t dpp_metadata = b.sel(
        dpp_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
               b.sel(dpp_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), b.linear_localid),
        dpp_metadata);
    b.barrier();

    // XOR 8 exchanges the two eight-lane halves without crossing an architectural DPP16 row.
    // This uses guest linear-local order, not the implementation-defined Vulkan subgroup lane ID.
    const uint32_t dpp_rotated_index = b.ibin(
        Op_BitwiseXor, b.linear_localid, b.uconst(8));
    const uint32_t dpp_source_in_bounds = b.ucmp(
        Op_ULessThan, dpp_rotated_index, b.uconst(b.local_count));
    // A partial final DPP16 row has no invocation to initialize the rotated slot. Address this
    // lane's initialized placeholder and let FI=0's validity gate supply zero for the missing peer.
    const uint32_t dpp_source_index = b.sel(
        dpp_source_in_bounds, dpp_rotated_index, b.linear_localid);
    const uint32_t dpp_rotated = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), dpp_source_index));
    const uint32_t dpp_source_metadata = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), dpp_source_index));
    const uint32_t dpp_source_event = b.ibin(
        Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active = b.ucmp(
        Op_INotEqual,
        b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    const uint32_t dpp_valid_source = b.land(
        dpp_source_in_bounds,
        b.land(dpp_source_active,
               b.ucmp(Op_IEqual, dpp_source_event, dpp_event)));
    // FI=0 requires an EXEC-active source. BOUND_CTRL=1 supplies zero when that source is invalid,
    // but the active destination still writes the operation's result. MOV uses the bounded source;
    // MIN/MAX combine it with the destination lane's unpermuted SRC1.
    const uint32_t dpp_bounded = b.sel(dpp_valid_source, dpp_rotated, zero);
    uint32_t dpp_result = dpp_bounded;
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation,
               b.uconst(static_cast<uint32_t>(DppRowRor8Op::MinF32))),
        b.fext2(Glsl_NMin, dpp_bounded, dpp_src1), dpp_result);
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation,
               b.uconst(static_cast<uint32_t>(DppRowRor8Op::MaxF32))),
        b.fext2(Glsl_NMax, dpp_bounded, dpp_src1), dpp_result);
    const uint32_t dpp_write = b.land(dpp_pending, dpp_active);
    const uint32_t dpp_dst = b.load_function(b.t_u32, dpp_ror8_dst_var);
    for (int reg : compute_dpp_row_ror8_dsts) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return reject_cfg(0, "missing-dpp-row-ror8-dst");
        const uint32_t selected = b.land(
            dpp_write, b.ucmp(Op_IEqual, dpp_dst,
                              b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // A physical destination definition invalidates scalar lane aliases even when EXEC suppresses
    // this invocation's data write, matching predicate_write and the existing DPP add phase.
    for (const auto& kv : lv) {
        if (!compute_dpp_row_ror8_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!compute_dpp_row_ror8_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // MBCNT common phase. Cases publish an event-tagged mask bit and accumulator, but never emit a
    // barrier themselves. Every invocation—including ended waves and lanes currently at a different
    // guest block—therefore executes these two barriers in identical structured control flow. Event
    // tags keep contributions from different static MBCNT sites isolated if malformed/nonuniform
    // scalar state ever lets waves reach different sites during the same dispatcher iteration.
    const uint32_t mbcnt_wave_index = b.ibin(
        Op_ShiftRightLogical, b.linear_localid,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t mbcnt_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1));

    if (has_portable_mask_ffbh) {
    // Portable Wave64 saved-mask FFBH phase. Each publishing lane contributes its one predicate bit
    // with a static-event tag. Lane zero assembles the selected architectural 32-bit half in LDS,
    // after which every lane applies the ordinary V_FFBH_U32 semantics and predicates the VGPR
    // write by its own EXEC. This deliberately does not use a host subgroup ballot: the portable
    // route has no exact-width contract, and a narrower ballot would silently lose guest lanes.
    const uint32_t mask_ffbh_pending =
        b.load_function(b.t_bool, mask_ffbh_pending_var);
    const uint32_t mask_ffbh_mask = b.load_function(b.t_bool, mask_ffbh_mask_var);
    const uint32_t mask_ffbh_tag = b.load_function(b.t_u32, mask_ffbh_event_var);
    const uint32_t mask_ffbh_encoded = b.sel(
        mask_ffbh_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, mask_ffbh_tag, b.uconst(1)),
               b.sel(mask_ffbh_mask, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, mask_ffbh_encoded);
    b.barrier();

    const uint32_t mask_ffbh_leader = b.id(), mask_ffbh_assembled = b.id();
    const uint32_t mask_ffbh_is_leader = b.land(
        mask_ffbh_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    b.emit_selmerge(mask_ffbh_assembled);
    b.emit_condbranch(mask_ffbh_is_leader, mask_ffbh_leader, mask_ffbh_assembled);
    b.emit_label(mask_ffbh_leader);
    const uint32_t mask_ffbh_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index, b.uconst(6));
    const uint32_t mask_ffbh_half = b.load_function(b.t_u32, mask_ffbh_half_var);
    uint32_t mask_ffbh_word = zero;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        const uint32_t candidate_lane = b.ibin(
            Op_IAdd, b.uconst(bit),
            b.ibin(Op_ShiftLeftLogical, mask_ffbh_half, b.uconst(5)));
        const uint32_t candidate_index = b.ibin(
            Op_IAdd, mask_ffbh_wave_base, candidate_lane);
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(Op_IEqual, candidate_tag, mask_ffbh_tag);
        include = b.land(
            include, b.ucmp(Op_ULessThan, candidate_index, b.uconst(b.local_count)));
        const uint32_t candidate_bit = b.ibin(
            Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t positioned = b.ibin(
            Op_ShiftLeftLogical, candidate_bit, b.uconst(bit));
        mask_ffbh_word = b.ibin(
            Op_BitwiseOr, mask_ffbh_word, b.sel(include, positioned, zero));
    }
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index),
        mask_ffbh_word);
    b.emit_branch(mask_ffbh_assembled);
    b.emit_label(mask_ffbh_assembled);
    b.barrier();

    const uint32_t mask_ffbh_result = b.ffbh_u32(b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index)));
    const uint32_t mask_ffbh_dst = b.load_function(b.t_u32, mask_ffbh_dst_var);
    const uint32_t mask_ffbh_write = b.land(
        mask_ffbh_pending, b.load_function(b.t_bool, mask_ffbh_write_var));
    for (int reg : portable_mask_ffbh_dsts) {
        const auto destination = vv.find(reg);
        if (destination == vv.end()) return reject_cfg(0, "missing-mask-ffbh-dst");
        const uint32_t selected = b.land(
            mask_ffbh_write,
            b.ucmp(Op_IEqual, mask_ffbh_dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, destination->second);
        b.store_function(destination->second, b.sel(selected, mask_ffbh_result, old));
    }
    // As for every ordinary VALU destination, the physical write ends scalar-spill aliases even
    // where EXEC suppresses this lane's data update.
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            mask_ffbh_pending,
            b.ucmp(Op_IEqual, mask_ffbh_dst,
                   b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            mask_ffbh_pending,
            b.ucmp(Op_IEqual, mask_ffbh_dst,
                   b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    if (!mbcnt_event_for_pc.empty()) {
    const uint32_t mbcnt_pending = b.load_function(b.t_bool, mbcnt_pending_var);
    const uint32_t mbcnt_mask = b.load_function(b.t_bool, mbcnt_mask_var);
    const uint32_t mbcnt_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, mbcnt_event_var), b.uconst(1));
    const uint32_t mbcnt_encoded = b.sel(
        mbcnt_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, mbcnt_tag, b.uconst(1)),
               b.sel(mbcnt_mask, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, mbcnt_encoded);
    b.barrier();

    const uint32_t mbcnt_low = b.load_function(b.t_bool, mbcnt_low_var);
    b.store_function(mbcnt_sum_var, zero);
    const uint32_t mbcnt_scan = b.id(), mbcnt_scanned = b.id();
    b.emit_selmerge(mbcnt_scanned);
    b.emit_condbranch(mbcnt_pending, mbcnt_scan, mbcnt_scanned);
    b.emit_label(mbcnt_scan);
    uint32_t mbcnt_sum = zero;
    const uint32_t mbcnt_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t half_size = std::min(32u, b.wave_size);
    for (uint32_t i = 0; i < half_size; ++i) {
        const uint32_t candidate_lane = b.sel(
            mbcnt_low, b.uconst(i), b.uconst(i + (b.wave_size == 64 ? 32u : 0u)));
        const uint32_t candidate = b.cfg_scratch_load(
            b.ibin(Op_IAdd, mbcnt_wave_base, candidate_lane));
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        const uint32_t candidate_bit = b.ibin(Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t below = b.ucmp(
            Op_ULessThan, candidate_lane, mbcnt_lane);
        uint32_t include = below;
        if (b.wave_size == 32) include = b.land(include, mbcnt_low);
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, mbcnt_tag));
        mbcnt_sum = b.b_iadd(mbcnt_sum, b.sel(include, candidate_bit, zero));
    }
    b.store_function(mbcnt_sum_var, mbcnt_sum);
    b.emit_branch(mbcnt_scanned);
    b.emit_label(mbcnt_scanned);
    const uint32_t mbcnt_result = b.b_iadd(
        b.load_function(b.t_u32, mbcnt_acc_var),
        b.load_function(b.t_u32, mbcnt_sum_var));
    const uint32_t mbcnt_dst = b.load_function(b.t_u32, mbcnt_dst_var);
    const uint32_t mbcnt_write = b.land(
        mbcnt_pending, b.load_function(b.t_bool, mbcnt_write_var));
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            mbcnt_write, b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, mbcnt_result, old));
    }
    // An ordinary VGPR write ends any scalar-spill lifetime even for lanes masked off by EXEC.
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    if (!append_event_for_pc.empty()) {
    // DS_APPEND/DS_CONSUME common phase. Each wave reduces popcount(EXEC), its lane zero performs
    // exactly one LDS atomic add/subtract, and the pre-op value is broadcast through the result slot.
    // The event tag prevents a wave at one static append from counting another event's lanes.
    const uint32_t append_pending = b.load_function(b.t_bool, append_pending_var);
    const uint32_t append_active = b.load_function(b.t_bool, append_active_var);
    const uint32_t append_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, append_event_var), b.uconst(1));
    const uint32_t append_encoded = b.sel(
        append_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, append_tag, b.uconst(1)),
               b.sel(append_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, append_encoded);
    b.barrier();

    b.store_function(append_count_var, zero);
    const uint32_t append_scan = b.id(), append_scanned = b.id();
    b.emit_selmerge(append_scanned);
    b.emit_condbranch(append_pending, append_scan, append_scanned);
    b.emit_label(append_scan);
    uint32_t append_count = zero;
    const uint32_t append_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    for (uint32_t i = 0; i < b.wave_size; ++i) {
        const uint32_t candidate_index = b.ibin(
            Op_IAdd, append_wave_base, b.uconst(i));
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(
            Op_ULessThan, candidate_index, b.uconst(b.local_count));
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, append_tag));
        append_count = b.b_iadd(
            append_count,
            b.sel(include, b.ibin(Op_BitwiseAnd, candidate, b.uconst(1)), zero));
    }
    b.store_function(append_count_var, append_count);
    b.emit_branch(append_scanned);
    b.emit_label(append_scanned);
    const uint32_t append_is_leader = b.land(
        append_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    const uint32_t append_leader = b.id(), append_reduced = b.id();
    b.emit_selmerge(append_reduced);
    b.emit_condbranch(append_is_leader, append_leader, append_reduced);
    b.emit_label(append_leader);
    const uint32_t append_count_value = b.load_function(b.t_u32, append_count_var);
    const uint32_t append_delta = b.sel(
        b.load_function(b.t_bool, append_consume_var),
        b.ibin(Op_ISub, zero, append_count_value), append_count_value);
    const uint32_t append_index = b.load_function(b.t_u32, append_idx_var);
    uint32_t append_old = 0;
    if (has_gds_append && has_lds_append) {
        const uint32_t gds_block = b.id(), lds_block = b.id(), memory_merge = b.id();
        b.emit_selmerge(memory_merge);
        b.emit_condbranch(b.load_function(b.t_bool, append_gds_var),
                          gds_block, lds_block);
        b.emit_label(gds_block);
        const uint32_t gds_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
        const uint32_t gds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(lds_block);
        const uint32_t lds_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
        const uint32_t lds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(memory_merge);
        append_old = b.emit_phi_2way(
            b.t_u32, gds_old, gds_end, lds_old, lds_end);
    } else if (has_gds_append) {
        append_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
    } else {
        append_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
    }
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index), append_old);
    b.emit_branch(append_reduced);
    b.emit_label(append_reduced);
    b.barrier();

    const uint32_t append_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index));
    const uint32_t append_dst = b.load_function(b.t_u32, append_dst_var);
    const uint32_t append_write = b.land(append_pending, append_active);
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            append_write,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, append_result, old));
    }
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // Publish this lane's pending vote bit and liveness. The switch merge is reached by every
    // invocation on every iteration, including lanes whose emulated wave has already ended.
    const uint32_t pending = b.load_function(b.t_bool, vote_pending_var);
    const uint32_t vote_value = b.load_function(b.t_bool, vote_value_var);
    const uint32_t vote_bit = b.sel(b.land(pending, vote_value), b.uconst(1), zero);
    const uint32_t active_bit = b.sel(b.load_function(b.t_bool, active_var), b.uconst(2), zero);
    b.cfg_scratch_store(b.linear_localid, b.ibin(Op_BitwiseOr, vote_bit, active_bit));
    b.barrier();

    const uint32_t wave_shift = b.wave_size == 32 ? 5u : 6u;
    const uint32_t wave_index = b.ibin(Op_ShiftRightLogical, b.linear_localid,
                                       b.uconst(wave_shift));
    const uint32_t wave_base = b.ibin(Op_ShiftLeftLogical, wave_index,
                                      b.uconst(wave_shift));
    const uint32_t lane_in_wave = b.ibin(Op_BitwiseAnd, b.linear_localid,
                                         b.uconst(b.wave_size - 1));

    // One lane per hardware wave reduces that wave's vote. Padding keeps every dynamic access in
    // bounds for a partial final wave; padded values are explicitly masked out.
    uint32_t wave_leader = b.id(), wave_reduced = b.id();
    const uint32_t is_wave_leader = b.ucmp(Op_IEqual, lane_in_wave, zero);
    b.emit_selmerge(wave_reduced);
    b.emit_condbranch(is_wave_leader, wave_leader, wave_reduced);
    b.emit_label(wave_leader);
    uint32_t wave_flags = zero;
    for (uint32_t lane = 0; lane < b.wave_size; ++lane) {
        const uint32_t idx = b.ibin(Op_IAdd, wave_base, b.uconst(lane));
        uint32_t flags = b.cfg_scratch_load(idx);
        if (padded_lanes != b.local_count)
            flags = b.sel(b.ucmp(Op_ULessThan, idx, b.uconst(b.local_count)), flags, zero);
        wave_flags = b.ibin(Op_BitwiseOr, wave_flags,
                            b.ibin(Op_BitwiseAnd, flags, b.uconst(1)));
    }
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index), wave_flags);
    b.emit_branch(wave_reduced);
    b.emit_label(wave_reduced);

    // Lane zero also reduces workgroup liveness. This uniform dispatcher loop is what makes the
    // internal barriers legal even after one hardware wave reaches S_ENDPGM before another.
    uint32_t group_leader = b.id(), group_reduced = b.id();
    const uint32_t is_group_leader = b.ucmp(Op_IEqual, b.linear_localid, zero);
    b.emit_selmerge(group_reduced);
    b.emit_condbranch(is_group_leader, group_leader, group_reduced);
    b.emit_label(group_leader);
    uint32_t group_flags = zero;
    for (uint32_t lane = 0; lane < b.local_count; ++lane) {
        const uint32_t flags = b.cfg_scratch_load(b.uconst(lane));
        group_flags = b.ibin(Op_BitwiseOr, group_flags,
                             b.ibin(Op_BitwiseAnd, flags, b.uconst(2)));
    }
    b.cfg_scratch_store(b.uconst(group_active_slot), group_flags);
    b.emit_branch(group_reduced);
    b.emit_label(group_reduced);
    b.barrier();

    const uint32_t wave_flags_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index));
    const uint32_t wave_any = b.ucmp(
        Op_INotEqual, b.ibin(Op_BitwiseAnd, wave_flags_result, b.uconst(1)), zero);
    const uint32_t vote_condition = b.bsel(
        b.load_function(b.t_bool, vote_invert_var), b.logical_not(wave_any), wave_any);
    const uint32_t selected_pc = b.sel(vote_condition,
        b.load_function(b.t_u32, vote_taken_var), b.load_function(b.t_u32, vote_next_var));
    const uint32_t vote_to_scc = b.load_function(b.t_bool, vote_to_scc_var);
    const uint32_t write_scc = b.land(pending, vote_to_scc);
    b.store_function(scc_var,
        b.bsel(write_scc, vote_condition, b.load_function(b.t_bool, scc_var)));
    const uint32_t vote_to_vcc = b.load_function(b.t_bool, vote_to_vcc_var);
    const uint32_t write_vcc = b.land(pending, vote_to_vcc);
    const uint32_t compared_vcc = b.land(
        b.load_function(b.t_bool, exec_var), vote_condition);
    b.store_function(vcc_var,
        b.bsel(write_vcc, compared_vcc, b.load_function(b.t_bool, vcc_var)));
    if (const auto saved_vcc = mv.find(106); saved_vcc != mv.end())
        b.store_function(saved_vcc->second,
            b.bsel(write_vcc, compared_vcc,
                   b.load_function(b.t_bool, saved_vcc->second)));
    const uint32_t write_pc = b.land(
        pending, b.land(b.logical_not(vote_to_scc), b.logical_not(vote_to_vcc)));
    b.store_function(pc_var, b.sel(write_pc, selected_pc, b.load_function(b.t_u32, pc_var)));
    uint32_t group_active = b.ucmp(
        Op_INotEqual, b.cfg_scratch_load(b.uconst(group_active_slot)), zero);
    group_active = apply_trip_bound(group_active);
    b.emit_condbranch(group_active, loop_header, loop_merge);
    }
    b.emit_label(loop_merge);

    // Expose the final emulated state to the caller. Graphics exports are emitted in their exact
    // cases, while any post-body bookkeeping still sees this invocation's final register values.
    initial = load_state();
    if (const auto terminal = block_for_pc.find(end_pc);
        terminal != block_for_pc.end() && scalar_reachable[terminal->second]) {
        initial.sreg_written = scalar_may_write_in[terminal->second];
        for (int reg : initial.sreg_written) initial.sreg_input.erase(reg);
    }
    initial.invalidated_vgpr_lane_slots =
        std::move(terminal_invalidated_vgpr_lane_slots);
    // sreg_srt and lds_addtid are path-sensitive SSA provenance without dispatcher function
    // variables. Not reconstructing them across a phase is intentional and fail-closed: a later
    // descriptor/LDS consumer rejects instead of reviving provenance from an arbitrary path.
    if (proven_wave32_masks) {
        const auto end_block = block_for_pc.find(end_pc);
        if (end_block != block_for_pc.end() &&
            (!b32_mask_reachable[end_block->second] ||
             !b32_mask_in[end_block->second].contains(106))) {
            initial.vcc = 0;
            initial.sreg_bool.erase(106);
            initial.sreg_bool_narrowed.erase(106);
            initial.sreg_bool_b32.erase(106);
        }
    }
    return true;
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
bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                             size_t guard_index) {
    // This is a textual backwards slice, not a reaching-definition analysis. Keep the prefix
    // branch-free so every scalar definition it finds dominates the terminal guard. In particular,
    // a VCC/EXEC branch may vary between guest waves even when the literal or SMEM value written in
    // one arm is itself uniform.
    for (size_t i = 0; i < guard_index; ++i) {
        const Rdna2Inst& prefix = ins[i];
        if ((prefix.fmt == Rdna2Format::SOPP && !sopp_is_noop(prefix)) ||
            (prefix.fmt == Rdna2Format::SOP1 && prefix.opcode >= 0x20u &&
             prefix.opcode <= 0x22u))
            return false;
    }

    std::function<bool(int, uint32_t, uint32_t)> uniform_reg_at;
    auto operand_uniform_at = [&](const Operand& operand, uint32_t words, uint32_t use_pc,
                                  auto&& self) -> bool {
        switch (operand.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal:
                return true;
            case OperandKind::Special:
                if (operand.value == 125) return true; // SGPR_NULL
                if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                    return false; // EXEC and SCC are not scalar launch data.
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!self(operand.value + static_cast<int>(word), use_pc, 0)) return false;
                return true;
            default:
                return false;
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
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::Special:
                        if (operand.value == 125) return true;
                        if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                            return false;
                        [[fallthrough]];
                    case OperandKind::SGPR:
                        for (uint32_t word = 0; word < words; ++word)
                            if (!uniform_reg_at(operand.value + static_cast<int>(word),
                                                writer.pc, depth + 1))
                                return false;
                        return true;
                    default:
                        return false;
                }
            };

            switch (writer.fmt) {
                case Rdna2Format::SOP1: {
                    // GETPC is identical for every wave. Mask/saveexec operations consume EXEC and
                    // are rejected by rdna2_instruction_may_change_exec; ordinary scalar operations trace
                    // their source at the destination's architectural width.
                    if (writer.opcode == 0x1f) return true;
                    if (rdna2_instruction_may_change_exec(writer) || writer.n_src != 1) return false;
                    return uniform_operand(writer.src[0], scalar_write_width(writer) == 2 ? 2u : 1u);
                }
                case Rdna2Format::SOP2: {
                    // Carry/cselect forms consume SCC in addition to their decoded operands. They are
                    // outside this narrow slice until SCC itself is represented as a recursive value.
                    if (writer.opcode == 0x04 || writer.opcode == 0x05 ||
                        writer.opcode == 0x0a || writer.opcode == 0x0b)
                        return false;
                    const uint32_t words = scalar_write_width(writer) == 2 ? 2u : 1u;
                    return writer.n_src == 2 && uniform_operand(writer.src[0], words) &&
                           uniform_operand(writer.src[1], words);
                }
                case Rdna2Format::SMEM: {
                    // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads use a
                    // base pair. The scalar offset is one dword in both forms.
                    if (!scalar_write_width(writer) || writer.n_src < 1) return false;
                    const uint32_t base_words = writer.opcode >= 0x8 ? 4u : 2u;
                    if (!uniform_operand(writer.src[0], base_words)) return false;
                    return writer.n_src < 2 || uniform_operand(writer.src[1], 1);
                }
                case Rdna2Format::SOPK:
                    // S_MOVK is a literal definition. Other SOPK data writers are read/modify/write
                    // or hardware-register reads and stay outside this proof.
                    return writer.opcode == 0x00;
                default:
                    return false;
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
                if (!operand_uniform_at(candidate.src[source], words, candidate.pc,
                                        uniform_reg_at))
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
BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
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
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a &&
            first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        ++branch_count;
        if (result.guard_index == ins.size() &&
            (in.opcode == 0x04 || in.opcode == 0x05) && branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool guarded = result.guard_index < first_barrier && first_barrier < result.end_index;
    bool scalar_prefix = guarded;
    for (size_t i = 0; scalar_prefix && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 ||
            in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::SMEM ||
            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            scalar_prefix = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR ||
                    operand.kind == OperandKind::Special) &&
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
    bool valid = result.end_index < ins.size() && branch_count > 2;

    for (size_t i = body_begin; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a)
            result.barriers.push_back(i);
    valid = valid && !result.barriers.empty();

    for (size_t i = body_begin; valid && i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // A trap deactivates its guest wave in the dispatcher. Emitting a host workgroup barrier
        // afterward would then require an invocation that has left the phase to participate.
        // Final-phase traps are fine because no outer barrier follows them.
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12 &&
            i < result.barriers.back())
            valid = false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            valid = false;
    }
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = body_begin; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a) continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid;
    return result;
}

// RDNA waves order their own LDS instructions, but the portable compute shell represents guest
// wave lanes as independent Vulkan invocations. A plain OpStore performed by lane 0 therefore does
// not publish its value to a following cross-lane atomic merely because every invocation reaches
// the atomic later in program order. GTA V's BVH bounds kernel uses this exact wave-synchronous
// idiom: select EXEC=1, initialize six adjacent dwords with one B64 and one B128 write, restore
// EXEC=-1, set vaddr=0, then issue three DS_MIN_F32 and three DS_MAX_F32 operations.
//
// Preserve the original byte-exact adjacent packet as a fast path: insert emitter-only S_BARRIERs
// before and after its six atomics. For a separated packet in one guest wave, ask emit_body to route
// every float atomic through the dispatcher's synchronized common phase. A proved lane-zero writer can
// retain ordinary stores; otherwise each preceding store becomes an atomic exchange, matching RDNA's
// serialized indexed bank conflicts without introducing a Vulkan write/write data race. The first
// common-phase barrier publishes those writes; each trailing barrier completes that atomic before the
// next dispatcher iteration or later gather. AcquireRelease on an individual atomic orders memory but
// is not an arrival barrier, so neither edge can be omitted. Every multi-wave separated shape rejects
// visibly. A real guest barrier, or an atomic with no preceding ordinary store in its phase, remains
// architectural and needs no synthesized edge.
struct LdsFminmaxSynchronization {
    bool needs_dispatcher = false;
    std::unordered_set<uint32_t> atomicized_store_pcs;
};

bool prepare_lds_fminmax_synchronization(std::vector<Rdna2Inst>& ins,
                                         RecompileDiagnosticContext diagnostic,
                                         bool at_most_one_guest_wave,
                                         LdsFminmaxSynchronization* synchronization = nullptr) {
    if (synchronization) *synchronization = {};
    auto ordinary_lds_store = [](const Rdna2Inst& in) {
        if (in.fmt != Rdna2Format::DS || in.ds_gds) return false;
        return in.opcode == 0x0d || in.opcode == 0x0e || in.opcode == 0x4d ||
               in.opcode == 0x4e ||
               in.opcode == 0xb0 || in.opcode == 0xde || in.opcode == 0xdf;
    };
    auto float_lds_atomic = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && !in.ds_gds &&
               (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32);
    };
    auto store_data_registers = [](const Rdna2Inst& in) {
        std::vector<int> registers;
        auto append = [&](int first, uint32_t count) {
            for (uint32_t word = 0; word < count; ++word)
                registers.push_back(first + static_cast<int>(word));
        };
        switch (in.opcode) {
            case 0x0d: case 0xb0: append(in.src[1].value, 1); break;
            case 0x0e:
                append(in.src[1].value, 1);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 1);
                break;
            case 0x4d: append(in.src[1].value, 2); break;
            case 0x4e:
                append(in.src[1].value, 2);
                if ((in.literal & 0xffu) != ((in.literal >> 8u) & 0xffu))
                    append(in.src[2].value, 2);
                break;
            case 0xde: append(in.src[1].value, 3); break;
            case 0xdf: append(in.src[1].value, 4); break;
            default: break;
        }
        return registers;
    };
    // Atomic exchange is equivalent to RDNA's indexed-bank serialization only when colliding lanes
    // write the same bits. Prove the narrow but generic form this family needs: every stored dword's
    // last writer is a plain v_mov from a scalar/literal source, and no control/EXEC edge can let a
    // store lane bypass that writer. Different addresses remain independent; equal addresses then
    // have identical candidate values, so the exchange winner is immaterial.
    auto store_data_are_wave_uniform = [&](size_t store_index) {
        const Rdna2Inst& store = ins[store_index];
        for (int reg : store_data_registers(store)) {
            size_t writer_index = ins.size();
            for (size_t j = store_index; j-- > 0;) {
                if (writes_vgpr(ins[j], reg)) {
                    writer_index = j;
                    break;
                }
            }
            if (writer_index == ins.size()) return false;
            const Rdna2Inst& writer = ins[writer_index];
            if (writer.fmt != Rdna2Format::VOP1 || writer.opcode != 0x01 ||
                writer.has_modifier || writer.has_sdwa || writer.has_dpp ||
                writer.src[0].kind == OperandKind::VGPR)
                return false;
            for (size_t j = writer_index + 1; j < store_index; ++j) {
                const Rdna2Inst& between = ins[j];
                if (rdna2_instruction_may_change_exec(between) ||
                    (between.fmt == Rdna2Format::SOPP && between.opcode >= 0x02u &&
                     between.opcode <= 0x12u && between.opcode != 0x03u &&
                     between.opcode != 0x0cu))
                    return false;
            }
            for (const Rdna2Inst& edge : ins) {
                if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                    edge.opcode <= 0x22u)
                    return false;
                if (edge.fmt != Rdna2Format::SOPP || edge.opcode < 0x02u ||
                    edge.opcode > 0x09u || edge.opcode == 0x03u)
                    continue;
                const uint32_t target = branch_target(edge);
                const bool source_outside = edge.pc < writer.pc || edge.pc >= store.pc;
                if (source_outside && target > writer.pc && target <= store.pc)
                    return false;
            }
        }
        return true;
    };
    auto words_are = [](const Rdna2Inst& in, uint32_t word0, uint32_t word1) {
        return in.words[0] == word0 && in.words[1] == word1;
    };

    static constexpr uint32_t kAtomicWord0[6] = {
        0xd8480000u, 0xd8480004u, 0xd8480008u,
        0xd84c000cu, 0xd84c0010u, 0xd84c0014u,
    };
    static constexpr uint32_t kAtomicWord1[6] = {
        0x00000900u, 0x00000a00u, 0x00000b00u,
        0x00000600u, 0x00000700u, 0x00000800u,
    };

    std::vector<size_t> phase_stores;
    bool phase_stores_are_single_lane = true;
    bool exec_is_single_lane = false;
    bool dispatcher_initializer_exec = false;
    uint32_t dispatcher_initializer_pc = UINT32_MAX;
    uint32_t phase_dispatcher_initializer_pc = UINT32_MAX;
    std::vector<size_t> synth_before;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) {
            phase_stores.clear();
            phase_stores_are_single_lane = true;
            phase_dispatcher_initializer_pc = UINT32_MAX;
            continue;
        }
        // These byte-exact scalar mask operations are the two lane-zero forms used by the known
        // initializers. A control-flow edge or any other EXEC writer ends the straight-line proof;
        // the dispatcher supplies ordering, not ownership of racing ordinary stores.
        if (in.words[0] == 0xbeea2481u || in.words[0] == 0xbefe0481u) {
            exec_is_single_lane = true; // s_and_saveexec_b64 vcc,1 / s_mov_b64 exec,1
            dispatcher_initializer_exec = in.words[0] == 0xbeea2481u;
            dispatcher_initializer_pc = dispatcher_initializer_exec ? in.pc : UINT32_MAX;
        } else if (rdna2_instruction_may_change_exec(in) ||
                   (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09) ||
                     in.opcode == 0x12))) {
            exec_is_single_lane = false;
            dispatcher_initializer_exec = false;
            dispatcher_initializer_pc = UINT32_MAX;
        }
        if (ordinary_lds_store(in)) {
            phase_stores.push_back(i);
            phase_stores_are_single_lane &= exec_is_single_lane;
            if (!dispatcher_initializer_exec ||
                (phase_dispatcher_initializer_pc != UINT32_MAX &&
                 phase_dispatcher_initializer_pc != dispatcher_initializer_pc))
                phase_stores_are_single_lane = false;
            else
                phase_dispatcher_initializer_pc = dispatcher_initializer_pc;
            continue;
        }
        if (!float_lds_atomic(in) || phase_stores.empty()) continue;

        bool exact = i >= 4 && i + 9 < ins.size() && phase_stores.size() == 2 &&
            phase_stores[0] == i - 4 && phase_stores[1] == i - 3 &&
            words_are(ins[i - 4], 0xd9340010u, 0x0000040cu) &&
            words_are(ins[i - 3], 0xdb7c0000u, 0x0000000cu) &&
            ins[i - 2].words[0] == 0xbefe04c1u && // s_mov_b64 exec, -1
            ins[i - 1].words[0] == 0x7e000280u;   // v_mov_b32 v0, 0
        for (size_t atomic = 0; exact && atomic < 6; ++atomic)
            exact = words_are(ins[i + atomic], kAtomicWord0[atomic], kAtomicWord1[atomic]);
        exact = exact &&
            ins[i + 6].words[0] == 0xbefe0481u && // pc81 s_mov_b64 exec, 1
            ins[i + 7].words[0] == 0x7e080280u && // pc82 v_mov_b32 v4, 0
            words_are(ins[i + 8], 0xdbfc0000u, 0x00000004u) && // pc83 ds_read_b128 v[0:3],v4
            words_are(ins[i + 9], 0xd9d80010u, 0x04000004u);   // pc85 ds_read_b64 v[4:5],v4

        bool found_lane0 = false;
        uint32_t lane0_pc = UINT32_MAX;
        if (exact) {
            for (size_t j = i - 4; j-- > 0;) {
                const Rdna2Inst& prefix = ins[j];
                if (prefix.words[0] == 0xbefe0481u) { // s_mov_b64 exec, 1
                    found_lane0 = true;
                    lane0_pc = prefix.pc;
                    for (size_t k = j + 1; k < i - 2; ++k) {
                        const Rdna2Inst& between = ins[k];
                        if (between.fmt == Rdna2Format::SOPP ||
                            (rdna2_instruction_may_change_exec(between) &&
                             between.words[0] != 0xbefe04c1u)) {
                            found_lane0 = false;
                            break;
                        }
                    }
                    break;
                }
                if (prefix.fmt == Rdna2Format::SOPP ||
                    rdna2_instruction_may_change_exec(prefix))
                    break;
            }
        }
        if (found_lane0) {
            const uint32_t last_store_pc = ins[i - 3].pc;
            for (const Rdna2Inst& edge : ins) {
                if (edge.pc >= lane0_pc || edge.fmt != Rdna2Format::SOPP ||
                    edge.opcode < 0x02 || edge.opcode > 0x09 || edge.opcode == 0x03)
                    continue;
                const uint32_t target = branch_target(edge);
                // The exact EXEC=1 writer must dominate both initializer stores. An edge from its
                // prefix may target the writer itself, but entering after it can leave EXEC full
                // and turn the supposedly single-writer OpStores into same-address races.
                if (target > lane0_pc && target <= last_store_pc) {
                    found_lane0 = false;
                    break;
                }
            }
        }
        if (!exact || !found_lane0) {
            bool dispatcher_initializer_dominates =
                phase_dispatcher_initializer_pc != UINT32_MAX;
            if (dispatcher_initializer_dominates) {
                const uint32_t last_store_pc = ins[phase_stores.back()].pc;
                for (const Rdna2Inst& edge : ins) {
                    // The deferred proof deliberately owns only a linear initializer region. Any
                    // scalar edge in that region could skip one of its lane-zero operations, while
                    // a later edge back into it could revisit a store after EXEC was restored full.
                    // Reject both, independent of whether the branch condition happens to look
                    // constant in this shader. GTA's captured initializer has no such edges.
                    if (edge.fmt == Rdna2Format::SOPP && edge.opcode >= 0x02 &&
                        edge.opcode <= 0x09 && edge.opcode != 0x03) {
                        const uint32_t target = branch_target(edge);
                        if (edge.pc <= last_store_pc || target <= last_store_pc) {
                            dispatcher_initializer_dominates = false;
                            break;
                        }
                    }
                    // An indirect PC update has no statically bounded target, so it cannot prove
                    // that the single-writer initializer is never re-entered.
                    if (edge.fmt == Rdna2Format::SOP1 && edge.opcode >= 0x20u &&
                        edge.opcode <= 0x22u) {
                        dispatcher_initializer_dominates = false;
                        break;
                    }
                }
            }
            if (synchronization) {
                if (!at_most_one_guest_wave) {
                    log_recompile_diagnostic(
                        diagnostic, "compute-recompile-reject", "terminal",
                        "pc=%u reason=multiwave-lds-fminmax-dispatcher", in.pc);
                    return false;
                }
                synchronization->needs_dispatcher = true;
                // A proven lane-zero initializer has one writer and can retain ordinary OpStores.
                // For a general one-wave initializer whose data is identical across active lanes,
                // make each exact preceding DS write an atomic exchange. RDNA serializes indexed
                // bank conflicts; equal colliding values make Vulkan's exchange winner immaterial.
                if (!phase_stores_are_single_lane || !dispatcher_initializer_dominates) {
                    if (!std::all_of(phase_stores.begin(), phase_stores.end(),
                                     store_data_are_wave_uniform)) {
                        log_recompile_diagnostic(
                            diagnostic, "compute-recompile-reject", "terminal",
                            "pc=%u reason=nonuniform-lds-store-before-ds-fminmax", in.pc);
                        return false;
                    }
                    for (size_t store : phase_stores)
                        synchronization->atomicized_store_pcs.insert(ins[store].pc);
                }
                phase_stores.clear();
                phase_stores_are_single_lane = true;
                phase_dispatcher_initializer_pc = UINT32_MAX;
                continue;
            }
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=unsynchronized-lds-store-before-ds-fminmax", in.pc);
            return false;
        }
        // EXEC=1 selects lane zero independently in every guest wave. The exact initializer uses
        // ordinary same-address stores, so more than one wave would race even though each wave has
        // only one active lane. Keep the title-observed single-wave workgroup admissible and leave a
        // different launch shape fail-visible until its cross-wave ownership can be proved.
        if (!at_most_one_guest_wave) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=multiwave-lds-fminmax-initializer", in.pc);
            return false;
        }

        synth_before.push_back(i);
        synth_before.push_back(i + 6);
        phase_stores.clear();
        phase_stores_are_single_lane = true;
        phase_dispatcher_initializer_pc = UINT32_MAX;
        i += 9; // both synthesized boundaries belong to this complete live atomic/gather group
    }

    std::vector<uint32_t> synth_pcs;
    synth_pcs.reserve(synth_before.size());
    for (size_t index : synth_before) synth_pcs.push_back(ins[index].pc);
    for (auto it = synth_before.rbegin(); it != synth_before.rend(); ++it) {
        Rdna2Inst barrier;
        barrier.pc = ins[*it].pc; // boundary immediately before the atomic at this guest PC
        barrier.fmt = Rdna2Format::SOPP;
        barrier.opcode = 0x0a;
        barrier.words[0] = 0xbf8a0000u;
        barrier.len_dwords = 0;   // emitter-only marker; never part of the guest byte stream
        ins.insert(ins.begin() + static_cast<std::ptrdiff_t>(*it), barrier);
    }

    // Prove each synthesized boundary is top-level even when the compact structurizer (rather than
    // the phase dispatcher) owns the program. Branches and loops may finish before the boundary,
    // but no edge may skip it, enter it from the far side, or carry only part of a workgroup back
    // across it. Traps/indirect PC changes before it also fail closed. This is the same edge
    // invariant used by the barrier-phase route without its unrelated >2-branch selection policy.
    for (uint32_t barrier_pc : synth_pcs) {
        bool uniform = true;
        for (const Rdna2Inst& in : ins) {
            if (in.pc < barrier_pc &&
                (in.is_end ||
                 (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) ||
                 (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)))
                uniform = false;
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                uniform = false;
        }
        if (!uniform) {
            log_recompile_diagnostic(
                diagnostic, "compute-recompile-reject", "terminal",
                "pc=%u reason=lds-fminmax-publication-barrier-not-workgroup-uniform",
                barrier_pc);
            return false;
        }
    }
    return true;
}

// Emit the instruction body (shared by every stage). Handles a single recognized COUNTED loop as a real
// structured SPIR-V loop (OpLoopMerge + OpPhi for loop-carried registers); loop-FREE streams walk straight
// through, byte-identical to the pre-loop-feature behavior. `exp_fn` handles an EXP instruction per stage
// (compute: reject; fragment: MRT color; vertex: POS/PARAM). Returns false if any instruction is
// unsupported. allow_exec_update / allow_smem match the stage's emit_alu flags.
bool emit_body(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
               const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
               bool allow_exec_update, bool allow_smem,
               const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
               const uint32_t* code = nullptr, size_t dwords = 0,
               const std::unordered_set<uint32_t>* inherited_dead_masks = nullptr,
               bool allow_cfg_dispatcher = true,
               uint32_t initial_dispatch_active = 0,
               bool force_barrier_phases = false,
               bool force_lds_fminmax_dispatcher = false) {
               // code/dwords: raw stream for forward-if target checks; inherited_dead_masks keeps
               // whole-shader liveness valid when a barrier-separated body is compiled in phases.
    rs.max_vgpr = std::max(rs.max_vgpr, shader_max_vgpr(ins));
    if (!b.cselect_b64_low_only_analysis_done) {
        b.cselect_b64_low_only_pcs = proven_cselect_b64_low_only_pcs(ins);
        b.cselect_b64_low_only_analysis_done = true;
    }
    if (!b.vcc_b32_low_only_analysis_done) {
        b.vcc_b32_low_only_pcs = proven_wave64_vcc_b32_low_only_pcs(ins);
        b.vcc_b32_low_only_analysis_done = true;
    }
    if (!b.structured_wave64_mask_reduction_analysis_done) {
        b.structured_wave64_mask_reduction_pcs =
            proven_structured_wave64_mask_reduction_pcs(ins);
        b.structured_wave64_mask_reduction_analysis_done = true;
    }
    if (!rs.smem_x16_descriptor_analysis_done) {
        rs.smem_x16_descriptor_loads = proven_smem_x16_descriptor_loads(ins, rt);
        rs.smem_x16_descriptor_analysis_done = true;
    }
    if (!rs.smem_x2_descriptor_fragment_analysis_done) {
        rs.smem_x2_descriptor_fragment_loads =
            proven_smem_x2_descriptor_fragment_loads(ins, rt, b.wave_size);
        rs.smem_x2_descriptor_fragment_analysis_done = true;
    }
    // Fold PC-relative embedded-table loads (s_getpc_b64-built V#s) before the walk — emit_alu's
    // SMEM/MUBUF/SOP1 handlers consult the proven table maps (#273/#1054).
    if (code) {
        PcrelTables tables = detect_pcrel_tables(ins, code, dwords);
        rs.mubuf_pcrel_tables = std::move(tables.mubuf);
        rs.smem_pcrel_tables = std::move(tables.smem);
    }
    std::unordered_set<uint32_t> local_dead_masks;
    if (!inherited_dead_masks) local_dead_masks = dead_wave_mask_writes(ins);
    const std::unordered_set<uint32_t>& dead_masks = inherited_dead_masks
        ? *inherited_dead_masks : local_dead_masks;

    // A large generated compute kernel may put a workgroup-uniform scalar early-out around several
    // barrier-separated phases, then use arbitrary (but barrier-free) control flow in its final
    // phase. The whole-stream CFG dispatcher cannot legally place an OpControlBarrier in one switch
    // case, while the narrow structurizer rejects the final phase's branch graph. Peel the terminal
    // scalar guard and compile each barrier-free phase independently. Every invocation in one
    // workgroup takes the guard together, and no guest edge may cross a split, so each explicit
    // barrier remains uniform while the final phase can use the ordinary CFG dispatcher.
    if (b.is_compute) {
        const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        if (phased.found &&
            (phased.guarded || initial_dispatch_active || force_barrier_phases)) {
            // Every phase shares one immutable Workgroup OpTypeArray. Size it from the complete
            // phased stream before the first dispatcher: a later portable DPP operation needs a
            // second per-lane plane even when the earlier phase needed only votes/liveness.
            if (!b.native_subgroup_size) {
                const uint32_t wave_count =
                    (b.local_count + b.wave_size - 1) / b.wave_size;
                const uint32_t padded_lanes = wave_count * b.wave_size;
                const bool has_portable_dpp = std::any_of(
                    ins.begin(), ins.begin() + phased.end_index,
                    [](const Rdna2Inst& in) {
                        return is_inplace_vadd_nc_u32_dpp_row_shr(in) ||
                            dpp_row_ror8_op(in) != DppRowRor8Op::None;
                    });
                const uint32_t scratch_dwords = padded_lanes +
                    (has_portable_dpp ? padded_lanes : 0u) + wave_count + 1;
                if (!b.declare_cfg_scratch(scratch_dwords)) return false;
            }
            uint32_t merge_label = 0;
            if (phased.guarded) {
                // Padded invocations cannot follow a divergent extent guard around this selection.
                // Keep that uncommon combination on the existing fail-closed path; the unguarded
                // phase form below is the one proved safe for partial workgroups.
                if (initial_dispatch_active) return false;
                const std::vector<Rdna2Inst> prefix(
                    ins.begin(), ins.begin() + phased.guard_index);
                if (!prefix.empty() &&
                    !emit_body(b, rs, prefix, safe, rt, allow_exec_update, allow_smem,
                               exp_fn, code, dwords, &dead_masks))
                    return false;
                if (!rs.scc) return false;
                const uint32_t execute_body = ins[phased.guard_index].opcode == 0x04
                    ? rs.scc : b.logical_not(rs.scc);
                const uint32_t body_label = b.id();
                merge_label = b.id();
                b.emit_selmerge(merge_label);
                b.emit_condbranch(execute_body, body_label, merge_label);
                b.emit_label(body_label);
            }

            auto emit_phase = [&](std::vector<Rdna2Inst>& phase) {
                // Unguarded phases are admitted specifically because the whole-stream exact-wave
                // dispatcher was blocked by a guest barrier. Use that dispatcher directly for each
                // now barrier-free region; re-entering the narrow structurizer could reject the same
                // backward-else/complex shape before it ever reaches the fallback.
                if (!phased.guarded || initial_dispatch_active)
                    return emit_cfg_state_machine(
                        b, rs, phase, safe, rt, allow_exec_update, allow_smem,
                        exp_fn, code, dwords, initial_dispatch_active,
                        force_lds_fminmax_dispatcher);
                return emit_body(b, rs, phase, safe, rt, allow_exec_update, allow_smem,
                                 exp_fn, code, dwords, &dead_masks, true, 0, false,
                                 force_lds_fminmax_dispatcher);
            };

            size_t phase_begin = phased.guarded ? phased.guard_index + 1 : 0;
            for (size_t barrier_index : phased.barriers) {
                std::vector<Rdna2Inst> phase(
                    ins.begin() + phase_begin, ins.begin() + barrier_index);
                if (!phase.empty()) {
                    // A phase is a complete control-flow region even though the guest's real
                    // S_ENDPGM follows the final phase. The arbitrary-CFG fallback requires an end
                    // block so its persistent dispatcher can become inactive and rejoin this outer
                    // barrier sequence. Give each proven split a synthetic, emitter-only terminator
                    // at the boundary; no branch crosses the boundary (proved above), and the raw
                    // barrier remains emitted exactly once by this outer shell.
                    Rdna2Inst phase_end;
                    phase_end.pc = ins[barrier_index].pc;
                    phase_end.fmt = Rdna2Format::SOPP;
                    phase_end.opcode = 0x01u;
                    phase_end.len_dwords = 1;
                    phase_end.is_end = true;
                    phase_end.synthetic_terminator = true;
                    phase.push_back(phase_end);
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[compute-phase] begin=%u end=%u barrier=%u\n",
                                     phase.front().pc, phase[phase.size() - 2].pc,
                                     ins[barrier_index].pc);
                    if (!emit_phase(phase)) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-phase-reject", "consequent",
                            "begin=%u end=%u", phase.front().pc, phase[phase.size() - 2].pc);
                        return false;
                    }
                }
                b.barrier();
                phase_begin = barrier_index + 1;
            }
            std::vector<Rdna2Inst> tail(ins.begin() + phase_begin, ins.end());
            if (!tail.empty()) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-phase] begin=%u end=%u tail=1\n",
                                 tail.front().pc, tail.back().pc);
                if (!emit_phase(tail)) {
                    log_recompile_diagnostic(b.diagnostic, "compute-phase-reject", "consequent",
                                             "begin=%u end=%u tail=1",
                                             tail.front().pc, tail.back().pc);
                    return false;
                }
            }
            if (phased.guarded) {
                b.emit_branch(merge_label);
                b.emit_label(merge_label);
            }
            if (initial_dispatch_active)
                b.partial_barrier_phases_emitted = true;
            return true;
        }
    }
    if (force_lds_fminmax_dispatcher)
        return emit_cfg_state_machine(
            b, rs, ins, safe, rt, allow_exec_update, allow_smem,
            exp_fn, code, dwords, initial_dispatch_active, true);
    std::unordered_set<uint32_t> effective_safe = safe;
    const CountedLoop L = detect_counted_loop(ins);
    size_t idx = 0;
    // Cross-lane wave ops (mbcnt) emit LDS + barriers, which are only valid at wave-uniform points — so
    // they're allowed ONLY in the straight-line path (no divergent loop/if around them). Set true below.
    bool wave_ok = false;
    auto emit_range = [&](uint32_t pc_lo, uint32_t pc_hi) -> bool {   // emit ins whose pc ∈ [pc_lo, pc_hi)
        for (; idx < ins.size(); ++idx) {
            const Rdna2Inst& in = ins[idx];
            if (in.is_end || in.pc >= pc_hi) break;
            if (in.pc < pc_lo) continue;
            if (dead_masks.count(in.pc)) continue;
            if (in.fmt == Rdna2Format::EXP) { if (!exp_fn(rs, in)) return false; continue; }
            bool ok = true;
            const bool handled = emit_alu(
                b, rs, in, ok, allow_exec_update, &effective_safe, allow_smem, rt, wave_ok);
            if (handled && ok)
                record_scalar_write(
                    rs, in,
                    allows_compute_scalar_vcc_bridge(b));
            // Shader I/O tap: snapshot this instruction's destination VGPR (+3) if it is the tapped PC.
            if (handled && ok && in.pc == b.tap_pc && in.dst.kind == OperandKind::VGPR) {
                auto tv = [&](int r) { auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                b.set_tap(tv(in.dst.value), tv(in.dst.value + 1), tv(in.dst.value + 2), tv(in.dst.value + 3));
            }
            if (!handled || !ok) {
                // PROSPER_DBG (gated, off by default): report the instruction that fails recompilation —
                // the first unsupported op / unresolved resource that makes a shader return empty.
                // NOT gated on PROSPER_DBG, for the reason at the CFG reject site above: the gate
                // suppressed the RECORDING as well as the printing, and the recording is what the
                // unconditional skip line reads. This site fires once per failing compile, so the
                // formatting cost it now always pays is one string per rejected shader.
                {
                    // `mode` separates the two rejections that used to print identically and want
                    // OPPOSITE work (#2412). `unknown-encoding` (handled=false) means no lowering
                    // exists — write the emitter. `unresolved-operand` (handled=true, ok=false)
                    // means the lowering exists and could not resolve an operand or a V#/T#/S#
                    // through the resource table — the emitter is fine and the descriptor is the
                    // defect. GTA V's black 3D world is the worked example: its top three rejected
                    // instructions are buffer_store_dword / buffer_load_dwordx2 / buffer_load_dword,
                    // all lowered at :9343-9362, so a census without this field reads as "implement
                    // MUBUF" when nothing about MUBUF is missing.
                    // Shader identity (#2412): the reject lines carry a program-local pc and nothing
                    // else, so a census cannot group them by SHADER -- which is the unit that matters,
                    // since 24,485 skipped GTA V draws turned out to be 43 distinct shaders. The first
                    // code dword plus the span identifies one cheaply and stably.
                    log_recompile_diagnostic(
                        b.diagnostic, "recompile-reject", "terminal",
                        "sh=%08x/%zu mode=%s pc=%u words=%s fmt=%d op=0x%x "
                        "dst=%d(kind%d) src=%d(k%d),%d(k%d),%d(k%d) dmask=0x%x "
                        "dim=%u glc=%d len=%u modifier=%d dpp=%d sdwa=%u/%u/%u/%u "
                        "sext=%d/%d",
                        dwords ? code[0] : 0u, dwords,
                        handled ? "unresolved-operand" : "unknown-encoding",
                        in.pc, reject_words_text(in).c_str(), (int)in.fmt, in.opcode,
                        in.dst.value, (int)in.dst.kind,
                        in.src[0].value, (int)in.src[0].kind,
                        in.src[1].value, (int)in.src[1].kind,
                        in.src[2].value, (int)in.src[2].kind,
                        in.mimg_dmask, in.mimg_dim, (int)in.mimg_glc, in.len_dwords, (int)in.has_modifier,
                        (int)in.has_dpp, in.sdwa_dst_sel, in.sdwa_dst_unused,
                        in.sdwa_src0_sel, in.sdwa_src1_sel,
                        // The four selects alone cannot distinguish the zero- from the sign-extending
                        // form of one encoding, and they are different operations (#2013).
                        (int)in.sdwa_src0_sext, (int)in.sdwa_src1_sext);
                    // The primary line historically printed only the fixed MIMG pair. That hid the
                    // address VGPRs which distinguished Asterix's rejected NSA form from the accepted
                    // consecutive-vaddr form. Keep a separate MIMG-only line so an address-shape
                    // diagnosis sees every decoded extra dword without adding zeros to unrelated ops.
                    if (in.fmt == Rdna2Format::MIMG && in.len_dwords > 2u)
                        log_recompile_diagnostic(
                            b.diagnostic, "recompile-reject-mimg-address", "terminal",
                            "pc=%u extra=%u words=%08x,%08x,%08x",
                            in.pc, in.len_dwords - 2u,
                            in.words[2], in.words[3], in.words[4]);
                }
                return false;
            }
        }
        return true;
    };
    auto& safe_branches = effective_safe;
    if (L.found) {
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        bool guarded_narrow_entry = false;
        // saveexec -> execz -> matching EXEC restore around a side-effect-free counted region is a
        // whole-wave empty-work optimization. In the per-invocation shell we may run the uniform
        // scalar loop for every invocation while narrowed EXEC predicates vector writes; inactive
        // lanes retain their old VGPRs until the exact restore. Reject stores/exports/barriers and
        // unclassified memory so this never becomes a general branch-linearization escape hatch.
        // Scan inside-out so an already-proven nested guard may contribute its balanced save/restore
        // pair without making an otherwise-safe outer guarded loop look like it leaks narrowed EXEC.
        struct GuardedExecRegion { uint32_t save_pc, restore_pc; };
        std::vector<GuardedExecRegion> guarded_exec_regions;
        for (size_t branch_index = ins.size(); branch_index-- > 0;) {
            const Rdna2Inst& branch = ins[branch_index];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 || branch.simm16 <= 0)
                continue;
            size_t previous = branch_index;
            while (previous > 0) {
                --previous;
                if (!sopp_is_noop(ins[previous])) break;
            }
            if (previous >= branch_index) continue;
            const Rdna2Inst& saveexec = ins[previous];
            if (saveexec.fmt != Rdna2Format::SOP1 ||
                (saveexec.opcode != 0x24 && saveexec.opcode != 0x25) ||
                saveexec.dst.kind != OperandKind::SGPR || saveexec.dst.value > 104) continue;
            const uint32_t target = branch_target(branch);
            const Rdna2Inst* restore = nullptr;
            for (const auto& candidate : ins) if (candidate.pc == target) { restore = &candidate; break; }
            if (!restore || restore->fmt != Rdna2Format::SOP1 || restore->opcode != 0x04 ||
                restore->dst.value < 126 || !reg_operand(restore->src[0], saveexec.dst.value)) continue;
            // A lexical save/restore pair is not necessarily balanced along the counted-loop CFG.
            // In particular, a save in the body with its restore after the backedge leaves EXEC
            // narrowed between iterations (EXEC has no loop phi), and a zero-trip path reaches an
            // undominated restore. Accept only a pair contained in one straight-line loop segment,
            // or a true preheader-to-postloop wrapper around the complete loop.
            const bool same_preloop = saveexec.pc < L.header_pc && target < L.header_pc;
            const bool same_condition = saveexec.pc >= L.header_pc && target < L.exit_branch_pc;
            const bool same_body = saveexec.pc > L.exit_branch_pc && target < L.backedge_pc;
            const bool same_postloop = saveexec.pc >= L.exit_pc;
            const bool wraps_loop = saveexec.pc < L.header_pc && target >= L.exit_pc;
            if (!same_preloop && !same_condition && !same_body && !same_postloop && !wraps_loop)
                continue;
            bool side_effect_free = true;
            for (const auto& candidate : ins) {
                if (candidate.pc <= branch.pc || candidate.pc >= target) continue;
                bool clobbers_guard_mask = false;
                for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                    clobbers_guard_mask |= base < saveexec.dst.value + 2 &&
                        saveexec.dst.value < base + static_cast<int>(width);
                });
                bool balanced_nested_exec = false;
                for (const auto& nested : guarded_exec_regions) {
                    if (nested.save_pc > branch.pc && nested.restore_pc < target &&
                        (candidate.pc == nested.save_pc || candidate.pc == nested.restore_pc)) {
                        balanced_nested_exec = true;
                        break;
                    }
                }
                if (candidate.fmt == Rdna2Format::EXP || candidate.fmt == Rdna2Format::DS ||
                    candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                    candidate.fmt == Rdna2Format::MIMG || candidate.fmt == Rdna2Format::FLAT ||
                    (rdna2_instruction_may_change_exec(candidate) && !balanced_nested_exec) ||
                    clobbers_guard_mask ||
                    (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0x0a)) {
                    side_effect_free = false;
                    break;
                }
            }
            if (!side_effect_free) continue;
            effective_safe.insert(branch.pc);
            guarded_exec_regions.push_back({saveexec.pc, target});
            if (branch.pc < L.header_pc && target >= L.exit_pc) guarded_narrow_entry = true;
        }
        // 1. Pre-loop body. A compiler may place one ordinary uniform if/else before the canonical
        // counted loop (Evergate selects one of two constant blocks this way; Astro's NGG culling
        // prelude also has a one-arm conditional). Structure that choice with the same two-arm PHIs
        // as the general forward-if path, then enter the existing counted-loop lowering. Anything
        // nested/more complex stays unsupported and rejects visibly.
        std::vector<Rdna2Inst> preloop;
        for (const auto& in : ins) {
            if (in.pc >= L.header_pc) break;
            // A forward execz spanning the counted loop is a redundant wave-empty guard when the
            // prelude reaches it with full EXEC. Leave it in the emitted stream (emit_alu proves
            // full EXEC and otherwise rejects), but do not ask the pre-loop uniform-if detector to
            // model a region whose body deliberately extends beyond its artificial end marker.
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= L.header_pc) continue;
            preloop.push_back(in);
        }
        Rdna2Inst preloop_end;
        preloop_end.pc = L.header_pc;
        preloop_end.is_end = true;
        preloop.push_back(preloop_end);
        bool preloop_rejected = false;
        const std::vector<ForwardIf> preloop_ifs = detect_forward_ifs(
            preloop, /*allow_vcc*/!b.is_compute, code, dwords, &effective_safe, nullptr,
            &preloop_rejected, /*compute_wave_branches*/b.is_compute, b.diagnostic);
        // detect_forward_ifs clamps a branch to an immediate s_endpgm at its artificial end marker
        // and records it as early_out. In this truncated prelude that can be a real branch over the
        // entire counted loop, so it cannot be structured as an ordinary one-arm conditional.
        const bool preloop_if_unsupported = std::any_of(
            preloop_ifs.begin(), preloop_ifs.end(), [&](const ForwardIf& branch) {
                return branch.early_out ||
                    (branch.has_else ? branch.merge_pc : branch.target_pc) > L.header_pc;
            });
        if (preloop_rejected || preloop_if_unsupported) {
            log_recompile_diagnostic(
                b.diagnostic, "recompile-reject", "terminal",
                "counted-loop prelude cfg rejected=%u ifs=%zu header=%u",
                preloop_rejected, preloop_ifs.size(), L.header_pc);
            return false;
        }
        if (preloop_ifs.empty()) {
            if (!emit_range(0, L.header_pc)) return false;
        } else if (preloop_ifs.size() > 1) {
            // The general structurizer already owns nested and disjoint ForwardIf trees, including
            // exact guest-wave EXEC/VCC tests in compute. Reuse it for a multi-choice prefix instead
            // of duplicating another recursive IF emitter inside the counted-loop path. Keep every
            // real prefix instruction here (including a proven-safe guard spanning the whole loop):
            // the truncated scan above intentionally omitted that guard only so it was not mistaken
            // for an early-out at the artificial end marker. effective_safe still makes emit_alu
            // linearize a proven spanning guard, and inherited dead masks retain whole-shader liveness.
            std::vector<Rdna2Inst> preloop_body;
            for (const auto& in : ins) {
                if (in.pc >= L.header_pc) break;
                preloop_body.push_back(in);
            }
            preloop_body.push_back(preloop_end);
            if (!emit_body(b, rs, preloop_body, effective_safe, rt, allow_exec_update,
                           allow_smem, exp_fn, code, dwords, &dead_masks,
                           /*allow_cfg_dispatcher*/false))
                return false;
            while (idx < ins.size() && ins[idx].pc < L.header_pc) ++idx;
            if (idx >= ins.size() || ins[idx].pc != L.header_pc) return false;
        } else {
            const ForwardIf F = preloop_ifs[0];
            if (!emit_range(0, F.branch_pc)) return false;
            if (idx >= ins.size() || ins[idx].pc != F.branch_pc) return false;
            ++idx; // consume the conditional branch
            // The structured PHIs below use zero for an SGPR absent on one predecessor. From this
            // point onward map presence alone is no longer a scalar-lifetime MUST fact.
            rs.scalar_presence_has_no_placeholders = false;
            // An scc-conditioned forward-if with a POISONED SCC (rs.scc == 0: a 64-bit mask op was
            // the last architectural SCC writer, unrepresentable per-lane) must reject — this is
            // exactly the non-adjacent stale-SCC consumer from the ISA audit (#879).
            if (!F.on_exec && !F.on_vcc && !rs.scc) return false;
            if (F.on_vcc && !rs.vcc) return false;
            uint32_t condition = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
            if (b.is_fragment && (F.on_exec || F.on_vcc))
                condition = b.fragment_wave_any(condition);
            condition = F.on_scc0 ? condition : b.logical_not(condition);
            const RegState before = rs;
            std::set<int> written_v, written_s;
            const uint32_t then_end = F.has_else ? F.sb_pc : F.target_pc;
            const uint32_t merge_pc = F.has_else ? F.merge_pc : F.target_pc;
            loop_written_regs(ins, F.branch_pc + 1, then_end, written_v, written_s);
            if (F.has_else)
                loop_written_regs(ins, F.target_pc, merge_pc, written_v, written_s);
            const uint32_t then_label = b.id(), else_label = b.id(), merge_label = b.id();
            b.emit_selmerge(merge_label);
            b.emit_condbranch(condition, then_label, else_label);

            b.emit_label(then_label);
            if (!emit_range(F.branch_pc + 1, then_end)) return false;
            if (F.has_else) {
                if (idx >= ins.size() || ins[idx].pc != F.sb_pc) return false;
                ++idx; // consume the then arm's jump to the merge
            }
            const uint32_t then_block = b.cur_block;
            std::unordered_map<int, uint32_t> then_v, then_s;
            for (int reg : written_v) then_v[reg] = vget(reg);
            for (int reg : written_s) then_s[reg] = sget(reg);
            const uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
            const bool then_narrowed = rs.exec_narrowed;
            const auto then_bool = rs.sreg_bool;
            const auto then_bool_b32 = rs.sreg_bool_b32;
            const auto then_written = rs.sreg_written;
            b.emit_branch(merge_label);

            rs = before;
            b.emit_label(else_label);
            if (F.has_else && !emit_range(F.target_pc, merge_pc)) return false;
            const uint32_t else_block = b.cur_block;
            b.emit_branch(merge_label);
            b.emit_label(merge_label);
            for (int reg : written_v) {
                const uint32_t else_value = vget(reg);
                if (then_v[reg] != else_value)
                    rs.vreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_v[reg], then_block, else_value, else_block);
            }
            for (int reg : written_s) {
                const uint32_t else_value = sget(reg);
                if (then_s[reg] != else_value)
                    rs.sreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_s[reg], then_block, else_value, else_block);
            }
            if (then_scc != rs.scc)
                // A poisoned (0) input degrades to bfalse across the merge — no invalid SPIR-V and
                // no stricter rejection than the pre-poison behavior; straight-line consumers of a
                // poisoned SCC are still rejected at their own sites.
                rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), then_block,
                                         rs.scc ? rs.scc : b.bfalse(), else_block);
            if (then_vcc != rs.vcc)
                rs.vcc = !then_vcc || !rs.vcc ? 0u : b.emit_phi_2way(
                    b.t_bool, then_vcc, then_block, rs.vcc, else_block);
            if (then_exec != rs.exec)
                rs.exec = b.emit_phi_2way(b.t_bool, then_exec, then_block, rs.exec, else_block);
            rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
            rs.sreg_written.insert(then_written.begin(), then_written.end());
            for (int reg : then_written) rs.sreg_input.erase(reg);
            if (then_bool != rs.sreg_bool || then_bool_b32 != rs.sreg_bool_b32) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "counted-loop prelude changes mask domain");
                return false; // no mask-domain PHIs in this narrow composition
            }
            if (!emit_range(merge_pc, L.header_pc)) return false;
        }
        // Ordinary counted loops require full EXEC at entry. An NGG vertex is already represented by
        // one independent Vulkan invocation, however, so its EXEC bit is an ordinary per-invocation
        // Boolean. Carry that bit through the loop just like SCC/VCC instead of rejecting Astro Bot's
        // fixed-trip culling loop merely because some guest lanes were masked before its header.
        const bool carry_vertex_exec = b.ngg_one_lane && rs.exec_narrowed;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "counted-loop enters with narrowed EXEC");
            return false;
        }
        // 2. Loop-carried registers -> a header OpPhi each. `cond_written` = regs written in the CONDITION
        // region [header, exit_branch): those execute on the exiting iteration too, so their post-loop
        // value is the condition-block value (which dominates the merge), NOT the phi (defect A). SCC/VCC
        // always get a phi so any cross-iteration carry is valid SSA (defect C); they're recomputed each
        // iteration in practice, so the phi is usually dead — harmless.
        std::set<int> cv, cs, condv, conds, scalar_may_writes;
        loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
        loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
        loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
        for (int reg : rs.sreg_bool_b32)
            if (scalar_may_writes.contains(reg)) return false;
        if (b.allow_b32_masks &&
            has_unpersisted_b32_mask_lifetime(
                ins, L.header_pc, L.backedge_pc, rs))
            return false;
        const uint32_t preheader = b.cur_block;
        const uint32_t hdr = b.id(), check = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        // Loop-carried PHIs likewise seed a missing preheader SGPR with zero.
        rs.scalar_presence_has_no_placeholders = false;
        b.emit_branch(hdr); b.emit_label(hdr);
        struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };   // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec
        std::vector<PhiRec> phis;
        for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
        for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
        // A poisoned (0) SCC live-in degrades to bfalse — the loop shapes re-produce SCC via their
        // in-loop s_cmp before any read, so the phi seed is dead in practice; 0 would be invalid SSA.
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
        if (rs.vcc) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
        if (carry_vertex_exec) {
            size_t p;
            uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p);
            rs.exec = ph;
            phis.push_back({0, 4, ph, p});
        }
        // The header executes again after the back-edge. A direct/SRT descriptor overwritten
        // anywhere in the loop is therefore not an invariant entry descriptor at header compile
        // time. An exact descriptor load in the header may establish fresh provenance afterward.
        invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
        b.emit_loopmerge(merge, cont); b.emit_branch(check); b.emit_label(check);
        // 3. Condition block: emit [header, exit_branch); the SCC exit becomes OpBranchConditional.
        if (!emit_range(L.header_pc, L.exit_branch_pc)) return false;
        if (!rs.scc) return false;   // condition region left SCC poisoned: the exit test is unknowable
        // Snapshot condition-region register values (these dominate the merge — see defect A above).
        std::unordered_map<int,uint32_t> condv_val, conds_val;
        for (int r : condv) condv_val[r] = vget(r);
        for (int r : conds) conds_val[r] = sget(r);
        const uint32_t cond_exec = rs.exec;
        const bool cond_exec_narrowed = rs.exec_narrowed;
        // s_cbranch_scc0 exits when SCC==0 (so the loop CONTINUES when SCC!=0); scc1 is the inverse.
        uint32_t loop_cond = L.exit_on_scc0 ? rs.scc : b.bsel(rs.scc, b.bfalse(), b.btrue());
        b.emit_condbranch(loop_cond, body, merge);
        while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;   // (already past it)
        if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;     // skip the exit branch itself
        b.emit_label(body);
        // 4. Body [after exit_branch, back-edge). Must restore EXEC before looping (bail if left narrowed).
        if (!emit_range(L.exit_branch_pc + 1, L.backedge_pc)) return false;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "counted-loop body leaves EXEC narrowed");
            return false;
        }
        if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;        // skip the back-edge branch
        // 5. Continue block branches back to the header; patch each phi's back-edge (value = current, cont).
        b.emit_branch(cont); b.emit_label(cont);
        for (auto& pr : phis) {
            uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                        : pr.dom == 1 ? sget(pr.reg)
                        : pr.dom == 2 ? rs.scc
                        : pr.dom == 3 ? rs.vcc : rs.exec;
            if (!nv && pr.dom == 3) return false;
            if (!nv && pr.dom == 2)
                nv = b.bfalse(); // poisoned SCC back-edge value: false when dead in practice
            b.patch_phi(pr.patch, nv, cont);
        }
        b.emit_branch(hdr);
        // 6. Merge (loop exit): a condition-region reg keeps its exit-iteration (%check) value; a body-only
        //    reg (and scc/vcc) takes the header phi (its value when the loop exited).
        b.emit_label(merge);
        for (auto& pr : phis) {
            if (pr.dom == 0)      rs.vreg[pr.reg] = condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi;
            else if (pr.dom == 1) rs.sreg[pr.reg] = conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi;
            // SCC/VCC take the phi (not a condition-region snapshot): reading a wave flag AFTER a loop is
            // not a real codegen pattern (flags are transient, consumed by their branch), so the A-class
            // exit-iteration refinement is intentionally omitted for them.
            else if (pr.dom == 2) rs.scc = pr.phi;
            else if (pr.dom == 3) rs.vcc = pr.phi;
            else                  rs.exec = cond_exec;
        }
        if (carry_vertex_exec) rs.exec_narrowed = cond_exec_narrowed;
        // 7. Post-loop body. Feed the suffix back through the ordinary body selector: with this
        // counted back-edge removed it can use the established nested forward-if/divergent-loop
        // structurizer. This composes a counted loop with a non-trivial shared postlude without
        // duplicating that CFG machinery or admitting arbitrary branches inside the counted loop.
        std::vector<Rdna2Inst> postloop;
        for (const auto& in : ins) if (in.pc >= L.exit_pc) postloop.push_back(in);
        if (!postloop.empty() &&
            !emit_body(b, rs, postloop, effective_safe, rt, allow_exec_update, allow_smem,
                       exp_fn, code, dwords)) return false;
    } else if (std::vector<DivLoop> Ls; true) {
        // EXEC/VCC/SCC-exit loops (#273/#615/#1554) + structured scalar IFs. Each is a real SPIR-V
        // loop with header phis for carried register/mask state. Fragment conditions are exact wave64
        // votes; vertex and the guarded compute cases retain their per-invocation form. The IF
        // machinery recurses into loop bodies and handles their nested forward-execz regions.
        Ls = detect_divergent_loops(ins, safe, b.is_fragment, b.diagnostic);
        if (b.is_compute) {
            // Compute VCC-exit loops (#590, extending #615): the fragment-stage uniformity proof is
            // data-provenance-based, not stage-based — vcc_exit_is_wave_uniform accepts a compare only
            // when every input is scalar/inline/literal or a VGPR whose nearest definition is an
            // unmodified uniform VOP1 move from a scalar. With that proven, every lane's compare bool
            // is identical, so the wave-empty vccz exit lowers to THIS invocation's !cond exactly as
            // in the fragment shell (tid-derived/varying inputs fail the proof and keep rejecting).
            // One compute-specific guard (see the per-condition detail below for why both Vcc- and
            // Exec-condition loops are safe here — the Exec case is GTA V's exec_cs_2042d47600 grid-stride
            // decode loop, #1183):
            //   * the body must be barrier/LDS/cross-lane-free — the proof is per-WAVE, and a barrier
            //     inside a loop whose trip count could differ across the workgroup's waves would be
            //     workgroup-divergent control flow (UB). DOLL's blocked light/fill kernels are
            //     straight-line bodies, so nothing observed is lost. CONFIDENCE: MED-HIGH (shared
            //     emit machinery; spirv-val + coverage tests + Messenger guard gate it).
            // SCC-condition loops use exact architectural scalar control and may retain ordinary
            // LDS effects; their synchronized/cross-lane operations remain rejected below.
            // Condition::Vcc loops are accepted under the detector's uniformity proof (#615/#590).
            // Condition::Exec loops (#590 — DOLL's nested post-process kernel is the observed compute
            // case) lower with the per-invocation model: this invocation
            // iterates while ITS EXEC bit holds after the header's v_cmpx recompute. Both flavors
            // require the barrier/LDS/cross-lane-free body below — the per-invocation trip count can
            // differ across a workgroup, so a barrier inside the loop would be workgroup-divergent
            // control flow (UB); barriers AFTER the loop are fine (the loop merge reconverges).
            bool compute_ok = !Ls.empty();
            const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
            auto has_stable_post_barrier_lds_reads = [&](const DivLoop& loop) {
                // The structured emitter performs the Workgroup load before predicating the VGPR
                // result. Limit this exception to a top-tested EXEC loop, where the body entry
                // proves this invocation active, and keep that proof live through every DS read.
                if (!phased.found || loop.condition != DivLoop::Condition::Exec ||
                    loop.bottom_tested) return false;
                size_t phase_begin = ins.size();
                size_t phase_end = phased.end_index;
                for (size_t barrier : phased.barriers) {
                    if (ins[barrier].pc < loop.header_pc) {
                        phase_begin = barrier + 1;
                        continue;
                    }
                    phase_end = barrier;
                    break;
                }
                if (phase_begin == ins.size() ||
                    (phase_end < ins.size() && loop.exit_pc > ins[phase_end].pc))
                    return false;
                // This narrow exception is read-only for the COMPLETE phase. A preceding proved
                // top-level barrier publishes earlier LDS initialization; with no later LDS write,
                // atomic, GDS access, or other DS opcode, each per-lane loop may repeat its own
                // ordinary DS_READ_B32 without introducing visibility or synchronization edges.
                for (size_t i = phase_begin; i < phase_end; ++i)
                    if (ins[i].fmt == Rdna2Format::DS &&
                        (ins[i].opcode != 0x36 || ins[i].ds_gds))
                        return false;
                for (const auto& read : ins) {
                    if (read.pc < loop.header_pc || read.pc >= loop.backedge_pc ||
                        read.fmt != Rdna2Format::DS) continue;
                    // The condition region runs before the top-of-loop EXEC test. A load there has
                    // no active-lane proof yet, so the emitter's unconditional Workgroup access is
                    // not safe even though the rest of the phase is read-only.
                    if (read.pc <= loop.exit_branch_pc) return false;
                    for (const auto& prior : ins) {
                        if (prior.pc <= loop.exit_branch_pc || prior.pc >= read.pc) continue;
                        if (rdna2_instruction_may_change_exec(prior)) return false;
                    }
                }
                return true;
            };
            for (const auto& L : Ls) {
                if (!compute_ok) break;
                const bool stable_post_barrier_reads =
                    has_stable_post_barrier_lds_reads(L);
                for (const auto& in : ins) {
                    if (in.is_end || in.pc >= L.exit_pc) break;
                    if (in.pc < L.header_pc) continue;
                    const bool ds_wave_collective = in.fmt == Rdna2Format::DS &&
                        (in.opcode == 0x35 || in.opcode == 0x3d || in.opcode == 0x3e ||
                         in.opcode == kDsOpcodeBpermuteB32);
                    // SCC is architectural scalar control, so an SCC loop executes ordinary LDS
                    // effects uniformly within each guest wave just like the existing CountedLoop
                    // path. EXEC/VCC loops retain their per-invocation approximation: only ordinary
                    // read-only LDS in the proved stable post-barrier phase above is safe. Cross-lane
                    // DS collectives, writes/atomics, MBCNT, and a guest barrier remain unavailable.
                    const bool stable_lds_read = stable_post_barrier_reads &&
                        in.fmt == Rdna2Format::DS && in.opcode == 0x36 && !in.ds_gds;
                    if ((in.fmt == Rdna2Format::DS &&
                         ((L.condition != DivLoop::Condition::Scc && !stable_lds_read) ||
                          ds_wave_collective)) ||
                        (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) ||
                        (in.fmt == Rdna2Format::VOP3 &&
                         (in.opcode == 0x365 || in.opcode == 0x366))) { compute_ok = false; break; }
                }
                // Fail-visible marker for the per-invocation approximation's one known divergence: a
                // POST-loop read of an SGPR advanced inside an Exec-condition loop. With a lane-varying
                // bound, hardware advances in-loop scalars to the wave's MAX trip count while the
                // per-invocation lowering yields each invocation its own exit value. Uniform bounds
                // (all observed shapes) are exact. Diagnose loudly instead of silently diverging; if a
                // real kernel trips this AND has a varying bound, that is the evidence to revisit.
                if (compute_ok && L.condition == DivLoop::Condition::Exec && getenv("PROSPER_DBG")) {
                    std::set<int> lv, lsr;
                    loop_written_regs(ins, L.header_pc, L.backedge_pc, lv, lsr);
                    for (const auto& in : ins) {
                        if (in.is_end) break;
                        if (in.pc < L.exit_pc) continue;
                        for (uint32_t oi = 0; oi < in.n_src; ++oi)
                            if (in.src[oi].kind == OperandKind::SGPR && lsr.count((int)in.src[oi].value))
                                fprintf(stderr,
                                        "[compute-exec-loop] post-loop read of loop-advanced s%u at pc=%u "
                                        "(per-invocation value; wave max-trip on hardware)\n",
                                        (unsigned)in.src[oi].value, in.pc);
                    }
                }
            }
            if (!compute_ok) Ls.clear();   // unchanged behavior: the branch reaches emit_alu -> loud reject
        }
        bool cf_rejected = false;
        const std::vector<ForwardIf> Fs = detect_forward_ifs(ins, /*allow_vcc*/!b.is_compute, code, dwords, &safe,
                                                             Ls.empty() ? nullptr : &Ls, &cf_rejected,
                                                             /*compute_wave_branches*/b.is_compute,
                                                             b.diagnostic);

        // UE4's volume-lighting kernels combine nested EXEC loops with a backward VCC vote loop.
        // That shape is intentionally outside the narrow pattern structurizer above. Use the
        // dispatcher fallback only when there is unmistakably complex compute control flow and the
        // body is free of operations whose workgroup-uniform placement the dispatcher cannot prove.
        // Raw DS operations are wave-local memory effects and are valid in a case. Guest barriers
        // remain unsupported; MBCNT is hoisted into the dispatcher's uniform synchronized phase.
        // Existing straight-line, single-if, and recognized single-loop shaders keep their old path.
        size_t cfg_branches = 0;
        const bool graphics_cfg = b.is_fragment || b.is_vertex;
        bool cfg_has_backedge = false, cfg_dispatch_safe = b.is_compute || graphics_cfg;
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a)
                cfg_dispatch_safe = false;
            if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
                in.opcode != 0x03) {
                ++cfg_branches;
                if (branch_target(in) <= in.pc) cfg_has_backedge = true;
            }
        }
        // A lone exit + back-edge is the ordinary single-loop shape. Keep rejecting it when its
        // wave-uniformity proof fails; the dispatcher is reserved for genuinely nested/multi-branch CFGs.
        // The fallback is equally valid for an acyclic branch tree: Astro Bot's screen-space kernels
        // contain several nested forward EXEC/SCC early-outs but no back-edge. Requiring a back-edge
        // left those valid CFGs in the straight-line path, where their first branch rejected.
        const bool complex_compute_cfg = b.is_compute && cfg_dispatch_safe && cfg_branches > 2;
        const bool complex_graphics_cfg = graphics_cfg && cfg_dispatch_safe && cfg_branches > 2;
        const bool exact_compute_wave_cfg = b.is_compute &&
            std::any_of(Fs.begin(), Fs.end(), [](const ForwardIf& branch) {
                return branch.on_exec || branch.on_vcc;
            });
        // A structured shader may contain top-level guest-wave branches and workgroup-uniform barriers
        // between those regions (DOLL's title grading kernel and UE4's barrier-separated reductions).
        // The generic CFG dispatcher cannot contain guest barriers, but routing the entire shader there
        // is unnecessary: reduce each top-level branch with exact 32/64-lane scratch votes and retain the
        // structured emitter. Loops are not required; a sequence of forward top-level reductions has the
        // same safety argument. Neither a vote nor a guest barrier may be nested in a varying region.
        auto top_level_pc = [&](uint32_t pc) {
            for (const auto& parent : Fs) {
                // A proved workgroup-uniform region is entered or skipped by EVERY invocation of the
                // workgroup together (#1554), so it does not make a nested barrier or scratch vote
                // divergent. Only genuinely per-wave regions hide their contents from this test.
                if (parent.uniform_workgroup) continue;
                const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
                if (parent.branch_pc < pc && pc < parent_end)
                    return false;
            }
            for (const auto& loop : Ls)
                if (pc >= loop.header_pc && pc <= loop.backedge_pc)
                    return false;
            return true;
        };
        const bool barriers_are_top_level =
            std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
                return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a ||
                       top_level_pc(in.pc);
            });
        const bool structured_has_cross_lane_mbcnt =
            std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
                return in.fmt == Rdna2Format::VOP3 &&
                       (in.opcode == 0x365 || in.opcode == 0x366) &&
                       !(in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1);
            });
        const bool structured_wave_forward_ifs_ok =
            std::all_of(Fs.begin(), Fs.end(), [&](const ForwardIf& branch) {
                // An exact native guest-size subgroup performs the vote without synthesized
                // workgroup barriers, so nested wave branches are safe. Portable scratch votes
                // must remain top-level so every workgroup invocation reaches their barriers.
                return (!branch.on_exec && !branch.on_vcc) || b.native_subgroup_size ||
                       top_level_pc(branch.branch_pc);
            });
        const bool structured_compute_wave_cfg = exact_compute_wave_cfg && !cf_rejected &&
            barriers_are_top_level && !structured_has_cross_lane_mbcnt &&
            structured_wave_forward_ifs_ok;
        // PROSPER_DBG: name WHICH conjunct denied structured wave emission, and what that
        // actually costs THIS stream.
        //
        // When `exact_compute_wave_cfg` holds and this conjunction does not, the stream loses
        // structured wave emission. What happens instead is NOT one outcome, and an earlier revision
        // of this reporter asserted the loop-emulating one unconditionally. That was wrong in a way
        // worth recording, because the wrong case is not a corner:
        //
        //   `barriers_are_top_level == false` requires a barrier inside a ForwardIf/DivLoop region.
        //   Any barrier before `is_end` also clears `cfg_dispatch_safe`, so the gate below always
        //   takes its `!cfg_dispatch_safe` arm -- and `analyze_barrier_phased_compute` marks a
        //   branch that crosses a barrier invalid, so the phased retry does not fire either and the
        //   shader is REJECTED OUTRIGHT. A line claiming "emulates loops" would then describe loop
        //   emulation for a program that emitted no SPIR-V at all.
        //
        // The census that accompanied that revision could not have caught it: the conjuncts it
        // measured were the two that leave `cfg_dispatch_safe` alone, so every sampled line landed
        // in the one arm where the sentence happened to be true. A control drawn from the arm that
        // works cannot test the arm that does not.
        //
        // So the outcome is computed rather than assumed, and it is in the KEY as well as the
        // message. `emit_body` reaches this point for one program address from the whole stream, the
        // counted-loop prelude (dispatcher deliberately withheld), the post-loop suffix and per-phase
        // sub-streams; keying on (program, conjunct) alone lets whichever ran first suppress the
        // rest, so the line with the wrong consequence would win. That is the same shared-key
        // suppression fixed on `divloop_reject` (#2684) and not inherited here until review.
        //
        // The neighbouring `[compute-cfg]` line prints `structured_wave=` as a single bool, which
        // cannot name a conjunct at all.
        if (b.is_compute && exact_compute_wave_cfg && !structured_compute_wave_cfg &&
            getenv("PROSPER_DBG") && b.diagnostic.program_address != 0) {
            const char* outcome =
                !allow_cfg_dispatcher
                    ? "dispatcher withheld for this sub-stream; emission continues here"
                    : (!cfg_dispatch_safe
                           ? "dispatcher unsafe (guest barrier): phased retry, else the whole "
                             "program is rejected and emits nothing"
                           : "stream goes to the CFG dispatcher, which emulates loops");
            const std::pair<const char*, bool> conjuncts[] = {
                {"cf-rejected",                  !cf_rejected},
                {"barrier-not-top-level",        barriers_are_top_level},
                {"cross-lane-mbcnt",             !structured_has_cross_lane_mbcnt},
                {"nested-wave-forward-if",       structured_wave_forward_ifs_ok},
            };
            static std::mutex swr_mu;
            static std::set<std::tuple<uint64_t, std::string, std::string>> swr_seen;
            std::lock_guard<std::mutex> lk(swr_mu);
            // Clearing, not refusing: past the cap the reporter degrades into REPEATING lines rather
            // than dropping them. For a diagnostic, losing tidiness beats losing findings.
            if (swr_seen.size() >= 4096u) swr_seen.clear();
            for (const auto& c : conjuncts) {
                // EVERY false conjunct is named, not just the first. They are independent, so
                // fixing the first one reported can leave the decision unchanged -- a report that
                // stopped at one would send a reader to widen a guard that changes nothing.
                if (c.second) continue;
                if (!swr_seen.emplace(b.diagnostic.program_address, c.first, outcome).second)
                    continue;
                std::fprintf(stderr,
                             "[structured-wave-reject] program=0x%llx %s (%s)\n",
                             (unsigned long long)b.diagnostic.program_address, c.first, outcome);
            }
        }
        if (b.is_compute && cfg_branches && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[compute-cfg] branches=%zu backedge=%d dispatch_safe=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d exact_wave=%d "
                         "structured_wave=%d local=%u wave=%u\n",
                         cfg_branches, cfg_has_backedge, cfg_dispatch_safe, complex_compute_cfg,
                         Fs.size(), Ls.size(), cf_rejected, exact_compute_wave_cfg,
                         structured_compute_wave_cfg,
                         b.local_count, b.wave_size);
        if (graphics_cfg && cfg_branches && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[graphics-cfg] stage=%s branches=%zu backedge=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d\n",
                         b.is_fragment ? "fragment" : "vertex", cfg_branches,
                         cfg_has_backedge, complex_graphics_cfg, Fs.size(), Ls.size(), cf_rejected);
        if (allow_cfg_dispatcher && exact_compute_wave_cfg && !structured_compute_wave_cfg) {
            // Native Vulkan subgroup widths may be 8/16/32 while the guest wave is 32/64. A native
            // subgroupAny would let different pieces of one guest wave take different scalar edges.
            // The dispatcher performs the reduction through Workgroup scratch and synchronized common
            // phases. Top-level wave branches use the compact structured reduction above even without
            // a guest barrier; nested wave branches still need this dispatcher. If a guest barrier
            // makes that transformation unsafe, reject rather than silently changing the branch domain.
            if (!cfg_dispatch_safe) {
                const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
                if (phased.found && !phased.guarded)
                    return emit_body(
                        b, rs, ins, safe, rt, allow_exec_update, allow_smem,
                        exp_fn, code, dwords, &dead_masks, allow_cfg_dispatcher,
                        /*initial_dispatch_active=*/0, /*force_barrier_phases=*/true);
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-reject", "terminal",
                    "reason=exact-wave-dispatcher-unsafe guest-barrier=1");
                return false;
            }
            if (!emit_cfg_state_machine(b, rs, ins, safe, rt,
                                        allow_exec_update, allow_smem, exp_fn, code, dwords,
                                        initial_dispatch_active))
                return false;
            return true;
        }
        // Attempting the dispatcher is only recoverable while it has emitted NOTHING.
        // emit_cfg_state_machine writes the dispatcher loop's OpLoopMerge, its OpSelectionMerge and
        // the OpSwitch into b.code *before* it emits the per-case block bodies, and it can still
        // reject inside a case body (an unlowerable instruction, an unresolvable successor). Falling
        // through to the structurizer after that leaves the half-written construct in the module: the
        // switch and both merges survive, referencing case / default / continue / merge labels that
        // are never emitted. That is structurally invalid SPIR-V, and nothing downstream catches it —
        // `spirv-val` is a CI gate over representative modules, not over game shaders, so the module
        // reaches the driver, `spirv_to_nir` returns NULL, and RADV dereferences that NULL (#2396).
        // So the attempt is transactional: fall back only when the buffer is untouched, and reject
        // loudly once anything has been written.
        auto try_cfg_dispatcher = [&]() -> int {           // 1 = emitted, 0 = clean reject, -1 = partial
            const size_t checkpoint = b.code.size();
            if (emit_cfg_state_machine(b, rs, ins, safe, rt,
                                       allow_exec_update, allow_smem, exp_fn, code, dwords,
                                       initial_dispatch_active))
                return 1;
            if (b.code.size() == checkpoint) return 0;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[cfg-dispatcher-partial] rejected after emitting %zu words; failing the "
                             "shader instead of shipping a half-written dispatcher\n",
                             b.code.size() - checkpoint);
            return -1;
        };
        const bool portable_compute_dpp_ror8 = b.is_compute &&
            !b.native_subgroup_size &&
            std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
                return dpp_row_ror8_op(in) != DppRowRor8Op::None;
            });
        if (allow_cfg_dispatcher && portable_compute_dpp_ror8) {
            // The generic straight-line/structured emitter can use ROW_ROR only when the backend
            // guarantees one exact native guest wave. Otherwise the complete program must enter the
            // synchronized dispatcher so every invocation reaches both scratch barriers uniformly.
            if (!cfg_dispatch_safe) {
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-reject", "terminal",
                    "reason=portable-dpp-row-ror8-dispatcher-unsafe guest-barrier=1");
                return false;
            }
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            return false;
        }
        if (allow_cfg_dispatcher && complex_compute_cfg && (cf_rejected || Ls.empty())) {
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            if (emitted < 0) return false;
        }
        // In graphics, the SPIR-V invocation already represents one guest lane. Complex reducible
        // control flow therefore needs no workgroup vote: the dispatcher selects the next block from
        // this pixel/vertex's SCC, VCC, or EXEC bit. Keep ordinary structured shaders on their compact
        // SSA path and use the Function-variable fallback only after the narrow structurizer rejects.
        // Only one of these two blocks can run: complex_compute_cfg requires b.is_compute and
        // complex_graphics_cfg requires graphics_cfg (b.is_fragment || b.is_vertex), and a module is
        // one or the other. That exclusivity is what makes a clean reject (0) from the compute block
        // safe to fall through — it reaches this block only in a stage that cannot enter it. If the
        // stage predicates above ever stop being disjoint, this second attempt would run against
        // builder state the first had already touched, and the checkpoint would no longer describe an
        // untouched buffer.
        if (allow_cfg_dispatcher && complex_graphics_cfg && cf_rejected) {
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            if (emitted < 0) return false;
        }
        if (cf_rejected) Ls.clear();   // unmodeled CF somewhere: fall through to straight-line (loud reject)
        if (Fs.empty() && Ls.empty()) {
            wave_ok = true;   // straight-line: barriers are wave-uniform, so cross-lane mbcnt is safe here
            if (!emit_range(0, UINT32_MAX)) return false;   // loop-free: straight-line, unchanged behavior
            (void)safe_branches;
            return true;
        }
        // Structured uniform IFs (forward s_cbranch_scc*/vcc*), possibly SEQUENTIAL and/or NESTED
        // (detect_forward_ifs verified the region tree). Each if emits as OpSelectionMerge +
        // OpBranchConditional on the SCC/VCC bool, with an OpPhi per register written in the
        // conditional block that is live after the merge — the same phi machinery as the original
        // single-if path (which this generalizes 1:1: a single if takes exactly the old shape).
        // Recursion handles nesting: the then-block emitter re-enters for inner branches.
        // CONFIDENCE: MED-HIGH — guarded by the test suite + exec-diff; DOLL's two-vccz color-grade
        // PS and nested-vccz lighting PS are the motivating real shaders (#273).
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        size_t bi = 0;   // next unconsumed branch in Fs (pc order; recursion consumes nested ones)
        size_t li = 0;   // next unconsumed loop in Ls (pc order)
        const DivLoop* active_direct_wave_loop = nullptr;
        uint32_t* active_direct_wave_continue = nullptr;
        // `cont` = the pc control flows to after `hi` — the enclosing construct's merge chain. An
        // if/else whose merge ESCAPES the current region (the shared-outer-merge cascade) is legal
        // only when it targets exactly this continuation (no skipped instructions); anything else
        // rejects, fail-visible. Fragment EXEC/VCC branches vote over the enforced wave64; compute
        // uses its exact guest-wave reduction paths. Either may enter/leave with EXEC narrowed, and
        // EXEC is phi'd across the merge like any other value.
        std::function<bool(uint32_t, uint32_t, uint32_t)> emit_structured;
        // Emit one EXEC/VCC/SCC-exit loop (#273/#615/#1554) as structured SPIR-V. Same block shape as
        // the counted-loop path (hdr -> chk -> body -> cont -> hdr, exit chk->merge) with three
        // differences: (1) fragment votes the complete wave's EXEC/VCC after the header recompute
        // (other guarded stages consume this lane's bool); (2) the body is
        // emitted RECURSIVELY (nested forward-execz if regions live inside it); (3) per-lane MASKS
        // (VCC/EXEC/saved sreg_bool pairs) are loop-carried too, so each gets a header phi. At the
        // merge, a register written in the condition region keeps its exit-iteration (chk) value —
        // it dominates the merge — while body-written state takes the header phi (its value when the
        // exiting check ran); masks CREATED inside the loop are dropped at the merge (an SSA id from
        // inside the body does not dominate it — a later read then rejects loudly instead of
        // emitting invalid SPIR-V). Execution tests cover both wave-vote outcomes and direct breaks.
        std::function<bool(const DivLoop&)> emit_divloop = [&](const DivLoop& L) -> bool {
            // A loop-carried SGPR absent at the preheader receives a zero PHI input, so compact map
            // presence inside or after the loop cannot stand in for a scalar-lifetime MUST proof.
            rs.scalar_presence_has_no_placeholders = false;
            const bool entry_exec_narrowed = rs.exec_narrowed;
            std::set<int> cv, cs, condv, conds, scalar_may_writes;
            loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
            loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
            loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
            for (int reg : rs.sreg_bool_b32)
                if (scalar_may_writes.contains(reg)) return false;
            if (b.allow_b32_masks &&
                has_unpersisted_b32_mask_lifetime(
                    ins, L.header_pc, L.backedge_pc, rs))
                return false;
            const uint32_t preheader = b.cur_block;
            const uint32_t hdr = b.id(), chk = b.id(), body = b.id(), cont = b.id(), merge = b.id();
            b.emit_branch(hdr); b.emit_label(hdr);
            struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };  // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec,5=mask
            std::vector<PhiRec> phis;
            for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
            for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
            // A poisoned (0) SCC live-in degrades to bfalse (invalid as an SSA phi input; dead in
            // practice — the loop shapes re-produce SCC before any read).
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
            if (rs.vcc) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p); rs.exec = ph; phis.push_back({0, 4, ph, p}); }
            std::vector<int> mask_keys;                        // saved masks live at entry: loop-carried bools
            for (auto& kv : rs.sreg_bool) mask_keys.push_back(kv.first);
            std::sort(mask_keys.begin(), mask_keys.end());     // deterministic emission order
            for (int k : mask_keys) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.sreg_bool[k], preheader, p); rs.sreg_bool[k] = ph; phis.push_back({k, 5, ph, p}); }
            invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
            b.emit_loopmerge(merge, cont); b.emit_branch(chk); b.emit_label(chk);
            // An EXEC-governed loop predicates vector writes. VCC/SCC-governed loops branch on their
            // represented predicate but do not themselves change EXEC, matching the hardware body.
            if (L.condition == DivLoop::Condition::Exec) rs.exec_narrowed = true;
            // A top-tested condition region is branch-free. For a bottom-tested EXEC loop the
            // same interval is the complete do-while body and may contain validated nested loops;
            // recurse so those children keep their own structured merges before the latch test.
            const uint32_t condition_entry_scc = rs.scc;
            if (L.bottom_tested) {
                if (!emit_structured(
                        L.header_pc, L.exit_branch_pc, L.exit_branch_pc)) return false;
            } else if (!emit_range(L.header_pc, L.exit_branch_pc)) {
                return false;
            }
            // A canonical SCC loop must compute a fresh representable scalar predicate on every
            // header visit. Reusing the header phi would admit stale SCC, while a B64 wave-mask
            // producer poisons rs.scc to zero; both remain fail-visible.
            if (L.condition == DivLoop::Condition::Scc &&
                (!rs.scc || rs.scc == condition_entry_scc))
                return false;
            // chk-end snapshots: the exit path flows THROUGH this block, so these dominate the merge.
            std::unordered_map<int, uint32_t> condv_val, conds_val;
            for (int r : condv) condv_val[r] = vget(r);
            for (int r : conds) conds_val[r] = sget(r);
            const uint32_t exec_chk = rs.exec, vcc_chk = rs.vcc, scc_chk = rs.scc;
            const std::unordered_map<int, uint32_t> bool_chk = rs.sreg_bool;
            uint32_t loop_cond = L.condition == DivLoop::Condition::Exec ? rs.exec
                               : L.condition == DivLoop::Condition::Vcc ? rs.vcc : rs.scc;
            if (!loop_cond) return false;
            if (!L.continue_on_set)
                loop_cond = b.logical_not(loop_cond);
            // EXECZ/VCCZ are scalar wave decisions. Keeping every fragment invocation in the loop
            // until the complete guest wave becomes empty makes scalar state and nested wave votes
            // exact; vector writes remain predicated by the per-lane EXEC bool.
            if (b.is_fragment && L.condition != DivLoop::Condition::Scc)
                loop_cond = b.fragment_wave_any(loop_cond);
            const uint32_t chk_end = b.cur_block;
            b.emit_condbranch(loop_cond, body, merge);         // canonical exit: branch on continue predicate
            while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;
            if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;   // consume the exit branch
            b.emit_label(body);
            // Body (recursive: nested if regions + breaks); ends just before the back-edge.
            uint32_t direct_wave_continue = b.btrue();
            const DivLoop* prior_direct_wave_loop = active_direct_wave_loop;
            uint32_t* prior_direct_wave_continue = active_direct_wave_continue;
            active_direct_wave_loop = &L;
            active_direct_wave_continue = &direct_wave_continue;
            const bool body_ok = L.bottom_tested || emit_structured(
                L.exit_branch_pc + 1, L.backedge_pc, L.backedge_pc);
            active_direct_wave_loop = prior_direct_wave_loop;
            active_direct_wave_continue = prior_direct_wave_continue;
            if (!body_ok) return false;
            const bool body_exec_narrowed = rs.exec_narrowed;
            if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;      // consume the back-edge
            const uint32_t body_end = b.cur_block;
            // An unconditional back-edge can re-enable EXEC at the header. When an interior EXECZ
            // targets the loop exit, send the inactive invocation straight to the loop merge; the
            // ordinary path still reaches the continue block and patches the loop-carried phis.
            uint32_t continue_condition = direct_wave_continue;
            if (L.direct_exec_breaks) continue_condition = b.land(continue_condition, rs.exec);
            if (L.direct_exec_breaks || L.direct_wave_breaks)
                b.emit_condbranch(continue_condition, cont, merge);
            else
                b.emit_branch(cont);
            b.emit_label(cont);
            for (auto& pr : phis) {
                uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                            : pr.dom == 1 ? sget(pr.reg)
                            : pr.dom == 2 ? rs.scc
                            : pr.dom == 3 ? rs.vcc
                            : pr.dom == 4 ? rs.exec
                            : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                if (!nv && pr.dom == 3) return false;
                if (!nv && pr.dom == 2) nv = b.bfalse();
                b.patch_phi(pr.patch, nv, cont);
            }
            b.emit_branch(hdr);
            b.emit_label(merge);
            for (auto& pr : phis) {
                if (pr.dom == 3 && (!vcc_chk || !rs.vcc)) return false;
                uint32_t chk_value = pr.dom == 0 ? (condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi)
                                   : pr.dom == 1 ? (conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi)
                                   : pr.dom == 2 ? (scc_chk ? scc_chk : b.bfalse())
                                   : pr.dom == 3 ? vcc_chk
                                   : pr.dom == 4 ? exec_chk
                                   : (bool_chk.count(pr.reg) ? bool_chk.at(pr.reg) : pr.phi);
                uint32_t body_value = pr.dom == 0 ? vget(pr.reg)
                                    : pr.dom == 1 ? sget(pr.reg)
                                    : pr.dom == 2 ? (rs.scc ? rs.scc : b.bfalse())
                                    : pr.dom == 3 ? rs.vcc
                                    : pr.dom == 4 ? rs.exec
                                    : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                const uint32_t merged = (L.direct_exec_breaks || L.direct_wave_breaks) &&
                                                chk_value != body_value
                    ? b.emit_phi_2way(pr.dom <= 1 ? b.t_u32 : b.t_bool,
                                      chk_value, chk_end, body_value, body_end)
                    : chk_value;
                if (pr.dom == 0)      rs.vreg[pr.reg] = merged;
                else if (pr.dom == 1) rs.sreg[pr.reg] = merged;
                else if (pr.dom == 2) rs.scc = merged;
                else if (pr.dom == 3) rs.vcc = merged;
                else if (pr.dom == 4) rs.exec = merged;
                else                  rs.sreg_bool[pr.reg] = merged;
            }
            // Masks CREATED inside the loop: their ids do not dominate the merge — drop them.
            for (auto it = rs.sreg_bool.begin(); it != rs.sreg_bool.end();) {
                if (!std::binary_search(mask_keys.begin(), mask_keys.end(), it->first)) {
                    rs.sreg_bool_narrowed.erase(it->first);
                    rs.sreg_bool_b32.erase(it->first);
                    it = rs.sreg_bool.erase(it);
                } else ++it;
            }
            // An execz exit leaves this lane inactive until the compiled restore. A vccz exit leaves
            // EXEC unchanged; preserve any narrowing that existed on entry or occurred in the body.
            rs.exec_narrowed = L.condition == DivLoop::Condition::Exec
                ? true : (entry_exec_narrowed || body_exec_narrowed);
            return true;
        };
        emit_structured =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool {
            for (;;) {
                const uint32_t next_br = (bi < Fs.size() && Fs[bi].branch_pc < hi) ? Fs[bi].branch_pc : hi;
                const uint32_t next_lp = (li < Ls.size() && Ls[li].header_pc < hi) ? Ls[li].header_pc : hi;
                if (next_lp < next_br) {                     // a loop begins before the next if
                    if (!emit_range(lo, next_lp)) return false;
                    const DivLoop& L = Ls[li++];
                    if (!emit_divloop(L)) return false;
                    lo = L.exit_pc;
                    continue;
                }
                if (!emit_range(lo, next_br)) return false;
                if (next_br == hi) return true;
                const ForwardIf F = Fs[bi++];
                if (idx < ins.size() && ins[idx].pc == F.branch_pc) ++idx;   // skip the branch itself
                // Arm merges synthesize zero for a scalar absent on either predecessor. Keep the
                // branch-free prefix eligible for exact path-local proofs, then turn them off for
                // both arms and every successor of this construct.
                rs.scalar_presence_has_no_placeholders = false;
                // scc0/vccz/execz: branch (skip block) taken when the flag==0 → the block runs when
                // flag!=0; scc1/vccnz are the inverse. Compute and fragment reduce per-invocation
                // mask bits to the architecture's wave-wide "any lane active" predicate.
                // A poisoned SCC (0: last written by a 64-bit mask op) cannot condition a real
                // structured if — reject (the ISA-audit #879 stale-SCC consumer).
                if (!F.on_exec && !F.on_vcc && !rs.scc) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "poisoned SCC at branch pc=%u", F.branch_pc);
                    return false;
                }
                if (F.on_vcc && !rs.vcc) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "missing VCC at branch pc=%u", F.branch_pc);
                    return false;
                }
                // Compute VCC/EXEC branches normally return through the exact guest-wave dispatcher.
                // The narrow structured-wave path above accepts only top-level vote sites, where all
                // workgroup invocations may participate in the scratch barriers uniformly.
                if (b.is_compute && (F.on_exec || F.on_vcc) && !structured_compute_wave_cfg) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "unavailable wave vote at branch pc=%u", F.branch_pc);
                    return false;
                }
                uint32_t cond_reg = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
                if (b.is_compute && (F.on_exec || F.on_vcc))
                    cond_reg = b.native_subgroup_size
                        ? b.native_wave_any(cond_reg)
                        : b.guest_wave_any(cond_reg);
                else if (b.is_fragment && (F.on_exec || F.on_vcc))
                    cond_reg = b.fragment_wave_any(cond_reg);
                uint32_t exec_cond = F.on_scc0 ? cond_reg : b.bsel(cond_reg, b.bfalse(), b.btrue());
                if (active_direct_wave_loop && active_direct_wave_continue &&
                    active_direct_wave_loop->direct_wave_breaks && F.on_vcc &&
                    std::find(active_direct_wave_loop->break_pcs.begin(),
                              active_direct_wave_loop->break_pcs.end(), F.branch_pc) !=
                        active_direct_wave_loop->break_pcs.end())
                    *active_direct_wave_continue =
                        b.land(*active_direct_wave_continue, exec_cond);
                const uint32_t preblock = b.cur_block;      // block holding the OpBranchConditional
                if (!F.has_else) {
                    std::set<int> ifv, ifs;
                    loop_written_regs(ins, F.branch_pc + 1, F.target_pc, ifv, ifs);
                    std::unordered_map<int,uint32_t> pre_v, pre_s;
                    for (int r : ifv) pre_v[r] = vget(r);
                    for (int r : ifs) pre_s[r] = sget(r);
                    uint32_t pre_scc = rs.scc, pre_vcc = rs.vcc, pre_exec = rs.exec;
                    const bool pre_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> pre_bool = rs.sreg_bool;   // mask-domain snapshot
                    const auto pre_bool_b32 = rs.sreg_bool_b32;
                    uint32_t thenL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, mergeL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.target_pc, F.target_pc)) return false;
                    const uint32_t thenEnd = b.cur_block;   // last block of the then-body (nested ifs move it)
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : ifv) then_v[r] = vget(r);
                    for (int r : ifs) then_s[r] = sget(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    // The skipped edge retains the entry-time physical-word lifetime. A B32 mask
                    // created or invalidated only in the taken arm therefore needs a validity phi,
                    // which this narrow merge does not represent. Reject rather than attach the
                    // taken arm's marker to the synthesized bool value on both paths.
                    std::set<int> dead_b32_at_merge;
                    std::set_symmetric_difference(
                        rs.sreg_bool_b32.begin(), rs.sreg_bool_b32.end(),
                        pre_bool_b32.begin(), pre_bool_b32.end(),
                        std::inserter(dead_b32_at_merge, dead_b32_at_merge.end()));
                    const bool dead_domain_difference =
                        std::all_of(dead_b32_at_merge.begin(), dead_b32_at_merge.end(),
                                    [&](int reg) {
                                        return sgpr_dead_at_merge(ins, F.target_pc, reg);
                                    });
                    if (!dead_domain_difference) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-struct-reject", "terminal",
                            "live b32 mask domain differs across branch pc=%u merge=%u",
                            F.branch_pc, F.target_pc);
                        return false;
                    }
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : ifv) rs.vreg[r] = b.emit_phi_2way(b.t_u32,  pre_v[r], preblock, then_v[r], thenEnd);
                    for (int r : ifs) rs.sreg[r] = b.emit_phi_2way(b.t_u32,  pre_s[r], preblock, then_s[r], thenEnd);
                    if (then_scc != pre_scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, pre_scc ? pre_scc : b.bfalse(), preblock,
                                                 then_scc ? then_scc : b.bfalse(), thenEnd);
                    if (then_vcc != pre_vcc)
                        rs.vcc = !pre_vcc || !then_vcc ? 0u : b.emit_phi_2way(
                            b.t_bool, pre_vcc, preblock, then_vcc, thenEnd);
                    // EXEC changed inside the arm (saveexec / v_cmpx / restore): merge it like any value.
                    // Narrowed-ness is sticky (either edge narrowed → post-merge writes stay predicated).
                    if (then_exec != pre_exec) rs.exec = b.emit_phi_2way(b.t_bool, pre_exec, preblock, then_exec, thenEnd);
                    rs.exec_narrowed = pre_narrowed || then_narrowed;
                    // A saved MASK (sreg_bool) created or changed inside the block must dominate both
                    // merge predecessors. A newly-created per-lane mask is false on the skipped edge.
                    for (auto& kv : rs.sreg_bool) {
                        auto p = pre_bool.find(kv.first);
                        const uint32_t before = p != pre_bool.end() ? p->second : b.bfalse();
                        if (before != kv.second) {
                            kv.second = b.emit_phi_2way(b.t_bool, before, preblock, kv.second, thenEnd);
                            rs.sreg_bool_narrowed[kv.first] = true;   // conservative: provenance now mixed
                        }
                    }
                    // A differing physical-word domain needs no validity phi when that word is
                    // provably overwritten before every post-merge read. Drop its stale typed view
                    // on both synthesized paths; the later defining instruction recreates the
                    // appropriate scalar or mask lifetime.
                    for (int reg : dead_b32_at_merge) {
                        rs.sreg_bool_b32.erase(reg);
                        rs.sreg_bool.erase(reg);
                        rs.sreg_bool_narrowed.erase(reg);
                        if (reg == 106) rs.vcc = 0;
                    }
                    lo = F.target_pc;   // continue after the merge (further sequential ifs handled here)
                } else {
                    // IF/ELSE: then = [branch_pc+1, sb_pc) (its s_branch terminator is consumed);
                    // else = [target_pc, merge). A merge escaping this region must be exactly the
                    // enclosing continuation `cont` (the cascade shape) — the else-arm then runs to
                    // `hi` and the merge coincides with the region end.
                    uint32_t else_hi = F.merge_pc;
                    if (F.merge_pc >= hi) {
                        if (F.merge_pc != cont && F.merge_pc != hi) {
                            log_recompile_diagnostic(
                                b.diagnostic, "compute-struct-reject", "terminal",
                                "escaping merge pc=%u merge=%u region=%u continuation=%u",
                                F.branch_pc, F.merge_pc, hi, cont);
                            return false;
                        }
                        else_hi = hi;
                    }
                    const RegState pre = rs;                // FULL snapshot: the else-arm re-runs from it
                    std::set<int> wv, ws;                   // regs written in EITHER arm
                    loop_written_regs(ins, F.branch_pc + 1, F.sb_pc, wv, ws);
                    loop_written_regs(ins, F.target_pc, else_hi, wv, ws);
                    uint32_t thenL = b.id(), elseL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, elseL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.sb_pc, F.merge_pc)) return false;
                    if (idx < ins.size() && ins[idx].pc == F.sb_pc) ++idx;   // consume the arm's s_branch
                    const uint32_t thenEnd = b.cur_block;
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : wv) then_v[r] = vget(r);
                    for (int r : ws) then_s[r] = sget(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> then_bool = rs.sreg_bool;
                    const auto then_bool_b32 = rs.sreg_bool_b32;
                    const auto then_written = rs.sreg_written;
                    b.emit_branch(mergeL);
                    rs = pre;                               // else-arm starts from the pre-branch state
                    b.emit_label(elseL);
                    if (!emit_structured(F.target_pc, else_hi, F.merge_pc)) return false;
                    const uint32_t elseEnd = b.cur_block;
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : wv) { uint32_t ev = vget(r);
                        if (then_v[r] != ev) rs.vreg[r] = b.emit_phi_2way(b.t_u32, then_v[r], thenEnd, ev, elseEnd); }
                    for (int r : ws) { uint32_t es = sget(r);
                        if (then_s[r] != es) rs.sreg[r] = b.emit_phi_2way(b.t_u32, then_s[r], thenEnd, es, elseEnd); }
                    if (then_scc != rs.scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), thenEnd,
                                                 rs.scc ? rs.scc : b.bfalse(), elseEnd);
                    if (then_vcc != rs.vcc)
                        rs.vcc = !then_vcc || !rs.vcc ? 0u : b.emit_phi_2way(
                            b.t_bool, then_vcc, thenEnd, rs.vcc, elseEnd);
                    if (then_exec != rs.exec) rs.exec = b.emit_phi_2way(b.t_bool, then_exec, thenEnd, rs.exec, elseEnd);
                    rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
                    rs.sreg_written.insert(then_written.begin(), then_written.end());
                    for (int reg : then_written) rs.sreg_input.erase(reg);
                    if (then_bool_b32 != rs.sreg_bool_b32) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-struct-reject", "terminal",
                            "b32 mask domain differs across if/else pc=%u merge=%u",
                            F.branch_pc, F.merge_pc);
                        return false;
                    }
                    // Merge the UNION of mask keys. A mask created in only one arm is false in the
                    // other arm; leaving that arm-local SSA id live after the merge is invalid SPIR-V.
                    std::set<int> bool_keys;
                    for (const auto& kv : then_bool) bool_keys.insert(kv.first);
                    for (const auto& kv : rs.sreg_bool) bool_keys.insert(kv.first);
                    for (int key : bool_keys) {
                        auto t = then_bool.find(key);
                        auto e = rs.sreg_bool.find(key);
                        const uint32_t tv = t != then_bool.end() ? t->second : b.bfalse();
                        const uint32_t ev = e != rs.sreg_bool.end() ? e->second : b.bfalse();
                        rs.sreg_bool[key] = tv == ev ? tv
                            : b.emit_phi_2way(b.t_bool, tv, thenEnd, ev, elseEnd);
                        if (tv != ev) rs.sreg_bool_narrowed[key] = true;
                    }
                    lo = else_hi;   // continue after the merge (== hi for the escaping-cascade shape)
                }
            }
        };
        // Every scalar branch and loop condition is subgroup-uniform in the fragment shell (SCC is
        // scalar already; EXECZ/VCCZ use fragment_wave_any), so native wave operations in any
        // structured arm observe the complete guest wave.
        wave_ok = b.is_fragment;
        if (!emit_structured(0, UINT32_MAX, UINT32_MAX)) {
            if (getenv("PROSPER_DBG")) {
                const uint32_t next_pc = idx < ins.size() ? ins[idx].pc : UINT32_MAX;
                const uint32_t next_if = bi < Fs.size() ? Fs[bi].branch_pc : UINT32_MAX;
                const uint32_t next_loop = li < Ls.size() ? Ls[li].header_pc : UINT32_MAX;
                log_recompile_diagnostic(
                    b.diagnostic, "compute-struct-reject", "consequent",
                    "structured emission stopped next-pc=%u next-if=%u next-loop=%u",
                    next_pc, next_if, next_loop);
            }
            return false;
        }
    }
    (void)safe_branches;
    return true;
}

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt, uint32_t lds_bytes,
                                     uint32_t compute_pgm_rsrc1) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // This shell publishes register state after the structured region, so both terminal arms can
    // converge in the host SPIR-V without inventing further guest execution. Graphics exports keep
    // separate side-effect bookkeeping and remain on the conservative reject path for this shape.
    (void)extend_terminating_if_else(code, dwords, ins);
    // The synthetic test shell is one Wave64 workgroup, matching the live GTA dispatch.
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, {}, true, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.compute_pgm_rsrc1 = compute_pgm_rsrc1;
    // Size the LDS array from the shader's real allocation when known (#130): bytes -> dwords, at
    // least the ds ops need, clamped to the RDNA2 64 KB (16384-dword) max. 0 keeps the 16 KB default.
    if (lds_bytes) {
        uint32_t dw = (lds_bytes + 3) / 4;
        b.lds_dwords = dw > 16384u ? 16384u : (dw ? dw : 1u);
    }
    b.begin(num_inputs ? num_inputs : 1, rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/true,
                   [](RegState&, const Rdna2Inst&){ return false; }, code, dwords,
                   nullptr, true, 0, false, lds_fminmax_synchronization.needs_dispatcher)) return {};
    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's prior
    // value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

std::vector<uint32_t> recompile_compute(const uint32_t* code, size_t dwords,
                                        const ShaderResourceTable* rt,
                                        const ComputeShaderConfig& config,
                                        RecompileDiagnosticContext diagnostic) {
    const bool has_null_guarded_raw_store = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_proven_null_guarded_raw_store);
    const bool has_nullable_output_raw_buffer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_nullable_raw_buffer_marker_candidate);
    const bool has_selected_sbuffer_descriptor = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_selected_sbuffer_marker_candidate);
    const bool has_gta5_packed_pointer = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_packed_pointer_marker_candidate);
    const bool has_indirect_pointer_relocation = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_indirect_pointer_relocation_marker_candidate);
    const bool has_gta5_cf9200_no_backing = rt &&
        std::any_of(rt->resources.begin(), rt->resources.end(),
                    is_gta5_cf9200_no_backing_marker_candidate);
    const ShaderResource* selected_sbuffer_descriptor =
        has_selected_sbuffer_descriptor ? rt->by_fetch_pc(153u) : nullptr;
    // A resource table is externally constructible and can outlive the shader bytes or dispatch
    // that produced it. Re-establish the complete static guard and dynamic null-entry contract at
    // the final translation boundary before any marker is permitted to erase a real store.
    if (has_null_guarded_raw_store &&
        !rdna2_gta5_null_guarded_raw_store_dispatch(
            code, dwords, config.user_sgprs.data(), config.user_sgprs.size()))
        return {};
    if (has_nullable_output_raw_buffer &&
        !rdna2_gta5_nullable_output_dispatch(code, dwords, config, *rt))
        return {};
    if (has_selected_sbuffer_descriptor &&
        !rdna2_gta5_selected_sbuffer_dispatch(code, dwords, config, *rt))
        return {};
    if (has_gta5_packed_pointer &&
        !rdna2_gta5_packed_pointer_dispatch(code, dwords, config, *rt))
        return {};
    IndirectPointerRelocationProof indirect_pointer_proof;
    IndirectBufferRelocationInfo indirect_pointer_info;
    if (has_indirect_pointer_relocation &&
        !validate_rdna2_indirect_pointer_relocations(
            code, dwords, config, *rt,
            &indirect_pointer_proof, &indirect_pointer_info))
        return {};
    if (has_gta5_cf9200_no_backing &&
        !rdna2_gta5_cf9200_no_backing_dispatch(code, dwords, config, *rt))
        return {};
    const uint32_t local_x = std::max(1u, config.local_x);
    const uint32_t local_y = std::max(1u, config.local_y);
    const uint32_t local_z = std::max(1u, config.local_z);
    const uint32_t wave_size = config.wave_size == 32 ? 32u : 64u;
    const uint64_t local_count = static_cast<uint64_t>(local_x) * local_y * local_z;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    // A dispatch-scoped proven-null BVH can collapse an exact no-hit exit and fully matched empty-stack
    // traversal cycle before generic shader-byte constant folding. Resource identity (including null
    // marker + fetch PC) is already part of the compute module cache key, so a later non-null dispatch
    // receives a distinct module.
    (void)rdna2_specialize_proven_null_bvh_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_zero_record_execz_paths(ins, rt, config.wave_size);
    (void)rdna2_specialize_shader_constant_branches(ins);
    // See recompile_valu: compute has no branch-external EXP state, so the common host-shell merge is
    // only a place to finish the invocation after either guest arm has terminated.
    (void)extend_terminating_if_else(code, dwords, ins);
    LdsFminmaxSynchronization lds_fminmax_synchronization;
    if (!prepare_lds_fminmax_synchronization(
            ins, diagnostic, local_count <= wave_size, &lds_fminmax_synchronization))
        return {};
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    SpirvCompute b;
    b.atomicized_lds_store_pcs = lds_fminmax_synchronization.atomicized_store_pcs;
    b.diagnostic = diagnostic;
    b.gta5_selected_sbuffer_dispatch_validated = has_selected_sbuffer_descriptor;
    if (selected_sbuffer_descriptor && has_selected_sbuffer_descriptor)
        b.gta5_selected_sbuffer_soffset = selected_sbuffer_descriptor->selected_sbuffer_soffset;
    b.gta5_cf9200_no_backing_dispatch_validated = has_gta5_cf9200_no_backing;
    b.indirect_buffer_dispatch_validated = has_gta5_packed_pointer;
    if (has_gta5_packed_pointer) {
        const ShaderResource* packed = rt->by_fetch_pc(kGta5PackedPointerSourcePc);
        if (!packed || !is_gta5_packed_pointer_resource(*packed)) return {};
        b.indirect_buffer_binding = packed->binding;
        b.indirect_buffer_source_bytes = static_cast<uint32_t>(packed->size);
        b.indirect_buffer_slot_count = packed->indirect_buffer_slot_count;
        b.indirect_buffer_contract_tag = packed->indirect_buffer_contract_tag;
        b.indirect_buffer_header_bytes = packed->indirect_buffer_header_bytes;
        b.indirect_buffer_slot_bytes = packed->indirect_buffer_slot_bytes;
        const ShaderResource* atomic = rt->by_fetch_pc(kGta5PackedPointerAtomicSourcePc);
        if (!atomic || atomic->size != kGta5PackedPointerAtomicBindingBytes ||
            (atomic->gpu_addr & 7u) != 0u)
            return {};
        b.indirect_buffer_atomic_binding = atomic->binding;
        b.indirect_buffer_atomic_byte_offset = kGta5PackedPointerAtomicByteOffset;
    }
    if (has_indirect_pointer_relocation) {
        const ShaderResource* relocated = rt->by_fetch_pc(
            indirect_pointer_proof.source_fetch_pc);
        const auto& marker = relocated
            ? relocated->indirect_pointer_relocation
            : IndirectPointerRelocationBinding{};
        const uint32_t expected_carrier_version =
            indirect_pointer_proof.bound_kind ==
                    IndirectPointerBoundKind::StaticFootprint
                ? kIndirectPointerStaticFootprintLayout.version
                : indirect_pointer_proof.bound_kind ==
                        IndirectPointerBoundKind::DescriptorRange
                    ? kIndirectPointerDescriptorRangeLayout.version
                    : 0u;
        const uint64_t record_directory_offset =
            static_cast<uint64_t>(indirect_pointer_info.source_bytes) +
            kIndirectBufferRelocationHeaderBytes;
        const uint64_t segment_directory_offset = record_directory_offset +
            static_cast<uint64_t>(indirect_pointer_info.records.size()) *
                kIndirectBufferRelocationRecordBytes;
        if (!relocated || !is_indirect_pointer_relocation_resource(*relocated) ||
            !expected_carrier_version || marker.carrier_version != expected_carrier_version ||
            marker.record_count != indirect_pointer_info.records.size() ||
            marker.record_count != indirect_pointer_proof.record_count ||
            marker.segment_count != indirect_pointer_info.segments.size() ||
            indirect_pointer_info.source_bytes != relocated->size ||
            record_directory_offset > UINT32_MAX ||
            segment_directory_offset > UINT32_MAX ||
            marker.segment_directory_byte_offset != segment_directory_offset)
            return {};
        b.indirect_pointer_proof = &indirect_pointer_proof;
        b.indirect_pointer_binding = relocated->binding;
        b.indirect_pointer_source_bytes = indirect_pointer_info.source_bytes;
        b.indirect_pointer_record_count = marker.record_count;
        b.indirect_pointer_record_directory_byte_offset =
            static_cast<uint32_t>(record_directory_offset);
        b.indirect_pointer_segment_count = marker.segment_count;
        b.indirect_pointer_segment_directory_byte_offset =
            marker.segment_directory_byte_offset;
        b.indirect_pointer_payload_byte_offset =
            indirect_pointer_info.payload_byte_offset;
        b.indirect_pointer_carrier_bytes = marker.binding_bytes;
        b.indirect_pointer_source_stride = indirect_pointer_proof.source_stride;
        b.indirect_pointer_source_pointer_byte_offset =
            indirect_pointer_proof.pointer_byte_offset;
    }
    if (config.lds_bytes) {
        uint32_t dw = (config.lds_bytes + 3) / 4;
        b.lds_dwords = std::min(16384u, std::max(1u, dw));
    }
    const bool has_partial_workgroup = config.threads_x % local_x != 0 ||
                                       config.threads_y % local_y != 0 ||
                                       config.threads_z % local_z != 0;
    const BarrierPhasedCompute barrier_phases = analyze_barrier_phased_compute(ins);
    const bool partial_barrier_phases = config.exact_thread_extent && has_partial_workgroup &&
        barrier_phases.found && !barrier_phases.guarded;
    const bool exact_partial_dispatcher = config.exact_thread_extent &&
        has_partial_workgroup && (b.gta5_selected_sbuffer_dispatch_validated ||
                                  b.indirect_buffer_dispatch_validated ||
                                  has_indirect_pointer_relocation);
    b.native_subgroup_size = config.native_subgroup_size == wave_size &&
        local_count <= UINT32_MAX && local_count % wave_size == 0 ? wave_size : 0u;
    // A partial guest wave needs the portable dispatcher's per-lane ACTIVE bit. Native subgroup
    // operations cannot be entered by only the real prefix of the final host subgroup.
    if (partial_barrier_phases || exact_partial_dispatcher)
        b.native_subgroup_size = 0;
    // PROSPER_DBG: report the inputs to that decision, not just its outcome (#2429).
    //
    // Every wave-width-dependent lowering in this file gates on `b.native_subgroup_size` -- the
    // VCC-as-scalar-data path at :5604 most consequentially, since when it is 0 the guest's
    // `s_add_u32 sN, sM, vcc_lo` never resolves, the descriptor never lands, and the draw is
    // skipped. Nothing printed any of this, under any variable, so "was that path active on this
    // device?" could only be inferred from the adapter's advertised width.
    //
    // `config.native_subgroup_size` is NOT the adapter's advertised width -- it is the OUTPUT of
    // select_native_compute_subgroup_size() (gpu_executor.cpp), an adoption decision with THREE
    // `return 0` sites comprising 22 clauses -- 25 if `adoptable`'s four ANDed device checks are
    // counted individually, which is defensible since each is independently sufficient. It spans
    // device features, queue support, workgroup limits and the dispatch's own dimensions, and
    // yields 0 when it declines. Zero therefore means "no native width was adopted", NOT "the
    // device is narrower than wave_size", and the three cases below are distinguished for that
    // reason.
    //
    // Counted rather than estimated, because two lanes published two different guesses at it on the
    // same day (#2483 "~14", #2484 "roughly eight") and neither had derived the number.
    //
    // This line dedupes on the exact input tuple, so it answers "which combinations exist" cheaply --
    // a couple of lines for a whole boot. For a per-dispatch CENSUS (how many dispatches fall in
    // each category, which this instrument's dedupe destroys by design) use PROSPER_SUBGROUP_LOG in
    // gpu_executor.cpp instead.
    //
    // That inference is WRONG, and reporting only the effective value would preserve the error:
    // the expression above is zero for THREE independent reasons -- the device width not matching
    // `wave_size`, an implausible `local_count`, or a workgroup that is not a whole number of waves
    // (`local_count % wave_size`). A dispatch with a partial final wave disables the path on an
    // adapter whose width matches perfectly. #2429 attributes it entirely to the first cause, and
    // that is checkable only if all three inputs are printed.
    //
    // Deduplicated on the exact tuple rather than rate-limited, because the interesting event is a
    // DISTINCT combination appearing, not the hundredth repeat of one -- and a kernel that disables
    // the path for a different reason than its predecessors is exactly what a rate limit would drop.
    if (getenv("PROSPER_DBG")) {
        static std::mutex mx;
        // Keyed on the EXACT inputs, local_count included. An earlier revision packed
        // `local_count % wave_size` instead, which collapsed dispatches that differ only in
        // workgroup shape -- and the line prints `local=`, so one row then named whichever
        // instance arrived first and stood silently for the rest. Measured on GTA V's six
        // native=0 shapes, that key produced three rows, one of which represented 1024, 256,
        // 256 and 256 while printing only 1024 -- and the 256-wide ones were the multi-wave
        // case that mattered. A diagnostic may aggregate, but it must not name one member of
        // a bucket as though it were the bucket.
        static std::set<std::tuple<uint32_t, uint32_t, uint64_t, uint32_t>> seen;
        std::lock_guard<std::mutex> lk(mx);
        if (seen.insert(std::make_tuple(config.native_subgroup_size, wave_size,
                                        local_count, b.native_subgroup_size)).second)
            std::fprintf(stderr,
                         "[subgroup-width] device=%u wave=%u local=%llu local%%wave=%llu -> "
                         "native_subgroup_size=%u (%s)\n",
                         config.native_subgroup_size, wave_size,
                         (unsigned long long)local_count,
                         (unsigned long long)(local_count % (wave_size ? wave_size : 1u)),
                         b.native_subgroup_size,
                         b.native_subgroup_size
                             ? "width-dependent lowerings ENABLED"
                             : (partial_barrier_phases
                                    ? "DISABLED: partial barrier phases require the portable "
                                      "dispatcher"
                                : (config.native_subgroup_size == 0
                                    ? "DISABLED: no native width adopted -- "
                                      "select_native_compute_subgroup_size() declined"
                                    : (config.native_subgroup_size != wave_size
                                           ? "DISABLED: adopted width != wave_size"
                                           : (local_count > UINT32_MAX
                                                  ? "DISABLED: local_count exceeds the plausibility "
                                                    "guard"
                                                  : "DISABLED: workgroup is not a whole number "
                                                    "of waves")))));
    }
    b.native_storage_format_support = config.native_storage_format_support;
    b.storage_buffer_int64_atomics = config.storage_buffer_int64_atomics;
    b.packed_r11_storage = config.packed_r11_storage;
    b.compute_pgm_rsrc1 = config.compute_pgm_rsrc1;
    b.begin(1, rt, local_x, local_y, local_z, wave_size,
            static_cast<uint32_t>(config.user_sgprs.size()));
    b.allow_b32_masks = wave_size == 32;
    if (has_indirect_pointer_relocation &&
        indirect_pointer_proof.bound_kind ==
            IndirectPointerBoundKind::DescriptorRange)
        b.declare_indirect_pointer_descriptor_capture();
    b.declare_guest_scratch(scratch);
    uint32_t initial_dispatch_active = 0;
    if (partial_barrier_phases || exact_partial_dispatcher)
        initial_dispatch_active = b.invocation_within_extent(
            config.threads_x, config.threads_y, config.threads_z);
    else if (config.exact_thread_extent && has_partial_workgroup)
        b.guard_invocation_extent(config.threads_x, config.threads_y, config.threads_z);

    RegState rs;
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.exec = b.btrue();
    // Inline descriptors are represented by the resource table, not scalar SSA values. Leaving
    // their SGPR range absent also preserves the existing direct-provenance rule: a format MUBUF may
    // fall back to by_sgpr_base only while its SRSRC has not been overwritten by shader code.
    std::set<uint32_t> descriptor_sgprs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            uint32_t words = (resource.cls == ResourceClass::Texture ||
                              resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; word++)
                descriptor_sgprs.insert(resource.sgpr_base + word);
        }
    }
    for (size_t i = 0; i < config.user_sgprs.size(); i++) {
        const uint32_t value = b.load_push_constant(static_cast<uint32_t>(i));
        if (descriptor_sgprs.count(static_cast<uint32_t>(i)))
            rs.sreg_input[static_cast<int>(i)] = value;
        else
            rs.sreg[static_cast<int>(i)] = value;
    }

    rs.vreg[0] = b.localid_comp[0];
    if (config.tidig_comp_cnt >= 1) rs.vreg[1] = b.localid_comp[1];
    if (config.tidig_comp_cnt >= 2) rs.vreg[2] = b.localid_comp[2];

    int system_sgpr = static_cast<int>(config.user_sgprs.size());
    if (config.tgid_x_en) rs.sreg[system_sgpr++] = b.groupid[0];
    if (config.tgid_y_en) rs.sreg[system_sgpr++] = b.groupid[1];
    if (config.tgid_z_en) rs.sreg[system_sgpr++] = b.groupid[2];
    if (config.tg_size_en)
        rs.sreg[system_sgpr] = b.uconst(local_x * local_y * local_z);

    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/true, [](RegState&, const Rdna2Inst&) { return false; },
                   code, dwords, nullptr, true, initial_dispatch_active, false,
                   lds_fminmax_synchronization.needs_dispatcher))
        return {};
    // Exact dispatch contracts execute their partial final wave through the CFG dispatcher's ACTIVE
    // bit. Padded Vulkan lanes stay in the dispatcher and its synthesized workgroup barriers, but
    // cannot execute guest memory effects. The full program, launch, and resource proof above is the
    // authority boundary for extending the selected-SBUFFER path to the packed-pointer program.
    if (exact_partial_dispatcher && b.uses_barrier)
        b.partial_barrier_phases_emitted = true;
    // The entry guard is intentionally divergent only in the final partial workgroup. Vulkan requires
    // every workgroup invocation to participate uniformly in OpControlBarrier, including barriers the
    // recompiler synthesizes for wave operations. Reject this uncommon combination instead of emitting
    // a module that could deadlock or observe undefined workgroup-memory behavior.
    if (has_partial_workgroup && b.uses_barrier && !b.partial_barrier_phases_emitted) {
        log_recompile_diagnostic(
            b.diagnostic, "compute-recompile-reject", "terminal",
            "reason=partial-workgroup-barrier threads=%ux%ux%u local=%ux%ux%u",
            config.threads_x, config.threads_y, config.threads_z,
            local_x, local_y, local_z);
        return {};
    }
    return b.finish();
}

bool compute_shader_prefers_native_multiwave(const std::vector<Rdna2Inst>& ins,
                                             const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    bool low = false;
    bool high = false;
    bool guest_barrier = false;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3) {
            low |= in.opcode == 0x365;
            high |= in.opcode == 0x366;
        }
        guest_barrier |= in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a;
        if (low && high) return true;
    }
    if (!guest_barrier || !code) return false;

    // The guarded phase splitter's shared proof is stronger than the whole-stream CFG shape:
    // branches and loops inside one barrier-free phase do not make the outer uniform guard or its
    // barriers divergent. Such kernels need an exact native subgroup for nested guest-wave votes.
    if (const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        phased.found && phased.guarded)
        return true;

    // Mirror the conservative, acyclic subset of emit_body's structured-compute admission. Counting
    // raw VCC/EXEC opcodes is insufficient: kill-mask branches may be safely linearized, loop exits
    // are owned by another emitter, and rejected CFGs never reach guest_wave_any. Requiring the same
    // accepted ForwardIf regions proves that portable lowering really emits two scratch barriers per
    // counted vote and that exact-subgroup lowering removes them. Loops deliberately stay behind the
    // explicit experiment until their additional compute guards are shared with this analysis.
    auto safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    const std::vector<DivLoop> loops =
        detect_divergent_loops(ins, safe, /*fragment*/false, diagnostic, "multiwave-probe");
    if (!loops.empty()) return false;

    bool rejected = false;
    const std::vector<ForwardIf> branches = detect_forward_ifs(
        ins, /*allow_vcc*/false, code, dwords, &safe, nullptr, &rejected,
        /*compute_wave_branches*/true, diagnostic);
    if (rejected) return false;

    auto top_level_pc = [&](uint32_t pc) {
        for (const ForwardIf& parent : branches) {
            const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
            if (parent.branch_pc < pc && pc < parent_end) return false;
        }
        return true;
    };
    const bool barriers_are_top_level = std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a || top_level_pc(in.pc);
    });
    if (!barriers_are_top_level) return false;

    size_t structured_wave_votes = 0;
    for (const ForwardIf& branch : branches) {
        if (!branch.on_exec && !branch.on_vcc) continue;
        if (!top_level_pc(branch.branch_pc)) return false;
        ++structured_wave_votes;
    }
    // Four proven scratch-emulated votes keep the default narrower than the all-multi-wave experiment.
    return structured_wave_votes >= 4;
}

bool compute_shader_prefers_native_multiwave(const uint32_t* code, size_t dwords,
                                             RecompileDiagnosticContext diagnostic) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    return compute_shader_prefers_native_multiwave(ins, code, dwords, diagnostic);
}

// See FlatLoadInfo / analyze_flat_loads in the header (#1171). A `base + offset` flat address VGPR pair
// v[N:N+1] is computed as vN = add(base_lo_sgpr, offset_lo) [carry-out] and v(N+1) =
// add(base_hi_sgpr, offset_hi, carry) [carry-in], where s[base_lo:base_hi] are consecutive user SGPRs
// holding a 64-bit guest pointer (the low kernel-arg pointer dword feeds the low address dword). We
// identify the base by the NEAREST prior definition of each address dword being an integer add that
// reads a user-range SGPR, and require the low dword to add the SGPR one below the high dword's.
// Anything else (a non-add producer, a non-user SGPR, a store/atomic/LDS/global-with-saddr form) leaves
// the load unresolved so the caller keeps failing visibly. flat_access_info lives in this TU's
// anonymous namespace but is visible here (internal linkage is still TU-wide).
// Resolution is over LINEAR program order (not CFG-aware). That is exact for the target decode kernels
// (the address adds sit in the same block immediately before the load) and the base is a loop-invariant
// kernel-arg pointer, so the resolved base is stable across loop iterations; the executor's
// guest_readable_mapping_containing validation is the runtime backstop against a bogus base.
FlatLoadAnalysis analyze_flat_loads(const uint32_t* code, size_t dwords, uint32_t user_sgpr_count) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    FlatLoadAnalysis out;

    auto writes_vgpr = [](const Rdna2Inst& d, uint32_t reg) {
        return d.dst.kind == OperandKind::VGPR && static_cast<uint32_t>(d.dst.value) == reg;
    };
    // If `d` is an integer add that could produce a 64-bit-pointer dword, return the user-range SGPR it
    // adds (else -1). Recognizes the add_co / add_co_ci / add_nc forms a compiler emits for pointer
    // arithmetic: VOP2 0x25 (add_nc) / 0x28 (add_co_ci), VOP3 0x30F (add_co) / 0x125 (add_nc) / 0x128
    // (add_co_ci).
    auto add_user_sgpr = [&](const Rdna2Inst& d) -> int32_t {
        const bool is_add =
            (d.fmt == Rdna2Format::VOP2 && (d.opcode == 0x25 || d.opcode == 0x28)) ||
            (d.fmt == Rdna2Format::VOP3 &&
             (d.opcode == 0x30F || d.opcode == 0x125 || d.opcode == 0x128));
        if (!is_add) return -1;
        for (int s = 0; s < 2; ++s)
            if (d.src[s].kind == OperandKind::SGPR && d.src[s].value >= 0 &&
                static_cast<uint32_t>(d.src[s].value) < user_sgpr_count)
                return d.src[s].value;
        return -1;
    };
    // The user-range SGPR added by the nearest prior writer of `reg` (or -1 if that writer is not a
    // user-SGPR add). We trust only the immediate definition — a later non-add redefinition breaks the
    // pattern and must not be skipped over.
    auto prior_add_sgpr = [&](uint32_t reg, size_t before) -> int32_t {
        for (size_t j = before; j-- > 0;)
            if (!ins[j].is_end && writes_vgpr(ins[j], reg))
                return add_user_sgpr(ins[j]);
        return -1;
    };

    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::FLAT) continue;
        if (in.flat_segment == 1u) continue;   // scratch spill: analyze_static_scratch owns it
        out.any = true;
        const FlatAccessInfo access = flat_access_info(in.opcode);
        FlatLoadInfo info;
        info.load_pc = in.pc;
        info.vaddr_lo_reg = in.src[0].kind == OperandKind::VGPR
                                ? static_cast<uint32_t>(in.src[0].value) : 0u;
        info.dst_reg = static_cast<uint32_t>(in.dst.value);
        info.bits = access.bits;
        info.components = access.components;
        info.sign_extend = access.sign_extend;
        // Resolvable only as a plain LOAD with a VGPR address pair and a null SADDR (true flat segment).
        const bool shape = access.valid && !access.store && !in.flat_lds &&
                           in.src[0].kind == OperandKind::VGPR &&
                           in.src[1].kind == OperandKind::Special && in.src[1].value == 125;
        if (shape) {
            const uint32_t lo = info.vaddr_lo_reg, hi = lo + 1;
            const int32_t hi_base = prior_add_sgpr(hi, i);
            const int32_t lo_base = prior_add_sgpr(lo, i);
            if (hi_base > 0 && lo_base == hi_base - 1)
                info.base_sgpr = lo_base;
        }
        if (info.base_sgpr < 0) out.all_resolved = false;
        out.loads.push_back(info);
    }
    return out;
}

std::vector<uint32_t> safe_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> s = safe_execz_branches(ins);
    return std::vector<uint32_t>(s.begin(), s.end());
}

std::vector<uint32_t> structured_execz_branches_for_test(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    bool rejected = false;
    const auto branches = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords, &safe,
                                             nullptr, &rejected,
                                             /*compute_wave_branches*/true);
    std::vector<uint32_t> pcs;
    if (!rejected) for (const auto& branch : branches) pcs.push_back(branch.branch_pc);
    return pcs;
}

std::vector<uint32_t> mask_test_branches_for_test(const uint32_t* code, size_t dwords,
                                                  bool wave32) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto branches = mask_test_branches(ins, wave32);
    return std::vector<uint32_t>(branches.begin(), branches.end());
}

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t synthetic_branch_pc = UINT32_MAX;
    (void)extend_terminating_if_else(code, dwords, ins, nullptr, &synthetic_branch_pc);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // A scratch builder/state so emit_alu can run; its emitted code is discarded — we only want `ok`.
    SpirvCompute b; b.begin(1);
    b.cselect_b64_low_only_pcs = proven_cselect_b64_low_only_pcs(ins);
    b.cselect_b64_low_only_analysis_done = true;
    // Coverage is deliberately a context-free instruction census. Preserve its historical
    // CSELECT exception, but do not let a newly recognized whole-CFG logical lifetime poison the
    // scratch compute state and make later instructions look unsupported. Real compute emission
    // runs the complete proof in emit_body.
    b.vcc_b32_low_only_pcs =
        proven_wave64_vcc_b32_low_only_pcs(ins, /*include_logical*/false);
    b.vcc_b32_low_only_analysis_done = true;
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // emit_alu is a per-instruction check and rejects control-flow branches, but the whole-stream emit_body
    // RECONSTRUCTS a counted loop and a forward uniform-if. Credit the branches emit_body consumes (loop
    // back-edge + exit, and the forward-if branch) as handled, so coverage matches what actually recompiles
    // (previously the MSAA-resolve loop shaders 031-034 were mis-flagged "blocked" at their s_cbranch_scc0).
    const CountedLoop cL = detect_counted_loop(ins);
    std::vector<ForwardIf> cFs;
    if (cL.found) {
        // Match emit_body's counted-loop composition: inspect the truncated prefix independently
        // from the loop exit/back-edge, then inspect the recursively-emitted suffix. Feeding the
        // complete stream to detect_forward_ifs makes the canonical exit look like an IF whose
        // alleged else terminator is the backward loop branch, so otherwise-valid prefix/postfix
        // branches are incorrectly counted as unsupported.
        std::vector<Rdna2Inst> prefix;
        for (const auto& in : ins) {
            if (in.pc >= cL.header_pc) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= cL.header_pc) continue;
            prefix.push_back(in);
        }
        Rdna2Inst prefix_end;
        prefix_end.pc = cL.header_pc;
        prefix_end.is_end = true;
        prefix.push_back(prefix_end);
        bool prefix_rejected = false;
        std::vector<ForwardIf> prefix_ifs = detect_forward_ifs(
            prefix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr,
            &prefix_rejected, /*compute_wave_branches*/true);
        const bool prefix_crosses_loop = std::any_of(
            prefix_ifs.begin(), prefix_ifs.end(), [&](const ForwardIf& branch) {
                return branch.early_out ||
                    (branch.has_else ? branch.merge_pc : branch.target_pc) > cL.header_pc;
            });
        if (!prefix_rejected && !prefix_crosses_loop)
            cFs.insert(cFs.end(), prefix_ifs.begin(), prefix_ifs.end());

        std::vector<Rdna2Inst> suffix;
        for (const auto& in : ins) if (in.pc >= cL.exit_pc) suffix.push_back(in);
        std::vector<ForwardIf> suffix_ifs = detect_forward_ifs(
            suffix, /*allow_vcc*/false, code, dwords, &safe_branches, nullptr, nullptr,
            /*compute_wave_branches*/true);
        cFs.insert(cFs.end(), suffix_ifs.begin(), suffix_ifs.end());
    } else {
        cFs = detect_forward_ifs(ins, /*allow_vcc*/false, code, dwords,
                                 nullptr, nullptr, nullptr,
                                 /*compute_wave_branches*/true);   // matches the compute shell (#590)
    }
    auto cf_reconstructed = [&](const Rdna2Inst& i) {
        if (cL.found && (i.pc == cL.backedge_pc || i.pc == cL.exit_branch_pc)) return true;
        for (const auto& F : cFs)
            if (i.pc == F.branch_pc || (F.has_else && i.pc == F.sb_pc)) return true;
        return false;
    };

    RecompileCoverage cov;
    for (const auto& in : ins) {
        if (in.is_end) break;
        // The rewritten first s_endpgm is compiler-only control flow. It lets the structured emitter
        // reuse its ordinary if/else path, but coverage describes decoded guest instructions and has
        // always excluded s_endpgm, so do not inflate total/ALU with this synthetic arm skip.
        if (in.pc == synthetic_branch_pc) continue;
        cov.total++;
        if (in.fmt == Rdna2Format::EXP) { cov.exports++; continue; }   // handled by the stage recompilers
        bool ok = true;
        const bool emitted = emit_alu(
            b, rs, in, ok, /*allow_exec_update*/true, &safe_branches,
            /*allow_smem*/true, /*rt*/nullptr, /*allow_wave*/true);
        if (emitted && ok)
            record_scalar_write(
                rs, in,
                allows_compute_scalar_vcc_bridge(b));
        bool handled = cf_reconstructed(in) || (emitted && ok);
        // Shapes the recompiler handles only in context (a resource table for MIMG sample/load/LOD/store
        // and buffer_load/store_format; a fragment stage for VINTRP). This table-less compute-shell pass
        // rejects them, so count them apart from truly-unsupported (cross-lane, etc.). Instruction-aware
        // for MIMG so deferred variants (NSA multi-dword addr; arrayed/cube/MSAA dims) are NOT overcounted
        // as recompilable — they still land in `unsupported`, matching what the recompiler actually accepts.
        auto table_dependent = [](const Rdna2Inst& i) {
            switch (i.fmt) {
                case Rdna2Format::MIMG: {
                    // Storage load/store handle 1D/2D/3D + 1D/2D_ARRAY (dims 0,1,2,4,5) and NSA.
                    // Sampled 2D_MSAA IMAGE_LOAD is narrower: only the exact consecutive-address or
                    // one-extra NSA [x,y,sample] shapes accepted by emit_alu are table-dependent.
                    // Do not credit dim7 or unused NSA address bytes merely because a T# could exist.
                    const bool st_dim = i.mimg_dim <= 2u || i.mimg_dim == 4u || i.mimg_dim == 5u;
                    if (i.opcode == 0x00u) {
                        if (st_dim) return true;
                        const bool msaa_address_shape = i.len_dwords == 2u ||
                            (i.len_dwords == 3u && (i.words[2] & 0xffff0000u) == 0u);
                        return i.mimg_dim == 6u && !i.mimg_unorm && !i.has_modifier &&
                               msaa_address_shape;
                    }
                    if (i.opcode == 0x01u || i.opcode == 0x09u)
                        return rdna2_mimg_zero_mip_shape(i);
                    if (i.opcode == 0x08u) return st_dim;                       // image_store (no per-sample MSAA store)
                    if (i.opcode == 0x0fu || i.opcode == 0x11u)   // image_atomic_swap/add R32_UINT 2D / 2D_ARRAY
                        // #2265: 2D_ARRAY (dim 5) admitted alongside 2D. This is the COVERAGE
                        // predicate -- it decides whether the instruction counts as supported for
                        // the census, and it was the last of the four sites still reporting the
                        // arrayed form as unsupported after #2272 widened the lowering. A 2D_ARRAY
                        // arrayed atomic reaches its layer through the address VGPRs, not through a
                        // longer ENCODING: CrossWorlds' own instruction is `dim=5 ... len=2`, so
                        // `len_dwords` is the NSA-vs-packed encoding length and pinning it to 3 for
                        // dim 5 would reject exactly the instruction this admits. Matches the
                        // lowering gate, which rejects only `len_dwords < 2`.
                        return (i.mimg_dim == 1u || i.mimg_dim == 5u) && i.mimg_dmask == 1u &&
                               !i.mimg_unorm && i.len_dwords >= 2u;
                    if (i.opcode == 0x0eu) return i.mimg_dim <= 2u;             // image_get_resinfo 1D/2D/3D
                    if (i.opcode == 0x60u)                                     // fragment image_get_lod 2D
                        return i.mimg_dim == 1u && i.len_dwords == 2u &&
                               !mimg_get_lod_has_unmodeled_controls(i) &&
                               (i.mimg_dmask & 0x3u) && !(i.mimg_dmask & ~0x3u);
                    // sample*: 2D (NSA ok); plus implicit-LOD image_sample (0x20) / LOD-0 image_sample_lz
                    // (0x27) from a 3D texture; sample_b (0x25) and gather4_lz (0x47) are 2D. 2D_ARRAY (dim 5)
                    // is accepted for all sample paths and handled as its base 2D slice (array index dropped,
                    // #325) — so array-sampling draws recompile+render instead of being skipped.
                    // 0xa0 is the high-bit sibling of image_sample (0x20), lowered identically (GTA V, #1145).
                    if (i.opcode == 0x20u || i.opcode == 0x27u || i.opcode == 0xa0u)
                        return i.mimg_dim == 1u || i.mimg_dim == 2u || i.mimg_dim == 5u;
                    if (i.opcode == 0x22u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    if (i.opcode == 0x2fu)
                        return i.mimg_dim == 1u || i.mimg_dim == 3u || i.mimg_dim == 5u;
                    if (i.opcode == 0x24u || i.opcode == 0x25u || i.opcode == 0x47u) return i.mimg_dim == 1u || i.mimg_dim == 5u;
                    return false;
                }
                case Rdna2Format::MUBUF:  return i.opcode <= 0x07u ||                    // load/store_format_*
                                                 (i.opcode >= 0x0Cu && i.opcode <= 0x0Fu);  // load_dword/x2/x4/x3 (need the V#)
                case Rdna2Format::MTBUF:  return i.opcode <= 0x07u && !i.mtbuf_tfe;
                // Wide scalar loads are descriptor-table fetches. A real resource table lets the
                // emitter preserve their provenance without reading a fallback buffer, including the
                // register-offset bindless form; the table-less coverage shell must not call that a
                // newly unsupported instruction.
                case Rdna2Format::SMEM:   return i.opcode == 0x02u || i.opcode == 0x03u;
                case Rdna2Format::VINTRP: return true;               // handled in the fragment shell
                default: return false;
            }
        };
        if (handled) { cov.alu++; }
        else if (table_dependent(in)) { cov.table_dependent++; }
        else {
            cov.unsupported++;
            if (cov.first_bad_fmt < 0) {
                cov.first_bad_fmt = (int)in.fmt;
                cov.first_bad_op = in.opcode;
                cov.first_bad_pc = in.pc;
            }
        }
    }
    return cov;
}

std::vector<uint32_t> cselect_b64_low_only_pcs_for_test(
        const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto proven = proven_cselect_b64_low_only_pcs(ins);
    std::vector<uint32_t> result(proven.begin(), proven.end());
    std::sort(result.begin(), result.end());
    return result;
}

static uint64_t shader_program_hash(const uint32_t* code, size_t dwords) {
    uint64_t hash = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const uint8_t*>(code);
    for (size_t i = 0; i < dwords * sizeof(uint32_t); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint32_t effective_fragment_wave_size(uint32_t requested_wave_size,
                                             size_t program_dwords,
                                             uint64_t program_hash) {
    if (requested_wave_size != 32 && requested_wave_size != 64) return 0;
    // Compatibility for the one captured Astro fragment whose older producer omitted
    // SPI_PS_IN_CONTROL.PS_W32_EN. Its complete byte identity proves the same Wave32 contract; that
    // contract must select both one-word mask semantics and a 32-lane native subgroup.
    const bool legacy_wave32 = requested_wave_size == 64 && program_dwords == 3142 &&
        program_hash == 0x616dd4c0b241fbb1ull;
    return requested_wave_size == 32 || legacy_wave32 ? 32u : 64u;
}

uint32_t fragment_effective_wave_size_for_test(uint32_t requested_wave_size,
                                               size_t program_dwords,
                                               uint64_t program_hash) {
    return effective_fragment_wave_size(requested_wave_size, program_dwords, program_hash);
}

uint32_t fragment_color_export_mask(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    uint32_t packed = 0;
    // One nibble per MRT, MRT0..MRT7. Sized 2 until 2026-08-15, which silently forced
    // `write_mask &= 0` for slots 2..7 at gpu_execute.hpp's EXP.EN gate -- so a shader exporting to
    // MRT2+ had those attachments dropped no matter what CB_TARGET_MASK and CB_SHADER_MASK said.
    std::array<bool, kFragmentColorOutputs> realized{};
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP || in.exp_target >= realized.size() ||
            realized[in.exp_target] || in.exp_en == 0)
            continue;
        packed |= (in.exp_en & 0xFu) << (in.exp_target * 4u);
        realized[in.exp_target] = true;
    }
    return packed;
}

static std::vector<uint32_t> recompile_fragment_impl(
        const uint32_t* code, size_t dwords,
        const ShaderResourceTable* rt,
        const PixelSystemInputMapping* system_inputs,
        uint32_t pcrel_dispatch_target,
        const FragmentInterpolationLayout* interpolation,
        uint32_t wave_size) {
    if (wave_size != 32 && wave_size != 64) return {};
    const RecompileDiagnosticContext diagnostic{RecompileDiagnosticStage::Fragment, 0};
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    if (pcrel_dispatch_target != UINT32_MAX) {
        const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(code, dwords);
        if (!specialize_pcrel_dispatch(ins, dispatch, pcrel_dispatch_target)) {
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "pcrel dispatch specialization target=%u",
                                     pcrel_dispatch_target);
            return {};
        }
    }
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    // Preserve hardware target locations. MRT0 and MRT1 are backed by real Vulkan attachments; later
    // targets remain unsupported and must never be silently remapped to location 0 (#635).
    constexpr uint32_t kMrtzDepth = 1u << 0;
    constexpr uint32_t kMrtzStencil = 1u << 1;
    constexpr uint32_t kMrtzSampleMask = 1u << 2;
    constexpr uint32_t kMrtzAlpha = 1u << 3;
    constexpr uint32_t kUnsupportedMrtz = kMrtzStencil | kMrtzAlpha;
    uint32_t color_mask = 0;
    bool has_null_export = false;
    bool has_depth_export = false;
    bool has_sample_mask_export = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target < kFragmentColorOutputs) color_mask |= 1u << in.exp_target;
        else if (in.exp_target == 8 && !in.exp_compr) {
            has_depth_export |= (in.exp_en & kMrtzDepth) != 0;
            has_sample_mask_export |= (in.exp_en & kMrtzSampleMask) != 0;
        }
        else if (in.exp_target == 9) has_null_export = true;
    }
    // A NULL export is a real fragment-shader terminator. Depth/stencil-only draws use it after
    // narrowing EXEC to the surviving samples, so the module intentionally has no color outputs.
    // Keep other unsupported MRT-only programs fail-visible instead of accepting every no-color PS.
    if (!color_mask && !has_null_export && !has_depth_export && !has_sample_mask_export) {
        if (pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(
                diagnostic, "recompile-reject", "terminal",
                "fragment has no supported export dwords=%zu ins=%zu first=%08x last=%08x "
                "pcrel-target=%u",
                dwords, ins.size(), dwords ? code[0] : 0u,
                dwords ? code[dwords - 1] : 0u, pcrel_dispatch_target);
        else
            log_recompile_diagnostic(
                diagnostic, "recompile-reject", "terminal",
                "fragment has no supported export dwords=%zu ins=%zu first=%08x last=%08x",
                dwords, ins.size(), dwords ? code[0] : 0u,
                dwords ? code[dwords - 1] : 0u);
        return {};
    }

    const FragmentInterpolationLayout derived_interpolation = interpolation
        ? *interpolation : fragment_interpolation_layout(code, dwords, system_inputs);
    if (!derived_interpolation.valid) {
        log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                 "invalid fragment interpolation layout");
        return {};
    }
    const uint32_t effective_wave_size = effective_fragment_wave_size(
        wave_size, program_dwords, shader_program_hash(code, program_dwords));
    SpirvCompute b;
    b.diagnostic = diagnostic;
    b.wave_size = effective_wave_size;
    b.begin_fragment(rt, color_mask);
    // SPI_PS_IN_CONTROL.PS_W32_EN proves that EXEC_HI/VCC_HI are unused and the low-half mask
    // operations below represent the complete wave. Keep the older byte-exact captured exception
    // until every replay/capture producer carries the stage register into this entry point.
    b.allow_b32_masks = effective_wave_size == 32;
    // Fragment I/O value tap (PROSPER_FS_TAP=draw:pc): redirect the MRT0 colour export to the intermediate
    // VGPR produced at that PC so the rendered frame visualises the value. The `draw:` prefix is consumed by
    // gpu_replay (which re-recompiles only that draw's FS). Parse the same complete selector here so an
    // invalid or overflowing PC cannot silently become PC zero or truncate to 32 bits.
    if (const char* tap = getenv("PROSPER_FS_TAP")) {
        uint64_t draw = 0;
        uint32_t pc = 0;
        if (parse_fragment_tap_selector(tap, draw, pc)) b.tap_pc = pc;
    }
    b.declare_guest_scratch(scratch);
    b.fragment_interpolation = &derived_interpolation;
    // P0-only attributes retain the cheap Flat varying path. Mixed smooth/explicit-parameter reads
    // are legal with the portable geometry stage and use separate packed locations there.
    if (!derived_interpolation.requires_geometry) {
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::VINTRP && in.opcode == 2)
                b.flat_attrs.insert(in.vintrp_attr);
        }
    }
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    // #2418: a static property of the decoded stream, set once and never mutated during emission.
    // Gates the fragment SCC re-arm after mask ops so only shaders that actually consume SCC pay the
    // exact-wave-vote's subgroup-size requirement.
    rs.reads_scc = shader_reads_scc(ins);
    if (system_inputs) {
        // RDNA2 PS system values are packed in field order. ADDR reserves each field's documented
        // width even when ENA is clear, allowing a driver to keep later VGPR numbers stable. Vulkan
        // exposes the four floating-point position terms directly as FragCoord.xyzw.
        static constexpr uint8_t widths[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t vgpr = 0;
        for (uint32_t field = 0; field < 16; ++field) {
            const uint32_t bit = 1u << field;
            if (!(system_inputs->addr & bit)) continue;
            if (system_inputs->ena & bit) {
                if (field <= 6 && derived_interpolation.requires_geometry) {
                    for (uint32_t component = 0; component < widths[field]; ++component) {
                        const uint32_t value = b.system_interpolation_component(field, component);
                        if (!value) {
                            log_recompile_diagnostic(
                                diagnostic, "recompile-reject", "terminal",
                                "missing fragment system interpolation field=%u component=%u",
                                field, component);
                            return {};
                        }
                        rs.vreg[(int)(vgpr + component)] = value;
                    }
                } else if (field >= 8 && field <= 11) {
                    rs.vreg[(int)vgpr] = b.fragcoord_component(field - 8);
                }
            }
            vgpr += widths[field];
        }
    }
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // FRAGMENT-only: also linearize alpha-test / clip() kill-mask s_cbranch_scc0/scc1 early-outs (Unity
    // cutout + text draws). Merged into the same set so emit_alu drops the branch and detect_forward_if
    // skips it; the block's EXEC narrow + the export's OpKill do the per-invocation discard (#102). This is
    // NOT added for the vertex/compute shells (their scc branches are real uniform-ifs / NGG culling).
    for (uint32_t pc : mask_test_branches(ins, b.allow_b32_masks)) safe_branches.insert(pc);
    std::array<bool, kFragmentColorOutputs> exported{};
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP MRT0/MRT1 -> matching output
        // An export while EXEC is narrowed (lanes killed by an alpha test / v_cmpx and not restored to
        // all-on) must not write the inactive lanes. Lower it to a real fragment discard: OpKill the lanes
        // whose EXEC bit is false, then export from the survivors under full EXEC. This is exactly the
        // alpha-tested-sprite shape (image_sample -> v_cmp alpha<ref -> s_andn2 saved,saved,vcc -> s_wqm
        // exec,saved -> shade -> export): the surviving lanes are the ones that passed the test. (When EXEC
        // was never narrowed this is a no-op — the common sRGB/tonemap restore-then-export path.)
        if (state.exec_narrowed) {
            b.discard_unless(state.exec);
            state.exec = b.btrue();
            state.exec_narrowed = false;
        }
        // MRTZ (target 8): EN bit 0 exports depth from VSRC0 and bit 2 exports sample coverage from
        // VSRC2. Vulkan represents those as FragDepth and SampleMask[0], respectively. Stencil
        // reference (bit 1) needs a separate extension-backed path and remains fail-visible; COMPR
        // MRTZ and the bit-3 alpha-to-coverage payload are likewise unmodeled.
        if (in.exp_target == 8) {
            if (in.exp_compr || !in.exp_en || (in.exp_en & kUnsupportedMrtz)) return false;
            const bool exports_depth = (in.exp_en & kMrtzDepth) != 0;
            const bool exports_sample_mask = (in.exp_en & kMrtzSampleMask) != 0;
            bool eok = true;
            const uint32_t z = exports_depth
                ? operand_bits(b, state, in, in.src[0], &eok) : 0;
            const uint32_t sample_mask = exports_sample_mask
                ? operand_bits(b, state, in, in.src[2], &eok) : 0;
            if (!eok) return false;
            if (exports_depth) b.export_depth(z);
            if (exports_sample_mask) b.export_sample_mask(sample_mask);
            return true;
        }
        if (in.exp_target < exported.size()) {
            // EN (Table 56) selects which VSRC channels the export sends; hardware does not update
            // disabled components. The executor maps this mask to Vulkan colorWriteMask, so the SPIR-V
            // value in a disabled channel is irrelevant and must not force a read of a stale VGPR.
            if (in.exp_en == 0) return true;
            bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
            if (in.exp_compr) {
                // COMPR: the 4 channels are two f16x2 pairs — src[0] holds (r,g), src[1] holds (b,a).
                // Unpack each half to a float and reassemble the vec4 (the pkrtz'd tonemap/sRGB output).
                const uint32_t p0 = (in.exp_en & 0x3u)
                    ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
                const uint32_t p1 = (in.exp_en & 0xCu)
                    ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? b.unpack_half(p0, 0) : b.uconst(0),
                               (in.exp_en & 0x2u) ? b.unpack_half(p0, 1) : b.uconst(0),
                               (in.exp_en & 0x4u) ? b.unpack_half(p1, 0) : b.uconst(0),
                               (in.exp_en & 0x8u) ? b.unpack_half(p1, 1) : b.uconst(0));
            } else {
                b.export_color(in.exp_target,
                               (in.exp_en & 0x1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0),
                               (in.exp_en & 0x2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0),
                               (in.exp_en & 0x4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0),
                               (in.exp_en & 0x8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0));
            }
            if (!eok) return false;
            exported[in.exp_target] = true;
        }
        return true;   // NULL carries the EXEC/discard effect above; ignore additional exports for now
    };
    // cmpx is now ALLOWED (allow_exec_update=true): a fragment divergent-if (v_cmpx ... s_mov exec,saved)
    // is handled by EXEC predication like compute, and the export is guarded above. Memory ops need a
    // resource table. Loops (if any) are reconstructed by emit_body.
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true, /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG") && pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(diagnostic, "recompile-reject", "consequent",
                                     "pcrel target=%u body failed", pcrel_dispatch_target);
        return {};
    }
    // ANY colour slot counts, not just the first two. This tested `exported[0] || exported[1]` while
    // the array was sized 2, which read as "did anything export"; once the shell carries eight
    // outputs it silently became "did MRT0 or MRT1 export", and a shader whose only colour output is
    // MRT2+ was rejected outright -- dropping its draws rather than its attachments. The guard's
    // purpose is unchanged: a fragment program that emits no colour, no NULL, no depth and no sample
    // mask is still fail-visible.
    const bool exported_any_color =
        std::any_of(exported.begin(), exported.end(), [](bool e) { return e; });
    if (!exported_any_color && !has_null_export && !has_depth_export &&
        !has_sample_mask_export) {
        if (pcrel_dispatch_target != UINT32_MAX)
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "emitted no fragment color pcrel-target=%u",
                                     pcrel_dispatch_target);
        else
            log_recompile_diagnostic(diagnostic, "recompile-reject", "terminal",
                                     "emitted no fragment color");
        return {};
    }
    return b.finish();
}

std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt,
                                         const PixelSystemInputMapping* system_inputs,
                                         uint32_t pcrel_dispatch_target,
                                         const FragmentInterpolationLayout* interpolation,
                                         bool wave32) {
    return recompile_fragment_impl(code, dwords, rt, system_inputs,
                                   pcrel_dispatch_target, interpolation,
                                   wave32 ? 32u : 64u);
}

std::vector<uint32_t> recompile_fragment_wave32_for_test(
        const uint32_t* code, size_t dwords) {
    return recompile_fragment_impl(code, dwords, nullptr, nullptr,
                                   UINT32_MAX, nullptr, 32);
}

uint32_t fragment_spirv_required_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.FragmentSubgroupSize=";
    bool legacy_arithmetic_marker = false;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2 &&
            spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
            legacy_arithmetic_marker = true;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(
                        text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length && (value == 32 || value == 64))
                        return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // Compatibility for cached/captured modules produced before explicit module metadata.
    return legacy_arithmetic_marker ? 64u : 0u;
}

uint32_t compute_spirv_min_subgroup_size(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    constexpr char prefix[] = "Prosper.ComputeSubgroupMin=";
    uint32_t required = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value = std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length &&
                        (value == 4 || value == 16 || value == 32 || value == 64))
                        required = std::max(required, static_cast<uint32_t>(value));
                }
            }
        }
        offset += words;
    }
    return required;
}

uint32_t fragment_spirv_required_subgroup_reasons(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return UINT32_MAX;
    constexpr char prefix[] = "Prosper.FragmentSubgroupWhy=";
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return UINT32_MAX;
        if (opcode == Op_ModuleProcessed && words > 1) {
            const char* text = reinterpret_cast<const char*>(&spirv[offset + 1]);
            const size_t bytes = static_cast<size_t>(words - 1) * sizeof(uint32_t);
            const void* terminator = std::memchr(text, '\0', bytes);
            if (terminator) {
                const size_t length = static_cast<const char*>(terminator) - text;
                if (length > sizeof(prefix) - 1 &&
                    std::memcmp(text, prefix, sizeof(prefix) - 1) == 0) {
                    char* end = nullptr;
                    const unsigned long value =
                        std::strtoul(text + sizeof(prefix) - 1, &end, 10);
                    if (end == text + length) return static_cast<uint32_t>(value);
                }
            }
        }
        offset += words;
    }
    // No marker: the module predates #2147. UINT32_MAX rather than 0, so a caller cannot read
    // 'unknown' as 'nothing required it'.
    return UINT32_MAX;
}

uint32_t fragment_spirv_required_subgroup_features(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return 0;
    uint32_t features = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return 0;
        if (opcode == Op_Capability && words == 2) {
            if (spirv[offset + 1] == Cap_GroupNonUniformVote)
                features |= kFragmentSubgroupVote;
            else if (spirv[offset + 1] == Cap_GroupNonUniformArithmetic)
                features |= kFragmentSubgroupArithmetic;
            else if (spirv[offset + 1] == Cap_GroupNonUniformShuffle)
                features |= kFragmentSubgroupShuffle;
            else if (spirv[offset + 1] == Cap_GroupNonUniformBallot)
                features |= kFragmentSubgroupBallot;
        }
        offset += words;
    }
    return features;
}

bool fragment_spirv_uses_internal_gds(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;
    std::unordered_map<uint32_t, uint32_t> sets, bindings;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t instruction = spirv[offset];
        const uint32_t words = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!words || words > spirv.size() - offset) return false;
        if (opcode == Op_Decorate && words == 4) {
            if (spirv[offset + 2] == Dec_DescriptorSet)
                sets[spirv[offset + 1]] = spirv[offset + 3];
            else if (spirv[offset + 2] == Dec_Binding)
                bindings[spirv[offset + 1]] = spirv[offset + 3];
        }
        offset += words;
    }
    for (const auto& [variable, set] : sets) {
        auto binding = bindings.find(variable);
        if (set == 1 && binding != bindings.end() && binding->second == 0) return true;
    }
    return false;
}

// Astro Bot's observed NGG wrappers use wave-shared plumbing for vertex allocation/compaction. The
// current Vulkan vertex shell intentionally projects those complete wrappers to one private guest
// lane; applying the projection to arbitrary NGG programs would silently miscompile peer-lane state.
// Keep each exception byte-exact and fail closed for every other wrapper. The hashes are FNV-1a over
// the little-endian instruction bytes through S_ENDPGM, matching the raw hashes in capture
// diagnostics. A proven PC-relative constant-table tail is deliberately excluded from the wrapper
// identity while remaining available to the recompiler.
static bool is_astro_bot_ngg_one_lane_wrapper(const uint32_t* code, size_t dwords) {
    if (!code) return false;
    std::vector<Rdna2Inst> instructions;
    const size_t program_dwords = rdna2_walk(code, dwords, instructions);
    if (program_dwords != 54 && program_dwords != 734 && program_dwords != 749 &&
        program_dwords != 3124 && program_dwords != 3435 && program_dwords != 3455 &&
        program_dwords != 3917)
        return false;
    const uint64_t hash = shader_program_hash(code, program_dwords);
    return (program_dwords == 54 && hash == 0x9e9d8e37bcc70607ull) ||
           (program_dwords == 734 && hash == 0x79eb2b954b07dc8eull) ||
           // The same 734-word culling wrapper is live-linked after its exact 15-word fetch prolog.
           (program_dwords == 749 && hash == 0xb440349937df751eull) ||
           (program_dwords == 3124 && hash == 0x41e6ac616c18d295ull) ||
           (program_dwords == 3435 && hash == 0xfad7a9f486523cfcull) ||
           // The same 3435-word wrapper is live-linked after its exact 20-word fetch prolog.
           // Hashing the complete concatenated program keeps the one-lane projection byte-exact.
           (program_dwords == 3455 && hash == 0x562ce5ad01c4c6e3ull) ||
           (program_dwords == 3917 && hash == 0x7f5f2349e2816f5eull);
}

VertexPrologInfo rdna2_vertex_prolog_info(const uint32_t* code, size_t dwords) {
    VertexPrologInfo result;
    if (!code || !dwords) return result;

    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    for (const Rdna2Inst& instruction : instructions) {
        // A fetch prolog has no architectural output or program termination of its own. Encountering
        // either before the transfer means this is a complete/different shader, not the split ABI.
        if (instruction.is_end || instruction.fmt == Rdna2Format::EXP ||
            instruction.fmt == Rdna2Format::Unknown)
            return {};
        if (instruction.fmt != Rdna2Format::SOP1 || instruction.opcode != 0x20)
            continue;

        // GFX9+ merged-stage fetch prologs receive the continuation PC in reserved s[6:7]. Keeping
        // this exact pair in the recognizer prevents an arbitrary indirect jump from becoming host
        // fallthrough merely because a second program happened to be bound.
        if (instruction.n_src != 1 || instruction.src[0].kind != OperandKind::SGPR ||
            instruction.src[0].value != 6 || instruction.len_dwords != 1)
            return {};
        result.valid = instruction.pc != 0;
        result.setpc_pc = instruction.pc;
        result.prefix_dwords = instruction.pc;
        break;
    }
    if (!result.valid) return {};

    // Replacing the transfer with fallthrough is valid only when every direct branch in the prolog
    // remains inside the retained prefix (or lands exactly on the transfer, which becomes main pc0).
    // A branch into discarded padding/data would otherwise be silently redirected into unrelated
    // main code. The linked body recompiler performs the remaining structured-CFG validation.
    for (const Rdna2Inst& instruction : instructions) {
        if (instruction.pc >= result.setpc_pc) break;
        if (instruction.fmt != Rdna2Format::SOPP ||
            (instruction.opcode != 0x02 &&
             (instruction.opcode < 0x04 || instruction.opcode > 0x09)))
            continue;
        const int64_t target = static_cast<int64_t>(instruction.pc) + instruction.len_dwords +
                               static_cast<int64_t>(instruction.simm16);
        if (target < 0 || target > result.setpc_pc) return {};
    }
    return result;
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.
struct NggPassthroughLayout {
    bool valid = false;
    uint32_t producer_base_vgpr = 0;
    uint32_t producer_base_byte = 0;
    uint32_t record_stride_bytes = 0;
    std::array<int32_t, 4> position = {-1, -1, -1, -1};
    std::array<std::array<int32_t, 4>, 32> params{};
    uint32_t param_mask = 0;

    NggPassthroughLayout() {
        for (auto& param : params) param.fill(-1);
    }
};

struct NggLdsSource {
    bool valid = false;
    uint32_t byte_offset = 0;
    uint32_t stride_bytes = 0;
    uint32_t index_vgpr = 0;
};

// Resolve the terminal wrapper's canonical `stride * exporter + constant` LDS address.  Requiring
// the nearest writer to be this exact u24 MAD shape keeps the optimization fail-closed when a user
// GS or a different compiler layout performs real output computation.
NggLdsSource ngg_terminal_lds_source(const std::vector<Rdna2Inst>& ins, size_t load_index,
                                     uint32_t output_vgpr) {
    const Rdna2Inst& load = ins[load_index];
    uint32_t address_vgpr = 0, component_byte = 0;
    if (load.fmt != Rdna2Format::DS || load.ds_gds) return {};
    if (load.opcode == 0x36u) {                         // ds_read_b32
        if (output_vgpr != static_cast<uint32_t>(load.dst.value)) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal;
    } else if (load.opcode == 0x37u) {                  // ds_read2_b32
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr > first + 1u) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        const uint32_t component = output_vgpr - first;
        component_byte = ((load.literal >> (component * 8u)) & 0xffu) * 4u;
    } else if (load.opcode == 0x76u || load.opcode == 0xfeu || load.opcode == 0xffu) {
        const uint32_t count = load.opcode == 0x76u ? 2u : load.opcode == 0xfeu ? 3u : 4u;
        const uint32_t first = static_cast<uint32_t>(load.dst.value);
        if (output_vgpr < first || output_vgpr >= first + count) return {};
        address_vgpr = static_cast<uint32_t>(load.src[0].value);
        component_byte = load.literal + (output_vgpr - first) * 4u;
    } else {
        return {};
    }

    for (size_t j = load_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.dst.kind != OperandKind::VGPR ||
            static_cast<uint32_t>(writer.dst.value) != address_vgpr)
            continue;
        // The compacted record address has either of the two canonical compiler forms below:
        //
        //   v_mad_u32_u24 addr, stride, exporter, constant
        //   v_mul_u32_u24 addr, stride, exporter; ds_read ... offset:constant
        //
        // The latter avoids a MAD when the entire constant fits in the DS instruction's immediate.
        // Both prove the same `stride * exporter + constant` identity; accepting only these exact
        // integer-u24 forms keeps arbitrary wrapper address arithmetic fail-closed.
        const bool mad = writer.fmt == Rdna2Format::VOP3 && writer.opcode == 0x143u &&
                         !writer.has_modifier && writer.has_literal;
        const bool mul = writer.fmt == Rdna2Format::VOP2 && writer.opcode == 0x0bu &&
                         !writer.has_modifier && !writer.has_literal;
        if (!mad && !mul) return {};
        // The first two operands may be swapped.
        int stride_src = -1, index_src = -1;
        for (int k = 0; k < 2; ++k) {
            if (writer.src[k].kind == OperandKind::InlineInt && writer.src[k].value > 0)
                stride_src = k;
            else if (writer.src[k].kind == OperandKind::VGPR)
                index_src = k;
        }
        if (stride_src < 0 || index_src < 0 ||
            (mad && writer.src[2].kind != OperandKind::Literal))
            return {};
        const uint32_t stride = static_cast<uint32_t>(writer.src[stride_src].value);
        if ((stride & 3u) || stride < 16u || stride > 4096u ||
            (mad && writer.literal > UINT32_MAX - component_byte))
            return {};
        return {true, (mad ? writer.literal : 0u) + component_byte, stride,
                static_cast<uint32_t>(writer.src[index_src].value)};
    }
    return {};
}

NggLdsSource ngg_find_terminal_output(const std::vector<Rdna2Inst>& ins, size_t export_index,
                                      uint32_t output_vgpr) {
    // The direct output loads sit in the export block.  Stop at the nearest writer; skipping a
    // transform or phi would turn a user GS into a false passthrough.
    for (size_t j = export_index; j-- > 0;) {
        const Rdna2Inst& writer = ins[j];
        if (writer.fmt == Rdna2Format::DS) {
            const NggLdsSource source = ngg_terminal_lds_source(ins, j, output_vgpr);
            if (source.valid) return source;
            const uint32_t first = static_cast<uint32_t>(writer.dst.value);
            const uint32_t count = writer.opcode == 0x37u || writer.opcode == 0x76u ? 2u
                                 : writer.opcode == 0xfeu ? 3u
                                 : writer.opcode == 0xffu ? 4u : 1u;
            if (output_vgpr >= first && output_vgpr < first + count) return {};
        }
        if (writer.dst.kind == OperandKind::VGPR &&
            static_cast<uint32_t>(writer.dst.value) == output_vgpr)
            return {};
    }
    return {};
}

NggPassthroughLayout analyze_ngg_passthrough(const uint32_t* prolog, size_t prefix_dwords,
                                             const uint32_t* main, size_t main_dwords) {
    NggPassthroughLayout out;
    auto reject = [&](const char* reason) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-ngg-passthrough-reject] %s\n", reason);
        return NggPassthroughLayout{};
    };
    std::vector<Rdna2Inst> producer, wrapper;
    rdna2_walk(prolog, prefix_dwords, producer);
    rdna2_walk(main, main_dwords, wrapper);
    if (producer.empty() || wrapper.empty()) return reject("empty producer/wrapper");

    // Prove that every LDS operation in the producer is a non-atomic store into one record base.
    // The private-LDS execution is valid only before the cross-lane wrapper starts reading it.
    bool saw_store = false;
    uint32_t base_vgpr = UINT32_MAX;
    std::set<uint32_t> stored_bytes;
    for (const Rdna2Inst& in : producer) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) return reject("producer terminates or exports");
        if (in.fmt != Rdna2Format::DS) continue;
        if (in.ds_gds || in.src[0].kind != OperandKind::VGPR)
            return reject("producer uses non-LDS DS address");
        const uint32_t base = static_cast<uint32_t>(in.src[0].value);
        if (base_vgpr == UINT32_MAX) base_vgpr = base;
        if (base != base_vgpr) return reject("producer has multiple LDS record bases");
        auto store_byte = [&](uint32_t byte) {
            if ((byte & 3u) == 0u) stored_bytes.insert(byte);
        };
        if (in.opcode == 0x0du) {
            store_byte(in.literal);
        } else if (in.opcode == 0x0eu) {
            store_byte((in.literal & 0xffu) * 4u);
            store_byte(((in.literal >> 8) & 0xffu) * 4u);
        } else if (in.opcode == 0x4du || in.opcode == 0xdeu || in.opcode == 0xdfu) {
            const uint32_t count = in.opcode == 0x4du ? 2u : in.opcode == 0xdeu ? 3u : 4u;
            for (uint32_t k = 0; k < count; ++k) store_byte(in.literal + k * 4u);
        } else {
            return reject("producer has a non-store/cross-lane DS operation");
        }
        saw_store = true;
    }
    if (!saw_store || stored_bytes.empty()) return reject("producer writes no aligned LDS record");

    bool saw_alloc = false, saw_primitive_export = false, saw_position = false;
    uint32_t common_stride = 0, common_index = UINT32_MAX;
    std::vector<uint32_t> absolute_offsets;
    auto accept_source = [&](const NggLdsSource& source, int32_t& destination) -> bool {
        if (!source.valid || (common_stride && source.stride_bytes != common_stride) ||
            (common_index != UINT32_MAX && source.index_vgpr != common_index))
            return false;
        common_stride = source.stride_bytes;
        common_index = source.index_vgpr;
        destination = static_cast<int32_t>(source.byte_offset);
        absolute_offsets.push_back(source.byte_offset);
        return true;
    };
    for (size_t i = 0; i < wrapper.size(); ++i) {
        const Rdna2Inst& in = wrapper[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10u) saw_alloc = true;
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_target == 20u) { saw_primitive_export = true; continue; }
        if (in.exp_target != 12u && in.exp_target < 32u) continue;
        if (in.exp_compr) return reject("terminal output uses a compressed export");
        if (in.exp_target == 12u) {
            if (saw_position || in.exp_en != 0xfu)
                return reject("POS0 is duplicated or incomplete");
            saw_position = true;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.position[component]))
                    return reject("POS0 does not directly load the terminal LDS record");
            }
        } else {
            const uint32_t param = in.exp_target - 32u;
            if (param >= out.params.size() || (out.param_mask & (1u << param)))
                return reject("PARAM export is out of range or duplicated");
            out.param_mask |= 1u << param;
            for (uint32_t component = 0; component < 4; ++component) {
                if (!(in.exp_en & (1u << component))) continue;
                if (!accept_source(ngg_find_terminal_output(
                                       wrapper, i, static_cast<uint32_t>(in.src[component].value)),
                                   out.params[param][component]))
                    return reject("PARAM does not directly load the terminal LDS record");
            }
        }
    }
    if (!saw_alloc || !saw_primitive_export || !saw_position || absolute_offsets.empty() ||
        !common_stride || base_vgpr == UINT32_MAX)
        return reject("wrapper lacks the no-GS allocation/export shape");

    const uint32_t wrapper_base = *std::min_element(absolute_offsets.begin(), absolute_offsets.end());
    const uint32_t producer_base = *stored_bytes.begin();
    auto normalize = [&](int32_t& byte) -> bool {
        if (byte < 0) return true;
        const uint32_t absolute = static_cast<uint32_t>(byte);
        if (absolute < wrapper_base) return false;
        const uint32_t relative = absolute - wrapper_base;
        if ((relative & 3u) || relative >= common_stride ||
            producer_base > UINT32_MAX - relative ||
            !stored_bytes.count(producer_base + relative))
            return false;
        byte = static_cast<int32_t>((producer_base + relative) / 4u);
        return true;
    };
    for (int32_t& component : out.position)
        if (!normalize(component)) return reject("POS0 offset does not match the producer record");
    for (auto& param : out.params)
        for (int32_t& component : param)
            if (!normalize(component)) return reject("PARAM offset does not match the producer record");

    out.valid = true;
    out.producer_base_vgpr = base_vgpr;
    out.producer_base_byte = producer_base;
    out.record_stride_bytes = common_stride;
    return out;
}

} // namespace

static std::vector<uint32_t> recompile_vertex_impl(const uint32_t* code, size_t dwords,
                                                   const ShaderResourceTable* rt,
                                                   const PixelInputMapping* pixel_inputs,
                                                   bool capture_position,
                                                   uint32_t virtual_lds_dwords,
                                                   const NggPassthroughLayout* passthrough,
                                                   bool allow_test_ngg_output_gate,
                                                   bool allow_test_ngg_one_lane) {
    const uint32_t passthrough_mask =
        pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Vertex, 0};
    b.capture_position = capture_position;   // geometry-probe: mark gl_Position for xfb capture (gated)
    b.vertex_lds_dwords = std::min(virtual_lds_dwords, 16384u);
    b.vertices_per_instance = rt ? rt->vertices_per_instance : 0u;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): redirect the position export to the intermediate VGPR
    // produced at that PC. Applies to the vertex stage only; captured via the geometry probe.
    if (const char* tap = getenv("PROSPER_SHADER_TAP")) b.tap_pc = static_cast<uint32_t>(strtoul(tap, nullptr, 0));
    b.begin_vertex(rt);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    // NGG vertex shaders (the exact GS_ALLOC_REQ message present) carry the vertex index in v5, not v0, and wrap
    // the body in wave-packing plumbing (s_sendmsg / exp prim / s_lshr_b64 exec) that lowers to no-ops in
    // our per-invocation model. Detect NGG and bind the index to v5 as well.
    bool ngg = passthrough && passthrough->valid;
    for (const auto& in : ins) { if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10 &&
            in.words[0] == 0xBF900009u) { ngg = true; break; } }
    const bool exact_ngg_projection = ngg && is_astro_bot_ngg_one_lane_wrapper(code, dwords);
    // The generic split-stage path uses private LDS only after the producer/wrapper analyzer has
    // proven that all visible outputs come from one per-vertex record. Wave-sensitive instruction
    // approximations remain restricted to the byte-exact projection above.
    b.ngg_private_lds = exact_ngg_projection || (passthrough && passthrough->valid) ||
                        (ngg && allow_test_ngg_output_gate);
    // The legacy byte-exact NGG projection and its explicit unit-test hook predate the linked-stage
    // ABI's exact LDS allocation. Preserve their proven 16 KiB private scratch contract when no
    // allocation was supplied; generic linked producers still require their real plumbed size.
    if (!b.vertex_lds_dwords && (exact_ngg_projection || allow_test_ngg_output_gate))
        b.vertex_lds_dwords = 4096;
    // Every wave/peer approximation is an exception for the one captured Astro wrapper, not a
    // property of the GS_ALLOC_REQ opcode. Other NGG programs retain only the ordinary merged-stage
    // ABI setup below and fail closed if they reach a lane-sensitive operation.
    b.ngg_one_lane = exact_ngg_projection || (ngg && allow_test_ngg_one_lane);
    // A LO+HI all-ones pair is the compiler's explicit wave64 lane-index construction. Infer the
    // width from that machine-code proof instead of assuming every NGG program is wave64. A low-only
    // producer may be wave32 or may use only half of a wave64 mask, so it remains fail-closed until
    // the graphics wave-size contract is plumbed independently.
    bool logical_mbcnt_lo = false, logical_mbcnt_hi = false, logical_mbcnt_invalid = false;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::VOP3 || (in.opcode != 0x365 && in.opcode != 0x366))
            continue;
        const bool all_ones = in.src[0].kind == OperandKind::InlineInt &&
                              in.src[0].value == -1;
        logical_mbcnt_invalid |= !all_ones;
        logical_mbcnt_lo |= all_ones && in.opcode == 0x365;
        logical_mbcnt_hi |= all_ones && in.opcode == 0x366;
    }
    b.ngg_logical_lane = passthrough && passthrough->valid && !logical_mbcnt_invalid &&
                         logical_mbcnt_lo && logical_mbcnt_hi;
    if (b.ngg_logical_lane) b.wave_size = 64;
    b.allow_b32_masks = b.ngg_one_lane;
    uint32_t ngg_output_gate_begin = UINT32_MAX;
    uint32_t ngg_output_gate_end = 0;
    if (ngg) {
        // A terminal compacted-output suffix may reconstruct its values from a shader-embedded
        // constant table. Detect those loads with the same bounded PC-relative proof used by the
        // emitter; an arbitrary external load or any buffer write must not broaden this gate.
        PcrelTables output_tables;
        if (b.ngg_private_lds)
            output_tables = detect_pcrel_tables(ins, code, dwords);
        auto scalar_output_setup = [](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::SOP1 &&
                candidate.fmt != Rdna2Format::SOP2 &&
                candidate.fmt != Rdna2Format::SOPK)
                return false;
            bool wrote_data = false;
            bool safe = !rdna2_instruction_may_change_exec(candidate);
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                wrote_data = true;
                safe &= base >= 0 && base + static_cast<int>(width) <= 106;
                safe &= !scalar_write_is_b64_mask(candidate, base);
            });
            return wrote_data && safe;
        };
        auto embedded_output_load = [&](const Rdna2Inst& candidate) {
            if (candidate.fmt != Rdna2Format::MUBUF || candidate.mubuf_lds)
                return false;
            const bool read_only = candidate.opcode <= 0x03u ||
                (candidate.opcode >= 0x0cu && candidate.opcode <= 0x0fu);
            return read_only && output_tables.mubuf.contains(candidate.pc);
        };
        uint32_t end_pc = UINT32_MAX;
        for (const auto& in : ins)
            if (in.is_end) { end_pc = in.pc; break; }
        // An NGG primitive shader finishes by compacting surviving vertices, CMPX-testing whether
        // this lane owns one of those vertices, then exporting POS/PARAM values before S_ENDPGM.
        // Vulkan's vertex shell already represents one surviving guest vertex per invocation, but
        // retaining the condition is still useful when the compacted count is zero. Permit exports
        // under narrowed EXEC only for this mechanically bounded terminal output gate; ordinary
        // vertex CMPX/export programs remain rejected below.
        for (size_t i = 1; i < ins.size(); ++i) {
            const Rdna2Inst& branch = ins[i];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                branch.simm16 <= 0 || branch_target(branch) != end_pc)
                continue;
            size_t previous = i;
            do { --previous; } while (previous > 0 && sopp_is_noop(ins[previous]));
            if (ins[previous].fmt != Rdna2Format::VOPC ||
                !vopc_is_cmpx(ins[previous].opcode))
                continue;
            // Production accepts data-dependent vertex suppression only for the byte-exact Astro
            // wrapper. The explicit test hook below exercises active/inactive export selection with
            // a tiny shader without turning that shader shape into a runtime allow-list exception.
            if (!b.ngg_private_lds && !allow_test_ngg_output_gate)
                continue;
            bool has_position = false;
            bool output_only = true;
            std::vector<uint32_t> trailing_vcc_branches;
            for (size_t j = i + 1; j < ins.size() && ins[j].pc < end_pc; ++j) {
                const Rdna2Inst& candidate = ins[j];
                if (candidate.fmt == Rdna2Format::EXP) {
                    has_position |= candidate.exp_target == 12;
                    continue;
                }
                // Astro's compacted-output suffix reconstructs the surviving vertex from private
                // LDS or a bounded shader-embedded table immediately before exporting it. Vector,
                // DS, and table-load destination writes are EXEC-predicated by emit_alu; scalar ALU
                // may only build ordinary data/descriptor registers. Admit them only for the
                // byte-exact wrapper (or the explicit test hook); arbitrary NGG shaders never reach
                // this exception, and buffer stores/external reads remain rejected.
                const bool output_rebuild = b.ngg_private_lds || allow_test_ngg_output_gate;
                if (output_rebuild &&
                    (candidate.fmt == Rdna2Format::VOP1 ||
                     candidate.fmt == Rdna2Format::VOP2 ||
                     candidate.fmt == Rdna2Format::VOP3 ||
                     candidate.fmt == Rdna2Format::VOP3P ||
                     (candidate.fmt == Rdna2Format::VOPC &&
                      !vopc_is_cmpx(candidate.opcode)) ||
                     candidate.fmt == Rdna2Format::DS ||
                     scalar_output_setup(candidate) ||
                     embedded_output_load(candidate)))
                    continue;
                if (candidate.fmt == Rdna2Format::SOPC) continue;
                if (sopp_is_noop(candidate)) continue;
                if (candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x04 || candidate.opcode == 0x05) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc)
                    continue;
                // The 7f5f wrapper exports POS, compares a per-vertex flag into VCC, then conditionally
                // skips only its trailing PARAM exports. Those exports cannot affect position/topology;
                // linearizing the branch merely supplies otherwise-undefined varyings for that path.
                if (has_position && candidate.fmt == Rdna2Format::SOPP &&
                    (candidate.opcode == 0x06 || candidate.opcode == 0x07) &&
                    candidate.simm16 > 0 && branch_target(candidate) == end_pc) {
                    trailing_vcc_branches.push_back(candidate.pc);
                    continue;
                }
                output_only = false;
                break;
            }
            if (has_position && output_only) {
                ngg_output_gate_begin = branch.pc;
                ngg_output_gate_end = end_pc;
                safe_branches.insert(trailing_vcc_branches.begin(), trailing_vcc_branches.end());
                break;
            }
        }
    }
    uint32_t vidx = b.load_vertex_index();
    uint32_t iidx = b.load_instance_index();
    rs.vreg[0] = vidx;                       // Legacy VS ABI: v0 = vertex index
    rs.vreg[3] = iidx;                       // Legacy VS ABI: v3 = instance index
    if (ngg) {
        // Locate the NGG prologue's LDS-to-ES vertex-index handoff without hard-coding a register or
        // program counter: the first MUBUF vaddr must be most recently defined by a scalar DS read.
        // The one-lane backend substitutes BuiltIn VertexIndex at that exact read (see DS lowering).
        for (size_t use = 0; use < ins.size(); ++use) {
            if (ins[use].is_end) break;
            if ((ins[use].fmt != Rdna2Format::MUBUF && ins[use].fmt != Rdna2Format::MTBUF) ||
                ins[use].src[0].kind != OperandKind::VGPR)
                continue;
            const int index_reg = ins[use].src[0].value;
            for (size_t def = use; def-- > 0;) {
                const Rdna2Inst& candidate = ins[def];
                if (candidate.dst.kind != OperandKind::VGPR || candidate.dst.value != index_reg)
                    continue;
                if (candidate.fmt == Rdna2Format::DS && candidate.opcode == 0x36) {
                    b.ngg_vertex_index_read_pc = candidate.pc;
                    b.ngg_vertex_index_value = vidx;
                }
                break;
            }
            break;
        }
        // GFX10's merged GS/ES ABI enters the ES prolog with vertex/instance indices in v5/v8 and
        // merged-wave info in s3: per-wave ES/GS counts [7:0]/[15:8], wave-in-TG [27:24], and
        // TG wave count [31:28]. The host draw has already omitted padding invocations, so expose a
        // full logical wave while deriving the architectural wave ID from the flattened invocation.
        rs.vreg[5] = vidx;
        rs.vreg[8] = iidx;
        if (exact_ngg_projection) {
            rs.sreg[3] = b.uconst(1);

        // The merged NGG wrapper guards its whole counted ES loop with EXECZ. In the single-lane
        // model above that one ES lane is active by construction, so the wave-empty shortcut cannot
        // be taken. Let the ordinary counted-loop lowering consume the loop instead of rejecting the
        // redundant outer guard merely because the original hardware mask was vector-shaped.
        const CountedLoop loop = detect_counted_loop(ins);
        if (loop.found) {
            // NGG culling unrolls several EXEC-predicated LDS blocks inside its counted loop. A
            // larger block may contain a smaller already-safe EXECZ block plus CMPX comparisons,
            // and ends by restoring a VCC-saved mask to EXEC. The generic safe-execz pass deliberately
            // rejects CMPX writes. Here they are exact: the branch immediately follows a CMPX that
            // narrowed EXEC, all effects in the skipped block are EXEC-predicated (further CMPX can
            // only narrow it again), and the common target performs the same EXEC=VCC restore on both
            // paths. Scan inside-out so nested blocks are proven before their parents.
            for (size_t branch_index = ins.size(); branch_index-- > 0;) {
                const Rdna2Inst& branch = ins[branch_index];
                if (branch.pc < loop.header_pc || branch.pc > loop.backedge_pc ||
                    branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 ||
                    branch.simm16 <= 0)
                    continue;
                size_t previous = branch_index;
                while (previous > 0) {
                    --previous;
                    if (!sopp_is_noop(ins[previous])) break;
                }
                if (previous >= branch_index || ins[previous].fmt != Rdna2Format::VOPC ||
                    !vopc_is_cmpx(ins[previous].opcode))
                    continue;
                const uint32_t target_pc = branch_target(branch);
                size_t target_index = ins.size();
                for (size_t i = 0; i < ins.size(); ++i)
                    if (ins[i].pc == target_pc) { target_index = i; break; }
                bool restores_saved_exec = false;
                for (size_t i = target_index; i < ins.size() && i < target_index + 3; ++i) {
                    const Rdna2Inst& candidate = ins[i];
                    if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == 0x04 &&
                        candidate.dst.value >= 126 &&
                        (candidate.src[0].value == 106 || candidate.src[0].value == 107)) {
                        restores_saved_exec = true;
                        break;
                    }
                    bool clobbers_saved_mask = rdna2_instruction_may_change_exec(candidate) ||
                        candidate.fmt == Rdna2Format::VOPC;
                    for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                        clobbers_saved_mask |= base < 108 && 106 < base + static_cast<int>(width);
                    });
                    if (clobbers_saved_mask ||
                        (candidate.fmt == Rdna2Format::SOPP && candidate.opcode >= 0x02 &&
                         candidate.opcode <= 0x09 && candidate.opcode != 0x03))
                        break;
                }
                if (!restores_saved_exec)
                    continue;
                bool safe_block = true;
                for (const auto& candidate : ins) {
                    if (candidate.pc <= branch.pc || candidate.pc >= target_pc) continue;
                    const bool predicated_or_masked =
                        candidate.fmt == Rdna2Format::VOP1 ||
                        candidate.fmt == Rdna2Format::VOP2 ||
                        candidate.fmt == Rdna2Format::VOP3 ||
                        candidate.fmt == Rdna2Format::VOP3P ||
                        (candidate.fmt == Rdna2Format::VOPC &&
                         vopc_is_cmpx(candidate.opcode)) ||
                        candidate.fmt == Rdna2Format::MIMG ||
                        candidate.fmt == Rdna2Format::MUBUF ||
                        candidate.fmt == Rdna2Format::MTBUF ||
                        candidate.fmt == Rdna2Format::DS ||
                        candidate.fmt == Rdna2Format::FLAT;
                    const bool nested_safe = candidate.fmt == Rdna2Format::SOPP &&
                        candidate.opcode == 0x08 && safe_branches.count(candidate.pc);
                    if (!predicated_or_masked && !nested_safe && !sopp_is_noop(candidate)) {
                        safe_block = false;
                        break;
                    }
                }
                if (safe_block) safe_branches.insert(branch.pc);
            }
            for (const auto& in : ins) {
                if (in.pc >= loop.header_pc) break;
                if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                    branch_target(in) == loop.exit_pc)
                    safe_branches.insert(in.pc);
            }
        }
        } else {
            const uint32_t wave = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, b.vertex_invocation_id(), b.uconst(6)),
                b.uconst(0xFu));
            rs.sreg[3] = b.ibin(Op_BitwiseOr, b.uconst(0x40004040u),
                                b.ibin(Op_ShiftLeftLogical, wave, b.uconst(24)));
        }
    }
    bool exported = false;
    auto exp_fn = [&](RegState& state, const Rdna2Inst& in) -> bool { // EXP POS0..3 -> gl_Position; PARAM -> varyings
        bool eok = true;   // a Special (wave-mask) source has no data value — reject, don't export 0 (#134)
        // COMPR pos/param exports carry two packed f16x2 pairs, not four f32 fields — reading the
        // VSRCs as full floats would pass packed-half bit patterns as x/y and stale registers as
        // z/w. Never observed in a vertex stage (compilers export positions/params at 32 bits);
        // reject fail-visibly until a live title exercises one.
        if (in.exp_compr) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex compressed export pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        // v_cmpx is now allowed in the vertex shell (allow_exec_update=true below): a divergent block
        // (v_cmpx … s_mov_b64 exec, saved — DOLL's per-vertex lighting/fog attenuation) predicates its
        // VGPR writes like compute. A vertex MUST still export from full EXEC — the compiled shape
        // always restores EXEC before its pos/param exports; if one ever arrives narrowed, reject
        // (fail-visibly) rather than export possibly-inactive-lane values.
        const bool terminal_ngg_output = ngg_output_gate_begin != UINT32_MAX &&
            in.pc > ngg_output_gate_begin && in.pc < ngg_output_gate_end;
        if (state.exec_narrowed && !terminal_ngg_output && (in.exp_target >= 32 ||
            (in.exp_target >= 12 && in.exp_target <= 16))) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "vertex export under narrowed exec pc=%u target=%u",
                                     in.pc, in.exp_target);
            return false;
        }
        if (in.exp_target == 12) {
            // POS0 is the mandatory x/y/z/w position vector. POS1..POS4 carry ancillary position
            // data (clip/cull distances, point size, viewport/layer selection according to the
            // programmed position format) and must never be mistaken for gl_Position merely because
            // an NGG shader emits one before POS0. Until those built-ins are modeled, retain the
            // existing deliberate behavior of ignoring them.
            // A position export must supply all four components (EN=0xF); a partial POS0 is not
            // meaningfully completable in the current model, so reject rather than invent components.
            if (in.exp_en != 0xFu) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "partial vertex position export pc=%u en=0x%x",
                                         in.pc, in.exp_en);
                return false;
            }
            uint32_t x = operand_bits(b, state, in, in.src[0], &eok);
            uint32_t y = operand_bits(b, state, in, in.src[1], &eok);
            uint32_t z = operand_bits(b, state, in, in.src[2], &eok);
            uint32_t w = operand_bits(b, state, in, in.src[3], &eok);
            if (terminal_ngg_output && state.exec_narrowed) {
                // The exact compacted-output wrapper emits a contiguous prefix of complete
                // primitives. Map every inactive suffix invocation to one identical clip point;
                // primitives assembled solely from that suffix are therefore degenerate instead of
                // accidentally reusing the last active vertex. Keep the real values on the true path.
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero);
                y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero);
                w = b.sel(state.exec, w, b.uconst(fbits(1.0f)));
            }
            b.export_position(x, y, z, w);
            exported = true;
        } else if (in.exp_target >= 32) {                    // PARAM0.. -> remapped PS input varying
            const uint32_t source = in.exp_target - 32;
            // EN gates which channels the export sends (vec2/vec3 varyings use EN=0x3/0x7):
            // hardware leaves disabled channels unwritten (undefined for the PS). Substitute a
            // deterministic 0.0 for them instead of exporting stale VGPR data.
            uint32_t x = (in.exp_en & 1u) ? operand_bits(b, state, in, in.src[0], &eok) : b.uconst(0);
            uint32_t y = (in.exp_en & 2u) ? operand_bits(b, state, in, in.src[1], &eok) : b.uconst(0);
            uint32_t z = (in.exp_en & 4u) ? operand_bits(b, state, in, in.src[2], &eok) : b.uconst(0);
            uint32_t w = (in.exp_en & 8u) ? operand_bits(b, state, in, in.src[3], &eok) : b.uconst(0);
            if (terminal_ngg_output && state.exec_narrowed) {
                const uint32_t zero = b.uconst(0);
                x = b.sel(state.exec, x, zero); y = b.sel(state.exec, y, zero);
                z = b.sel(state.exec, z, zero); w = b.sel(state.exec, w, zero);
            }
            const uint32_t gx = x, gy = y, gz = z, gw = w;
            if (!pixel_inputs || source >= 32 || !(pixel_inputs->valid_mask & (1u << source)))
                b.export_param(source, gx, gy, gz, gw);       // absent control retains identity wiring
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    const uint32_t raw_offset = pixel_inputs->controls[ps_input] & 0x3Fu;
                    const uint32_t offset = (passthrough_mask & (1u << ps_input))
                        ? (raw_offset & 0x1fu) : raw_offset;
                    if (offset == source) b.export_param(ps_input, gx, gy, gz, gw);
                }
            }
        }
        if (!eok && getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-export-reject] pc=%u target=%u unresolved export operand\n",
                         in.pc, in.exp_target);
        return eok;
    };
    if (!emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                   /*allow_smem*/rt != nullptr, exp_fn, code, dwords)) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] body or export lowering failed\n");
        return {};
    }
    if (passthrough && passthrough->valid) {
        const auto base_it = rs.vreg.find(static_cast<int>(passthrough->producer_base_vgpr));
        if (base_it == rs.vreg.end() || !b.vertex_lds_dwords) {
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[vertex-ngg-passthrough-reject] missing producer LDS base/allocation\n");
            return {};
        }
        b.declare_lds();
        const uint32_t base_dword = b.ibin(
            Op_ShiftRightLogical, base_it->second, b.uconst(2u));
        auto record_load = [&](int32_t dword) -> uint32_t {
            return b.lds_load(dword == 0
                ? base_dword
                : b.ibin(Op_IAdd, base_dword, b.uconst(static_cast<uint32_t>(dword))));
        };
        b.export_position(record_load(passthrough->position[0]),
                          record_load(passthrough->position[1]),
                          record_load(passthrough->position[2]),
                          record_load(passthrough->position[3]));
        exported = true;

        for (uint32_t source = 0; source < passthrough->params.size(); ++source) {
            if (!(passthrough->param_mask & (1u << source))) continue;
            std::array<uint32_t, 4> value{};
            for (uint32_t component = 0; component < 4; ++component) {
                const int32_t dword = passthrough->params[source][component];
                value[component] = dword >= 0 ? record_load(dword) : b.uconst(0u);
            }
            if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source)))
                b.export_param(source, value[0], value[1], value[2], value[3]);
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if ((pixel_inputs->controls[ps_input] & 0x3fu) == source)
                        b.export_param(ps_input, value[0], value[1], value[2], value[3]);
                }
            }
        }
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr,
                         "[vertex-ngg-passthrough] base=v%u stride=%u params=%08x\n",
                         passthrough->producer_base_vgpr,
                         passthrough->record_stride_bytes, passthrough->param_mask);
    }
    if (!exported) {
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[vertex-recompile-reject] shader emitted no POS0\n");
        return {};
    }
    // OFFSET=0x20 asks the interpolator to synthesize a constant instead of consuming a PARAM
    // export. GFX10 DEFAULT_VAL encodes 0000, 0001, 1110, and 1111. Materialize those outputs in
    // the Vulkan vertex stage, whose fixed-function interface has no equivalent default source.
    if (pixel_inputs) {
        for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
            if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
            const uint32_t control = pixel_inputs->controls[ps_input];
            if ((passthrough_mask & (1u << ps_input)) ||
                (control & 0x3Fu) != 0x20u) continue;
            const uint32_t one = b.uconst(0x3F800000u), zero = b.uconst(0u);
            const uint32_t default_val = (control >> 8) & 0x3u;
            const bool xyz_one = (default_val & 0x2u) != 0;
            const bool w_one = (default_val & 0x1u) != 0;
            b.export_param(ps_input, xyz_one ? one : zero, xyz_one ? one : zero,
                           xyz_one ? one : zero, w_one ? one : zero);
        }
    }
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt,
                                       const PixelInputMapping* pixel_inputs,
                                       bool capture_position,
                                       uint32_t virtual_lds_dwords) {
    return recompile_vertex_impl(code, dwords, rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false);
}

std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, true, false);
}

std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, false, true);
}

std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt,
                                             const PixelInputMapping* pixel_inputs,
                                             bool capture_position,
                                             uint32_t virtual_lds_dwords) {
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog, prolog_dwords);
    if (!info.valid || !main || !main_dwords) return {};

    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span || info.prefix_dwords > SIZE_MAX - main_span) return {};
    const NggPassthroughLayout passthrough =
        analyze_ngg_passthrough(prolog, info.prefix_dwords, main, main_span);
    if (passthrough.valid) {
        return recompile_vertex_impl(prolog, info.prefix_dwords, rt, pixel_inputs,
                                     capture_position, virtual_lds_dwords, &passthrough, false, false);
    }
    std::vector<uint32_t> linked;
    linked.reserve(info.prefix_dwords + main_span);
    linked.insert(linked.end(), prolog, prolog + info.prefix_dwords);
    linked.insert(linked.end(), main, main + main_span);
    return recompile_vertex_impl(linked.data(), linked.size(), rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false);
}

} // namespace prosper::gpu
