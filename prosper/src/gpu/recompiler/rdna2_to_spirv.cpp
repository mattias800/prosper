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
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

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
    if (pixel_inputs) {
        layout.passthrough_mask =
            pixel_inputs->effective_passthrough_mask() & layout.attribute_mask;
        // #3051: the guest's own FLAT_SHADE control bit, independent of which VINTRP opcode the
        // shader used to read the attribute (see PixelInputMapping::effective_flat_mask()). Computed
        // here, before the requires_geometry early-out below, because it must apply to every
        // fragment program -- not only ones that also need the portable interpolation-geometry
        // fallback.
        layout.flat_mask = pixel_inputs->effective_flat_mask() & layout.attribute_mask;
    }

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

uint32_t fragment_consumed_attribute_mask(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return 0;
    // Decoded instruction by instruction rather than through `rdna2_walk`, and this is the whole
    // point of the function. `rdna2_walk` stops at the first `is_end` AND at the first Unknown
    // encoding (rdna2_decode.cpp), which is exactly the prefix `fragment_interpolation_layout`
    // already sees -- so routing through it would make this mask EQUAL to `attribute_mask`, not a
    // superset, and the margin the caller's safety argument depends on would be zero. Stepping past
    // both terminators is what buys the margin.
    //
    // The bias is deliberate and one-directional. Decoding past a terminator can decode data as
    // instructions, so a spurious VINTRP hit is possible; that costs one dead output varying.
    // MISSING an attribute would hand the fragment stage an input the vertex stage no longer
    // exports, which is the regression this analysis must never cause. Over-report, never under.
    const size_t span = rdna2_recompile_code_span(code, dwords);
    uint32_t mask = 0;
    for (size_t pc = 0; pc < span;) {
        const Rdna2Inst instruction = rdna2_decode_one(code + pc, span - pc);
        if (instruction.fmt == Rdna2Format::VINTRP && instruction.vintrp_attr < 32)
            mask |= 1u << instruction.vintrp_attr;
        if (!instruction.len_dwords) break;   // safety: never advance 0
        pc += instruction.len_dwords;
    }
    return mask;
}

bool dead_varying_elimination_enabled() {
    static const bool disabled = getenv("PROSPER_NO_DEAD_VARYING_ELIM") != nullptr;
    return !disabled;
}

void apply_fragment_consumption(PixelInputMapping& mapping,
                                const uint32_t* fragment_code, size_t dwords) {
    if (!dead_varying_elimination_enabled() || !mapping.valid_mask || !fragment_code || !dwords)
        return;
    mapping.consumed_mask = fragment_consumed_attribute_mask(fragment_code, dwords);
    mapping.consumed_known = true;
}

std::vector<uint32_t> recompile_interpolation_geometry(
        const FragmentInterpolationLayout& layout, bool capture_position,
        bool synthesize_rect) {
    SpirvCompute builder;
    return builder.build_interpolation_geometry(layout, capture_position, synthesize_rect);
}

namespace {
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

namespace {
// Defined after the scalar-writer inventory it depends on; used by detect_forward_ifs above it.
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

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

// S_MOV_B64 from VCC publishes both an exact Bool-domain saved mask and its two ballot words under
// native Wave64. The compact structured emitter preserves both views through its SSA/PHI machinery,
// but unlike the dispatcher it previously had no lifetime tag telling S_FF1/S_BCNT which view owns
// the pair at an exact consumer. Compute a small forward MUST analysis over the decoded scalar CFG:
// a saved-mask fact is generated by an exact VCC copy reached from a proved mask-domain VCC or by a
// SAVEEXEC destination (which receives OLD_EXEC independently of its logical source). Every
// overlapping scalar write kills it, and joins retain it only when every reachable predecessor
// agrees. Indirect PC updates have no successor, so they cannot manufacture a dominance fact beyond
// an unknown transfer.

} // namespace

namespace {


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

std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt, uint32_t lds_bytes,
                                     uint32_t compute_pgm_rsrc1,
                                     bool force_cfg_for_test,
                                     uint32_t local_x_for_test,
                                     uint32_t threads_x_for_test) {
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
    if (!local_x_for_test || local_x_for_test > 1024) return {};
    b.begin(num_inputs ? num_inputs : 1, rt, local_x_for_test, 1, 1, 64, 0);
    b.declare_guest_scratch(scratch);
    RegState rs; rs.vcc = b.bfalse(); rs.scc = b.bfalse(); rs.exec = b.btrue();
    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t wpc : waterfall_branches(ins)) safe_branches.insert(wpc);   // readfirstlane waterfalls (#273)
    for (uint32_t k = 0; k < num_inputs; k++) rs.vreg[(int)k] = b.load_input(k);
    // Compute kernels have no EXP output; reject if one appears.
    const auto no_export = [](RegState&, const Rdna2Inst&){ return false; };
    const uint32_t initial_active =
        force_cfg_for_test && threads_x_for_test &&
                threads_x_for_test % local_x_for_test != 0
            ? b.invocation_within_extent(threads_x_for_test, 1, 1)
            : 0;
    const bool emitted = force_cfg_for_test
        ? emit_cfg_state_machine(
              b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
              /*allow_smem*/true, no_export, code, dwords, initial_active, false)
        : emit_body(b, rs, ins, safe_branches, rt, /*allow_exec_update*/true,
                    /*allow_smem*/true, no_export, code, dwords,
                    nullptr, true, 0, false, lds_fminmax_synchronization.needs_dispatcher);
    if (!emitted) return {};
    auto it = rs.vreg.find((int)out_vgpr);
    uint32_t outbits = it == rs.vreg.end() ? b.uconst(0) : it->second;
    // If EXEC is still narrowed (a v_cmpx with no restore), masked-off lanes keep the output slot's prior
    // value; if it was restored to all-lanes-on, every lane stores.
    if (!rs.exec_narrowed) b.store_output(outbits);
    else                   b.store_output_pred(outbits, rs.exec);
    return b.finish();
}

bool fragment_vcc_branch_is_wave_uniform_for_test(
        const uint32_t* code, size_t dwords, uint32_t branch_pc) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    return vcc_exit_is_wave_uniform(instructions, branch_pc);
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

RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords,
                                     std::vector<RecompileUnsupportedSite>* sites) {
    if (sites) sites->clear();
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
        const SavedB64MaskSnapshot saved_masks = snapshot_saved_b64_masks(rs, in);
        const bool emitted = emit_alu(
            b, rs, in, ok, /*allow_exec_update*/true, &safe_branches,
            /*allow_smem*/true, /*rt*/nullptr, /*allow_wave*/true);
        if (emitted && ok)
            record_scalar_write(
                rs, in,
                allows_compute_scalar_vcc_bridge(b), saved_masks);
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
                    // image_get_resinfo 1D/2D/3D, plus 2D_ARRAY (dim 5), which lowers as Dim_2D with
                    // Arrayed and reports the layer count as its third result (#2790). CUBE (dim 3) is
                    // still declined: its stacked-face lowering (#273) would have to promise face-count
                    // semantics for the third result that this has not established.
                    if (i.opcode == 0x0eu) return i.mimg_dim <= 2u || i.mimg_dim == 5u;
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
            // Recorded from the SAME branch that increments the counter, so the enumeration cannot
            // disagree with `unsupported` about which instructions are in the class.
            if (sites) sites->push_back({(int)in.fmt, in.opcode, in.pc});
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
    uint32_t packed = 0;
    // One nibble per MRT, MRT0..MRT7. Sized 2 until 2026-08-15, which silently forced
    // `write_mask &= 0` for slots 2..7 at gpu_execute.hpp's EXP.EN gate -- so a shader exporting to
    // MRT2+ had those attachments dropped no matter what CB_TARGET_MASK and CB_SHADER_MASK said.
    std::array<bool, kFragmentColorOutputs> realized{};
    // This property needs no retained instructions. Match rdna2_walk's bounded decode and
    // termination rules without allocating/copying a complete instruction vector on every draw.
    for (size_t pc = 0; pc < dwords;) {
        const Rdna2Inst in = rdna2_decode_one(code + pc, dwords - pc);
        if (in.is_end || in.fmt == Rdna2Format::Unknown) break;
        if (in.fmt == Rdna2Format::EXP && in.exp_target < realized.size() &&
            !realized[in.exp_target] && in.exp_en != 0) {
            packed |= (in.exp_en & 0xFu) << (in.exp_target * 4u);
            realized[in.exp_target] = true;
        }
        if (in.len_dwords == 0) break;
        pc += in.len_dwords;
    }
    return packed;
}

static std::vector<uint32_t> recompile_fragment_impl(
        const uint32_t* code, size_t dwords,
        const ShaderResourceTable* rt,
        const PixelSystemInputMapping* system_inputs,
        uint32_t pcrel_dispatch_target,
        const FragmentInterpolationLayout* interpolation,
        uint32_t wave_size,
        RecompileDiagnosticContext diagnostic) {
    if (wave_size != 32 && wave_size != 64) return {};
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
                                         bool wave32,
                                         RecompileDiagnosticContext diagnostic) {
    return recompile_fragment_impl(code, dwords, rt, system_inputs,
                                   pcrel_dispatch_target, interpolation,
                                   wave32 ? 32u : 64u, diagnostic);
}

std::vector<uint32_t> recompile_fragment_wave32_for_test(
        const uint32_t* code, size_t dwords) {
    return recompile_fragment_impl(code, dwords, nullptr, nullptr,
                                   UINT32_MAX, nullptr, 32,
                                   {RecompileDiagnosticStage::Fragment, 0});
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

namespace {

struct SpirvInst { uint32_t op; size_t at; uint32_t len; };

// ---------------------------------------------------------------------------------------------
// A shared SPIR-V word model. Following every trailing word as an SSA id treats LITERALS as ids,
// and a literal that happens to equal a result id fabricates a dataflow edge: measured, that made
// a vote "reach" an output through OpExtInst, whose word 4 is a GLSL instruction NUMBER in the
// same numeric range as low result ids. It produced 8 false positives across 49 real modules.
bool spirv_has_result(uint32_t op) {
    switch (op) {
        case Op_Store: case Op_Branch: case Op_BranchConditional:
        case Op_SelectionMerge: case Op_LoopMerge: case Op_Return: case Op_Kill:
        case Op_Decorate: case Op_MemberDecorate: case Op_Switch:
            return false;
        default: return true;
    }
}

// Type instructions carry their result at word 1 (they have no type operand).
uint32_t spirv_result_word(uint32_t op) {
    switch (op) {
        case Op_TypeVoid: case Op_TypeBool: case Op_TypeInt: case Op_TypeFloat:
        case Op_TypeVector: case 24 /*OpTypeMatrix*/: case Op_TypeImage:
        case 26 /*OpTypeSampler*/:
        case Op_TypeSampledImage: case Op_TypeArray: case Op_TypeRuntimeArray:
        case Op_TypeStruct: case Op_TypePointer: case Op_TypeFunction:
        case Op_ExtInstImport:
            return 1;
        default: return 2;
    }
}

bool spirv_is_id_operand(uint32_t op, uint32_t i) {
    switch (op) {
        case Op_ExtInst:        return i == 3 || i >= 5;          // set id, LITERAL instr, args
        case Op_Constant:       return false;                     // value words
        case Op_CompositeExtract: return i == 3;                  // then literal indices
        case 82 /*OpCompositeInsert*/: return i == 3 || i == 4;
        case 79 /*OpVectorShuffle*/: return i == 3 || i == 4;     // then literal components
        case Op_ImageSampleImplicitLod:
        case Op_ImageSampleExplicitLod:
        case Op_ImageFetch: case Op_ImageGather:
            return i == 3 || i == 4 || i >= 6;                    // word 5 is a LITERAL mask
        case Op_TypeInt: case Op_TypeFloat: return false;
        case Op_TypeVector: case 24 /*OpTypeMatrix*/: case Op_TypeImage: return i == 2;
        case Op_TypePointer:    return i == 3;                    // word 2 is LITERAL storage
        case Op_Variable:       return i >= 4;                    // word 3 is LITERAL storage
        default:                return i >= 3;
    }
}

// Values that differ between the invocations of one draw, and so cannot carry uniformity. Listing
// the DIVERGENT sources rather than the uniform ones is deliberate: an opcode nobody thought about
// then has to earn uniformity through its operands instead of inheriting it, so an unknown case
// fails towards "not proven" rather than towards "safe".
bool spirv_op_is_divergent_source(uint32_t op) {
    switch (op) {
        case Op_ImageSampleImplicitLod: case Op_ImageSampleExplicitLod:
        case Op_ImageSampleDrefImplicitLod: case Op_ImageSampleDrefExplicitLod:
        case Op_ImageFetch: case Op_ImageGather: case Op_ImageRead:
        case Op_ImageQuerySizeLod: case Op_ImageQuerySize: case Op_ImageQueryLod:
        case Op_ImageQueryLevels:
        case Op_DPdx: case Op_DPdy:
        case 209 /*OpFwidth*/: case 210 /*OpDPdxFine*/: case 211 /*OpDPdyFine*/:
        case 212 /*OpFwidthFine*/: case 213 /*OpDPdxCoarse*/: case 214 /*OpDPdyCoarse*/:
        case 215 /*OpFwidthCoarse*/:
        case Op_AtomicLoad: case Op_AtomicExchange: case Op_AtomicCompareExchange:
        case Op_AtomicIAdd: case Op_AtomicISub: case Op_AtomicSMin: case Op_AtomicUMin:
        case Op_AtomicSMax: case Op_AtomicUMax: case Op_AtomicAnd: case Op_AtomicOr:
        case Op_AtomicXor:
        case 57 /*OpFunctionCall*/:
            return true;
        default:
            // Every subgroup instruction: its result depends on which lanes share the group, which
            // is the very thing under question here.
            return op >= 333 && op <= 366;
    }
}

bool spirv_op_is_constant(uint32_t op) {
    switch (op) {
        case 41 /*OpConstantTrue*/: case 42 /*OpConstantFalse*/: case Op_Constant:
        case 44 /*OpConstantComposite*/: case 45 /*OpConstantSampler*/:
        case 46 /*OpConstantNull*/:
        case 48 /*OpSpecConstantTrue*/: case 49 /*OpSpecConstantFalse*/:
        case 50 /*OpSpecConstant*/: case 51 /*OpSpecConstantComposite*/:
        case 52 /*OpSpecConstantOp*/:
            return true;
        default: return false;
    }
}

// Atomics, named rather than taken as the numeric window `227..242`: the float and flag forms sit
// outside it. ONE predicate, because the observable check and the leaving-scan were two spellings of
// this set and had already drifted apart once.
bool spirv_op_is_atomic(uint32_t op) {
    return (op >= Op_AtomicLoad && op <= Op_AtomicXor) ||
           op == 318 /*OpAtomicFlagTestAndSet*/ || op == 319 /*OpAtomicFlagClear*/ ||
           op == 5614 /*OpAtomicFMinEXT*/ || op == 5615 /*OpAtomicFMaxEXT*/ ||
           op == 6035 /*OpAtomicFAddEXT*/;
}

// Every way one pointer is derived from another. Missing one splits a single slot into two, which
// loses the dependence between a store through one id and a load through another.
bool spirv_op_derives_pointer(uint32_t op) {
    return op == Op_AccessChain || op == 66 /*OpInBoundsAccessChain*/ ||
           op == 67 /*OpPtrAccessChain*/ || op == 70 /*OpInBoundsPtrAccessChain*/ ||
           op == 83 /*OpCopyObject*/;
}

bool spirv_storage_is_uniform(uint32_t sc, bool module_writes_memory) {
    if (sc == SC_StorageBuffer) return !module_writes_memory;   // a UAV this draw writes is not
    return sc == 2 /*Uniform*/ || sc == SC_PushConstant || sc == SC_UniformConstant;
}

}  // namespace

bool fragment_spirv_wave_width_independent(const std::vector<uint32_t>& spirv) {
    // Unparseable is not proven: a module we cannot read has not been shown to be width-independent,
    // and the whole point of this predicate is that its POSITIVE answer is load-bearing.
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) return false;

    std::vector<SpirvInst> insts;
    for (size_t off = 5; off < spirv.size();) {
        const uint32_t len = spirv[off] >> 16;
        const uint32_t op = spirv[off] & 0xffffu;
        if (!len || len > spirv.size() - off) return false;   // truncated: not proven, see above
        insts.push_back({op, off, len});
        off += len;
    }

    // --- the votes ------------------------------------------------------------------------------
    std::vector<std::pair<uint32_t, uint32_t>> votes;   // (result, operand)
    std::unordered_set<uint32_t> ballots;               // results whose WIDTH is the answer
    for (const SpirvInst& in : insts) {
        if (in.op == Op_GroupNonUniformAny && in.len >= 5) {
            votes.push_back({spirv[in.at + 2], spirv[in.at + 4]});
        } else if (in.op == Op_GroupNonUniformBallot && in.len >= 5) {
            votes.push_back({spirv[in.at + 2], spirv[in.at + 4]});
            ballots.insert(spirv[in.at + 2]);
        }
    }
    if (votes.empty()) return true;                     // nothing width-dependent to prove

    // --- storage classes ------------------------------------------------------------------------
    std::unordered_map<uint32_t, uint32_t> var_storage;
    std::unordered_set<uint32_t> outputs, locals;
    for (const SpirvInst& in : insts) {
        if (in.op != Op_Variable || in.len < 4) continue;
        const uint32_t storage = spirv[in.at + 3];
        var_storage[spirv[in.at + 2]] = storage;
        if (storage == SC_Output) outputs.insert(spirv[in.at + 2]);
        else if (storage == SC_Function) locals.insert(spirv[in.at + 2]);
    }
    // Access chains inherit their base's class, transitively. Two passes over the instruction list
    // are not enough in general, so this runs to a fixed point.
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_op_derives_pointer(in.op) || in.len < 4) continue;
            const uint32_t base = spirv[in.at + 3], result = spirv[in.at + 2];
            if (outputs.count(base) && outputs.insert(result).second) changed = true;
            if (locals.count(base) && locals.insert(result).second) changed = true;
        }
    }

    // The VARIABLE a pointer roots in. Taint through memory has to be keyed on the slot, not on the
    // pointer id: the emitter mints a fresh OpAccessChain per guest-scratch access, so a
    // read-modify-write of one slot goes through two different ids and a store/load pair keyed by id
    // drops the dependence between them entirely.
    std::unordered_map<uint32_t, uint32_t> pointer_root;
    for (const SpirvInst& in : insts)
        if (in.op == Op_Variable && in.len >= 3) pointer_root[spirv[in.at + 2]] = spirv[in.at + 2];
    // Every way one pointer is derived from another. OpCopyObject and the InBounds/Ptr chain forms
    // each split one slot into two if they are not followed.
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_op_derives_pointer(in.op) || in.len < 4) continue;
            const auto base = pointer_root.find(spirv[in.at + 3]);
            if (base == pointer_root.end()) continue;
            if (pointer_root.emplace(spirv[in.at + 2], base->second).second) changed = true;
        }
    }
    // An UNKNOWN pointer must not root at itself. That was an identity fallback dressed as a
    // lookup: a pointer the map has never seen became its own slot, so a store through one id and a
    // load through another lost the dependence between them -- exactly the defect the root map was
    // added to fix, reintroduced by its own default. Unknown now means "could be any local", which
    // is conservative on both sides. (The asymmetry that hid this: an unrooted pointer was already
    // safe on the STORE side, because a pointer not in `locals` reads as an observable effect.)
    const auto pointer_is_known = [&](uint32_t ptr) { return pointer_root.count(ptr) != 0; };
    const auto root_of = [&](uint32_t ptr) {
        const auto it = pointer_root.find(ptr);
        return it == pointer_root.end() ? ptr : it->second;
    };

    // What counts as leaving the shader. A store through anything that is not a Function-storage
    // pointer is observable -- a colour attachment, a storage buffer, an image. Restricting this to
    // Output would make a UAV write invisible to the analysis, which is the same shape of blind spot
    // as the control-dependence one, one level down.
    const auto is_observable_effect = [&](const SpirvInst& in) {
        switch (in.op) {
            case Op_Store:
                return in.len >= 3 && !locals.count(spirv[in.at + 1]);
            case 63 /*OpCopyMemory*/:
                return in.len >= 3 && !locals.count(spirv[in.at + 1]);
            case 64 /*OpCopyMemorySized*/:
                return in.len >= 4 && !locals.count(spirv[in.at + 1]);
            case Op_ImageWrite: case Op_Kill:
            case 4416 /*OpTerminateInvocation*/: case 5380 /*OpDemoteToHelperInvocation*/:
            case Op_ControlBarrier: case Op_MemoryBarrier:
                return true;
            // Atomics both read and write memory the rest of the draw can see. Named explicitly
            // rather than as the numeric window `227..242`: the float and flag atomics sit outside
            // it, so a vote-guarded OpAtomicFAddEXT admitted while the byte-identical module using
            // OpAtomicIAdd refused. One opcode number was the whole difference.
            default:
                return spirv_op_is_atomic(in.op);
        }
    };

    // --- arm (a): the vote's operand is wave-uniform ---------------------------------------------
    // Any(P) reduces P over whatever lanes the group holds. When P takes the same value in every
    // invocation of the draw, that reduction IS P -- at 64 lanes, at 32, at any width. So a vote
    // over a provably uniform operand answers identically however the hardware groups lanes, and
    // nothing downstream of it can move. This is the arm that carries the corpus: the guest's
    // `s_cbranch_execz`-style tests are overwhelmingly comparisons of constant-buffer scalars.
    //
    // One qualification, and it is the difference between a predicate that happens to be right on
    // the shaders in hand and one whose argument holds: "uniform" must mean the same value in EVERY
    // invocation of this draw. A read-only constant buffer satisfies that. A storage buffer this
    // shader also WRITES does not -- one invocation's store changes what another's load returns --
    // so when the module writes buffer or image memory at all, StorageBuffer loads stop counting.
    // Measured, 0 of the 49 corpus modules write any, so the guard costs nothing today.
    // Defined in terms of is_observable_effect rather than as its own list. These are the same
    // question at two scopes -- "can the rest of the draw see this?" versus "can this shader read it
    // back?" -- and keeping them as separate lists let them drift twice: this one kept the
    // `227..242` atomic window after the other moved to a named set, and never learned about
    // OpCopyMemory at all. A write to memory is an observable effect that is not a store to a colour
    // output, which nothing reads back.
    bool writes_memory = false;
    for (const SpirvInst& in : insts) {
        if (!is_observable_effect(in)) continue;
        // Observable and yet writes NOTHING a later load could read back: ending the invocation,
        // or ordering other threads' accesses. Measured -- folding these in cost Evergate 10 of its
        // 11 provable modules, because a module containing a barrier stopped being able to treat
        // any constant buffer as uniform.
        switch (in.op) {
            case Op_Kill: case 4416 /*OpTerminateInvocation*/:
            case 5380 /*OpDemoteToHelperInvocation*/:
            case Op_ControlBarrier: case Op_MemoryBarrier:
                continue;
            default: break;
        }
        // A store to a colour output is observable but is not memory this shader reads back.
        if ((in.op == Op_Store || in.op == 63 /*OpCopyMemory*/ || in.op == 64) &&
            in.len >= 2 && outputs.count(spirv[in.at + 1]))
            continue;
        writes_memory = true;
        break;
    }
    // Pointers that root in a read-only uniform storage class, and the access chains over them.
    // Kept SEPARATE from the value set on purpose: a pointer's uniformity says something about
    // where it points, and a load's uniformity is about what is stored there. Conflating the two
    // made a Function local with a uniform initializer -- an initializer is an id operand, so the
    // generic rule marked the variable uniform -- lend that uniformity to values written into it
    // per lane. Constructed by hand; the corpus has no instance, which is exactly why.
    std::unordered_set<uint32_t> uniform_ptrs;
    for (const SpirvInst& in : insts)
        if (in.op == Op_Variable && in.len >= 4 &&
            spirv_storage_is_uniform(spirv[in.at + 3], writes_memory))
            uniform_ptrs.insert(spirv[in.at + 2]);

    std::unordered_set<uint32_t> uniform;
    for (const SpirvInst& in : insts)
        if (spirv_op_is_constant(in.op) && in.len >= 3) uniform.insert(spirv[in.at + 2]);
    for (bool changed = true; changed;) {
        changed = false;
        for (const SpirvInst& in : insts) {
            if (!spirv_has_result(in.op)) continue;
            const uint32_t rw = spirv_result_word(in.op);
            if (in.len <= rw) continue;
            const uint32_t result = spirv[in.at + rw];
            if (uniform.count(result)) continue;
            if (in.op == Op_GroupNonUniformAny) {
                // A vote's own result is uniform once its operand is: an all-lanes reduction of a
                // value every lane already agrees on. Chained votes resolve by fixed point.
                if (in.len >= 5 && uniform.count(spirv[in.at + 4])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            if (spirv_op_is_divergent_source(in.op)) continue;
            if (in.op == Op_Load) {
                // ONLY from a pointer that roots in a read-only uniform class. Never from "this
                // pointer value is uniform": a Function local's pointer is perfectly uniform and
                // what it holds is whatever was last stored into it, which may be per lane.
                if (in.len >= 4 && uniform_ptrs.count(spirv[in.at + 3])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            if (in.op == Op_AccessChain) {
                // A chain inherits its base's class, and is a uniform POINTER only when every
                // index is uniform too -- a per-lane index selects a different element per lane.
                if (in.len >= 4 && uniform_ptrs.count(spirv[in.at + 3]) &&
                    !uniform_ptrs.count(result)) {
                    bool indices_uniform = true;
                    for (uint32_t i = 4; i < in.len; ++i)
                        if (!uniform.count(spirv[in.at + i])) { indices_uniform = false; break; }
                    if (indices_uniform) { uniform_ptrs.insert(result); changed = true; }
                }
                continue;
            }
            if (in.op == Op_Variable) continue;   // a pointer, never a value; see uniform_ptrs
            if (in.op == Op_Phi) {
                // IDENTICAL incoming values, not merely uniform ones. A phi's whole job is to
                // choose between its edges, so "every value is uniform" says nothing about the
                // result when the branch doing the choosing is divergent:
                //
                //     p = divergent;  if (p) {} else {};  q = phi(true, false);  a = Any(q)
                //
                // both constants are uniform, and q is exactly p. Admitting that vote lets a
                // 64-lane wave and two 32-lane groups answer differently, which is the defect this
                // predicate exists to prevent. Identical ids are safe for the opposite reason:
                // which edge ran cannot matter when every edge carries the same value.
                //
                // Proving the SELECTION uniform would admit more, and is left undone rather than
                // guessed: measured over the 49-module corpus this restriction costs nothing.
                bool all = in.len > 4;
                for (uint32_t i = 5; i + 1 < in.len; i += 2)
                    if (spirv[in.at + i] != spirv[in.at + 3]) { all = false; break; }
                if (all && uniform.count(spirv[in.at + 3])) {
                    uniform.insert(result);
                    changed = true;
                }
                continue;
            }
            bool any_operand = false, all_uniform = true;
            for (uint32_t i = 1; i < in.len; ++i) {
                if (i == rw || !spirv_is_id_operand(in.op, i)) continue;
                any_operand = true;
                if (!uniform.count(spirv[in.at + i])) { all_uniform = false; break; }
            }
            if (any_operand && all_uniform) { uniform.insert(result); changed = true; }
        }
    }

    // --- the CFG, for arm (b) -------------------------------------------------------------------
    std::unordered_map<uint32_t, std::vector<SpirvInst>> blocks;
    std::unordered_map<uint32_t, std::vector<uint32_t>> succ;
    std::unordered_map<uint32_t, uint32_t> merge_of, branch_cond;
    uint32_t current = 0;
    for (const SpirvInst& in : insts) {
        if (in.op == Op_Label && in.len >= 2) {
            current = spirv[in.at + 1];
            blocks[current];
            continue;
        }
        if (!current) continue;
        blocks[current].push_back(in);
        if ((in.op == Op_SelectionMerge || in.op == Op_LoopMerge) && in.len >= 2) {
            merge_of[current] = spirv[in.at + 1];
        } else if (in.op == Op_Branch && in.len >= 2) {
            succ[current] = {spirv[in.at + 1]};
        } else if (in.op == Op_BranchConditional && in.len >= 4) {
            branch_cond[current] = spirv[in.at + 1];
            succ[current] = {spirv[in.at + 2], spirv[in.at + 3]};
        } else if (in.op == Op_Switch && in.len >= 3) {
            // The SELECTOR is a branch condition too. Modelling only the successors left a
            // vote-tainted switch with no control dependence at all, and the CFG emitter really
            // does build one -- on the guest PC (rdna2_emit_cfg.cpp's dispatcher).
            branch_cond[current] = spirv[in.at + 1];
            std::vector<uint32_t> targets{spirv[in.at + 2]};
            for (uint32_t i = 4; i < in.len; i += 2) targets.push_back(spirv[in.at + i]);
            succ[current] = targets;
        }
    }

    // --- arm (b): the vote cannot influence a colour output --------------------------------------
    // Data flow alone is NOT enough, and believing it was is the defect this replaced. A vote used
    // only as a branch condition never flows into a store, yet it decides WHICH store runs: with
    // EXEC true in the upper half and false in the lower, a 64-lane any() takes the branch where two
    // independent 32-lane groups disagree, and the constant the taken block writes lands in one half
    // only. Measured over the same 49 modules, the value-reachability question clears 49 of 49 while
    // this one clears 1 -- the earlier predicate was not incomplete at the margin, it was blind to
    // the whole population.
    const auto influences_output = [&](uint32_t seed) {
        std::unordered_set<uint32_t> tainted{seed}, tainted_ptrs;
        // A bounded fixed point, and the bound is conservative on purpose: falling out of the loop
        // still growing means the closure was NOT complete, so the honest answer is "it might" --
        // returning false there would admit a module on an unfinished analysis.
        bool settled = false;
        for (int round = 0; round < 32 && !settled; ++round) {
            for (bool changed = true; changed;) {           // data flow to a fixed point
                changed = false;
                for (const SpirvInst& in : insts) {
                    if (in.op == Op_Store && in.len >= 3) {
                        const uint32_t raw = spirv[in.at + 1], val = spirv[in.at + 2];
                        if (tainted.count(val)) {
                            if (!pointer_is_known(raw)) {          // unknown: could be any of them
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) changed = true;
                            } else if (locals.count(root_of(raw)) &&
                                       tainted_ptrs.insert(root_of(raw)).second) {
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if ((in.op == 63 /*OpCopyMemory*/ || in.op == 64 /*OpCopyMemorySized*/) &&
                        in.len >= 3) {
                        // Target <- Source, both pointers. A tainted source slot makes the target
                        // slot tainted; no OpLoad/OpStore pair appears, which is why the closure
                        // could not see it.
                        const uint32_t dst = spirv[in.at + 1], src = spirv[in.at + 2];
                        const bool src_tainted = pointer_is_known(src)
                            ? tainted_ptrs.count(root_of(src)) != 0
                            : !tainted_ptrs.empty();
                        if (src_tainted) {
                            if (!pointer_is_known(dst)) {
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) changed = true;
                            } else if (locals.count(root_of(dst)) &&
                                       tainted_ptrs.insert(root_of(dst)).second) {
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if (in.op == Op_Load && in.len >= 4) {
                        const uint32_t raw = spirv[in.at + 3];
                        const bool reads_tainted = pointer_is_known(raw)
                            ? tainted_ptrs.count(root_of(raw)) != 0
                            : !tainted_ptrs.empty();               // unknown: could be any of them
                        if (reads_tainted && tainted.insert(spirv[in.at + 2]).second)
                            changed = true;
                        continue;
                    }
                    if (!spirv_has_result(in.op)) continue;
                    const uint32_t rw = spirv_result_word(in.op);
                    if (in.len <= rw) continue;
                    const uint32_t result = spirv[in.at + rw];
                    if (tainted.count(result)) continue;
                    for (uint32_t i = 1; i < in.len; ++i) {
                        if (i == rw || !spirv_is_id_operand(in.op, i)) continue;
                        if (!tainted.count(spirv[in.at + i])) continue;
                        tainted.insert(result);
                        changed = true;
                        break;
                    }
                }
            }
            for (const SpirvInst& in : insts) {              // a tainted value leaving the shader
                if (in.op == Op_Store && in.len >= 3 && !locals.count(spirv[in.at + 1]) &&
                    tainted.count(spirv[in.at + 2]))
                    return true;
                if (in.op == Op_ImageWrite && in.len >= 4 &&
                    (tainted.count(spirv[in.at + 2]) || tainted.count(spirv[in.at + 3])))
                    return true;
                // EVERY id operand, not word 3. Word 3 is an atomic's POINTER; the value it
                // writes is further along and its index differs by opcode, so checking one word
                // let a vote-derived value be atomically written to a UAV in straight-line code --
                // the scan fired and read the wrong word. Testing all of them needs no per-opcode
                // value index and cannot drift as opcodes are added.
                if (spirv_op_is_atomic(in.op)) {
                    for (uint32_t i = 3; i < in.len; ++i)
                        if (tainted.count(spirv[in.at + i])) return true;
                }
                // OpCopyMemory moves a value without an OpLoad/OpStore pair, so a tainted local
                // copied straight out was invisible here even though is_observable_effect knew the
                // opcode. Source is word 2; a tainted SOURCE leaving through a non-local target is
                // the same event as a tainted store.
                if ((in.op == 63 /*OpCopyMemory*/ || in.op == 64 /*OpCopyMemorySized*/) &&
                    in.len >= 3 && !locals.count(spirv[in.at + 1])) {
                    // The SOURCE is a POINTER, so its taint lives in tainted_ptrs -- a slot is
                    // tainted, not a value id. Testing `tainted` here made the arm inert, which is
                    // what its own fixture caught.
                    const uint32_t src = spirv[in.at + 2];
                    if (pointer_is_known(src) ? tainted_ptrs.count(root_of(src)) != 0
                                              : !tainted_ptrs.empty())
                        return true;
                }
            }

            bool grew = false;                               // then control dependence
            for (const auto& entry : branch_cond) {
                if (!tainted.count(entry.second)) continue;
                const uint32_t head = entry.first;
                const auto merge_it = merge_of.find(head);
                if (merge_it == merge_of.end())
                    return true;   // unstructured: the region has no bound we can trust
                const uint32_t merge = merge_it->second;
                // Every block strictly inside the construct executes only because the vote said so.
                std::unordered_set<uint32_t> inside;
                std::vector<uint32_t> stack;
                if (succ.count(head)) stack = succ[head];
                while (!stack.empty()) {
                    const uint32_t b = stack.back();
                    stack.pop_back();
                    if (b == merge || !blocks.count(b) || !inside.insert(b).second) continue;
                    if (succ.count(b))
                        for (uint32_t s : succ[b]) stack.push_back(s);
                }
                for (uint32_t b : inside) {
                    for (const SpirvInst& in : blocks[b]) {
                        if (is_observable_effect(in)) return true;
                        // A store into a local inside the region is control-dependent too: whether
                        // it happened at all is the vote's answer, so every later load of that
                        // local carries the vote.
                        if (in.op == Op_Store && in.len >= 3) {
                            const uint32_t raw = spirv[in.at + 1];
                            if (!pointer_is_known(raw)) {
                                for (uint32_t local : locals)
                                    if (tainted_ptrs.insert(local).second) grew = true;
                            } else if (locals.count(root_of(raw)) &&
                                       tainted_ptrs.insert(root_of(raw)).second) {
                                grew = true;
                            }
                        }
                    }
                }
                const auto merge_blk = blocks.find(merge);
                if (merge_blk == blocks.end()) continue;
                for (const SpirvInst& in : merge_blk->second) {
                    if (in.op != Op_Phi || in.len < 5) continue;
                    const uint32_t result = spirv[in.at + 2];
                    if (tainted.count(result)) continue;
                    bool differs = false;
                    for (uint32_t i = 5; i + 1 < in.len; i += 2)
                        if (spirv[in.at + i] != spirv[in.at + 3]) { differs = true; break; }
                    if (differs) { tainted.insert(result); grew = true; }
                }
            }
            settled = !grew;
        }
        return !settled;
    };

    for (const auto& vote : votes) {
        // A ballot is never cleared by uniformity: its result is a per-lane bit MASK whose width is
        // the subgroup's, so a 32-lane ballot reports half a mask as though it were whole however
        // uniform the value being balloted. Only deadness can clear one.
        if (!ballots.count(vote.first) && uniform.count(vote.second)) continue;
        if (!influences_output(vote.first)) continue;
        return false;
    }
    return true;
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

    const bool prologlog = getenv("PROSPER_PROLOGLOG") != nullptr;
    uint64_t phash = 0xcbf29ce484222325ull;
    if (prologlog)
        for (size_t i = 0; i < dwords && i < 4096; ++i) phash = (phash ^ code[i]) * 0x100000001b3ull;
    auto prolog_note = [&](const char* what, const Rdna2Inst* at) {
        if (!prologlog) return;
        static std::set<uint64_t> seen; static std::mutex mx;
        std::lock_guard<std::mutex> lk(mx);
        if (!seen.insert(phash).second) return;
        fprintf(stderr, "[prologlog] hash=%016llx dwords=%zu %s pc=%u fmt=%d\n",
                (unsigned long long)phash, dwords, what, at ? at->pc : 0u, at ? (int)at->fmt : -1);
    };
    // Only the prefix before the transfer can belong to a fetch prolog. Track its branch-target
    // bounds while decoding, so neither the prefix nor the unused shader tail needs a vector.
    int64_t min_branch_target = 0;
    int64_t max_branch_target = 0;
    for (size_t pc = 0; pc < dwords;) {
        Rdna2Inst instruction = rdna2_decode_one(code + pc, dwords - pc);
        instruction.pc = static_cast<uint32_t>(pc);
        // A fetch prolog has no architectural output or program termination of its own. Encountering
        // either before the transfer means this is a complete/different shader, not the split ABI.
        if (instruction.is_end || instruction.fmt == Rdna2Format::EXP ||
            instruction.fmt == Rdna2Format::Unknown) {
            prolog_note("BAIL", &instruction);
            return {};
        }
        if (instruction.fmt == Rdna2Format::SOP1 && instruction.opcode == 0x20) {
            // GFX9+ merged-stage fetch prologs receive the continuation PC in reserved s[6:7].
            // Keep this exact pair: an arbitrary indirect jump must not become host fallthrough.
            if (instruction.n_src != 1 || instruction.src[0].kind != OperandKind::SGPR ||
                instruction.src[0].value != 6 || instruction.len_dwords != 1)
                return {};
            prolog_note("TRANSFER", &instruction);
            // Every direct branch must remain inside the retained prefix or land exactly on the
            // transfer (main pc0 after linking), never in discarded padding/data. The linked body
            // recompiler performs the remaining structured-CFG validation.
            if (instruction.pc == 0 || min_branch_target < 0 ||
                max_branch_target > instruction.pc)
                return {};
            result.valid = true;
            result.setpc_pc = instruction.pc;
            result.prefix_dwords = instruction.pc;
            return result;
        }
        if (instruction.fmt == Rdna2Format::SOPP &&
            (instruction.opcode == 0x02 ||
             (instruction.opcode >= 0x04 && instruction.opcode <= 0x09))) {
            const int64_t target = static_cast<int64_t>(instruction.pc) + instruction.len_dwords +
                                   static_cast<int64_t>(instruction.simm16);
            min_branch_target = std::min(min_branch_target, target);
            max_branch_target = std::max(max_branch_target, target);
        }
        if (instruction.len_dwords == 0) break;
        pc += instruction.len_dwords;
    }
    return {};
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
                                                   bool allow_test_ngg_one_lane,
                                                   RecompileDiagnosticContext diagnostic) {
    const uint32_t passthrough_mask =
        pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const StaticScratchLayout scratch = analyze_static_scratch(ins);

    SpirvCompute b;
    // The vertex stage recorded its rejects at address 0, and record_terminal_reject_reason()
    // early-returns on a zero address -- so `last_terminal_reject_reason()` came back empty for
    // EVERY vertex program and the unconditional skip line could only print `reason=unrecorded`.
    // #3130 fixed exactly this for the fragment stage; the vertex half was left behind.
    b.diagnostic = diagnostic;
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
        // Which instruction disqualified the model, not just that something did. The all-ones pair
        // is lowerable on its own, so on a mixed program the reject surfaces at IT and names a pc
        // that is not the cause -- Stray's 0x300f190000 rejects at pc=4 while the general-mask
        // forms that disqualified it are at pc 277-286 (#3135).
        if (!all_ones && b.vertex_general_mask_mbcnt_pc == UINT32_MAX)
            b.vertex_general_mask_mbcnt_pc = in.pc;
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
        // MUBUF-only is DELIBERATE, not an oversight now that `output_tables.mtbuf` exists three
        // lines up (#2859). This gate is a narrowing allow-list for the byte-exact Astro Bot
        // compacted-output wrapper; admitting a typed consumer would widen an NGG output gate with no
        // title evidence behind it. A typed embedded table still FOLDS -- it just does not open this
        // gate, which is the conservative direction.
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
            // #2945: skip a slot the fragment program never reads. See PixelInputMapping::consumes
            // -- SPI_PS_INPUT_CNTL is sticky, so `valid_mask` alone fans one export out to 32
            // locations and overruns maxVertexOutputComponents.
            if (!pixel_inputs || source >= 32 || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))
                    b.export_param(source, gx, gy, gz, gw);   // absent control retains identity wiring
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;
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
            if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source))) {
                if (!pixel_inputs || pixel_inputs->consumes(source))   // #2945
                    b.export_param(source, value[0], value[1], value[2], value[3]);
            }
            if (pixel_inputs) {
                for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input) {
                    if (!(pixel_inputs->valid_mask & (1u << ps_input))) continue;
                    if (!pixel_inputs->consumes(ps_input)) continue;    // #2945
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
            if (!pixel_inputs->consumes(ps_input)) continue;            // #2945
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
        // A PS can read a PARAM that this VS never exports (#3416: a position-only guest VS
        // paired with a PS reading attr0). Vulkan requires the matching declaration even when
        // its value is unwritten. Complete only the proven consumed interface; do not invent a
        // store/default or expand sticky controls into 32 outputs. Real exports and explicit
        // DEFAULT_VAL controls above still supply their values through the same variables.
        if (pixel_inputs->consumed_known) {
            for (uint32_t ps_input = 0; ps_input < pixel_inputs->controls.size(); ++ps_input)
                if (pixel_inputs->consumes(ps_input)) b.vtx_output(ps_input);
        }
    }
    return b.finish();
}

std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt,
                                       const PixelInputMapping* pixel_inputs,
                                       bool capture_position,
                                       uint32_t virtual_lds_dwords,
                                       RecompileDiagnosticContext diagnostic) {
    return recompile_vertex_impl(code, dwords, rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

std::vector<uint32_t> recompile_vertex_terminal_ngg_gate_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, true, false,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_ngg_one_lane_for_test(
    const uint32_t* code, size_t dwords) {
    return recompile_vertex_impl(code, dwords, nullptr, nullptr, false, 0, nullptr, false, true,
                                 {RecompileDiagnosticStage::Vertex, 0});
}

std::vector<uint32_t> recompile_vertex_chain(const uint32_t* prolog, size_t prolog_dwords,
                                             const uint32_t* main, size_t main_dwords,
                                             const ShaderResourceTable* rt,
                                             const PixelInputMapping* pixel_inputs,
                                             bool capture_position,
                                             uint32_t virtual_lds_dwords,
                                             RecompileDiagnosticContext diagnostic) {
    const VertexPrologInfo info = rdna2_vertex_prolog_info(prolog, prolog_dwords);
    if (!info.valid || !main || !main_dwords) return {};

    const size_t main_span = rdna2_recompile_code_span(main, main_dwords);
    if (!main_span || info.prefix_dwords > SIZE_MAX - main_span) return {};
    const NggPassthroughLayout passthrough =
        analyze_ngg_passthrough(prolog, info.prefix_dwords, main, main_span);
    if (passthrough.valid) {
        return recompile_vertex_impl(prolog, info.prefix_dwords, rt, pixel_inputs,
                                     capture_position, virtual_lds_dwords, &passthrough, false,
                                     false, diagnostic);
    }
    std::vector<uint32_t> linked;
    linked.reserve(info.prefix_dwords + main_span);
    linked.insert(linked.end(), prolog, prolog + info.prefix_dwords);
    linked.insert(linked.end(), main, main + main_span);
    return recompile_vertex_impl(linked.data(), linked.size(), rt, pixel_inputs, capture_position,
                                 virtual_lds_dwords, nullptr, false, false, diagnostic);
}

} // namespace prosper::gpu
