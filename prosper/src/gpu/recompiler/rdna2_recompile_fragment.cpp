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



FragmentInterpolationLayout::FragmentInterpolationLayout() {
    for (auto& locations : parameter_locations) locations.fill(kUnusedLocation);
    system_locations.fill(kUnusedLocation);
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

PcrelDispatchInfo rdna2_pcrel_dispatch_info(const uint32_t* code, size_t dwords) {
    PcrelDispatchInfo out;
    if (!code || !dwords) return out;
    std::vector<Rdna2Inst> ins;
    const size_t program_dwords = rdna2_walk(code, dwords, ins);
    return detect_pcrel_dispatch(ins, code, dwords, program_dwords);
}

std::vector<uint32_t> mask_test_branches_for_test(const uint32_t* code, size_t dwords,
                                                  bool wave32) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    const auto branches = mask_test_branches(ins, wave32);
    return std::vector<uint32_t>(branches.begin(), branches.end());
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

namespace {


}  // namespace

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
