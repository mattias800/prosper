#include "gpu/recompiler/rdna2_wave32_dispatch_words.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
namespace prosper::gpu {
namespace {
bool native_wave32_vote_launch(const SpirvCompute& b) {
    const auto& launch = b.guest_compute_launch;
    if (!b.is_compute || b.is_fragment || b.is_vertex || b.is_fragment_packet() ||
        b.ngg_workgroup_shell || b.ngg_one_lane || b.wave_size != 32 ||
        b.native_subgroup_size != 32 || b.guest_stage != GuestShaderStage::Compute ||
        b.physical_domain != PhysicalExecutionDomain::Workgroup ||
        launch.wave_size != b.wave_size || launch.local != b.guest_local_size ||
        !compute_dispatch_has_complete_groups(launch))
        return false;
    const uint64_t count = uint64_t{launch.local[0]} * launch.local[1] * launch.local[2];
    return count >= 32 && count % 32 == 0 && count == b.local_count;
}
} // namespace
Wave32DispatchWords::Wave32DispatchWords(std::size_t count)
    : masks(count), ambiguous(count), pairs(count), ambiguous_pairs(count), scalar_words(count),
      marks(count), active_data(count), reachable(count, false), scalar_scc(count, false) {}
bool analyze_wave32_dispatch_words(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const RegState& initial,
    const std::vector<uint32_t>& starts, const std::vector<std::vector<uint32_t>>& successors,
    const std::set<int>& direct_descriptor_sregs, Wave32DispatchWords& words,
    std::set<int>& static_mask_keys, std::unordered_set<uint32_t>& proven_pair_compares,
    const std::function<std::array<int, 2>(const Rdna2Inst&)>& pair_compare_sources,
    const std::unordered_set<uint32_t>& native_mask_scc_vote_pcs, bool barrier_half_phase) {
    auto& b32_mask_in = words.masks;
    auto& b32_mask_ambiguous_in = words.ambiguous;
    auto& b64_mask_in = words.pairs;
    auto& b64_mask_ambiguous_in = words.ambiguous_pairs;
    auto& b32_mask_reachable = words.reachable;
    const bool proven_wave32_masks =
        b.allow_b32_masks && (b.is_fragment || (b.is_compute && b.wave_size == 32));
    const bool compute_scalar_vcc_bridge = allows_compute_scalar_vcc_bridge(b);
    const int max_vgpr = shader_max_vgpr(ins);
    const bool track_words = (b.is_compute || b.is_fragment) && b.wave_size == 32;
    const bool complete_native_vote_launch = native_wave32_vote_launch(b);
    // READFIRST outside an exact common-phase service remains an approximate scalar route.
    // Decline only this new DATA recovery; no ordinary admission or old mask fact is changed.
    const bool approximate_readfirst = std::any_of(ins.begin(), ins.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02;
    });
    words.tracks_active_data =
        tracks_wave32_active_data(b) && !barrier_half_phase && !approximate_readfirst;
    if (b.allow_b32_masks && !starts.empty()) {
        b32_mask_in.front().insert(initial.sreg_bool_b32.begin(), initial.sreg_bool_b32.end());
        for (const auto& mask : initial.sreg_bool) {
            if (!initial.sreg_bool_b32.contains(mask.first) && mask.first != 106 &&
                mask.first != 107)
                b64_mask_in.front().insert(mask.first);
        }
        // `vcc` is the implicit VCC_LO mask even when no instruction has needed an explicit
        // sreg_bool[106] alias yet. Preserve that valid entry lifetime in the same physical-domain
        // analysis as explicit Wave32 masks. A zero SSA id instead means the caller currently has
        // ordinary scalar data in the VCC words, so it deliberately does not seed the mask domain.
        if (initial.vcc) b32_mask_in.front().insert(106);
        if (track_words) {
            seed_native_logical_entry_marks(words.marks.front(), initial);
            if (words.tracks_active_data)
                for (int reg = 0; reg < 3; ++reg) {
                    const auto input = initial.vreg.find(reg);
                    // Only actual compute local-ID Inputs published by this shell, not a
                    // register map's default storage or a guessed launch profile.
                    if (input != initial.vreg.end() && b.localid_comp[reg] &&
                        input->second == b.localid_comp[reg] &&
                        !words.marks.front().relayed_vectors.contains(reg))
                        words.active_data.front().current.insert(reg);
                }
            for (const auto& value : initial.sreg) words.scalar_words.front().insert(value.first);
            for (const auto& value : initial.sreg_input)
                words.scalar_words.front().insert(value.first);
            words.scalar_words.front().insert(direct_descriptor_sregs.begin(),
                                              direct_descriptor_sregs.end());
            words.scalar_scc.front() = initial.scc && !initial.scc_merge_placeholder;
        }
        b32_mask_reachable.front() = true;
        auto implicit_vcc_mask_source = [](const Rdna2Inst& in) -> int {
            if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
                return 106;   // s_cbranch_vccz/nz
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)))
                return 106;   // e32 cndmask and carry-in/out forms use implicit VCC
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x101 || (in.opcode >= 0x128 && in.opcode <= 0x12a))) {
                const Operand& mask = in.src[2];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            if (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x365 || in.opcode == 0x366)) {
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
            auto scalar_words = words.scalar_words[block];
            auto marks = words.marks[block];
            auto active_data = words.active_data[block];
            bool scalar_scc = words.scalar_scc[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;

                // Snapshot actual consumed word domains before an overlapping destination write.
                std::set<int> source_words;
                SpilledVectorSourceMarks source;
                bool definite_scc = false;
                bool genuine_active_sources = false;
                if (track_words) {
                    source_words = scalar_words;
                    source_words.insert(masks.begin(), masks.end());
                    for (int pair : b64_masks) {
                        source_words.insert(pair);
                        source_words.insert(pair + 1);
                    }
                    definite_scc = scalar_scc && !marks.relayed_words.contains(kRelayedScc);
                    source =
                        spilled_vector_source_marks(marks, in, {}, source_words, definite_scc, 32);
                    if (words.tracks_active_data) {
                        ActiveVectorWords view;
                        view.current = active_data.current;
                        auto numeric = scalar_words;
                        for (int mask : masks) numeric.erase(mask);
                        for (int mask : b64_masks) {
                            numeric.erase(mask);
                            numeric.erase(mask + 1);
                        }
                        for (int descriptor : direct_descriptor_sregs) numeric.erase(descriptor);
                        genuine_active_sources = active_vector_sources_defined(
                            view, in, marks.relayed_vectors, masks, numeric, marks.relayed_words,
                            definite_scc);
                    }
                    if (in.fmt == Rdna2Format::SOPP && (in.opcode == 4 || in.opcode == 5)) {
                        if (definite_scc)
                            words.scalar_scc_branches.insert(in.pc);
                        else
                            words.scalar_scc_branches.erase(in.pc);
                    }
                }
                const auto compare_sources = pair_compare_sources(in);
                if (compare_sources[0] >= 0) {
                    const bool proven = b64_masks.contains(compare_sources[0]) &&
                                        b64_masks.contains(compare_sources[1]) &&
                                        !b64_ambiguous.contains(compare_sources[0]) &&
                                        !b64_ambiguous.contains(compare_sources[1]);
                    if (proven)
                        proven_pair_compares.insert(in.pc);
                    else
                        proven_pair_compares.erase(in.pc);
                }

                // The dispatcher may use a false backing value for a physical VCC lifetime that is
                // provably dead. Do not let that implementation placeholder become observable: every
                // instruction whose ISA encoding requires a VCC mask must see one on all incoming
                // paths. Explicit SGPR operands remain governed by the data/mask-domain checks below.
                const int implicit_vcc = implicit_vcc_mask_source(in);
                if (implicit_vcc >= 0 &&
                    (!masks.contains(implicit_vcc) || ambiguous.contains(implicit_vcc))) {
                    log_recompile_diagnostic(
                        b.diagnostic, b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
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
                if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07) &&
                    ambiguous.contains(106))
                    reads_ambiguous = true;
                if (reads_ambiguous) {
                    log_recompile_diagnostic(
                        b.diagnostic, b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
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
                            (masks.contains(source.value) || b64_masks.contains(source.value)));
                };
                auto source_mask = [&](const Operand& source) {
                    return register_mask(source) || source.kind == OperandKind::InlineInt ||
                           source.kind == OperandKind::Literal;
                };
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
                    const bool source_is_mask =
                        register_mask(in.src[0]) ||
                        (in.dst.value == 126 && in.src[0].kind == OperandKind::InlineInt);
                    writes_b32_mask =
                        source_is_mask && in.dst.value != 127 &&
                        (!sop1_opcode_is_emitted_saveexec_b32(in.opcode) || in.dst.value != 126);
                }
                if (in.fmt == Rdna2Format::SOP2 &&
                    (in.opcode == 0x0a ||
                     (in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0))) {
                    const bool scalar_vcc_bridge =
                        compute_scalar_vcc_bridge && is_scalar_cselect_b32_to_vcc_lo(in);
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
                    writes_b32_mask =
                        !scalar_vcchi_packet &&
                        (scalar_vcc_bridge || (mask_domain && source_mask(in.src[0]) &&
                                               source_mask(in.src[1]) && in.dst.value != 127));
                }

                int b32_write_reg = writes_b32_mask ? in.dst.value : -1;
                const bool wave32_b64_vcc_mask_write =
                    in.fmt == Rdna2Format::SOP2 && in.dst.value == 106 && in.opcode >= 0x0f &&
                    in.opcode <= 0x1d && (in.opcode & 1u) == 1 && source_mask(in.src[0]) &&
                    source_mask(in.src[1]);
                if (wave32_b64_vcc_mask_write) b32_write_reg = 106;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 && in.opcode <= 0x12a &&
                    in.sdst.kind == OperandKind::SGPR) {
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
                    else if ((in.opcode >= 0x24 && in.opcode <= 0x2b) || in.opcode == 0x37 ||
                             in.opcode == 0x38)
                        writes_b64_mask = true;
                }
                if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 105) {
                    if (in.opcode == 0x25)
                        writes_b64_mask = true;
                    else if (in.opcode == 0x0b ||
                             (in.opcode >= 0x0f && in.opcode <= 0x1d && (in.opcode & 1u) == 1))
                        writes_b64_mask = source_mask(in.src[0]) && source_mask(in.src[1]);
                }
                const bool vop3_b32_carry =
                    in.fmt == Rdna2Format::VOP3 &&
                    ((in.opcode >= 0x128 && in.opcode <= 0x12a) || vop3b_fresh_carry_output(in));
                // rdna2_decode populates `sdst` only for its explicit ten-opcode VOP3B whitelist:
                // add/sub carry, div-scale flag, and 64-bit multiply-add carry outputs. Each SDST
                // is a per-lane scalar mask/flag; ordinary VOP3A scalar data never appears here
                // (v_readlane uses `dst`, not `sdst`).
                if (vop3_writes_mask_sdst(in) && !vop3_b32_carry && in.sdst.value <= 105)
                    writes_b64_mask = true;
                int b64_write_reg =
                    writes_b64_mask ? (in.fmt == Rdna2Format::VOP3 ? in.sdst.value : in.dst.value)
                                    : -1;

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

                for_each_scalar_write(
                    in,
                    [&](int base, uint32_t width) {
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
                    },
                    proven_wave32_masks);

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
                if (track_words) {
                    std::vector<std::pair<int, uint32_t>> scalar_writes;
                    for_each_scalar_write(
                        in,
                        [&](int base, uint32_t width) { scalar_writes.emplace_back(base, width); },
                        proven_wave32_masks);
                    std::vector<int> vector_writes;
                    for_each_possible_vector_write(in, max_vgpr,
                                                   [&](int reg) { vector_writes.push_back(reg); });
                    if (words.tracks_active_data) {
                        const int mask_write = b32_write_reg >= 0 ? b32_write_reg : b64_write_reg;
                        advance_wave32_active_data(active_data, in, scalar_writes, vector_writes,
                                                   mask_write, genuine_active_sources);
                    }
                    advance_defined_slots(marks, in, {}, source_words, scalar_writes, vector_writes,
                                          source, definite_scc, 32);
                    for (const auto& [base, width] : scalar_writes)
                        for (uint32_t word = 0; word < width; ++word)
                            scalar_words.insert(base + static_cast<int>(word));
                    if (scalar_instruction_writes_scc(in)) {
                        // The numeric inputs must be genuine AND the current lowering must publish
                        // scalar SCC, not the backing false used for an unavailable mask reduction.
                        const bool mask_result =
                            writes_b32_mask || writes_b64_mask || wave32_b64_vcc_mask_write ||
                            ((in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2) &&
                             (in.dst.value == 126 || in.dst.value == 127));
                        const bool fragment_vote =
                            b.is_fragment && initial.reads_scc &&
                            ((in.fmt == Rdna2Format::SOP1 &&
                              (sop1_opcode_is_emitted_saveexec_b32(in.opcode) ||
                               in.opcode == kSop1OpcodeAndSaveexecB64)) ||
                             (in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0e &&
                              in.opcode <= 0x1d));
                        // CONFIDENCE: HIGH -- the dispatcher emits SCC=ANY(result) at these exact
                        // last B32 logicals before a scalar branch. Admit only a current one-word
                        // mask with genuine sources and a complete producing Wave32 launch. An
                        // inverted mask may have set bits outside a partial launch's host cohort.
                        // The existing MAY marks and CFG MUST meet still reject tainted/one-arm data.
                        const bool compute_vote = complete_native_vote_launch && writes_b32_mask &&
                                                  native_mask_scc_vote_pcs.contains(in.pc);
                        scalar_scc =
                            !source.scalar_relay && (!mask_result || fragment_vote || compute_vote);
                    } else if (!scalar_instruction_preserves_scc(in)) {
                        scalar_scc = scalar_scc && !source.scalar_relay;
                    }
                }
            }

            for (uint32_t successor : successors[block]) {
                if (!b32_mask_reachable[successor]) {
                    words.scalar_words[successor] = scalar_words;
                    words.marks[successor] = marks;
                    words.active_data[successor] = active_data;
                    words.scalar_scc[successor] = scalar_scc;
                    b32_mask_reachable[successor] = true;
                    b32_mask_in[successor] = masks;
                    b32_mask_ambiguous_in[successor] = ambiguous;
                    b64_mask_in[successor] = b64_masks;
                    b64_mask_ambiguous_in[successor] = b64_ambiguous;
                    pending.push_back(successor);
                } else {
                    std::set<int> joined_masks;
                    std::set<int> joined_ambiguous = b32_mask_ambiguous_in[successor];
                    std::set_intersection(b32_mask_in[successor].begin(),
                                          b32_mask_in[successor].end(), masks.begin(), masks.end(),
                                          std::inserter(joined_masks, joined_masks.end()));
                    for (int reg : b32_mask_in[successor])
                        if (!masks.contains(reg)) joined_ambiguous.insert(reg);
                    for (int reg : masks)
                        if (!b32_mask_in[successor].contains(reg)) joined_ambiguous.insert(reg);
                    joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                    for (int reg : joined_ambiguous) joined_masks.erase(reg);
                    std::set<int> joined_b64_masks;
                    std::set<int> joined_b64_ambiguous = b64_mask_ambiguous_in[successor];
                    std::set_intersection(b64_mask_in[successor].begin(),
                                          b64_mask_in[successor].end(), b64_masks.begin(),
                                          b64_masks.end(),
                                          std::inserter(joined_b64_masks, joined_b64_masks.end()));
                    for (int reg : b64_mask_in[successor])
                        if (!b64_masks.contains(reg)) joined_b64_ambiguous.insert(reg);
                    for (int reg : b64_masks)
                        if (!b64_mask_in[successor].contains(reg)) joined_b64_ambiguous.insert(reg);
                    joined_b64_ambiguous.insert(b64_ambiguous.begin(), b64_ambiguous.end());
                    for (int reg : joined_b64_ambiguous) joined_b64_masks.erase(reg);
                    auto joined_words = words.scalar_words[successor];
                    std::erase_if(joined_words,
                                  [&](int reg) { return !scalar_words.contains(reg); });
                    auto joined_active = words.active_data[successor];
                    meet_wave32_active_data(joined_active, active_data);
                    auto joined_marks = words.marks[successor];
                    meet_spilled_mask_halves(joined_marks, marks);
                    const bool joined_scc = words.scalar_scc[successor] && scalar_scc;
                    if (joined_active != words.active_data[successor] ||
                        joined_words != words.scalar_words[successor] ||
                        joined_marks != words.marks[successor] ||
                        joined_scc != words.scalar_scc[successor] ||
                        joined_masks != b32_mask_in[successor] ||
                        joined_ambiguous != b32_mask_ambiguous_in[successor] ||
                        joined_b64_masks != b64_mask_in[successor] ||
                        joined_b64_ambiguous != b64_mask_ambiguous_in[successor]) {
                        words.scalar_words[successor] = std::move(joined_words);
                        words.marks[successor] = std::move(joined_marks);
                        words.active_data[successor] = std::move(joined_active);
                        words.scalar_scc[successor] = joined_scc;
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

    return true;
}
void restore_wave32_dispatch_words(const Wave32DispatchWords& words, uint32_t block,
                                   RegState& state) {
    if (block >= words.reachable.size() || !words.reachable[block]) {
        for (const auto& [reg, value] : state.sreg) state.sreg_merge_placeholder.insert(reg);
        for (const auto& [reg, value] : state.vreg) {
            auto& mark = state.vreg_source_marks[reg];
            mark.placeholder = mark.hard = true;
        }
        state.scc_merge_placeholder = true;
        state.vector_full_exec = false;
        return;
    }
    mark_relayed_words(words.marks[block], state);
    for (const auto& [reg, value] : state.sreg)
        if (!words.scalar_words[block].contains(reg)) state.sreg_merge_placeholder.insert(reg);
    if (!words.scalar_scc[block]) state.scc_merge_placeholder = true;
}
}   // namespace prosper::gpu
