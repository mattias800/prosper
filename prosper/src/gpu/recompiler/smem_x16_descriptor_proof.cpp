// smem_x16_descriptor_proof.cpp -- which s_load_dwordx16 loads are PAIRS OF IMAGE DESCRIPTORS?
//
// A wide scalar load is not intrinsically a descriptor fetch: admitting every x16 as two T#s would
// replace real scalar data with zero placeholders. The proof below admits a load only when every one
// of its sixteen words is later consumed as an image descriptor (or overwritten first). Split out of
// rdna2_emit_cfg.cpp, whose file-size cap it was pushing against; the emitter consumes the result
// through smem_x16_descriptor_proof.hpp.
#include "gpu/recompiler/smem_x16_descriptor_proof.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu {
namespace {

// Prove the narrow S_LOAD_DWORDX16 descriptor-bundle shape used by compute kernels. A wide
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

// Branches are common between a pair of descriptor loads and their final register overwrite.
// This proof deliberately admits only unmodified x16 bundles. The separate linear proof below
// owns the word3 patch shape; a branch through that patch needs a distinct path-sensitive proof.
bool smem_x16_branch_lifetime_is_descriptors(const std::vector<Rdna2Inst>& ins,
                                             const std::unordered_map<uint32_t, size_t>& by_pc,
                                             size_t start, const ShaderResourceTable& rt,
                                             uint32_t wave_size) {
    // The exact-PC image resources describe dispatch-time CPU bytes. A preceding shader write
    // could change the descriptor before this scalar load observes it, even when the load
    // dominates every image use. Unknown aliases keep the ordinary unresolved path.
    for (size_t i = 0; i < start; ++i)
        if (rdna2_may_write_guest_memory(ins[i])) return false;
    const int base = ins[start].dst.value;
    struct State { size_t index; uint16_t live; };
    std::vector<State> pending{{start + 1, 0xffffu}};
    std::unordered_set<uint64_t> visited;
    bool consumed[2] = {false, false};
    std::unordered_set<size_t> consumer_indices;
    auto overlaps = [base](uint16_t live, int first, uint32_t width) {
        if (first < 0) return false;
        for (uint32_t word = 0; word < 16; ++word)
            if ((live & (1u << word)) && first <= base + static_cast<int>(word) &&
                base + static_cast<int>(word) < first + static_cast<int>(width)) return true;
        return false;
    };
    while (!pending.empty()) {
        const State state = pending.back();
        pending.pop_back();
        if (!state.live) continue;
        if (state.index >= ins.size()) return false;
        const uint64_t key = (static_cast<uint64_t>(state.index) << 16u) | state.live;
        if (!visited.insert(key).second) continue;
        if (visited.size() > 8192) return false; // bound pathological branch/liveness products
        const Rdna2Inst& in = ins[state.index];
        if (in.fmt == Rdna2Format::Unknown || !in.len_dwords ||
            (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16u)) return false;
        if (in.is_end) continue;

        bool image_source = false;
        if (in.fmt == Rdna2Format::MIMG) {
            const int tbase = in.src[1].kind == OperandKind::SGPR ? in.src[1].value : -1;
            const int half = tbase == base ? 0 : tbase == base + 8 ? 1 : -1;
            if (overlaps(state.live, in.src[2].value,
                         in.src[2].kind == OperandKind::SGPR ? 4u : 0u)) return false;
            if (overlaps(state.live, tbase, 8)) {
                const ShaderResource* resource = rt.by_fetch_pc(in.pc);
                const uint16_t mask = static_cast<uint16_t>(0xffu << (half == 1 ? 8 : 0));
                if (half < 0 || (state.live & mask) != mask || !resource ||
                    resource->fetch_pc != in.pc || resource->srt_offset != 0xffffffffu ||
                    resource->sgpr_base != 0xffffffffu ||
                    (resource->cls != ResourceClass::Texture &&
                     resource->cls != ResourceClass::StorageImage)) return false;
                consumed[half] = true;
                consumer_indices.insert(state.index);
                image_source = true;
            }
        }
        if (scalar_implicit_destination_read_width(in) &&
            overlaps(state.live, in.dst.value, scalar_implicit_destination_read_width(in)))
            return false;
        if (in.fmt == Rdna2Format::SOPK && in.dst.kind == OperandKind::SGPR &&
            overlaps(state.live, in.dst.value, 1)) return false;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const Operand& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special &&
                  operand.value >= 106 && operand.value <= 124)) continue;
            if (image_source && source == 1) continue;
            uint32_t width = 1;
            if (in.fmt == Rdna2Format::MIMG && source == 2) width = 4;
            else if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                     source == 1) width = 4;
            else if (in.fmt == Rdna2Format::SMEM && source == 0)
                width = in.opcode >= 8u ? 4u : 2u;
            else if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                     in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3 ||
                     in.fmt == Rdna2Format::VOPC) width = 2;
            if (overlaps(state.live, operand.value, width)) return false;
        }
        uint16_t live = state.live;
        for_each_scalar_write(in, [&](int first, uint32_t width) {
            for (uint32_t word = 0; word < 16; ++word)
                if (first <= base + static_cast<int>(word) &&
                    base + static_cast<int>(word) < first + static_cast<int>(width))
                    live &= static_cast<uint16_t>(~(1u << word));
        }, /*wave32_one_word_masks*/wave_size == 32);
        if (!live) continue;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX) return false;
            const auto found = by_pc.find(static_cast<uint32_t>(target));
            if (found == by_pc.end()) return false;
            pending.push_back({found->second, live});
            if (in.opcode == kSoppOpcodeBranch) continue;
        } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                   in.opcode != kSoppOpcodeBarrier) return false;
        if (state.index + 1 >= ins.size()) return false;
        pending.push_back({state.index + 1, live});
    }
    if (!consumed[0] || !consumed[1]) return false;
    // A valid descriptor lifetime after the load says nothing about a predecessor that skips
    // that load. Prove every admitted image use is unreachable from entry with this producer
    // removed. The executor normally makes the same check while constructing exact-PC resources,
    // but the recompiler must also be safe with an independently supplied resource table.
    std::vector<size_t> entry_pending{0};
    std::vector<uint8_t> entry_seen(ins.size());
    while (!entry_pending.empty()) {
        const size_t index = entry_pending.back();
        entry_pending.pop_back();
        if (index == start || entry_seen[index]) continue;
        entry_seen[index] = 1;
        if (consumer_indices.contains(index)) return false;
        const Rdna2Inst& in = ins[index];
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX) return false;
            const auto found = by_pc.find(static_cast<uint32_t>(target));
            if (found == by_pc.end()) return false;
            entry_pending.push_back(found->second);
            if (in.opcode == kSoppOpcodeBranch) continue;
        }
        if (index + 1 < ins.size()) entry_pending.push_back(index + 1);
    }
    return true;
}

} // namespace

std::unordered_set<uint32_t> proven_smem_x16_descriptor_loads(
        const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt,
        uint32_t wave_size) {
    std::unordered_set<uint32_t> proven;
    if (!rt || ins.empty()) return proven;

    // Alternate entries would require path-sensitive lifetime/provenance joins. Keep this first
    // admission linear: hints, waits and barriers are transparent; every real scalar branch or
    // indirect PC transfer makes the whole candidate ineligible.
    bool branched = false;
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            return proven;
        if (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16u) return proven;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode))
            branched = true;
        else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) && in.opcode != 0x0au)
            return proven;
    }

    if (branched) {
        std::unordered_map<uint32_t, size_t> by_pc;
        for (size_t i = 0; i < ins.size(); ++i)
            if (ins[i].fmt == Rdna2Format::Unknown || !ins[i].len_dwords ||
                !by_pc.emplace(ins[i].pc, i).second) return proven;
        for (size_t i = 0; i < ins.size(); ++i) {
            const Rdna2Inst& load = ins[i];
            if (load.fmt != Rdna2Format::SMEM || load.opcode != 0x04u ||
                load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
                load.dst.value + 15 > 105 || load.src[1].kind != OperandKind::Special ||
                load.src[1].value != 125 || static_cast<int32_t>(load.literal) < 0)
                continue;
            if (smem_x16_branch_lifetime_is_descriptors(ins, by_pc, i, *rt, wave_size))
                proven.insert(load.pc);
        }
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
        // The branch-free admission uses the same dispatch-time descriptor snapshot as the
        // branched proof. A preceding shader store can make that snapshot stale at this load.
        bool prior_guest_write = false;
        for (size_t i = 0; i < load_index; ++i)
            prior_guest_write |= rdna2_may_write_guest_memory(ins[i]);
        if (prior_guest_write) continue;

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
                        in.fmt == Rdna2Format::VOP3 ||
                        in.fmt == Rdna2Format::VOPC ? 2u : 1u;
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
            // A VOPC mask replaces both scalar words in Wave64 and only the low word in
            // Wave32. Keep a surviving Wave32 high word live for later observations.
            for_each_scalar_write(in, clear_written,
                                  /*wave32_one_word_masks*/wave_size == 32);
        }
        if (valid && consumed[0] && consumed[1]) proven.insert(load.pc);
    }
    return proven;
}

} // namespace prosper::gpu
