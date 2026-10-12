#include "gpu/recompiler/rdna2_barrier_mask_halves.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace prosper::gpu {
NativeBarrierHalfProof::NativeBarrierHalfProof(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                               const uint32_t* code, size_t dwords, bool guarded,
                                               uint32_t initial_active)
    : builder_(&b), launch_(b.guest_compute_launch) {
    if (guarded || initial_active || b.invocation_guard_merge ||
        !saved_mask_physical_launch_matches_original(b) || !code ||
        dwords != b.saved_mask_original_words.size() ||
        !std::equal(b.saved_mask_original_words.begin(), b.saved_mask_original_words.end(), code))
        return;
    std::vector<Rdna2Inst> original;
    if (!rdna2_walk(code, dwords, original) || ins != original) return;
    words_.assign(code, code + dwords);
    participant_pcs_ = b.saved_mask_entry_prefix_pcs;
    std::vector<Rdna2Inst> phase;
    bool found_barrier = false;
    for (const auto& in : original) {
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x0au) {
            phase.push_back(in);
            continue;
        }
        found_barrier = true;
        if (phase.empty()) continue;
        Rdna2Inst end;
        end.pc = in.pc;
        end.fmt = Rdna2Format::SOPP;
        end.opcode = 0x01u;
        end.len_dwords = 1;
        end.is_end = true;
        end.synthetic_terminator = true;
        phase.push_back(end);
        phases_.push_back(std::move(phase));
        phase.clear();
    }
    if (!phase.empty()) phases_.push_back(std::move(phase));
    valid_ = found_barrier && !participant_pcs_.empty();
}

bool NativeBarrierHalfProof::begin_phase(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                         const uint32_t* code, size_t dwords,
                                         uint32_t initial_active) {
    // CONFIDENCE: HIGH -- complete physical launch and original branch-free entry PCs are
    // independent obligations. A required subgroup width or a matching reload PC supplies neither.
    if (!valid_ || phase_open_ || builder_ != &b || initial_active || b.invocation_guard_merge ||
        !native_compute_has_complete_entry(b) || !native_vcc_copy_execution_domain_matches(b) ||
        b.guest_compute_launch != launch_ || !code || dwords != words_.size() ||
        !std::equal(words_.begin(), words_.end(), code) || b.saved_mask_original_words != words_ ||
        next_phase_ >= phases_.size() || ins != phases_[next_phase_]) {
        valid_ = false;
        slots_.clear();
        return false;
    }
    phase_open_ = true;
    ++next_phase_;
    // Beyond the immutable entry prefix a host dispatcher route may contain only some lanes.
    // Already materialized scalar DATA remains usable; no newly imported late ballot follows.
    if (!participant_pcs_.contains(ins.front().pc)) slots_.clear();
    return true;
}

void NativeBarrierHalfProof::seed_entry(SpilledMaskHalves& into, SpillSlotDomains& domains) const {
    if (!valid_ || !phase_open_) return;
    // Preserve MAY poison independently of the imported half facts. Copying a storage slot cannot
    // erase a relay/memory mark from the preceding phase or the current entry state.
    into.relayed_words.insert(marks_.relayed_words.begin(), marks_.relayed_words.end());
    into.lane_written_vgprs.insert(marks_.lane_written_vgprs.begin(),
                                   marks_.lane_written_vgprs.end());
    into.relayed_vectors.insert(marks_.relayed_vectors.begin(), marks_.relayed_vectors.end());
    into.memory_vectors.insert(marks_.memory_vectors.begin(), marks_.memory_vectors.end());
    into.memory_words.insert(marks_.memory_words.begin(), marks_.memory_words.end());
    into.memory_slots.insert(marks_.memory_slots.begin(), marks_.memory_slots.end());
    // A genuine numeric spill must accompany its MAY owner with the terminal MUST definition
    // and current domain. Otherwise importing the owner alone newly poisons its scalar/SCC use.
    for (const auto& [slot, domain] : domains_) {
        into.defined_slots.insert(slot);
        domains[slot] = domain;
    }
    for (const auto& [slot, half] : slots_) {
        into.defined_slots.insert(slot);
        domains[slot] = half ? SpillSlotDomain::MaskHi : SpillSlotDomain::MaskLo;
    }
}

void NativeBarrierHalfProof::finish_phase(const std::vector<Rdna2Inst>& ins,
                                          const PhysicalMaskHalfSlots& halves,
                                          const SpilledMaskHalves& marks,
                                          const SpillSlotDomains& domains) {
    if (!valid_ || !phase_open_) return;
    phase_open_ = false;
    slots_.clear();
    domains_.clear();
    // A phase with any original control boundary cannot export participation authority. Its
    // terminal MUST facts still protect local reads, but no later phase may turn them into ballots.
    if (!std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
            return in.synthetic_terminator || participant_pcs_.contains(in.pc);
        })) {
        valid_ = false;
        return;
    }
    marks_ = marks;
    for (const auto& slot : marks.defined_slots) {
        const auto domain = domains.find(slot);
        if (domain != domains.end() &&
            (domain->second == SpillSlotDomain::Data || is_mask_domain(domain->second)))
            domains_.emplace(slot, domain->second);
    }
    for (const auto& [slot, half] : halves) {
        const auto domain = domains.find(slot);
        if (half <= 1 && marks.defined_slots.contains(slot) && !marks.memory_slots.contains(slot) &&
            domain != domains.end() &&
            (domain->second == SpillSlotDomain::MaskAny ||
             domain->second == (half ? SpillSlotDomain::MaskHi : SpillSlotDomain::MaskLo)))
            slots_.emplace(slot, half);
    }
}
}   // namespace prosper::gpu
