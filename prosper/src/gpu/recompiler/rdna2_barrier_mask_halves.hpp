// A private handoff owned by the actual, unguarded native-compute barrier shell. Persisted
// dispatcher variables are storage, never evidence of a current physical EXEC half.
#pragma once

#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_spill_slot_domain.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct ShaderResourceTable;
using PhysicalMaskHalfSlots = std::map<std::pair<int, int>, uint32_t>;

class NativeBarrierHalfProof {
    friend bool emit_body(SpirvCompute&, RegState&, const std::vector<Rdna2Inst>&,
                          const std::unordered_set<uint32_t>&, const ShaderResourceTable*, bool,
                          bool, const std::function<bool(RegState&, const Rdna2Inst&)>&,
                          const uint32_t*, size_t, const std::unordered_set<uint32_t>*, bool,
                          uint32_t, bool, bool);

    SpirvCompute* builder_;
    ComputeDispatchLaunchProfile launch_;
    std::vector<uint32_t> words_;
    std::vector<std::vector<Rdna2Inst>> phases_;
    std::set<uint32_t> participant_pcs_;
    PhysicalMaskHalfSlots slots_;
    SpillSlotDomains domains_;
    SpilledMaskHalves marks_;
    size_t next_phase_ = 0;
    bool valid_ = false, phase_open_ = false;

    NativeBarrierHalfProof(SpirvCompute&, const std::vector<Rdna2Inst>&, const uint32_t*, size_t,
                           bool guarded, uint32_t initial_active);
    NativeBarrierHalfProof(const NativeBarrierHalfProof&) = delete;
    NativeBarrierHalfProof& operator=(const NativeBarrierHalfProof&) = delete;

public:
    // Each exact phase can consume one permit, in original order. Initialization still revokes
    // the builder's broad physical proof; this object grants only the typed spill handoff.
    bool begin_phase(SpirvCompute&, const std::vector<Rdna2Inst>&, const uint32_t*, size_t,
                     uint32_t initial_active);
    void finish_phase(const std::vector<Rdna2Inst>&, const PhysicalMaskHalfSlots&,
                      const SpilledMaskHalves&, const SpillSlotDomains&);
    const PhysicalMaskHalfSlots& slots() const { return slots_; }
    const std::set<uint32_t>& participant_pcs() const { return participant_pcs_; }
    void seed_entry(SpilledMaskHalves&, SpillSlotDomains&) const;
};
} // namespace prosper::gpu
