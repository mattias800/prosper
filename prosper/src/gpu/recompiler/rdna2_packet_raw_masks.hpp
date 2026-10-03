#pragma once
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
// Ordinary saved masks may have both a lane-predicate view and two physical DATA words. Only the
// owned packet dispatcher proves full logical64 participation; native subgroup ballots are separate.
struct PacketRawMasks {
    std::set<int> roots;
    std::map<uint32_t, int> sites;
    uint32_t pending_var = 0, mask_var = 0, event_var = 0, dst_var = 0, deferred_var = 0;
    uint32_t result_base = 0;
    PacketRawMasks(const SpirvCompute&, const std::vector<Rdna2Inst>&);
    bool contains(uint32_t pc) const { return sites.contains(pc); }
    void begin(SpirvCompute&, uint32_t result_base);
    void reset(SpirvCompute&);
    // Called AFTER the actual writer and lifetime bookkeeping, never from supplied entry words.
    bool stage(SpirvCompute&, const RegState&, const Rdna2Inst&, bool deferred = false);
    void expire(RegState&, const Rdna2Inst&) const;
    void phase(SpirvCompute&, const std::map<int, uint32_t>& scalar_vars,
               const std::map<int, uint32_t>& mask_vars);
};
// Existing portable FFBH phase extracted without changing its native/legacy instruction stream.
bool emit_cfg_mask_ffbh_phase(SpirvCompute&, uint32_t pending_var, uint32_t mask_var,
                              uint32_t write_var, uint32_t event_var, uint32_t half_var,
                              uint32_t dst_var, uint32_t lane, uint32_t wave_index,
                              uint32_t result_base, const std::set<int>& destinations,
                              const std::map<int, uint32_t>& vector_vars,
                              const std::map<std::pair<int, int>, uint32_t>& spill_vars,
                              const std::map<std::pair<int, int>, uint32_t>& spill_mask_vars);
}   // namespace prosper::gpu
