// Whole-stream CFG use queries. These preserve the emitter's existing proofs; they do not
// initialize data, grant resource authority, or change dispatcher/structured route selection.
#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>
namespace prosper::gpu {
struct Rdna2Inst;
struct ShaderResourceTable;
struct RegState;
std::unordered_set<uint32_t>
proven_smem_x2_descriptor_fragment_loads(const std::vector<Rdna2Inst>& ins,
                                         const ShaderResourceTable* rt, uint32_t wave_size);
bool has_unpersisted_b32_mask_lifetime(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                                       const RegState& entry);
int entry_m0_save_in_range(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi);
} // namespace prosper::gpu
