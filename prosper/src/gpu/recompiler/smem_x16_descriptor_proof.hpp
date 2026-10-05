// smem_x16_descriptor_proof.hpp -- see smem_x16_descriptor_proof.cpp.
#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

struct Rdna2Inst;
struct ShaderResourceTable;

// Load PCs of the s_load_dwordx16 descriptor bundles the whole-stream proof admits: two adjacent T#s
// loaded at once and consumed only as image descriptors. Empty without a resource table.
std::unordered_set<uint32_t> proven_smem_x16_descriptor_loads(
    const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt, uint32_t wave_size);

} // namespace prosper::gpu
