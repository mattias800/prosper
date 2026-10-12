// Independent B32/B64 mask-domain facts at Wave32 dispatcher block entries.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <unordered_set>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Wave32DispatchWords {
    std::vector<std::set<int>> masks, ambiguous, pairs, ambiguous_pairs;
    std::vector<bool> reachable;
    explicit Wave32DispatchWords(std::size_t count);
};
bool analyze_wave32_dispatch_words(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const RegState& initial,
    const std::vector<uint32_t>& starts, const std::vector<std::vector<uint32_t>>& successors,
    bool proven_wave32_masks, bool compute_scalar_vcc_bridge, Wave32DispatchWords& words,
    std::set<int>& static_mask_keys,
    std::unordered_set<uint32_t>& proven_saved_mask_pair_compare_pcs,
    const std::function<std::array<int, 2>(const Rdna2Inst&)>& saved_mask_pair_compare_sources);
}   // namespace prosper::gpu
