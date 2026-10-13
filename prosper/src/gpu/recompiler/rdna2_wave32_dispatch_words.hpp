// Current numeric word provenance beside the independent Wave32 mask-domain facts.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
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
    std::vector<std::set<int>> masks, ambiguous, pairs, ambiguous_pairs, scalar_words;
    std::vector<SpilledMaskHalves> marks;
    std::vector<bool> reachable, scalar_scc;
    std::unordered_set<uint32_t> scalar_scc_branches;
    explicit Wave32DispatchWords(std::size_t count);
};
bool analyze_wave32_dispatch_words(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const RegState& initial,
    const std::vector<uint32_t>& starts, const std::vector<std::vector<uint32_t>>& successors,
    const std::set<int>& direct_descriptor_sregs, Wave32DispatchWords& words,
    std::set<int>& static_mask_keys, std::unordered_set<uint32_t>& proven_pair_compares,
    const std::function<std::array<int, 2>(const Rdna2Inst&)>& pair_compare_sources,
    const std::unordered_set<uint32_t>& native_mask_scc_vote_pcs);
void restore_wave32_dispatch_words(const Wave32DispatchWords& words, uint32_t block,
                                   RegState& state);
} // namespace prosper::gpu
