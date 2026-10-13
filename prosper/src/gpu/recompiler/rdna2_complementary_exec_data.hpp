// A complete word under A and its exact complement P&~A is genuine under P.
// Relations name current architectural mask values; no PC, SSA ID or launch width is authority.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <map>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct ActiveVectorWords;
struct SpilledMaskHalves;

struct ComplementaryExecData {
    std::set<int> current_parents;
    std::map<int, std::set<int>> parents;
    int parent = -1;
    std::set<int> first_words;
    bool operator==(const ComplementaryExecData&) const = default;
};

void meet_complementary_exec_data(ComplementaryExecData& into,
                                  const ComplementaryExecData& incoming);
void advance_complementary_exec_data(ActiveVectorWords& state, const Rdna2Inst& in,
                                     const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                     int mask_write);
void publish_complementary_exec_word(ActiveVectorWords& state, int word);
SpilledMaskHalves active_vector_edge_state(const SpilledMaskHalves& outgoing,
                                           std::span<const Rdna2Inst> instructions, uint32_t lo,
                                           uint32_t hi, uint32_t next,
                                           const std::set<int>& vector_words);
} // namespace prosper::gpu
